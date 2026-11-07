#!/usr/bin/env bash
# Runs the full analysis in order, one heavy job at a time with a cool-down pause between jobs.
# Each step is skipped when its main output already exists, so the script is safe to re-run.
#
# Steps marked [GPS] need the collar fixes (data/gps/GPS_points.shp, not distributed).
# Without them the script starts from the fitted models in data/derived/ and runs the
# prediction, connectivity, network and corridor analyses.
#
# Usage:  bash code/run_all.sh            (python on PATH)
#         PY=/path/to/python bash code/run_all.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
COOL=${COOL:-120}
OUT=../results
export PYTHONIOENCODING=utf-8
cool() { echo "[$(date +%H:%M)] cooling down ${COOL} s"; sleep "$COOL"; }
step() { echo; echo "[$(date +%H:%M)] ===== $* ====="; }
GPS=0; [ -f ../data/gps/GPS_points.shp ] && GPS=1
WIN="88020 8230050 133020 8266050"   # 45 x 36 km floodplain-escarpment window (resolution test)

step "stack predictor bands"
$PY -u build_predictor_stack.py

if [ $GPS = 1 ]; then
  step "[GPS] presences, pseudo-absences, validation pool"
  [ -f ../data/derived/model_points.gpkg ] || $PY -u r1_data.py
  step "[GPS] hyper-parameter grids"
  [ -f "$OUT/tables/S16_grid_summary.csv" ] || $PY -u r1_grid.py
  cool
  step "[GPS] model fitting, validation schemes, importance, ALE"
  [ -f "$OUT/tables/T10_collar_profiles.csv" ] || $PY -u r1_models.py
  cool
else
  echo "No GPS fixes found: using the fitted models in data/derived/ and skipping [GPS] steps."
fi

step "probability, resistance and binary corridor rasters"
[ -f "$OUT/rasters/svm_binary_corridor.tif" ] || $PY -u r1_predict.py
cool

for M in RF SVM; do
  step "Circuitscape $M 90 m"
  [ -f "$OUT/cs/${M}_90m/cum_current.tif" ] || $PY -u r1_circuit.py $M
  cool
done

for M in RF SVM; do
  step "network $M"
  D=$OUT/cs/${M}_90m
  [ -f "$OUT/tables/network_new_$M/N9_node_removal_merged_touching_cores.csv" ] || \
    $PY -u r1_network.py new_$M "$D" "$D/resistance.asc" "$D/focal_nodes.asc" "$D/${M}_90m_resistances_3columns.out"
  cool
done

step "corridor agreement, current agreement, pinch points, context"
$PY -u r1_corridors.py agreement current pinch context

if [ $GPS = 1 ]; then
  step "[GPS] corridor occupancy and step-selection validation"
  $PY -u r1_validation.py
  cool
fi

for M in RF SVM; do
  step "Circuitscape $M clipped to the study polygon (extent test)"
  [ -f "$OUT/cs/${M}_90m_studyclip/cum_current.tif" ] || $PY -u r1_circuit.py $M --clip-to-study
  cool
done

for M in RF SVM; do
  for F in 1 3; do
    L=${M}_win$((30 * F))m
    step "Circuitscape $L (resolution test)"
    [ -f "$OUT/cs/$L/cum_current.tif" ] || $PY -u r1_circuit.py $M --factor $F --window $WIN --label $L
    cool
  done
done

step "extent and resolution sensitivity"
$PY -u r1_corridors.py extent resolution
cool

step "non-linear resistance transforms"
[ -f "$OUT/tables/T21_transform_sensitivity.csv" ] || $PY -u r1_transform.py make cs compare

if [ $GPS = 1 ]; then
  cool
  step "[GPS] equal weight per individual"
  [ -f "$OUT/tables/T22_weighting_sensitivity.csv" ] || $PY -u r1_weighting.py
fi
echo "[$(date +%H:%M)] ALL STEPS COMPLETE"
