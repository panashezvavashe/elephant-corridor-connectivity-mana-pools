"""Step 1: rebuild the presence / pseudo-absence / validation datasets without collar 4557.

Follows the sampling design in manuscript §2.2-2.3 exactly, so only the collar set changes:
  - 5000 presences by simple random subsampling of the ten collars' fixes (seed 42)
  - 5000 pseudo-absences at random valid 30 m cells inside the study polygon,
    >= 500 m from every presence (seed 42)
  - every fix not drawn as a presence is kept as the validation pool, with its
    collar ID and timestamp (needed for the individual-level bootstrap, #3).
Unlike rf.py, collar IDs are carried straight from GPS_points.shp, so no
nearest-neighbour recovery is needed.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree

import r1_config as C


def extract(gdf, tif=C.PREDICTOR_TIF):
    with rasterio.open(tif) as src:
        vals = np.array(list(src.sample(zip(gdf.geometry.x, gdf.geometry.y))))
    df = pd.DataFrame(vals, columns=C.BAND_NAMES, index=gdf.index)
    return pd.concat([gdf, df], axis=1)


def main():
    gps = gpd.read_file(C.GPS_SHP).to_crs(C.CRS)
    n_raw = len(gps)
    gps = gps[~gps['elephant_i'].isin(C.EXCLUDED_IDS)]
    n_after_4557 = len(gps)
    gps = gps.drop_duplicates(['elephant_i', 'timestamp']).reset_index(drop=True)
    gps['timestamp'] = pd.to_datetime(gps['timestamp'], format='mixed')
    gps = gps.rename(columns={'elephant_i': 'collar'})[['collar', 'timestamp', 'geometry']]
    assert sorted(gps['collar'].unique()) == C.IDS

    rng = np.random.default_rng(C.SEED)
    pres_idx = rng.choice(len(gps), C.N_PRESENCE, replace=False)
    is_pres = np.zeros(len(gps), bool)
    is_pres[pres_idx] = True
    pres = gps[is_pres].copy()
    pool = gps[~is_pres].copy()

    # Pseudo-absences: valid cells inside the study polygon, >= 500 m from every presence
    study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
    with rasterio.open(C.PREDICTOR_TIF) as src:
        inside = geometry_mask(study.geometry, out_shape=src.shape,
                               transform=src.transform, invert=True)
        rows, cols = np.nonzero(inside)
        cand = rng.choice(len(rows), 60000, replace=False)
        xs, ys = rasterio.transform.xy(src.transform, rows[cand], cols[cand])
    xs, ys = np.asarray(xs), np.asarray(ys)
    d, _ = cKDTree(np.c_[pres.geometry.x, pres.geometry.y]).query(np.c_[xs, ys])
    keep = d >= C.ABSENCE_EXCLUSION_M
    absn = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs[keep], ys[keep]), crs=C.CRS)
    absn = extract(absn)
    absn = absn.dropna(subset=C.BAND_NAMES)
    absn = absn[(absn[C.BAND_NAMES] != -9999).all(axis=1)].iloc[:C.N_ABSENCE]
    assert len(absn) == C.N_ABSENCE
    absn['collar'] = -1
    absn['class'] = 0

    pres = extract(pres)
    pres['class'] = 1
    assert pres[C.BAND_NAMES].notna().all().all()

    pool = extract(pool)
    pool['class'] = 1

    pts = pd.concat([pres.drop(columns='timestamp'), absn], ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry='geometry', crs=C.CRS)
    pts['x'], pts['y'] = pts.geometry.x, pts.geometry.y
    pts.to_file(C.OUT_DATA + '/model_points.gpkg', driver='GPKG')
    pres.to_file(C.OUT_DATA + '/presences.gpkg', driver='GPKG')
    pool.to_file(C.OUT_DATA + '/validation_pool.gpkg', driver='GPKG')

    summary = pd.DataFrame({
        'item': ['raw fixes (11 collars)', 'after removing 4557', 'after removing duplicate timestamps',
                 'presences drawn', 'validation pool', 'pseudo-absence candidates >= 500 m',
                 'pseudo-absences kept', 'date min', 'date max'],
        'value': [n_raw, n_after_4557, len(gps), len(pres), len(pool), int(keep.sum()),
                  len(absn), gps['timestamp'].min(), gps['timestamp'].max()]})
    summary.to_csv(C.OUT_TAB + '/S0_data_summary.csv', index=False)
    per = gps.groupby('collar').agg(fixes=('timestamp', 'size'), start=('timestamp', 'min'),
                                    end=('timestamp', 'max'))
    per['presences'] = pres.groupby('collar').size()
    per['sex'] = per.index.map(C.SEX)
    per.to_csv(C.OUT_TAB + '/S0_fixes_per_collar.csv')
    print(summary.to_string(index=False))
    print(per)


if __name__ == '__main__':
    main()
