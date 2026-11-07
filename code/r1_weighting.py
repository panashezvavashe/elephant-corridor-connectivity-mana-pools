"""Reviewer 2 #4: does unequal sampling among individuals (hourly / long-running collars contribute
more presences) drive the models?

Refits both classifiers with presence weights that give every individual the same total weight
(pseudo-absences keep weight 1, so the class balance is unchanged) and compares with the
unweighted models: holdout AUC (also an individual-balanced AUC), LOIO AUC per collar, and
agreement of predicted suitability and binary corridors on 200,000 random cells of the study area.
Outputs: tables/T22_weighting_sensitivity.csv, T22b_weighting_loio_by_collar.csv
"""
import r1_config as C  # noqa: F401  (sets PROJ paths and thread caps first)

C.lower_priority()

import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import r1_stats as S

pts = gpd.read_file(C.OUT_DATA + '/model_points.gpkg')
for c in C.LULC_CLASSES:
    pts[f'LULC_{c}'] = (pts['LandCover'] == c).astype(int)
X = pts[C.FEATURES].astype(float)
y = pts['class'].astype(int).values
collar = pts['collar'].values
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=C.SEED, stratify=y)
settings = json.load(open(C.OUT_DATA + '/model_settings.json'))
thr = settings['thresholds']
base = {m: joblib.load(f'{C.OUT_DATA}/{m.lower()}_model.joblib') for m in ('RF', 'SVM')}


def weights(idx):
    """Equal total weight per individual among presences; mean presence weight = 1."""
    w = np.ones(len(idx))
    pres = y[idx] == 1
    ids, counts = np.unique(collar[idx][pres], return_counts=True)
    per = dict(zip(ids, pres.sum() / (len(ids) * counts)))
    w[pres] = [per[c] for c in collar[idx][pres]]
    return w


def fit(m, idx, weighted):
    mdl = clone(base[m])
    kw = {'classifier__sample_weight': weights(idx)} if weighted else {}
    return mdl.fit(X.iloc[idx], y[idx], **kw)


# ── 200,000 random cells inside the study polygon ─────────────────────────────
rng = np.random.default_rng(C.SEED)
study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
with rasterio.open(C.PREDICTOR_TIF) as src:
    inside = geometry_mask(study.geometry, out_shape=(src.height, src.width), transform=src.transform, invert=True)
    rr, cc = np.nonzero(inside)
    pick = rng.choice(len(rr), 200_000, replace=False)
    rr, cc = rr[pick], cc[pick]
    order = np.lexsort((cc, rr))
    rr, cc = rr[order], cc[order]
    bands = {}
    for i, b in enumerate(C.BAND_NAMES):
        bands[b] = src.read(i + 1)[rr, cc].astype(float)
cells = pd.DataFrame(bands)
for c in C.LULC_CLASSES:
    cells[f'LULC_{c}'] = (cells['LandCover'] == c).astype(int)
cells = cells[C.FEATURES].dropna()

rows, loio_rows = [], []
w_te = weights(idx_te)
for m in ('RF', 'SVM'):
    fitted = {'unweighted': base[m], 'equal weight per individual': fit(m, idx_tr, True)}
    p_cells = {}
    for k, mdl in fitted.items():
        p = mdl.predict_proba(X.iloc[idx_te])[:, 1]
        p_cells[k] = mdl.predict_proba(cells)[:, 1]
        rows.append({'Model': m, 'Fit': k, 'holdout_AUC': roc_auc_score(y[idx_te], p),
                     'holdout_AUC_individual_balanced': roc_auc_score(y[idx_te], p, sample_weight=w_te),
                     'holdout_TSS_at_nested_threshold': S.tss(y[idx_te], (p >= thr[m]).astype(int)),
                     'corridor_share_of_study_area': np.mean(p_cells[k] >= thr[m])})
    a, b = p_cells['unweighted'], p_cells['equal weight per individual']
    rows[-1].update({'Spearman_suitability_vs_unweighted': stats.spearmanr(a, b).statistic,
                     'Jaccard_corridor_vs_unweighted': np.sum((a >= thr[m]) & (b >= thr[m]))
                     / np.sum((a >= thr[m]) | (b >= thr[m]))})
    # LOIO with and without weights (same folds as r1_models.py: one collar + a tenth of absences)
    absn = np.where(y == 0)[0]
    shuf = np.random.default_rng(C.SEED).permutation(absn)
    for i, cid in enumerate(C.IDS):
        te = np.concatenate([np.where(collar == cid)[0], np.array_split(shuf, len(C.IDS))[i]])
        tr = np.setdiff1d(np.arange(len(y)), te)
        r = {'Model': m, 'collar': cid}
        for k, wtd in (('AUC_unweighted', False), ('AUC_equal_weight', True)):
            r[k] = roc_auc_score(y[te], fit(m, tr, wtd).predict_proba(X.iloc[te])[:, 1])
        loio_rows.append(r)
    print('done', m, flush=True)

df = pd.DataFrame(rows)
pres = pd.Series(collar[y == 1]).value_counts().sort_index()
df['presences_per_individual'] = f'{pres.min()}–{pres.max()}'
north = [3905, 3906, 3907, 4141]  # floodplain cores (touching cores, Section 2.9)
df['floodplain_collars_share_of_presences'] = pres.loc[north].sum() / pres.sum()
df.round(4).to_csv(f'{C.OUT_TAB}/T22_weighting_sensitivity.csv', index=False)
lo = pd.DataFrame(loio_rows)
lo.round(4).to_csv(f'{C.OUT_TAB}/T22b_weighting_loio_by_collar.csv', index=False)
print(df.round(3).T.to_string())
print(lo.round(3).to_string())
