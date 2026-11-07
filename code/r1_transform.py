"""Reviewer 2 #1: sensitivity of RF–SVM connectivity agreement to the resistance transform.

Negative-exponential transforms (Keeley et al., 2016) of predicted suitability P (0–1):
    R = 100 − 99 · (1 − exp(−c·P)) / (1 − exp(−c)),   c = 4 and 8
giving resistance 1–100 with much higher contrast than the linear 1 − P. Each 30 m surface is
block-averaged to 90 m exactly as in the main runs (same grid, same focal cores).

Usage: python r1_transform.py [make] [cs] [compare]
Outputs: rasters/{rf,svm}_resistance_negexp{c}.tif, cs/{RF,SVM}_90m_negexp{c}/, tables/T21_transform_sensitivity.csv
"""
import r1_config as C  # noqa: F401  (sets PROJ paths and thread caps first)

C.lower_priority()

import os
import subprocess
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy import ndimage, stats

CS_DIR = C.OUT + '/cs'
MODELS = ('RF', 'SVM')
CS_ = (4, 8)
SURFACES = {'linear (1 − P)': ''} | {f'negative exponential, c = {c}': f'_negexp{c}' for c in CS_}
STRIP = 300


def negexp(p, c):
    return 100 - 99 * (1 - np.exp(-c * p)) / (1 - np.exp(-c))


def make():
    for m in MODELS:
        with rasterio.open(f'{C.OUT_RAS}/{m.lower()}_probability.tif') as src:
            prof = src.profile | {'dtype': 'float32', 'nodata': np.nan, 'compress': 'deflate'}
            outs = {c: rasterio.open(f'{C.OUT_RAS}/{m.lower()}_resistance_negexp{c}.tif', 'w', **prof) for c in CS_}
            for r0 in range(0, src.height, STRIP):
                win = rasterio.windows.Window(0, r0, src.width, min(STRIP, src.height - r0))
                p = src.read(1, window=win, masked=True).filled(np.nan).astype(float)
                for c, d in outs.items():
                    d.write(negexp(p, c).astype('float32'), 1, window=win)
            for d in outs.values():
                d.close()
        print('made', m, flush=True)


def cs():
    py = sys.executable
    for c in CS_:
        for m in MODELS:
            label = f'{m}_90m_negexp{c}'
            if os.path.exists(f'{CS_DIR}/{label}/cum_current.tif'):
                continue
            subprocess.run([py, '-u', 'r1_circuit.py', m, '--surface', f'_negexp{c}', '--label', label], check=True)
            print(f'[{time.strftime("%H:%M")}] cooling down 120 s', flush=True)
            time.sleep(120)


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(float)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform


def current(label):
    cur, t = read(f'{CS_DIR}/{label}/cum_current.tif')
    nodes, _ = read(f'{CS_DIR}/{label}/focal_nodes.tif')
    cur[np.nan_to_num(nodes) > 0] = np.nan
    return cur, t


def jaccard(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else np.nan


def pinch(cur, valid, pct, win=5):
    c0 = np.where(valid, cur, 0.0)
    n = ndimage.uniform_filter(valid.astype(float), win, mode='constant')
    s = ndimage.uniform_filter(c0, win, mode='constant')
    mean = np.where(n > 0, s / np.maximum(n, 1e-12), np.nan)
    return valid & (cur > np.nanpercentile(cur[valid], pct)) & (cur > mean)


def compare():
    study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
    cur = {(m, k): current(f'{m}_90m{sfx}') for k, sfx in SURFACES.items() for m in MODELS}
    t = cur[('RF', 'linear (1 − P)')][1]
    shape = cur[('RF', 'linear (1 − P)')][0].shape
    sm = geometry_mask(study.geometry, out_shape=shape, transform=t, invert=True)
    rows = []
    for k, sfx in SURFACES.items():
        a, b = cur[('RF', k)][0], cur[('SVM', k)][0]
        ok = sm & np.isfinite(a) & np.isfinite(b)
        r = {'Resistance transform': k, 'Comparison': 'RF vs SVM',
             'Spearman_rho': stats.spearmanr(a[ok], b[ok]).statistic,
             'Pearson_r_log10': stats.pearsonr(np.log10(a[ok] + 1e-12), np.log10(b[ok] + 1e-12)).statistic}
        for q in (90, 80):
            r[f'Jaccard_top{100 - q}pct'] = jaccard(ok & (a >= np.percentile(a[ok], q)),
                                                   ok & (b >= np.percentile(b[ok], q)))
        r['pinch_Jaccard_median_5x5'] = jaccard(pinch(a, ok, 50), pinch(b, ok, 50))
        r['pinch_Jaccard_top10_5x5'] = jaccard(pinch(a, ok, 90), pinch(b, ok, 90))
        for m, arr in (('RF', a), ('SVM', b)):
            res, _ = read(f'{CS_DIR}/{m}_90m{sfx}/resistance.tif')
            v = res[sm & np.isfinite(res) & (res > -9998)]
            r[f'{m}_resistance_p90_p10_ratio'] = np.percentile(v, 90) / np.percentile(v, 10)
            r[f'{m}_current_top10pct_share_of_total'] = np.sort(arr[ok])[::-1][:int(ok.sum() * 0.1)].sum() / arr[ok].sum()
        rows.append(r)
    for k, sfx in list(SURFACES.items())[1:]:  # each transform against the linear surface, same classifier
        for m in MODELS:
            a, b = cur[(m, 'linear (1 − P)')][0], cur[(m, k)][0]
            ok = sm & np.isfinite(a) & np.isfinite(b)
            rows.append({'Resistance transform': k, 'Comparison': f'{m}: transform vs linear',
                         'Spearman_rho': stats.spearmanr(a[ok], b[ok]).statistic,
                         'Pearson_r_log10': stats.pearsonr(np.log10(a[ok] + 1e-12),
                                                           np.log10(b[ok] + 1e-12)).statistic,
                         'Jaccard_top10pct': jaccard(ok & (a >= np.percentile(a[ok], 90)),
                                                     ok & (b >= np.percentile(b[ok], 90))),
                         'Jaccard_top20pct': jaccard(ok & (a >= np.percentile(a[ok], 80)),
                                                     ok & (b >= np.percentile(b[ok], 80))),
                         'pinch_Jaccard_median_5x5': jaccard(pinch(a, ok, 50), pinch(b, ok, 50)),
                         'pinch_Jaccard_top10_5x5': jaccard(pinch(a, ok, 90), pinch(b, ok, 90))})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f'{C.OUT_TAB}/T21_transform_sensitivity.csv', index=False)
    print(df.round(3).to_string())


if __name__ == '__main__':
    for step in sys.argv[1:] or ['make', 'cs', 'compare']:
        globals()[step]()
