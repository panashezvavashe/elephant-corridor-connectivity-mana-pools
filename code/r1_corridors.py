"""Step 6: corridor and current-flow analyses on the revised outputs.

  agreement   binary corridors (nested MaxTSS) inside the study polygon: areas, Jaccard,
              Dice, kappa, chance expectation, 25 km block-bootstrap CIs (Table 5)
  current     RF vs SVM cumulative current correlation, focal regions excluded
  pinch       #9 pinch-point sensitivity: current thresholds (median, top 25/20/10 %) x
              neighbourhood (3, 5, 7, 9 cells); RF-SVM agreement and stability
  context     #12 consensus / RF-only / SVM-only vs park boundary, water, riparian
              land cover, built-up distance, night-time lights
  extent      #14 full predictor rectangle vs study-polygon-clipped circuit
  resolution  #8  30 m vs 90 m inside a sub-window

Usage: python r1_corridors.py [agreement current pinch context extent resolution]
"""
import glob
import os
import sys

import r1_config as C
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from scipy import ndimage, stats

CS = C.OUT + '/cs'
MODELS = ('RF', 'SVM')
PCTS = {'median (submitted)': 50, 'top 25%': 75, 'top 20%': 80, 'top 10%': 90}
WINDOWS = (3, 5, 7, 9)
REF = ('median (submitted)', 5)


def study_mask(shape, transform):
    study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
    return geometry_mask(study.geometry, out_shape=shape, transform=transform, invert=True)


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(float)
        if s.nodata is not None and not np.isnan(s.nodata):
            a[a == s.nodata] = np.nan
        return a, s.transform, s.profile


def jaccard(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else np.nan


# ── Binary corridor agreement (30 m) ──────────────────────────────────────────
def agreement():
    rf, t, _ = read(f'{C.OUT_RAS}/rf_binary_corridor.tif')
    svm, _, _ = read(f'{C.OUT_RAS}/svm_binary_corridor.tif')
    m = study_mask(rf.shape, t) & np.isfinite(rf) & np.isfinite(svm)
    a, b = rf[m] == 1, svm[m] == 1
    cell = abs(t.a * t.e) / 1e6
    A = m.sum()
    inter, union = np.sum(a & b), np.sum(a | b)
    pa, pb = a.mean(), b.mean()
    po = np.mean(a == b)
    pe = pa * pb + (1 - pa) * (1 - pb)
    exp_inter = pa * pb * A
    rows = {'study_area_km2': A * cell, 'RF_km2': a.sum() * cell, 'RF_pct': 100 * pa,
            'SVM_km2': b.sum() * cell, 'SVM_pct': 100 * pb, 'consensus_km2': inter * cell,
            'consensus_pct': 100 * inter / A, 'RF_only_km2': np.sum(a & ~b) * cell,
            'SVM_only_km2': np.sum(~a & b) * cell, 'consensus_share_of_RF': inter / a.sum(),
            'consensus_share_of_SVM': inter / b.sum(), 'Jaccard': inter / union,
            'Dice': 2 * inter / (a.sum() + b.sum()), 'kappa': (po - pe) / (1 - pe),
            'Jaccard_chance': exp_inter / (a.sum() + b.sum() - exp_inter),
            'Dice_chance': 2 * exp_inter / (a.sum() + b.sum())}
    # 25 km block bootstrap
    rr, cc = np.nonzero(m)
    xs = t.c + (cc + 0.5) * t.a
    ys = t.f + (rr + 0.5) * t.e
    blk = ((xs - xs.min()) // 25000).astype(int) * 1000 + ((ys - ys.min()) // 25000).astype(int)
    ub, inv = np.unique(blk, return_inverse=True)
    I = np.bincount(inv, weights=a & b)
    U = np.bincount(inv, weights=a | b)
    Sa, Sb = np.bincount(inv, weights=a), np.bincount(inv, weights=b)
    rng = np.random.default_rng(C.SEED)
    J, D = [], []
    for _ in range(5000):
        k = rng.integers(0, len(ub), len(ub))
        J.append(I[k].sum() / U[k].sum())
        D.append(2 * I[k].sum() / (Sa[k].sum() + Sb[k].sum()))
    rows['Jaccard_CI'] = f'{np.percentile(J, 2.5):.3f}-{np.percentile(J, 97.5):.3f}'
    rows['Dice_CI'] = f'{np.percentile(D, 2.5):.3f}-{np.percentile(D, 97.5):.3f}'
    rows['n_blocks'] = len(ub)
    out = pd.Series(rows)
    out.to_csv(f'{C.OUT_TAB}/T11_corridor_agreement.csv', header=['value'])
    print(out)
    cls = np.full(rf.shape, np.nan, dtype='float32')  # 0 neither, 1 RF only, 2 SVM only, 3 consensus
    cls[m] = (a & ~b) * 1 + (~a & b) * 2 + (a & b) * 3
    prof = dict(driver='GTiff', height=rf.shape[0], width=rf.shape[1], count=1, dtype='float32', crs=C.CRS,
                transform=t, nodata=np.nan, compress='deflate')
    with rasterio.open(f'{C.OUT_RAS}/corridor_agreement_classes.tif', 'w', **prof) as d:
        d.write(cls, 1)


# ── Current-flow surfaces ─────────────────────────────────────────────────────
def load_current(label):
    cur, t, _ = read(f'{CS}/{label}/cum_current.tif')
    nodes, _, _ = read(f'{CS}/{label}/focal_nodes.tif')
    cur[np.nan_to_num(nodes) > 0] = np.nan  # focal regions excluded
    return cur, t


def current():
    cur = {m: load_current(f'{m}_90m') for m in MODELS}
    t = cur['RF'][1]
    m = study_mask(cur['RF'][0].shape, t) & np.isfinite(cur['RF'][0]) & np.isfinite(cur['SVM'][0])
    a, b = cur['RF'][0][m], cur['SVM'][0][m]
    rows = {'Spearman_rho': stats.spearmanr(a, b).statistic,
            'Pearson_r_log10': stats.pearsonr(np.log10(a + 1e-12), np.log10(b + 1e-12)).statistic,
            'Pearson_r_raw': stats.pearsonr(a, b).statistic}
    for q in (90, 80):
        ta, tb = a >= np.percentile(a, q), b >= np.percentile(b, q)
        rows[f'Jaccard_top{100 - q}pct_current'] = jaccard(ta, tb)
    pd.Series(rows).to_csv(f'{C.OUT_TAB}/T12_current_agreement.csv', header=['value'])
    print(pd.Series(rows))


# ── #9 Pinch-point sensitivity ────────────────────────────────────────────────
def pinch_map(cur, valid, pct, win):
    c0 = np.where(valid, cur, 0.0)
    n = ndimage.uniform_filter(valid.astype(float), win, mode='constant')
    s = ndimage.uniform_filter(c0, win, mode='constant')
    local_mean = np.where(n > 0, s / np.maximum(n, 1e-12), np.nan)
    thr = np.nanpercentile(cur[valid], pct)
    return valid & (cur > thr) & (cur > local_mean)


def pinch():
    cur = {m: load_current(f'{m}_90m') for m in MODELS}
    t = cur['RF'][1]
    sm = study_mask(cur['RF'][0].shape, t)
    valid = {m: sm & np.isfinite(cur[m][0]) for m in MODELS}
    cell = abs(t.a * t.e) / 1e6
    maps, rows = {}, []
    for name, pct in PCTS.items():
        for w in WINDOWS:
            for m in MODELS:
                maps[(m, name, w)] = pinch_map(cur[m][0], valid[m], pct, w)
    for name in PCTS:
        for w in WINDOWS:
            r = {'threshold': name, 'window_cells': w, 'window_m': w * 90}
            for m in MODELS:
                pm = maps[(m, name, w)]
                lab, n = ndimage.label(pm, np.ones((3, 3)))
                r[f'{m}_area_km2'] = pm.sum() * cell
                r[f'{m}_n_patches'] = n
                r[f'{m}_Jaccard_vs_submitted'] = jaccard(pm, maps[(m,) + REF])
                r[f'{m}_share_of_submitted_retained'] = (pm & maps[(m,) + REF]).sum() / maps[(m,) + REF].sum()
            r['RF_SVM_Jaccard'] = jaccard(maps[('RF', name, w)], maps[('SVM', name, w)])
            rows.append(r)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f'{C.OUT_TAB}/T13_pinch_point_sensitivity.csv', index=False)
    print(df.round(3).to_string())
    # Robust pinch points: flagged in >= 75% of the 16 configurations in BOTH models
    freq = {m: sum(maps[(m, n, w)].astype(int) for n in PCTS for w in WINDOWS) / (len(PCTS) * len(WINDOWS))
            for m in MODELS}
    robust = (freq['RF'] >= 0.75) & (freq['SVM'] >= 0.75)
    lab, n = ndimage.label(robust, np.ones((3, 3)))
    sizes = ndimage.sum(robust, lab, range(1, n + 1)) * cell
    cs = ndimage.center_of_mass(robust, lab, range(1, n + 1))
    rp = pd.DataFrame({'patch': range(1, n + 1), 'area_km2': sizes,
                       'x': [t.c + (c[1] + 0.5) * t.a for c in cs], 'y': [t.f + (c[0] + 0.5) * t.e for c in cs],
                       'mean_RF_current': ndimage.mean(cur['RF'][0], lab, range(1, n + 1))})
    rp = rp.sort_values('mean_RF_current', ascending=False)
    rp.round(4).to_csv(f'{C.OUT_TAB}/S9_robust_pinch_patches.csv', index=False)
    pd.Series({'robust_area_km2': robust.sum() * cell, 'robust_patches': n,
               'share_of_submitted_RF_pinch_that_is_robust': (robust & maps[('RF',) + REF]).sum() / maps[('RF',) + REF].sum()}
              ).to_csv(f'{C.OUT_TAB}/S9b_robust_pinch_summary.csv', header=['value'])
    prof = dict(driver='GTiff', height=robust.shape[0], width=robust.shape[1], count=1, dtype='float32',
                crs=C.CRS, transform=t, nodata=np.nan, compress='deflate')
    for m in MODELS:
        with rasterio.open(f'{C.OUT_RAS}/{m.lower()}_pinch_frequency.tif', 'w', **prof) as d:
            d.write(np.where(sm, freq[m], np.nan).astype('float32'), 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ext = (t.c / 1000, (t.c + t.a * robust.shape[1]) / 1000, (t.f + t.e * robust.shape[0]) / 1000, t.f / 1000)
    for a_, m in zip(ax[:2], MODELS):
        im = a_.imshow(np.where(sm, freq[m], np.nan), cmap='magma_r', vmin=0, vmax=1, extent=ext)
        a_.set_title(f'({"ab"[m == "SVM"]}) {m}: pinch-point frequency')
        fig.colorbar(im, ax=a_, shrink=0.7)
    ax[2].imshow(np.where(sm, robust.astype(float), np.nan), cmap='Greys', extent=ext)
    ax[2].set_title('(c) Robust pinch points')
    for a_ in ax:
        a_.set(xlabel='Easting (km)', ylabel='Northing (km)')
    fig.tight_layout()
    fig.savefig(f'{C.OUT_FIG}/FigS_pinch_sensitivity.png', dpi=250)
    plt.close(fig)


# ── #12 Context of consensus and classifier-specific corridors ────────────────
def context():
    cls, t, _ = read(f'{C.OUT_RAS}/corridor_agreement_classes.tif')
    names = {0: 'Neither', 1: 'RF only', 2: 'SVM only', 3: 'Consensus'}
    cell = abs(t.a * t.e) / 1e6
    with rasterio.open(C.PREDICTOR_TIF) as s:
        bands = {b: s.read(i + 1).astype(float) for i, b in enumerate(C.BAND_NAMES)
                 if b in ('Dist_Perm_Water', 'Dist_Season_Water', 'LandCover', 'Dist_Built_Up', 'Nighttime_Lights')}
    # repaired copy of DATA/Manapools.shp (original lacks its .shx index); shipped in data/boundaries
    parks = [f'{C.BOUNDARY_DIR}/mana_pools_np.gpkg'] if os.path.exists(f'{C.BOUNDARY_DIR}/mana_pools_np.gpkg') else []
    parks += sorted(glob.glob(C.DATA_DIR + '/*[Pp]ark*.shp') + glob.glob(C.DATA_DIR + '/*[Mm]ana*[Pp]ool*.shp')
                   + glob.glob(C.DATA_DIR + '/*MPNP*.shp') + glob.glob(C.DATA_DIR + '/*[Bb]oundar*.shp'))
    parks = [p for p in parks if 'createdmana' not in p.lower()]
    in_park = None
    if parks:
        park = gpd.read_file(parks[0]).to_crs(C.CRS)
        in_park = geometry_mask(park.geometry, out_shape=cls.shape, transform=t, invert=True)
        print('park layer:', parks[0], f'{park.area.sum() / 1e6:.1f} km2')
    pa_masks = {}
    if os.path.exists(f'{C.BOUNDARY_DIR}/mid_zambezi_pa.gpkg'):  # repaired DATA/Mid zambezi.shp, shipped in data/boundaries
        pas = gpd.read_file(f'{C.BOUNDARY_DIR}/mid_zambezi_pa.gpkg').to_crs(C.CRS)
        for _, row in pas.iterrows():
            pa_masks[row['Park'].replace(' ', '_')] = geometry_mask([row.geometry], out_shape=cls.shape, transform=t,
                                                                   invert=True)
        any_pa = np.logical_or.reduce(list(pa_masks.values()))
        pa_masks['outside_all_protected_areas'] = ~any_pa
    rows = []
    for k, nm in names.items():
        sel = cls == k
        r = {'class': nm, 'area_km2': sel.sum() * cell}
        if in_park is not None:
            r['area_outside_park_km2'] = (sel & ~in_park).sum() * cell
            r['share_outside_park'] = (sel & ~in_park).sum() / max(sel.sum(), 1)
        for pa_name, pa_mask in pa_masks.items():  # R2 #9: which protected area holds each class
            r[f'km2_{pa_name}'] = (sel & pa_mask).sum() * cell
        r['share_within_1km_perm_water'] = np.mean(bands['Dist_Perm_Water'][sel] <= 1000)
        r['share_within_1km_seasonal_water'] = np.mean(bands['Dist_Season_Water'][sel] <= 1000)
        r['median_dist_perm_water_km'] = np.median(bands['Dist_Perm_Water'][sel]) / 1000
        r['median_dist_built_up_km'] = np.median(bands['Dist_Built_Up'][sel]) / 1000
        r['share_within_5km_built_up'] = np.mean(bands['Dist_Built_Up'][sel] <= 5000)
        r['share_nightlight_gt0.5'] = np.mean(bands['Nighttime_Lights'][sel] > 0.5)
        for lc, lname in ((10, 'tree'), (20, 'shrub'), (30, 'grass'), (40, 'cropland'), (90, 'herb_wetland'),
                          (80, 'water')):
            r[f'share_LC_{lname}'] = np.mean(bands['LandCover'][sel] == lc)
        # riparian proxy: tree cover within 1 km of permanent or seasonal water
        rip = (bands['LandCover'] == 10) & ((bands['Dist_Perm_Water'] <= 1000) | (bands['Dist_Season_Water'] <= 1000))
        r['share_riparian_tree_proxy'] = np.mean(rip[sel])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f'{C.OUT_TAB}/T14_corridor_context.csv', index=False)
    print(df.round(3).T.to_string())
    if in_park is None:
        print('NOTE: no park boundary found in DATA folder; park columns omitted.')


# ── #14 Extent: rectangle vs study-polygon clip ───────────────────────────────
def extent():
    rows = []
    for m in MODELS:
        a, t = load_current(f'{m}_90m')
        b, _ = load_current(f'{m}_90m_studyclip')
        sm = study_mask(a.shape, t) & np.isfinite(a) & np.isfinite(b)
        x, y = a[sm], b[sm]
        r = {'Model': m, 'Spearman_rho': stats.spearmanr(x, y).statistic,
             'Pearson_r_log10': stats.pearsonr(np.log10(x + 1e-12), np.log10(y + 1e-12)).statistic}
        for q in (90, 80):
            r[f'Jaccard_top{100 - q}pct'] = jaccard(x >= np.percentile(x, q), y >= np.percentile(y, q))
        va = sm.copy()
        pa, pb = pinch_map(np.where(sm, a, np.nan), va, 50, 5), pinch_map(np.where(sm, b, np.nan), va, 50, 5)
        r['pinch_Jaccard'] = jaccard(pa, pb)
        edge = ndimage.distance_transform_edt(study_mask(a.shape, t)) * 90
        for lo, hi in ((0, 5000), (5000, 10000), (10000, 1e9)):
            z = sm & (edge >= lo) & (edge < hi)
            r[f'Spearman_{int(lo / 1000)}-{"inf" if hi > 1e8 else int(hi / 1000)}km_from_boundary'] = \
                stats.spearmanr(a[z], b[z]).statistic
        rows.append(r)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f'{C.OUT_TAB}/T15_extent_sensitivity.csv', index=False)
    print(df.round(3).T)


# ── #8 Resolution: 30 m vs 90 m in a window ───────────────────────────────────
def resolution():
    rows = []
    for m in MODELS:
        lab30, lab90 = f'{m}_win30m', f'{m}_win90m'
        if not glob.glob(f'{CS}/{lab30}/cum_current.tif'):
            continue
        c30, t30 = load_current(lab30)
        c90, t90 = load_current(lab90)
        agg = np.full(c90.shape, np.nan)
        reproject(np.nan_to_num(c30), agg, src_transform=t30, dst_transform=t90, src_crs=C.CRS, dst_crs=C.CRS,
                  resampling=Resampling.average)
        ok = np.isfinite(c90) & np.isfinite(agg) & (agg > 0)
        r = {'Model': m, 'cells_90m': int(ok.sum()), 'Spearman_rho': stats.spearmanr(agg[ok], c90[ok]).statistic,
             'Pearson_r_log10': stats.pearsonr(np.log10(agg[ok]), np.log10(c90[ok] + 1e-12)).statistic}
        for q in (90, 80):
            r[f'Jaccard_top{100 - q}pct'] = jaccard(agg[ok] >= np.percentile(agg[ok], q),
                                                    c90[ok] >= np.percentile(c90[ok], q))
        v90 = np.isfinite(c90)
        p90 = pinch_map(c90, v90, 50, 5)
        v30 = np.isfinite(c30)
        p30 = pinch_map(c30, v30, 50, 15)  # same 450 m neighbourhood at 30 m
        p30_90 = np.zeros(c90.shape)
        reproject(p30.astype(float), p30_90, src_transform=t30, dst_transform=t90, src_crs=C.CRS, dst_crs=C.CRS,
                  resampling=Resampling.max)
        r['pinch_Jaccard_450m_neighbourhood'] = jaccard(p30_90 > 0, p90)
        r['share_90m_pinch_cells_also_pinch_at_30m'] = (p90 & (p30_90 > 0)).sum() / max(p90.sum(), 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f'{C.OUT_TAB}/T16_resolution_sensitivity.csv', index=False)
    print(df.round(3).T)


if __name__ == '__main__':
    steps = sys.argv[1:] or ['agreement', 'current', 'pinch', 'context', 'extent', 'resolution']
    for s in steps:
        print(f'===== {s} =====', flush=True)
        globals()[s]()
