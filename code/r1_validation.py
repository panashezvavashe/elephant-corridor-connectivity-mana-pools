"""Step 7: corridor and connectivity validation against withheld GPS data (#3, #4).

Uses only the 102,574 fixes never drawn as presences (validation pool from r1_data.py).

#3 Corridor occupancy
  A  submitted test reproduced: 1,000 random fixes, binomial vs corridor share of study area
  B  autocorrelation-reduced: one fix per collar per day
  C  availability null: for each collar, expected = corridor share of that collar's own
     AKDE 95% range (Objective 1); enrichment = observed / available. Inference across
     the ten collars (Wilcoxon) and by resampling whole collars (cluster bootstrap).

#4 Do observed movements follow predicted connectivity? (step-selection test)
  Consecutive fixes <= 3.5 h apart form used steps. Each gets 10 control steps from the
  same start, with step length and turning angle drawn from that collar's own empirical
  distributions. Mean log cumulative current (90 m) and mean resistance (30 m) are
  sampled along every step. Conditional logistic regression (strata = steps) per
  collar and pooled; RF and SVM compared on the same strata.
"""
import os

import r1_config as C
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy import stats
from statsmodels.discrete.conditional_models import ConditionalLogit

AKDE_DIR = C.AKDE_DIR
MODELS = ('RF', 'SVM')
N_CONTROL = 10
MAX_DT_H = 3.5
B = 2000
rng = np.random.default_rng(C.SEED)


def sample(path, x, y):
    with rasterio.open(path) as s:
        r, c = rasterio.transform.rowcol(s.transform, x, y)
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < s.height) & (c >= 0) & (c < s.width)
        a = s.read(1)
        out = np.full(len(x), np.nan)
        out[ok] = a[r[ok], c[ok]]
        if s.nodata is not None and not np.isnan(s.nodata):
            out[out == s.nodata] = np.nan
    return out


pool = gpd.read_file(C.OUT_DATA + '/validation_pool.gpkg')
pool['timestamp'] = pd.to_datetime(pool['timestamp'])
pool['x'], pool['y'] = pool.geometry.x, pool.geometry.y
study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
for m in MODELS:
    pool[f'in_{m}'] = sample(f'{C.OUT_RAS}/{m.lower()}_binary_corridor.tif', pool.x.values, pool.y.values)


def corridor_share(m, geom=None):
    with rasterio.open(f'{C.OUT_RAS}/{m.lower()}_binary_corridor.tif') as s:
        a = s.read(1)
        mask = geometry_mask(study.geometry, out_shape=a.shape, transform=s.transform, invert=True)
        if geom is not None:
            mask &= geometry_mask([geom], out_shape=a.shape, transform=s.transform, invert=True)
    v = a[mask]
    v = v[np.isfinite(v)]
    return float(np.mean(v == 1))


area_share = {m: corridor_share(m) for m in MODELS}

# ── A. submitted test, reproduced on the revised corridors ────────────────────
rows = []
sub = pool.sample(C.N_VALIDATION_FIXES, random_state=C.SEED)
for m in MODELS:
    k = int((sub[f'in_{m}'] == 1).sum())
    bt = stats.binomtest(k, len(sub), area_share[m], alternative='greater')
    rows.append({'Test': 'A: 1,000 random fixes vs study-area share (submitted design)', 'Model': m,
                 'n': len(sub), 'observed_share': k / len(sub), 'expected_share': area_share[m],
                 'enrichment': k / len(sub) / area_share[m], 'p': bt.pvalue})

# ── B. one fix per collar per day ─────────────────────────────────────────────
daily = pool.sort_values('timestamp').groupby(['collar', pool.timestamp.dt.date]).head(1)
for m in MODELS:
    k = int((daily[f'in_{m}'] == 1).sum())
    bt = stats.binomtest(k, len(daily), area_share[m], alternative='greater')
    rows.append({'Test': 'B: one fix per collar-day vs study-area share', 'Model': m, 'n': len(daily),
                 'observed_share': k / len(daily), 'expected_share': area_share[m],
                 'enrichment': k / len(daily) / area_share[m], 'p': bt.pvalue})

# ── C. availability null from each collar's AKDE 95% range ────────────────────
per = []
for cid in C.IDS:
    hr = gpd.read_file(f'{AKDE_DIR}/AKDE_95_{cid}.gpkg').to_crs(C.CRS)
    geom = hr.geometry.union_all()  # one point-estimate polygon per collar
    fx = daily[daily.collar == cid]
    r = {'collar': cid, 'sex': C.SEX[cid], 'n_daily_fixes': len(fx), 'AKDE95_km2': geom.area / 1e6}
    for m in MODELS:
        avail = corridor_share(m, geom)
        obs = float(np.mean(fx[f'in_{m}'] == 1))
        r[f'{m}_available'] = avail
        r[f'{m}_observed'] = obs
        r[f'{m}_enrichment'] = obs / avail if avail > 0 else np.nan
    per.append(r)
per = pd.DataFrame(per)
per.round(4).to_csv(f'{C.OUT_TAB}/T17_corridor_use_vs_availability_by_collar.csv', index=False)
for m in MODELS:
    le = np.log(per[f'{m}_enrichment'].values)
    w = stats.wilcoxon(le, alternative='greater')
    boots = [np.exp(np.mean(rng.choice(le, len(le)))) for _ in range(B)]
    rows.append({'Test': 'C: per-collar use vs own AKDE95 availability (collars as units)', 'Model': m,
                 'n': f'{len(le)} collars', 'observed_share': per[f'{m}_observed'].mean(),
                 'expected_share': per[f'{m}_available'].mean(),
                 'enrichment': float(np.exp(le.mean())),
                 'enrichment_CI': f'{np.percentile(boots, 2.5):.3f}-{np.percentile(boots, 97.5):.3f}',
                 'collars_enriched': f'{int((le > 0).sum())} of {len(le)}', 'p': w.pvalue})
le_d = np.log(per['RF_enrichment']) - np.log(per['SVM_enrichment'])
rows.append({'Test': 'C: RF vs SVM enrichment, paired across collars (Wilcoxon, two-sided)', 'Model': 'RF-SVM',
             'n': '10 collars', 'enrichment': float(np.exp(le_d.mean())),
             'collars_enriched': f'RF higher in {int((le_d > 0).sum())} of 10', 'p': stats.wilcoxon(le_d).pvalue})
occ = pd.DataFrame(rows)
occ.round(5).to_csv(f'{C.OUT_TAB}/T18_corridor_validation.csv', index=False)
print(occ.round(4).to_string())

# ── #4 Step-selection test of current and resistance ──────────────────────────
g = pool.sort_values(['collar', 'timestamp']).reset_index(drop=True)
same = g.collar.values[1:] == g.collar.values[:-1]
dt = (g.timestamp.values[1:] - g.timestamp.values[:-1]) / np.timedelta64(1, 'h')
dx, dy = np.diff(g.x.values), np.diff(g.y.values)
st = pd.DataFrame({'collar': g.collar.values[:-1], 'x0': g.x.values[:-1], 'y0': g.y.values[:-1],
                   'x1': g.x.values[1:], 'y1': g.y.values[1:], 'dt': dt, 'sl': np.hypot(dx, dy),
                   'head': np.arctan2(dy, dx)})
st = st[same & (dt > 0) & (dt <= MAX_DT_H)].reset_index(drop=True)
st['ta'] = np.nan
prev_head = np.r_[np.nan, st['head'].values[:-1]]
contig = np.r_[False, (st.collar.values[1:] == st.collar.values[:-1]) &
               np.isclose(st.x0.values[1:], st.x1.values[:-1])]
st.loc[contig, 'ta'] = np.angle(np.exp(1j * (st['head'].values - prev_head)))[contig]
st = st[st.sl > 0].dropna(subset=['ta']).reset_index(drop=True)
st['step_id'] = np.arange(len(st))

ctrl = []
for cid, d in st.groupby('collar'):
    n = len(d)
    sl = rng.choice(d.sl.values, (n, N_CONTROL))
    ta = rng.choice(d.ta.values, (n, N_CONTROL))
    h = (d['head'].values - d.ta.values)[:, None] + ta  # previous heading + random turn
    ctrl.append(pd.DataFrame({'collar': cid, 'step_id': np.repeat(d.step_id.values, N_CONTROL),
                              'x0': np.repeat(d.x0.values, N_CONTROL), 'y0': np.repeat(d.y0.values, N_CONTROL),
                              'x1': (d.x0.values[:, None] + sl * np.cos(h)).ravel(),
                              'y1': (d.y0.values[:, None] + sl * np.sin(h)).ravel(), 'sl': sl.ravel()}))
ctrl = pd.concat(ctrl)
st['used'] = 1
ctrl['used'] = 0
allst = pd.concat([st[['collar', 'step_id', 'x0', 'y0', 'x1', 'y1', 'sl', 'used']], ctrl], ignore_index=True)

NPTS = 11  # points sampled along each step (start, end and 9 between)
fr = np.linspace(0, 1, NPTS)
px = allst.x0.values[:, None] + (allst.x1.values - allst.x0.values)[:, None] * fr
py = allst.y0.values[:, None] + (allst.y1.values - allst.y0.values)[:, None] * fr
for m in MODELS:
    cur = sample(f'{C.OUT}/cs/{m}_90m/cum_current.tif', px.ravel(), py.ravel()).reshape(px.shape)
    res = sample(f'{C.OUT_RAS}/{m.lower()}_resistance.tif', px.ravel(), py.ravel()).reshape(px.shape)
    allst[f'logcur_{m}'] = np.nanmean(np.log10(np.where(cur > 0, cur, np.nan)), axis=1)
    allst[f'res_{m}'] = np.nanmean(res, axis=1)
allst['log_sl'] = np.log(allst.sl)
ok = allst.groupby('step_id')[[f'logcur_{m}' for m in MODELS] + [f'res_{m}' for m in MODELS]] \
    .transform(lambda s: s.notna().all()).all(axis=1)
allst = allst[ok & allst.groupby('step_id').used.transform('sum').eq(1)]
allst.to_csv(C.OUT_DATA + '/ssf_steps.csv', index=False)

ssf = []
for cov in [f'logcur_{m}' for m in MODELS] + [f'res_{m}' for m in MODELS]:
    z = (allst[cov] - allst[cov].mean()) / allst[cov].std()
    for unit, d in [('pooled', allst)] + [(str(c), dd) for c, dd in allst.groupby('collar')]:
        zz = z.loc[d.index]
        X = np.column_stack([zz, d.log_sl])
        fit = ConditionalLogit(d.used.values, X, groups=d.step_id.values).fit(disp=0)
        used_v = zz.where(d.used == 1).groupby(d.step_id).transform('max')
        ctl = d.used == 0
        rank = ((zz < used_v) & ctl).groupby(d.step_id).sum() / ctl.groupby(d.step_id).sum()
        ssf.append({'covariate': cov, 'unit': unit, 'n_steps': d.step_id.nunique(),
                    'beta_per_SD': fit.params[0], 'se': fit.bse[0], 'p': fit.pvalues[0],
                    'loglik': fit.llf, 'mean_rank_of_used_step': rank.mean()})
ssf = pd.DataFrame(ssf)
ssf.round(5).to_csv(f'{C.OUT_TAB}/T19_step_selection.csv', index=False)
summ = []
for cov, d in ssf[ssf.unit != 'pooled'].groupby('covariate'):
    b = d.beta_per_SD.values
    boots = [np.mean(rng.choice(b, len(b))) for _ in range(B)]
    summ.append({'covariate': cov, 'mean_beta_across_collars': b.mean(),
                 'CI_collar_bootstrap': f'{np.percentile(boots, 2.5):.3f} to {np.percentile(boots, 97.5):.3f}',
                 'collars_positive': int((b > 0).sum()), 'collars_sig_positive': int(((b > 0) & (d.p < 0.05)).sum()),
                 'collars_sig_negative': int(((b < 0) & (d.p < 0.05)).sum()),
                 'pooled_beta': ssf[(ssf.covariate == cov) & (ssf.unit == 'pooled')].beta_per_SD.iloc[0],
                 'pooled_loglik': ssf[(ssf.covariate == cov) & (ssf.unit == 'pooled')].loglik.iloc[0]})
summ = pd.DataFrame(summ)
summ.round(4).to_csv(f'{C.OUT_TAB}/T20_step_selection_summary.csv', index=False)
pd.Series({'used_steps': int(allst.used.sum()), 'control_steps': int((allst.used == 0).sum()),
           'max_dt_h': MAX_DT_H, 'median_step_m': float(st.sl.median())}).to_csv(
    f'{C.OUT_TAB}/S10_ssf_design.csv', header=['value'])
print(summ.round(3).to_string())
print('done')
