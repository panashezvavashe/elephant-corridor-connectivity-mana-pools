"""Step 2: refit RF and SVM on the ten-collar data and run every model-level analysis
requested by Reviewer 1.

  #16  thresholds chosen from out-of-fold training predictions (nested), applied to
       the untouched holdout; calibration (Brier, reliability, slope/intercept)
  #2   spatial-block CV: 25 km blocks, systematic fold allocation, 12.5 km buffer,
       plus a buffer-distance sensitivity curve
  #15  paired cluster bootstraps resampling whole individuals (LOIO) and whole
       spatial blocks; DeLong / McNemar kept as secondary tests
  #6   grouped permutation importance, identical for both classifiers; Spearman
       matrix + VIF; models refitted without elevation
  #11  ALE and ICE alongside partial dependence
  #13  environmental / spatial profile of each collar vs the other nine

Outputs go to results_revision1/{tables,figures,data}. Run r1_data.py first.
"""
import json
import os
import time

import r1_config as C  # noqa: F401  (sets thread caps + low priority before numpy loads)
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss)
from sklearn.model_selection import (train_test_split, GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import r1_stats as S

T0 = time.time()
BLOCK_M = 25000
BUFFER_M = 12500
BUFFER_SENSITIVITY_M = [0, 2500, 5000, 10000, 12500]
N_FOLDS = 5


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


def save(df, name, index=False):
    df.to_csv(f'{C.OUT_TAB}/{name}.csv', index=index)
    log(f'saved {name}.csv')


# ── Data ──────────────────────────────────────────────────────────────────────
pts = gpd.read_file(C.OUT_DATA + '/model_points.gpkg')
SMOKE = bool(os.environ.get('R1_SMOKE'))  # quick end-to-end check on a subsample
B_BOOT = 50 if SMOKE else 2000
if SMOKE:
    pts = pts.groupby('class', group_keys=False).sample(600, random_state=1).reset_index(drop=True)
    C.OUT_TAB, C.OUT_FIG, C.OUT_DATA = (C.OUT + '/_smoke',) * 3
    os.makedirs(C.OUT_TAB, exist_ok=True)
for c in C.LULC_CLASSES:
    pts[f'LULC_{c}'] = (pts['LandCover'] == c).astype(int)
X = pts[C.FEATURES].astype(float)
y = pts['class'].astype(int)
collar = pts['collar'].values
xy = pts[['x', 'y']].values

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=C.SEED, stratify=y)
X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
log(f'{len(y)} points ({y.sum()} presences), train {len(idx_tr)}, test {len(idx_te)}')


# ── Tuning (same grids as the submitted manuscript) ───────────────────────────
def rf_pipe(**kw):
    return Pipeline([('classifier', RandomForestClassifier(random_state=C.SEED, n_jobs=C.N_JOBS, **kw))])


def svm_pipe(**kw):
    return Pipeline([('scaler', StandardScaler()),
                     ('classifier', SVC(probability=True, random_state=C.SEED, **kw))])


inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
rf_grid = GridSearchCV(Pipeline([('classifier', RandomForestClassifier(random_state=C.SEED, n_jobs=1))]),
                       {'classifier__n_estimators': [100, 300, 500], 'classifier__max_depth': [None, 10, 20],
                        'classifier__min_samples_split': [2, 5],
                        'classifier__max_features': ['sqrt', 0.33, 0.5, 1.0]},  # R2 #6: grid widened (mtry)
                       cv=inner, scoring='roc_auc', n_jobs=C.N_JOBS).fit(X_tr, y_tr)
log(f'RF tuned {rf_grid.best_params_}')
svm_grid = GridSearchCV(svm_pipe(cache_size=1000),
                        {'classifier__C': [0.1, 1, 10, 100, 1000],  # R2 #6: widened beyond the old edge
                         'classifier__gamma': ['scale', 0.01, 0.1, 1, 10], 'classifier__kernel': ['rbf']},
                        cv=inner, scoring='roc_auc', n_jobs=C.N_JOBS).fit(X_tr, y_tr)
log(f'SVM tuned {svm_grid.best_params_}')
RF_P = {k.split('__')[1]: v for k, v in rf_grid.best_params_.items()}
SVM_P = {k.split('__')[1]: v for k, v in svm_grid.best_params_.items()}
make = {'RF': lambda: rf_pipe(**RF_P), 'SVM': lambda: svm_pipe(**SVM_P)}
best = {'RF': make['RF']().fit(X_tr, y_tr), 'SVM': make['SVM']().fit(X_tr, y_tr)}
joblib.dump(best['RF'], C.OUT_DATA + '/rf_model.joblib')
joblib.dump(best['SVM'], C.OUT_DATA + '/svm_model.joblib')

# ── #16 Thresholds from out-of-fold TRAINING predictions, never the holdout ───
thr, thr_rows = {}, []
for m in ('RF', 'SVM'):
    oof = cross_val_predict(make[m](), X_tr, y_tr, cv=inner, method='predict_proba', n_jobs=1)[:, 1]
    thr[m], tss_oof = S.max_tss_threshold(y_tr.values, oof)
    fold_thr = []
    for tr, te in inner.split(X_tr, y_tr):
        fold_thr.append(S.max_tss_threshold(y_tr.values[te], oof[te])[0])
    p_te = best[m].predict_proba(X_te)[:, 1]
    thr_te_opt, tss_te_opt = S.max_tss_threshold(y_te.values, p_te)  # the old, optimistic way
    thr_rows.append({'Model': m, 'MaxTSS_threshold_nested': thr[m], 'TSS_train_OOF': round(tss_oof, 3),
                     'threshold_range_across_inner_folds': f'{min(fold_thr):.2f}-{max(fold_thr):.2f}',
                     'TSS_holdout_at_nested_threshold': round(S.tss(y_te.values, (p_te >= thr[m]).astype(int)), 3),
                     'old_threshold_selected_on_holdout': thr_te_opt,
                     'old_TSS_at_holdout_selected_threshold': round(tss_te_opt, 3)})
save(pd.DataFrame(thr_rows), 'T1_thresholds_nested')
json.dump({'thresholds': thr, 'RF_params': {k: (v if v is not None else 'None') for k, v in RF_P.items()},
           'SVM_params': SVM_P, 'block_m': BLOCK_M, 'buffer_m': BUFFER_M},
          open(C.OUT_DATA + '/model_settings.json', 'w'), indent=2, default=str)

# ── Holdout evaluation (Table 2 equivalent) ───────────────────────────────────
p_te = {m: best[m].predict_proba(X_te)[:, 1] for m in best}
yhat_default = {m: best[m].predict(X_te) for m in best}
yhat_thr = {m: (p_te[m] >= thr[m]).astype(int) for m in best}
yt = y_te.values
rows = []
for m in best:
    rows.append({'Model': m, 'AUC': roc_auc_score(yt, p_te[m]),
                 'TSS_default': S.tss(yt, yhat_default[m]), 'TSS_nested_threshold': S.tss(yt, yhat_thr[m]),
                 'Accuracy': accuracy_score(yt, yhat_default[m]), 'Precision': precision_score(yt, yhat_default[m]),
                 'Recall': recall_score(yt, yhat_default[m]), 'F1': f1_score(yt, yhat_default[m]),
                 'Brier': brier_score_loss(yt, p_te[m])})
hold = pd.DataFrame(rows).round(4)
fns = {'AUC': lambda yy, pp: roc_auc_score(yy, pp), 'Brier': lambda yy, pp: brier_score_loss(yy, pp)}
ci = S.paired_bootstrap_metrics(yt, p_te['RF'], p_te['SVM'], fns, B=B_BOOT)
for k, (a, b, d) in ci.items():
    hold.loc[hold.Model == 'RF', f'{k}_CI'] = f'{a[1]:.3f}-{a[2]:.3f}'
    hold.loc[hold.Model == 'SVM', f'{k}_CI'] = f'{b[1]:.3f}-{b[2]:.3f}'
    hold.loc[hold.Model == 'RF', f'{k}_diff_RF_minus_SVM'] = f'{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f})'
save(hold, 'T2_holdout_performance')
aucs, z, p = S.delong(yt, p_te['RF'], p_te['SVM'])
mc05 = S.mcnemar(yt, yhat_default['RF'], yhat_default['SVM'])
mcthr = S.mcnemar(yt, yhat_thr['RF'], yhat_thr['SVM'])
save(pd.DataFrame([{'test': 'DeLong (holdout, row-level)', 'stat': z, 'p': p},
                   {'test': 'McNemar default rule', 'stat': mc05[0], 'p': mc05[1], 'b_RF_only_correct': mc05[2], 'c_SVM_only_correct': mc05[3]},
                   {'test': 'McNemar nested MaxTSS', 'stat': mcthr[0], 'p': mcthr[1], 'b_RF_only_correct': mcthr[2], 'c_SVM_only_correct': mcthr[3]}]),
     'T3_row_level_tests')

# ── #16 Calibration ───────────────────────────────────────────────────────────
cal_rows = []
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for m, col in (('RF', '#1f6fb4'), ('SVM', '#c0392b')):
    fp, mp = calibration_curve(yt, p_te[m], n_bins=10, strategy='quantile')
    ax[0].plot(mp, fp, 'o-', color=col, label=m)
    ax[1].hist(p_te[m], bins=20, range=(0, 1), histtype='step', color=col, lw=1.6, label=m)
    lp = np.log(np.clip(p_te[m], 1e-6, 1 - 1e-6) / (1 - np.clip(p_te[m], 1e-6, 1 - 1e-6)))
    fit = sm.Logit(yt, sm.add_constant(lp)).fit(disp=0)
    ece = np.sum(np.abs(fp - mp)) / len(fp)
    cal_rows.append({'Model': m, 'Brier': brier_score_loss(yt, p_te[m]), 'Calibration_intercept': fit.params[0],
                     'Calibration_slope': fit.params[1], 'ECE_10_quantile_bins': ece})
ax[0].plot([0, 1], [0, 1], '--', color='grey')
ax[0].set(xlabel='Mean predicted probability', ylabel='Observed proportion of presences', title='(a) Reliability diagram')
ax[1].set(xlabel='Predicted probability', ylabel='Holdout points', title='(b) Distribution of predictions')
for a in ax:
    a.legend(frameon=False)
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/FigS_calibration.png', dpi=300)
plt.close(fig)
save(pd.DataFrame(cal_rows).round(4), 'T4_calibration')


# ── Cross-validation schemes, storing out-of-fold predictions for bootstraps ──
def blocks_and_folds(xy_, size=BLOCK_M):
    col_ = ((xy_[:, 0] - xy[:, 0].min()) // size).astype(int)
    row_ = ((xy_[:, 1] - xy[:, 1].min()) // size).astype(int)
    return col_ * 1000 + row_, (col_ + 2 * row_) % N_FOLDS


block_id, sfold = blocks_and_folds(xy)
pts['block_id'], pts['spatial_fold'] = block_id, sfold


def run_spatial(m, buffer_m):
    oof = np.full(len(y), np.nan)
    fold_rows = []
    for f in range(N_FOLDS):
        te = np.where(sfold == f)[0]
        cand = np.where(sfold != f)[0]
        d, _ = cKDTree(xy[te]).query(xy[cand])
        tr = cand[d >= buffer_m]
        mdl = make[m]().fit(X.iloc[tr], y.iloc[tr])
        oof[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        fold_rows.append({'Model': m, 'buffer_m': buffer_m, 'fold': f + 1,
                          'n_blocks_test': len(np.unique(block_id[te])), 'n_test': len(te),
                          'test_presence_share': y.values[te].mean(), 'n_train': len(tr),
                          'train_share_kept': len(tr) / len(cand), 'min_train_test_dist_m': d[d >= buffer_m].min(),
                          'AUC': roc_auc_score(y.values[te], oof[te]),
                          'TSS': S.tss(y.values[te], (oof[te] >= thr[m]).astype(int))})
    return oof, fold_rows


def run_stratified(m):
    oof = np.full(len(y), np.nan)
    rows_ = []
    for f, (tr, te) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=C.SEED).split(X, y)):
        oof[te] = make[m]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        rows_.append({'Model': m, 'fold': f + 1, 'AUC': roc_auc_score(y.values[te], oof[te]),
                      'TSS': S.tss(y.values[te], (oof[te] >= thr[m]).astype(int))})
    return oof, rows_


abs_idx = np.where(y.values == 0)[0]
abs_fold_of = np.empty(len(y), dtype=int)
_shuf = np.random.default_rng(C.SEED).permutation(abs_idx)
for i, a in enumerate(np.array_split(_shuf, len(C.IDS))):
    abs_fold_of[a] = C.IDS[i]
loio_unit = np.where(y.values == 1, collar, abs_fold_of)  # each LOIO fold = one collar + 1/10 absences


def run_loio(m):
    oof = np.full(len(y), np.nan)
    rows_ = []
    for cid in C.IDS:
        te = np.where(loio_unit == cid)[0]
        tr = np.setdiff1d(np.arange(len(y)), te)
        oof[te] = make[m]().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        rows_.append({'Model': m, 'collar': cid, 'sex': C.SEX[cid], 'n_presences': int(y.values[te].sum()),
                      'AUC': roc_auc_score(y.values[te], oof[te]),
                      'TSS': S.tss(y.values[te], (oof[te] >= thr[m]).astype(int))})
    return oof, rows_


oof_store = pd.DataFrame({'class': y.values, 'collar': collar, 'block_id': block_id,
                          'spatial_fold': sfold, 'loio_unit': loio_unit, 'x': xy[:, 0], 'y': xy[:, 1]})
strat_rows, spatial_rows, loio_rows = [], [], []
for m in ('RF', 'SVM'):
    o, r = run_stratified(m)
    oof_store[f'strat_{m}'] = o
    strat_rows += r
    log(f'{m} stratified CV done')
    for b in BUFFER_SENSITIVITY_M:
        o, r = run_spatial(m, b)
        oof_store[f'spatial_buf{b}_{m}'] = o
        spatial_rows += r
        log(f'{m} spatial CV buffer {b} m done')
    o, r = run_loio(m)
    oof_store[f'loio_{m}'] = o
    loio_rows += r
    log(f'{m} LOIO done')
    oof_store.to_csv(C.OUT_DATA + '/oof_predictions.csv', index=False)

strat_df, spatial_df, loio_df = map(pd.DataFrame, (strat_rows, spatial_rows, loio_rows))
save(spatial_df.round(4), 'S2_spatial_cv_folds')
save(loio_df.round(4), 'T5_loio_by_collar')
summary = []
for scheme, df_ in (('Stratified 5-fold', strat_df),
                    ('Spatial block, no buffer', spatial_df[spatial_df.buffer_m == 0]),
                    (f'Spatial block, {BUFFER_M / 1000:g} km buffer', spatial_df[spatial_df.buffer_m == BUFFER_M]),
                    ('LOIO (n = 10)', loio_df)):
    for m in ('RF', 'SVM'):
        d = df_[df_.Model == m]
        summary.append({'CV_scheme': scheme, 'Model': m, 'AUC_mean': d.AUC.mean(), 'AUC_sd': d.AUC.std(ddof=0),
                        'AUC_min': d.AUC.min(), 'AUC_max': d.AUC.max(),
                        'TSS_mean': d.TSS.mean(), 'TSS_sd': d.TSS.std(ddof=0)})
save(pd.DataFrame(summary).round(3), 'T6_cv_summary')
buf_curve = spatial_df.groupby(['Model', 'buffer_m']).agg(AUC_mean=('AUC', 'mean'), AUC_sd=('AUC', lambda a: a.std(ddof=0)),
                                                          TSS_mean=('TSS', 'mean'),
                                                          train_kept=('train_share_kept', 'mean')).reset_index()
save(buf_curve.round(3), 'S3_buffer_sensitivity')

# Block design description for the response letter (#2)
bx = pd.DataFrame({'block_id': block_id, 'fold': sfold, 'x': xy[:, 0], 'y': xy[:, 1], 'cls': y.values})
blk = bx.groupby('block_id').agg(fold=('fold', 'first'), n_points=('cls', 'size'), n_presences=('cls', 'sum'))
save(blk.reset_index(), 'S1_spatial_blocks')

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cmap = plt.get_cmap('Set2')
for f in range(N_FOLDS):
    s = sfold == f
    ax[0].scatter(xy[s, 0] / 1000, xy[s, 1] / 1000, s=2, color=cmap(f), label=f'Fold {f + 1}')
x0, y0 = xy[:, 0].min(), xy[:, 1].min()
for gx in np.arange(x0, xy[:, 0].max() + BLOCK_M, BLOCK_M):
    ax[0].axvline(gx / 1000, color='k', lw=0.4)
for gy in np.arange(y0, xy[:, 1].max() + BLOCK_M, BLOCK_M):
    ax[0].axhline(gy / 1000, color='k', lw=0.4)
ax[0].set(xlabel='Easting (km)', ylabel='Northing (km)', title='(a) 25 km blocks, systematic fold allocation')
ax[0].legend(markerscale=5, frameon=False, fontsize=8)
for m, col in (('RF', '#1f6fb4'), ('SVM', '#c0392b')):
    d = buf_curve[buf_curve.Model == m]
    ax[1].errorbar(d.buffer_m / 1000, d.AUC_mean, yerr=d.AUC_sd, marker='o', color=col, capsize=3, label=m)
ax[1].set(xlabel='Buffer between training and test points (km)', ylabel='Spatial-block AUC (fold mean ± SD)',
          title='(b) Performance vs train–test separation')
ax[1].legend(frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/FigS_spatial_blocks.png', dpi=300)
plt.close(fig)

# ── #15 Cluster bootstraps: individuals (LOIO OOF) and spatial blocks ─────────
boot_rows = []
for label, prefix, clusters in (('Individual (LOIO predictions, resample collars)', 'loio', loio_unit),
                                (f'Spatial block (buffered block-CV predictions, resample {BLOCK_M // 1000} km blocks)',
                                 f'spatial_buf{BUFFER_M}', block_id),
                                ('Spatial block (holdout, resample blocks)', 'holdout', block_id[idx_te])):
    if prefix == 'holdout':
        r = S.cluster_bootstrap(yt, p_te['RF'], p_te['SVM'], clusters, thr['RF'], thr['SVM'], B=B_BOOT)
    else:
        r = S.cluster_bootstrap(y.values, oof_store[f'{prefix}_RF'].values, oof_store[f'{prefix}_SVM'].values,
                                clusters, thr['RF'], thr['SVM'], B=B_BOOT)
    row = {'Resampling unit': label, 'n_clusters': r['n_clusters'], 'B': r['B_effective']}
    for k in ('auc_rf', 'auc_svm', 'auc_diff', 'tss_rf', 'tss_svm', 'tss_diff'):
        row[k] = f'{r[k][0]:.3f} ({r[k][1]:.3f} to {r[k][2]:.3f})'
    row['p_auc_diff'], row['p_tss_diff'] = r['p_auc_diff'], r['p_tss_diff']
    boot_rows.append(row)
    log(f'bootstrap done: {label}')
lw = loio_df.pivot(index='collar', columns='Model', values='AUC')
wil = stats.wilcoxon(lw['RF'], lw['SVM'])
boot_rows.append({'Resampling unit': 'Per-collar LOIO AUC, Wilcoxon signed-rank (RF vs SVM)',
                  'n_clusters': len(lw), 'auc_diff': f"{(lw['RF'] - lw['SVM']).median():.3f} median",
                  'p_auc_diff': wil.pvalue, 'B': f"RF > SVM in {(lw['RF'] > lw['SVM']).sum()} of {len(lw)} collars"})
save(pd.DataFrame(boot_rows), 'T7_cluster_bootstrap')

# ── #6 Importance, correlation, VIF, elevation-free refit ─────────────────────
groups = {c: [c] for c in C.CONTINUOUS}
groups['Land cover (8 classes)'] = C.LULC_COLS
imp_rows = []
for m in best:
    base, res = S.grouped_permutation_importance(best[m], X_te, y_te.values, groups)
    for g, (mu, sd) in res.items():
        imp_rows.append({'Model': m, 'Predictor': g, 'AUC_drop_mean': mu, 'AUC_drop_sd': sd})
imp = pd.DataFrame(imp_rows)
imp['Rank'] = imp.groupby('Model')['AUC_drop_mean'].rank(ascending=False).astype(int)
save(imp.sort_values(['Model', 'Rank']).round(4), 'T8_permutation_importance')
piv = imp.pivot(index='Predictor', columns='Model', values='Rank')
rho_rank = stats.spearmanr(piv['RF'], piv['SVM'])

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
order = imp[imp.Model == 'RF'].sort_values('AUC_drop_mean')['Predictor']
for a, m, col in ((ax[0], 'RF', '#1f6fb4'), (ax[1], 'SVM', '#c0392b')):
    d = imp[imp.Model == m].set_index('Predictor').loc[order]
    a.barh(d.index, d.AUC_drop_mean, xerr=d.AUC_drop_sd, color=col, alpha=0.85)
    a.set(xlabel='Mean decrease in holdout AUC (20 permutations)', title=f'({"ab"[m == "SVM"]}) {m}')
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/Fig5_permutation_importance.png', dpi=300)
plt.close(fig)

corr = pts[C.CONTINUOUS].corr(method='spearman')
save(corr.round(3), 'S4_predictor_spearman', index=True)
Z = sm.add_constant(pts[C.CONTINUOUS].astype(float))
vif = pd.DataFrame({'Predictor': C.CONTINUOUS,
                    'VIF': [variance_inflation_factor(Z.values, i + 1) for i in range(len(C.CONTINUOUS))]})
save(vif.round(2), 'S5_vif')
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha='right')
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman's ρ")
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/FigS_predictor_correlation.png', dpi=300)
plt.close(fig)

noelev = [f for f in C.FEATURES if f != 'Elevation']
ne_rows = []
for m in best:
    mdl = (rf_pipe(**RF_P) if m == 'RF' else svm_pipe(**SVM_P)).fit(X_tr[noelev], y_tr)
    pp = mdl.predict_proba(X_te[noelev])[:, 1]
    sp_auc = []
    for f in range(N_FOLDS):
        te = np.where(sfold == f)[0]
        cand = np.where(sfold != f)[0]
        d, _ = cKDTree(xy[te]).query(xy[cand])
        tr = cand[d >= BUFFER_M]
        mm = clone(mdl).fit(X.iloc[tr][noelev], y.iloc[tr])
        sp_auc.append(roc_auc_score(y.values[te], mm.predict_proba(X.iloc[te][noelev])[:, 1]))
    base, res = S.grouped_permutation_importance(mdl, X_te[noelev], y_te.values,
                                                 {k: v for k, v in groups.items() if k != 'Elevation'})
    top = sorted(res.items(), key=lambda kv: -kv[1][0])[:3]
    ne_rows.append({'Model': m, 'Holdout_AUC_with_elevation': hold.loc[hold.Model == m, 'AUC'].iloc[0],
                    'Holdout_AUC_without_elevation': roc_auc_score(yt, pp),
                    'SpatialCV_AUC_without_elevation': np.mean(sp_auc),
                    'Top3_without_elevation': '; '.join(f'{k} ({v[0]:.3f})' for k, v in top)})
ne = pd.DataFrame(ne_rows)
ne['Spearman_rank_RF_vs_SVM_importance'] = f'{rho_rank.statistic:.3f} (p = {rho_rank.pvalue:.3f})'
save(ne.round(4), 'T9_without_elevation')
log('importance / correlation / no-elevation done')

# ── #11 PDP + ICE + ALE ───────────────────────────────────────────────────────
resp_feats = ['Dist_Perm_Water', 'Wet_NDVI', 'Elevation']
Xs = X_tr.sample(min(2000, len(X_tr)), random_state=C.SEED)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
ale_rows = []
for r_, m in enumerate(('RF', 'SVM')):
    for c_, f in enumerate(resp_feats):
        a = axes[r_, c_]
        disp = PartialDependenceDisplay.from_estimator(
            best[m], Xs, [f], kind='both', subsample=150, ax=a, random_state=C.SEED, centered=True,
            ice_lines_kw={'alpha': 0.08, 'color': 'grey'},
            pd_line_kw={'color': 'black', 'lw': 2, 'label': 'PDP (centred)'})
        a = disp.axes_[0, 0]  # the display draws on its own child axes and hides `a`
        edges, ale = S.ale_1d(best[m], Xs, f)
        # ALE shifted to start at 0 so it shares the centred-PDP/ICE baseline
        a.plot(edges, ale - ale[0], color='#1f6fb4' if m == 'RF' else '#c0392b', lw=2.2, label='ALE')
        a.set_title(f'{m}: {f}')
        a.legend(fontsize=7, frameon=False)
        for e, v in zip(edges, ale):
            ale_rows.append({'Model': m, 'Predictor': f, 'x': e, 'ALE': v})
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/Fig6_ALE_ICE_PDP.png', dpi=300)
plt.close(fig)
save(pd.DataFrame(ale_rows).round(5), 'S6_ale_values')
log('response curves done')

# ── #13 Collar profiles ───────────────────────────────────────────────────────
pres = pts[pts['class'] == 1]
env = pres[C.CONTINUOUS].astype(float)
prof = []
for cid in C.IDS:
    me, ot = env[pres.collar == cid], env[pres.collar != cid]
    mu, cov = ot.mean().values, np.cov(ot.values.T)
    icov = np.linalg.pinv(cov)
    dd = me.values - mu
    mahal = np.sqrt(np.einsum('ij,jk,ik->i', dd, icov, dd))
    d_ot = ot.values - mu
    mahal_ref = np.sqrt(np.einsum('ij,jk,ik->i', d_ot, icov, d_ot))
    lo, hi = ot.quantile(0.01), ot.quantile(0.99)
    outside = ((me < lo) | (me > hi)).any(axis=1).mean()
    xy_c = pres.loc[pres.collar == cid, ['x', 'y']].values
    xy_o = pres.loc[pres.collar != cid, ['x', 'y']].values
    dnn, _ = cKDTree(xy_o).query(xy_c)
    hull = gpd.GeoSeries(gpd.points_from_xy(xy_c[:, 0], xy_c[:, 1])).union_all().convex_hull
    row = {'collar': cid, 'sex': C.SEX[cid], 'n_presences': len(me),
           'MCP100_presences_km2': hull.area / 1e6,
           'centroid_x_km': xy_c[:, 0].mean() / 1000, 'centroid_y_km': xy_c[:, 1].mean() / 1000,
           'median_km_to_nearest_other_collar_fix': np.median(dnn) / 1000,
           'share_fixes_>5km_from_other_collars': np.mean(dnn > 5000),
           'median_Mahalanobis_to_other_collars': np.median(mahal),
           'Mahalanobis_ratio_vs_others_own': np.median(mahal) / np.median(mahal_ref),
           'share_outside_1-99pct_env_range_of_others': outside}
    for v in ('Elevation', 'Dist_Perm_Water', 'Wet_NDVI', 'Dry_NDVI', 'Dist_Built_Up', 'Slope'):
        row[f'median_{v}'] = me[v].median()
    for m in ('RF', 'SVM'):
        row[f'LOIO_AUC_{m}'] = loio_df[(loio_df.Model == m) & (loio_df.collar == cid)].AUC.iloc[0]
    prof.append(row)
prof = pd.DataFrame(prof)
for v in ('median_Mahalanobis_to_other_collars', 'median_km_to_nearest_other_collar_fix', 'MCP100_presences_km2'):
    r = stats.spearmanr(prof[v], prof['LOIO_AUC_RF'])
    prof.loc[0, f'spearman_with_RF_LOIO_AUC__{v}'] = f'{r.statistic:.3f} (p = {r.pvalue:.3f})'
save(prof.round(4), 'T10_collar_profiles')

# Environmental-space plot (PCA of presence predictors), poorly predicted collars highlighted
Zs = (env - env.mean()) / env.std()
U, Sv, Vt = np.linalg.svd(Zs.values, full_matrices=False)
pcs = Zs.values @ Vt[:2].T
expl = Sv[:2] ** 2 / np.sum(Sv ** 2)
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
poor = prof.loc[prof.LOIO_AUC_RF < 0.6, 'collar'].tolist()
for cid in C.IDS:
    s = (pres.collar == cid).values
    kw = dict(s=4, alpha=0.6) if cid in poor else dict(s=2, alpha=0.15, color='grey')
    ax[0].scatter(pcs[s, 0], pcs[s, 1], label=f'{cid} ({C.SEX[cid]})' if cid in poor else None, **kw)
    ax[1].scatter(pres.x[s] / 1000, pres.y[s] / 1000, label=f'{cid} ({C.SEX[cid]})' if cid in poor else None, **kw)
ax[0].set(xlabel=f'PC1 ({expl[0]:.0%})', ylabel=f'PC2 ({expl[1]:.0%})', title='(a) Environmental space of presences')
ax[1].set(xlabel='Easting (km)', ylabel='Northing (km)', title='(b) Geographic space of presences')
for a in ax:
    a.legend(markerscale=4, frameon=False, fontsize=8)
    a.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig(C.OUT_FIG + '/FigS_collar_profiles.png', dpi=300)
plt.close(fig)
pd.DataFrame(Vt[:2].T, index=C.CONTINUOUS, columns=['PC1', 'PC2']).round(3).to_csv(C.OUT_TAB + '/S7_pca_loadings.csv')

pts.drop(columns='geometry').to_csv(C.OUT_DATA + '/model_points_with_folds.csv', index=False)
log('ALL DONE')
