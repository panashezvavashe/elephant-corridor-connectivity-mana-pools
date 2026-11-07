"""Reviewer 2 #6: widened hyperparameter grids.

RF adds max_features (mtry); SVM extends C and gamma beyond the edge of the original grid.
Both grids contain every original value, so the selected (current) settings are scored on the
same folds as the alternatives. Same training split, inner folds and scoring as r1_models.py.
Outputs: tables/S16_grid_rf.csv, S16_grid_svm.csv, S16_grid_summary.csv
"""
import r1_config as C  # noqa: F401  (sets PROJ paths and thread caps first)

C.lower_priority()

import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

T0 = time.time()


def log(msg):
    print(f'[{(time.time() - T0) / 60:6.1f} min] {msg}', flush=True)


pts = gpd.read_file(C.OUT_DATA + '/model_points.gpkg')
for c in C.LULC_CLASSES:
    pts[f'LULC_{c}'] = (pts['LandCover'] == c).astype(int)
X = pts[C.FEATURES].astype(float)
y = pts['class'].astype(int)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=C.SEED, stratify=y)
X_tr, X_te, y_tr, y_te = X.iloc[idx_tr], X.iloc[idx_te], y.iloc[idx_tr], y.iloc[idx_te]
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)
current = json.load(open(C.OUT_DATA + '/model_settings.json'))

grids = {
    'RF': (Pipeline([('classifier', RandomForestClassifier(random_state=C.SEED, n_jobs=1))]),
           {'classifier__n_estimators': [100, 300, 500], 'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__max_features': ['sqrt', 0.33, 0.5, 1.0]}),
    'SVM': (Pipeline([('scaler', StandardScaler()),
                      ('classifier', SVC(probability=True, random_state=C.SEED, cache_size=1000))]),
            {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__gamma': ['scale', 0.01, 0.1, 1, 10],
             'classifier__kernel': ['rbf']}),
}
cur_params = {
    'RF': {'n_estimators': 300, 'max_depth': None, 'min_samples_split': 5, 'max_features': 'sqrt'},  # sqrt = default
    'SVM': {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'},
}
summary = []
for m, (pipe, grid) in grids.items():
    gs = GridSearchCV(pipe, grid, cv=inner, scoring='roc_auc', n_jobs=C.N_JOBS).fit(X_tr, y_tr)
    res = pd.DataFrame(gs.cv_results_)
    keep = [c for c in res.columns if c.startswith('param_')] + ['mean_test_score', 'std_test_score',
                                                                 'rank_test_score', 'mean_fit_time']
    res = res[keep].rename(columns=lambda c: c.replace('param_classifier__', ''))
    res.sort_values('rank_test_score').round(5).to_csv(f'{C.OUT_TAB}/S16_grid_{m.lower()}.csv', index=False)
    is_cur = np.ones(len(res), bool)
    for k, v in cur_params[m].items():
        col = res[k].astype(object)
        is_cur &= np.array([(a is None and v is None) or a == v for a in col])
    cur_row = res[is_cur].iloc[0]
    best_p = {k.split('__')[1]: v for k, v in gs.best_params_.items()}
    best_hold = roc_auc_score(y_te, gs.best_estimator_.predict_proba(X_te)[:, 1])
    pipe_cur = pipe.set_params(**{f'classifier__{k}': v for k, v in cur_params[m].items()}).fit(X_tr, y_tr)
    cur_hold = roc_auc_score(y_te, pipe_cur.predict_proba(X_te)[:, 1])
    summary.append({'Model': m, 'combinations': len(res),
                    'best_params_wide_grid': json.dumps(best_p, default=str),
                    'best_CV_AUC': round(gs.best_score_, 4),
                    'current_params': json.dumps(cur_params[m], default=str),
                    'current_CV_AUC': round(cur_row.mean_test_score, 4),
                    'current_rank_in_wide_grid': int(cur_row.rank_test_score),
                    'CV_AUC_gain': round(gs.best_score_ - cur_row.mean_test_score, 4),
                    'holdout_AUC_best': round(best_hold, 4), 'holdout_AUC_current': round(cur_hold, 4)})
    log(f'{m} done: best {best_p} CV {gs.best_score_:.4f}; current CV {cur_row.mean_test_score:.4f} '
        f'(rank {int(cur_row.rank_test_score)}); holdout best {best_hold:.4f} vs current {cur_hold:.4f}')
pd.DataFrame(summary).to_csv(f'{C.OUT_TAB}/S16_grid_summary.csv', index=False)
log('ALL DONE')
