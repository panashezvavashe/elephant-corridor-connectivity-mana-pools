"""Statistical helpers for the Objective 2 revision: DeLong, McNemar, TSS, MaxTSS,
paired and cluster bootstraps, grouped permutation importance, 1-D ALE."""
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, confusion_matrix

THRESHOLDS = np.round(np.arange(0.10, 0.90, 0.01), 2)  # same search grid as the manuscript


def tss(y, yhat):
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return tp / (tp + fn) + tn / (tn + fp) - 1


def max_tss_threshold(y, p):
    scores = [tss(y, (p >= t).astype(int)) for t in THRESHOLDS]
    k = int(np.argmax(scores))
    return float(THRESHOLDS[k]), float(scores[k])


# ── DeLong test for two correlated ROC curves (Sun & Xu 2014 fast algorithm) ──
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, p1, p2):
    y = np.asarray(y)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs, float(z), float(2 * stats.norm.sf(abs(z)))


def mcnemar(y, yhat1, yhat2):
    c1, c2 = yhat1 == y, yhat2 == y
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return chi2, float(stats.chi2.sf(chi2, 1)), b, c


# ── Bootstraps ────────────────────────────────────────────────────────────────
def _summ(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def _p_two_sided(diff):
    diff = np.asarray(diff)
    diff = diff[np.isfinite(diff)]
    return float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))


def cluster_bootstrap(y, p_rf, p_svm, clusters, thr_rf, thr_svm, B=2000, seed=42):
    """Resample whole clusters (individuals or spatial blocks) with replacement,
    keeping RF and SVM predictions paired on the same rows in every replicate."""
    rng = np.random.default_rng(seed)
    y, p_rf, p_svm, clusters = map(np.asarray, (y, p_rf, p_svm, clusters))
    ids = np.unique(clusters)
    rows = {c: np.where(clusters == c)[0] for c in ids}
    out = {k: [] for k in ('auc_rf', 'auc_svm', 'auc_diff', 'tss_rf', 'tss_svm', 'tss_diff')}
    for _ in range(B):
        idx = np.concatenate([rows[c] for c in rng.choice(ids, len(ids), replace=True)])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        a1, a2 = roc_auc_score(yy, p_rf[idx]), roc_auc_score(yy, p_svm[idx])
        t1 = tss(yy, (p_rf[idx] >= thr_rf).astype(int))
        t2 = tss(yy, (p_svm[idx] >= thr_svm).astype(int))
        for k, v in zip(out, (a1, a2, a1 - a2, t1, t2, t1 - t2)):
            out[k].append(v)
    res = {k: _summ(v) for k, v in out.items()}
    res['p_auc_diff'] = _p_two_sided(out['auc_diff'])
    res['p_tss_diff'] = _p_two_sided(out['tss_diff'])
    res['n_clusters'] = len(ids)
    res['B_effective'] = len(out['auc_diff'])
    return res


def paired_bootstrap_metrics(y, p_rf, p_svm, metric_fns, B=2000, seed=42):
    """Row-level stratified paired bootstrap (the manuscript's original Table 2 CIs)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    store = {k: ([], [], []) for k in metric_fns}
    for _ in range(B):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, f in metric_fns.items():
            a, b = f(y[idx], p_rf[idx]), f(y[idx], p_svm[idx])
            store[k][0].append(a)
            store[k][1].append(b)
            store[k][2].append(a - b)
    return {k: tuple(_summ(v) for v in s) for k, s in store.items()}


# ── Grouped permutation importance (identical procedure for both classifiers) ──
def grouped_permutation_importance(model, X, y, groups, n_repeats=20, seed=42):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    res = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].values[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        res[name] = (float(np.mean(drops)), float(np.std(drops)))
    return base, res


# ── First-order accumulated local effects (Apley & Zhu 2020) ──────────────────
def ale_1d(model, X, feat, n_bins=20):
    x = X[feat].values
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    k = np.clip(np.searchsorted(edges, x, side='left'), 1, len(edges) - 1)
    lo, hi = X.copy(), X.copy()
    lo[feat] = edges[k - 1]
    hi[feat] = edges[k]
    d = model.predict_proba(hi)[:, 1] - model.predict_proba(lo)[:, 1]
    n_k = np.bincount(k, minlength=len(edges))[1:]
    mean_d = np.bincount(k, weights=d, minlength=len(edges))[1:] / np.maximum(n_k, 1)
    ale = np.concatenate([[0.0], np.cumsum(mean_d)])
    mid_ale = (ale[:-1] + ale[1:]) / 2
    ale -= np.sum(mid_ale * n_k) / n_k.sum()
    return edges, ale
