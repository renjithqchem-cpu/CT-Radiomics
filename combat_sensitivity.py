import sys, os, json, time
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, kruskal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from combat import combat
SEED = 20260826
THR = 0.9

def impute(X):
    X = X.replace([np.inf, -np.inf], np.nan); return X.fillna(X.median())
def onepass(X, thr=THR):
    from scipy.stats import spearmanr as sp
    C = np.abs(sp(X.values).statistic); n = X.shape[1]; drop = set()
    for j in range(n):
        for i in range(j):
            if C[i, j] > thr: drop.add(j); break
    return [c for k, c in enumerate(X.columns) if k not in drop]
def km(Z, k, seed=SEED, n_init=10): return KMeans(k, n_init=n_init, random_state=seed).fit_predict(Z)
def sils(Z): return {k: float(silhouette_score(Z, km(Z, k))) for k in (2, 3, 4, 5)}
def gap(Z, B=50, seed=SEED):
    rng = np.random.default_rng(seed); n = Z.shape[0]
    P = PCA().fit(Z); T = P.transform(Z); lo, hi = T.min(0), T.max(0)
    def W(X, k):
        m = KMeans(k, n_init=5, random_state=seed).fit(X); return m.inertia_
    ks = [1, 2, 3, 4, 5]; g = []; s = []
    for k in ks:
        lw = np.log(W(Z, k)); ref = []
        for b in range(B):
            R = rng.uniform(lo, hi, size=T.shape); ref.append(np.log(W(R, k)))
        g.append(np.mean(ref) - lw); s.append(np.std(ref) * np.sqrt(1 + 1 / B))
    sel = next((k for i, k in enumerate(ks[:-1]) if g[i] >= g[i + 1] - s[i + 1]), ks[-1])
    return g, s, sel
def null_p(X, obs, nperm, seed=SEED):
    rng = np.random.default_rng(seed); cnt = 0; vals = []
    A = X.values.copy()
    for _ in range(nperm):
        P = np.column_stack([rng.permutation(A[:, j]) for j in range(A.shape[1])])
        Z = StandardScaler().fit_transform(P)
        v = max(silhouette_score(Z, KMeans(k, n_init=3, random_state=int(rng.integers(1e9))).fit_predict(Z)) for k in (2, 3, 4, 5))
        vals.append(v); cnt += v >= obs
    return (cnt + 1) / (nperm + 1), cnt, float(np.mean(vals)), float(np.percentile(vals, 95))
def jaccard_boot(Xfull, ref, B, seed=SEED):
    rng = np.random.default_rng(seed); n = len(ref); J = [[], []]
    for _ in range(B):
        idx = rng.integers(0, n, n); u = np.unique(idx)
        Xb = impute(Xfull.iloc[idx]); keep = onepass(Xb)
        Zb = StandardScaler().fit_transform(Xb[keep].values)
        lb = KMeans(2, n_init=5, random_state=int(rng.integers(1e9))).fit_predict(Zb)
        # bootstrap clusters restricted to patients in the resample (use first occurrence)
        first = {}
        for pos, i in enumerate(idx):
            first.setdefault(i, lb[pos])
        bl = np.array([first[i] for i in u]); rl = ref[u]
        M = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                A = rl == a; Bc = bl == b; M[a, b] = (A & Bc).sum() / max((A | Bc).sum(), 1)
        r, c = linear_sum_assignment(-M)
        for a, b in zip(r, c): J[a].append(M[a, b])
    return [float(np.mean(j)) for j in J], [float(np.percentile(j, 2.5)) for j in J], [float(np.percentile(j, 97.5)) for j in J], [float(np.mean(np.array(j) < 0.5)) for j in J]
def batch_tests(Xk, batch):
    """number of retained features with a batch effect (Kruskal-Wallis, Bonferroni at 0.05)."""
    ps = []
    for c in Xk.columns:
        groups = [Xk[c].values[batch == b] for b in np.unique(batch)]
        ps.append(kruskal(*groups).pvalue)
    ps = np.array(ps); return int((ps < 0.05 / len(ps)).sum()), float(np.median(ps))
def run(name, feats, acq, ref, logvol, batch, harm_cols, covar, nperm, nboot, out, size_removed=False):
    t0 = time.time(); X = impute(feats)
    if batch is not None:
        H = X.copy()
        H[harm_cols] = combat(X[harm_cols].values, batch, covars=(logvol.reshape(-1, 1) if covar else None))
        X = H
    if size_removed:   # regress every descriptor on log volume (linear + quadratic); drop volume defining descriptors
        L = np.column_stack([np.ones(len(logvol)), logvol, logvol ** 2])
        R = X.values - L @ np.linalg.lstsq(L, X.values, rcond=None)[0]
        X = pd.DataFrame(R, index=X.index, columns=X.columns).drop(columns=['original_shape_VoxelVolume', 'original_shape_MeshVolume'])
    keep = onepass(X); Xk = X[keep]; Z = StandardScaler().fit_transform(Xk.values)
    S = sils(Z); l = km(Z, 2); sizes = sorted(np.bincount(l).tolist(), reverse=True)
    ari = adjusted_rand_score(ref, l)
    g, s, sel = gap(Z)
    p, cnt, nm, n95 = null_p(Xk, S[2], nperm)
    jm, jlo, jhi, jb5 = jaccard_boot(X, l, nboot)
    # associations of the new partition
    mdl = acq['model_grp'].values
    ct = pd.crosstab(l, mdl); chi_m = chi2_contingency(ct)
    ct2 = pd.crosstab(l, acq['kernel_grp'].values); chi_k = chi2_contingency(ct2)
    vol = np.exp(logvol)
    mw = mannwhitneyu(vol[l == 0], vol[l == 1]).pvalue; rho = spearmanr(l, logvol).statistic
    pc1 = PCA(1).fit_transform(Z)[:, 0]; rho_pc1 = abs(spearmanr(pc1, logvol).statistic)
    nb, med_p = batch_tests(Xk, mdl)
    # threshold sensitivity
    thr = {}
    for t in (0.80, 0.85, 0.95):
        kt = onepass(X, t); Zt = StandardScaler().fit_transform(X[kt].values)
        thr[t] = dict(features=len(kt), ARI_vs_ref090=float(adjusted_rand_score(l, km(Zt, 2))), sil=float(silhouette_score(Zt, km(Zt, 2))))
    res = dict(name=name, retained=len(keep), sil=S, sizes=sizes, ARI_vs_reported=float(ari), gap=g, gap_se=s, gap_k=sel,
               null_p=p, null_exceed=cnt, null_mean=nm, null_p95=n95, jaccard_mean=jm, jaccard_lo=jlo, jaccard_hi=jhi, jaccard_below05=jb5,
               chi2_model=float(chi_m.statistic), p_model=float(chi_m.pvalue), df_model=int(chi_m.dof),
               chi2_kernel=float(chi_k.statistic), p_kernel=float(chi_k.pvalue), df_kernel=int(chi_k.dof),
               p_volume=float(mw), rho_label_logvol=float(rho), pc1_logvol_abs_rho=float(rho_pc1),
               batch_effect_features_after=nb, median_kw_p=med_p, threshold=thr, seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(out, f"harm_{name}.json"), "w"), indent=1, default=float)
    pd.DataFrame({'patient_id': feats.index, 'cluster': l}).to_csv(os.path.join(out, f"harm_{name}_labels.csv"), index=False)
    print(name, json.dumps({k: res[k] for k in ('retained', 'sil', 'sizes', 'ARI_vs_reported', 'gap_k', 'null_p', 'jaccard_mean', 'p_model', 'p_kernel', 'p_volume', 'batch_effect_features_after')}, default=float), flush=True)
    return res
if __name__ == "__main__":
    feat_csv, acq_csv, lab_csv, outdir = sys.argv[1:5]
    nperm = int(sys.argv[5]) if len(sys.argv) > 5 else 2000; nboot = int(sys.argv[6]) if len(sys.argv) > 6 else 500
    os.makedirs(outdir, exist_ok=True)
    f = pd.read_csv(feat_csv).set_index('patient_id'); a = pd.read_csv(acq_csv).set_index('patient_id').loc[f.index]
    ref = pd.read_csv(lab_csv).set_index('patient_id').loc[f.index, 'phenotype_class'].values
    logvol = np.log(f['original_shape_VoxelVolume'].values)
    m = a['model'].fillna('XiO').replace({'Sensation 10': 'Sensation 10/16', 'Sensation 16': 'Sensation 10/16'})
    a['model_grp'] = m.values
    k = a['kernel'].fillna('NA'); k = k.where(~k.isin(['B30s', 'B19s', 'B10f']), 'other'); a['kernel_grp'] = k.values
    mk = (m.astype(str) + '|' + a['kvp'].fillna(0).astype(int).astype(str)).values
    nonshape = [c for c in f.columns if '_shape_' not in c]; allc = list(f.columns)
    schemes = [('baseline', None, None, False, False), ('sizeonly', None, None, False, True),
               ('model', a['model_grp'].values, nonshape, True, False),
               ('model_sizeremoved', a['model_grp'].values, nonshape, True, True),
               ('kernel_sizeremoved', a['kernel_grp'].values, nonshape, True, True),
               ('model_nocov', a['model_grp'].values, nonshape, False, False),
               ('model_allfeat', a['model_grp'].values, allc, True, False),
               ('kernel', a['kernel_grp'].values, nonshape, True, False),
               ('model_kvp', mk, nonshape, True, False)]
    only = sys.argv[7].split(',') if len(sys.argv) > 7 else None
    for name, b, cols, cov, sr in schemes:
        if only and name not in only: continue
        run(name, f, a, ref, logvol, b, cols, cov, nperm, nboot, outdir, sr)
