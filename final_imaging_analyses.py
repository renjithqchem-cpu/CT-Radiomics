import argparse, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--null_scores")
ap.add_argument("--observed", type=float, default=0.228)
ap.add_argument("--seed", type=int, default=20260826)
ap.add_argument("--out", default="results_final")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)

F = pd.read_csv(a.features); L = pd.read_csv(a.labels)
df = F.merge(L[["patient_id", "phenotype_class"]], on="patient_id")
y = df.pop("phenotype_class").values; df.pop("patient_id")
X = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.nunique() > 1]
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
print(f"patients={len(y)}, numeric features={X.shape[1]}, class sizes={np.bincount(y)}")


def one_shot(M, thr):
    """Archived rule: drop column j if ANY earlier column i (kept or dropped) has |Spearman r| > thr."""
    C = np.abs(spearmanr(M.values).correlation)
    keep = [j for j in range(C.shape[0]) if not np.any(C[:j, j] > thr)]
    return M.columns[keep].tolist()


def km(Z, k):
    return KMeans(k, n_init=50, random_state=a.seed).fit_predict(Z)


# ---------------------------------------------------------------- 1. phenotype description
feats = one_shot(X, 0.90)
print(f"\n[1] one-shot retained at 0.90: {len(feats)} features")
Z = StandardScaler().fit_transform(X[feats])
rf = RandomForestClassifier(500, class_weight="balanced", random_state=a.seed).fit(Z, y)
rows = []
for j, f in enumerate(feats):
    v0, v1 = X[f].values[y == 0], X[f].values[y == 1]
    sp = np.sqrt(((len(v0) - 1) * v0.var(ddof=1) + (len(v1) - 1) * v1.var(ddof=1)) / (len(y) - 2))
    rows.append(dict(feature=f, RF_importance=rf.feature_importances_[j],
                     cohens_d_class1_minus_class0=(v1.mean() - v0.mean()) / sp if sp > 0 else np.nan,
                     median_class0=np.median(v0), median_class1=np.median(v1)))
d1 = pd.DataFrame(rows).sort_values("RF_importance", ascending=False)
d1.to_csv(f"{a.out}/phenotype_feature_description.csv", index=False)
print(d1.round(4).to_string(index=False))
print("retained features:", ", ".join(feats))

# ---------------------------------------------------------------- 2. size-removed threshold comparison
vol_col = next(c for c in X.columns if c.lower().endswith("shape_voxelvolume"))
logv = np.log(X[vol_col].values.astype(float))
D = np.c_[np.ones(len(logv)), logv, logv ** 2]
drop_cols = [c for c in X.columns if c.lower().endswith(("shape_voxelvolume", "shape_meshvolume"))]
R = X.drop(columns=drop_cols).copy()
for c in R.columns:
    beta, *_ = np.linalg.lstsq(D, R[c].values.astype(float), rcond=None)
    R[c] = R[c].values - D @ beta
R = R.loc[:, R.std() > 1e-9 * (X[R.columns].std() + 1e-300)]
print(f"\n[2] size-removed matrix: {R.shape[1]} features (volume-defining columns {drop_cols} dropped; "
      f"residuals of log-volume linear+quadratic fit)")
thr_list = [0.80, 0.85, 0.90, 0.95]
res, store = [], {}
for t in thr_list:
    fs = one_shot(R, t)
    Zt = StandardScaler().fit_transform(R[fs])
    sil = {k: silhouette_score(Zt, km(Zt, k)) for k in (2, 3, 4, 5)}
    lab = km(Zt, 2)
    pc1 = PCA(n_components=1, random_state=0).fit_transform(Zt)[:, 0]
    store[t] = dict(lab=lab, pc1=pc1)
    res.append(dict(threshold=t, n_features=len(fs), S2=sil[2], S3=sil[3], best_k=max(sil, key=sil.get),
                    sizes="%d:%d" % tuple(np.bincount(lab)),
                    ARI_vs_reported=adjusted_rand_score(y, lab),
                    partition_vs_logvol_abs_rho=abs(spearmanr(lab, logv).correlation),
                    PC1_vs_logvol_abs_rho=abs(spearmanr(pc1, logv).correlation)))
ref = store[0.90]
for r in res:
    t = r["threshold"]
    r["ARI_vs_sizeremoved_ref"] = adjusted_rand_score(ref["lab"], store[t]["lab"])
    r["PC1_abs_rho_vs_ref"] = abs(spearmanr(store[t]["pc1"], ref["pc1"]).correlation)
res = pd.DataFrame(res)
res.to_csv(f"{a.out}/size_removed_threshold_comparison.csv", index=False)
print(res.round(3).to_string(index=False))
print("Reading: high ARI/PC1 agreement across thresholds => non-size structure is reproducible; low => analysis dependent.")

# ---------------------------------------------------------------- 3. null distribution summary
if a.null_scores and os.path.exists(a.null_scores):
    N = pd.read_csv(a.null_scores)
    col = N.select_dtypes(include=[np.number]).columns[-1]
    v = N[col].dropna().values
    b = int(np.sum(v >= a.observed)); m = len(v)
    print(f"\n[3] null scores: m={m}, mean={v.mean():.4f}, sd={v.std(ddof=1):.4f}, min={v.min():.3f}, max={v.max():.3f}")
    print("    quantiles 50/90/95/99/99.9%:", np.round(np.percentile(v, [50, 90, 95, 99, 99.9]), 3))
    print(f"    exceedances b={b}; p=(b+1)/(m+1)={(b + 1) / (m + 1):.4f}; observed={a.observed}")
    edges = np.linspace(0.02, 0.42, 21)
    cnt, _ = np.histogram(v, bins=edges)
    print("    histogram bin edges:", np.round(edges, 3).tolist())
    print("    histogram counts   :", cnt.tolist())
print("\nDone. CSVs in", a.out)
