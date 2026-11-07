#!/usr/bin/env python3
import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kruskal, chi2_contingency, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--clinical")
ap.add_argument("--seed", type=int, default=20260826); ap.add_argument("--nboot", type=int, default=500)
ap.add_argument("--thr", type=float, default=0.9); ap.add_argument("--rule", choices=["one_shot","iterative"], default="one_shot", help="Spearman rule for the bootstrap blocks; one_shot applies the one-pass Spearman rule"); ap.add_argument("--out", default="results_phenotype_stability")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)

F = pd.read_csv(a.features); L = pd.read_csv(a.labels)
df = F.merge(L[["patient_id", "phenotype_class"]], on="patient_id")
y = df.pop("phenotype_class").values; ids = df.pop("patient_id").values
X_all = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
X_all = X_all.loc[:, X_all.nunique() > 1]
X_all = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_all), columns=X_all.columns)   # median imputation (state in text)
print(f"patients={len(y)}, numeric columns={X_all.shape[1]}, classes={np.bincount(y)}")

# ---------- Spearman filters: archived one-shot rule vs iterative 'keep earlier, drop later vs retained' rule ----------
def corr_abs(X): return np.abs(spearmanr(X.values).correlation)
def one_shot(X, thr):                      # archived rule: drop column j if ANY earlier column i has |r|>thr (even if i was dropped)
    C = corr_abs(X); keep = [j for j in range(C.shape[0]) if not np.any(C[:j, j] > thr)]
    return X.columns[keep].tolist()
def iterative(X, thr):                     # manuscript wording: compare only against features already retained
    C = corr_abs(X); keep = []
    for j in range(C.shape[0]):
        if not keep or np.all(C[keep, j] <= thr): keep.append(j)
    return X.columns[keep].tolist()
f1, f2 = one_shot(X_all, a.thr), iterative(X_all, a.thr)
print(f"one-shot retains {len(f1)}, iterative retains {len(f2)}; identical sets: {set(f1)==set(f2)}")
pd.DataFrame({"feature": X_all.columns, "one_shot": X_all.columns.isin(f1), "iterative": X_all.columns.isin(f2)}).to_csv(f"{a.out}/spearman_rule_comparison.csv", index=False)
feats = f1                                                      # retained one-pass feature set (27 features for the 421-patient cohort)
Xs = StandardScaler().fit_transform(X_all[feats])

# ---------- R1-3 / R2-5: classifier restricted to the clustering features, scaling inside each fold ----------
cv = StratifiedKFold(5, shuffle=True, random_state=a.seed); res = {}
for nm, mdl in {"RF": RandomForestClassifier(500, class_weight="balanced", random_state=a.seed),
                "LR": LogisticRegression(max_iter=5000, class_weight="balanced")}.items():
    pipe = make_pipeline(StandardScaler(), mdl)
    pred = cross_val_predict(pipe, X_all[feats], y, cv=cv); res[nm] = balanced_accuracy_score(y, pred)
    # drop-shape ablation: does the partition survive without shape/volume features? (Welch 2019 volume-surrogate concern)
    nos = [f for f in feats if "_shape_" not in f]
    pred2 = cross_val_predict(make_pipeline(StandardScaler(), mdl), X_all[nos], y, cv=cv); res[nm+"_noShape"] = balanced_accuracy_score(y, pred2)
print("restricted, in-fold-scaled balanced accuracy:", res)
pd.Series(res).to_csv(f"{a.out}/classifier_restricted.csv")

# ---------- R1-3: volume confound + clinical associations ----------
vol_col = next((c for c in X_all.columns if c.lower().endswith("shape_voxelvolume")), None) or next((c for c in X_all.columns if "MeshVolume" in c), None)
if vol_col:
    v = X_all[vol_col].values
    rho = spearmanr(v, y).correlation; u = mannwhitneyu(v[y == 0], v[y == 1])
    print(f"volume ({vol_col}) median phenotype0={np.median(v[y==0]):.0f}, phenotype1={np.median(v[y==1]):.0f}; MWU p={u.pvalue:.3g}; point-biserial rho={rho:.3f}")
    # clustering on features after removing every volume-dependent descriptor (|rho|>0.9 with volume) -> ARI vs reported labels
    dep = [f for f in feats if abs(spearmanr(X_all[f], v).correlation) > 0.9]
    rest = [f for f in feats if f not in dep]
    lab = KMeans(2, n_init=50, random_state=a.seed).fit_predict(StandardScaler().fit_transform(X_all[rest]))
    print(f"removed {len(dep)} volume-dependent features; ARI of k=2 on the remainder vs reported = {adjusted_rand_score(y, lab):.3f}")
if a.clinical:
    C = pd.read_csv(a.clinical); C.columns = [c.strip() for c in C.columns]
    key = next(c for c in C.columns if "patient" in c.lower() or c.lower() in ("patientid", "id"))
    C = C.rename(columns={key: "patient_id"}); M = pd.DataFrame({"patient_id": ids, "pheno": y}).merge(C, on="patient_id")
    out = []
    for c in M.columns.difference(["patient_id", "pheno"]):
        s = M[c].dropna()
        if s.nunique() < 2 or s.nunique() > 12 and not pd.api.types.is_numeric_dtype(s): continue
        sub = M.loc[s.index]
        try:
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8: p = mannwhitneyu(sub.loc[sub.pheno == 0, c], sub.loc[sub.pheno == 1, c]).pvalue; t = "MWU"
            else: p = chi2_contingency(pd.crosstab(sub[c], sub.pheno))[1]; t = "chi2"
            out.append({"variable": c, "test": t, "n": len(sub), "p": p})
        except Exception: pass
    o = pd.DataFrame(out).sort_values("p"); o["p_bonf"] = np.minimum(1, o.p*len(o)); o.to_csv(f"{a.out}/clinical_association.csv", index=False)
    print(o.head(12).to_string(index=False))

# ---------- R1-3: gap statistic (Tibshirani 2001), uniform-over-PCA-box reference ----------
def wk(X, k): return KMeans(k, n_init=10, random_state=a.seed).fit(X).inertia_
U, S_, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False); Z = (Xs-Xs.mean(0)) @ Vt.T
B = 50; gaps = []
for k in range(1, 6):
    ref = []
    for b in range(B):
        R = rng.uniform(Z.min(0), Z.max(0), Z.shape); ref.append(np.log(wk(R, k)))
    gaps.append((k, np.mean(ref)-np.log(wk(Z, k)), np.std(ref)*np.sqrt(1+1/B)))
g = pd.DataFrame(gaps, columns=["k", "gap", "s_k"]); g["gap_k_minus_(gap_k+1-s_k+1)"] = g.gap-(g.gap.shift(-1)-g.s_k.shift(-1))
g.to_csv(f"{a.out}/gap_statistic.csv", index=False); print(g.round(4).to_string(index=False), "\n(Tibshirani rule: smallest k with the last column >= 0)")

# ---------- R2-3: alternative algorithms and matched k=2 vs k=3 stability ----------
alt = {"Ward": AgglomerativeClustering(2, linkage="ward").fit_predict(Xs),
       "GMM(full)": GaussianMixture(2, covariance_type="full", n_init=5, random_state=a.seed, reg_covar=1e-3).fit_predict(Xs),
       "GMM(diag)": GaussianMixture(2, covariance_type="diag", n_init=5, random_state=a.seed).fit_predict(Xs)}
rows = [{"method": m, "ARI_vs_reported": adjusted_rand_score(y, l), "sizes": str(np.bincount(l)), "silhouette": silhouette_score(Xs, l)} for m, l in alt.items()]
pd.DataFrame(rows).to_csv(f"{a.out}/alternative_algorithms.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))

from scipy.optimize import linear_sum_assignment
filt = one_shot if a.rule == "one_shot" else iterative
def hennig_like(ref, bl, k):
    """Same estimator as exact_pipeline_bootstrap.py: labels of the bootstrap-sample rows,
    Hungarian matching to the reference labels of the same rows, per-cluster Jaccard on those rows."""
    cont = np.zeros((k, k), int)
    for r_, b_ in zip(ref, bl): cont[r_, b_] += 1
    rows, cols = linear_sum_assignment(-cont); mp = {c: r for r, c in zip(rows, cols)}
    m = np.array([mp[x] for x in bl])
    return [np.sum((ref == c) & (m == c))/max(1, np.sum((ref == c) | (m == c))) for c in range(k)]
def boot_stability(k, nb, full_pipeline):
    ref_all = KMeans(k, n_init=20, random_state=42).fit_predict(Xs) if k != 2 else y
    J = []; nfeat = []; n = len(ref_all)
    for b in range(nb):
        idx = rng.integers(0, n, n)
        if full_pipeline:
            Xb = X_all.iloc[idx].reset_index(drop=True); fb = filt(Xb, a.thr); nfeat.append(len(fb)); Z_ = StandardScaler().fit_transform(Xb[fb])
        else: Z_ = StandardScaler().fit_transform(X_all.iloc[idx][feats])   # standardize WITHIN the bootstrap sample, as exact_pipeline_bootstrap.py does
        bl = KMeans(k, n_init=20, random_state=43+b).fit_predict(Z_)
        J.append(hennig_like(ref_all[idx], bl, k))
    return np.array(J), nfeat
for k in (2, 3):
    J, _ = boot_stability(k, a.nboot, False); print(f"fixed-feature bootstrap Jaccard (fixed-feature estimator) k={k}: {J.mean(0).round(3)}")
    pd.DataFrame(J).to_csv(f"{a.out}/jaccard_fixed_k{k}.csv", index=False)
J, nfeat = boot_stability(2, a.nboot, True)
lo, hi = np.percentile(J, [2.5, 97.5], axis=0)
print(f"FULL-pipeline bootstrap ({a.nboot}, rule={a.rule}): mean Jaccard class0 {J[:,0].mean():.3f} [2.5-97.5%: {lo[0]:.3f}-{hi[0]:.3f}], class1 {J[:,1].mean():.3f} [{lo[1]:.3f}-{hi[1]:.3f}]; retained features {min(nfeat)}-{max(nfeat)} (median {np.median(nfeat)})")
print(f"  share of resamples with Jaccard < 0.5: class0 {np.mean(J[:,0] < 0.5):.2f}, class1 {np.mean(J[:,1] < 0.5):.2f}; share >= 0.75: class0 {np.mean(J[:,0] >= 0.75):.2f}, class1 {np.mean(J[:,1] >= 0.75):.2f}")
pd.DataFrame(J, columns=["J_cluster0", "J_cluster1"]).assign(n_features=nfeat).to_csv(f"{a.out}/full_pipeline_bootstrap.csv", index=False)
