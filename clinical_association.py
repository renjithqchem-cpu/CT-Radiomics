import argparse, os, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2, chi2_contingency, fisher_exact, mannwhitneyu, norm
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True); ap.add_argument("--clinical", required=True)
ap.add_argument("--features"); ap.add_argument("--out", default="clinical_outputs")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

L = pd.read_csv(a.labels); L.columns = [c.strip() for c in L.columns]
lab_id = next(c for c in L.columns if "patient" in c.lower() or c.lower() == "id")
lab_y = next(c for c in L.columns if "pheno" in c.lower() or "class" in c.lower() or "cluster" in c.lower())
L = L.rename(columns={lab_id: "patient_id", lab_y: "pheno"})[["patient_id", "pheno"]]
C = pd.read_csv(a.clinical); C.columns = [c.strip() for c in C.columns]
C = C.rename(columns={"PatientID": "patient_id", "clinical.T.Stage": "T", "Clinical.N.Stage": "N", "Clinical.M.Stage": "M",
                      "Overall.Stage": "stage", "Survival.time": "time", "deadstatus.event": "event"})
M = L.merge(C, on="patient_id", how="inner")
print(f"labels {len(L)}, clinical rows {len(C)}, matched {len(M)}; unmatched labels: {sorted(set(L.patient_id) - set(C.patient_id))}")
print("phenotype sizes:", M.pheno.value_counts().sort_index().to_dict())
if a.features:
    F = pd.read_csv(a.features); F.columns = [c.strip() for c in F.columns]
    fid = next(c for c in F.columns if "patient" in c.lower() or c.lower() == "id")
    vc = next((c for c in F.columns if c.lower().endswith("shape_voxelvolume")), None) or next(c for c in F.columns if c.lower().endswith("shape_meshvolume"))  # VoxelVolume first, as in the manuscript
    M = M.merge(F[[fid, vc]].rename(columns={fid: "patient_id", vc: "vol"}), on="patient_id", how="left")
    M["logvol"] = np.log(M.vol)

def cramers_v(tab):
    chi = chi2_contingency(tab, correction=False)[0]; n = tab.values.sum()
    return float(np.sqrt(chi / (n * (min(tab.shape) - 1))))
rows = []
def cat_test(col, name, drop=()):
    s = M[~M[col].isin(drop)].dropna(subset=[col]); tab = pd.crosstab(s[col], s.pheno)
    if tab.shape[0] < 2: return
    chi, p, dof, exp = chi2_contingency(tab)
    t = "chi2"
    if (exp < 5).any() and tab.shape == (2, 2): p = fisher_exact(tab)[1]; t = "Fisher"
    elif (exp < 5).any(): t = "chi2 (some expected counts <5)"
    rows.append(dict(variable=name, test=t, n=len(s), statistic=chi, dof=dof, p=p, effect=f"Cramer V={cramers_v(tab):.3f}"))
    print(f"\n{name} (n={len(s)})\n{tab.to_string()}\n  row % in phenotype 1: " + ", ".join(f"{i}:{100*tab.loc[i].iloc[-1]/tab.loc[i].sum():.0f}%" for i in tab.index) + f"   {t} p={p:.4g}")
def ord_test(col, name):
    s = M.dropna(subset=[col]); x0 = s.loc[s.pheno == s.pheno.min(), col]; x1 = s.loc[s.pheno == s.pheno.max(), col]
    u = mannwhitneyu(x0, x1); auc = u.statistic / (len(x0) * len(x1))
    rows.append(dict(variable=name, test="Mann-Whitney", n=len(s), statistic=u.statistic, dof=np.nan, p=u.pvalue, effect=f"P(phenotype0 > phenotype1)={auc:.3f}"))
    print(f"\n{name} (n={len(s)}): median phenotype0 {x0.median():.2f}, phenotype1 {x1.median():.2f}; MWU p={u.pvalue:.4g}; P(0>1)={auc:.3f}")

M["stage_n"] = M.stage.map({"I": 1, "II": 2, "IIIa": 3, "IIIb": 4})
cat_test("stage", "Overall stage"); ord_test("stage_n", "Overall stage (ordinal I<II<IIIa<IIIb)")
cat_test("T", "Clinical T stage"); ord_test("T", "Clinical T stage (ordinal)")
cat_test("N", "Clinical N stage", drop=(4,)); ord_test("N", "Clinical N stage (ordinal)")
cat_test("Histology", "Histology (all categories, NA excluded)")
cat_test("Histology", "Histology (squamous vs adenocarcinoma vs large cell; nos excluded)", drop=("nos",))
M["histo_sq"] = M.Histology.map(lambda h: np.nan if pd.isna(h) or h == "nos" else ("squamous" if h.startswith("squam") else "non-squamous"))
cat_test("histo_sq", "Squamous vs non-squamous (nos and NA excluded)")
cat_test("gender", "Sex"); ord_test("age", "Age (years)")
if "logvol" in M: ord_test("logvol", "log tumour volume (for reference)")
R = pd.DataFrame(rows); R["p_bonf"] = np.minimum(1, R.p * len(R)); R.to_csv(f"{a.out}/clinical_association.csv", index=False)
print("\nSummary (Bonferroni over", len(R), "tests)\n", R[["variable", "test", "n", "p", "p_bonf", "effect"]].round(4).to_string(index=False))

# ---------------- survival ----------------
def logrank(t, e, g):
    t, e, g = map(np.asarray, (t, e, g)); O = E = V = 0.0
    for tt in np.unique(t[e == 1]):
        r = t >= tt; n = r.sum(); n1 = (r & (g == 1)).sum(); d = ((t == tt) & (e == 1)).sum(); d1 = ((t == tt) & (e == 1) & (g == 1)).sum()
        O += d1; E += d * n1 / n
        if n > 1: V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    z2 = (O - E) ** 2 / V; return z2, chi2.sf(z2, 1), O, E
def cox(t, e, X, names):
    t = np.asarray(t, float); e = np.asarray(e, int); X = np.asarray(X, float); n, p = X.shape
    order = np.argsort(-t, kind="stable"); t, e, X = t[order], e[order], X[order]
    beta = np.zeros(p)
    def ll_grad_hess(b):
        eta = X @ b; w = np.exp(eta); ll = 0.0; g = np.zeros(p); H = np.zeros((p, p))
        # risk set sums over times >= current (including ties): process unique times
        cw = np.cumsum(w); cwx = np.cumsum(w[:, None] * X, axis=0); cwxx = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        ut = np.unique(t[e == 1])
        for tt in ut:
            idx = np.where((t == tt) & (e == 1))[0]; last = np.where(t >= tt)[0].max()
            d = len(idx); s0 = cw[last]; s1 = cwx[last]; s2 = cwxx[last]
            ll += (X[idx] @ b).sum() - d * np.log(s0); g += X[idx].sum(0) - d * s1 / s0
            H -= d * (s2 / s0 - np.outer(s1, s1) / s0 ** 2)
        return ll, g, H
    for _ in range(50):
        ll, g, H = ll_grad_hess(beta); step = np.linalg.solve(-H, g); beta = beta + step
        if np.abs(step).max() < 1e-9: break
    ll, g, H = ll_grad_hess(beta); se = np.sqrt(np.diag(np.linalg.inv(-H))); z = beta / se
    return pd.DataFrame(dict(term=names, HR=np.exp(beta), lo=np.exp(beta - 1.96 * se), hi=np.exp(beta + 1.96 * se), p=2 * norm.sf(abs(z))))
S = M.dropna(subset=["time", "event"]).copy(); S["g"] = (S.pheno == S.pheno.max()).astype(int)
z2, p, O, E = logrank(S.time, S.event, S.g)
med = {k: float(np.median(v.time)) for k, v in S.groupby("g")}
print(f"\nSURVIVAL (n={len(S)}, events={int(S.event.sum())}): log-rank chi2={z2:.3f}, p={p:.4g}; observed/expected deaths in phenotype 1 = {O:.0f}/{E:.1f}; median follow-up/survival time (days, crude) phenotype0 {med[0]:.0f}, phenotype1 {med[1]:.0f}")
out = [dict(model="log-rank, phenotype 1 vs 0", n=len(S), p=p, HR=np.nan, lo=np.nan, hi=np.nan)]
c1 = cox(S.time, S.event, S[["g"]], ["phenotype 1 vs 0"]); print("Cox, phenotype alone:\n", c1.round(4).to_string(index=False))
out.append(dict(model="Cox phenotype alone", n=len(S), p=c1.p[0], HR=c1.HR[0], lo=c1.lo[0], hi=c1.hi[0]))
S2 = S.dropna(subset=["stage_n"]).copy(); cov = ["g", "stage_n"]; nm = ["phenotype 1 vs 0", "stage (per category)"]
if "logvol" in S2: cov.append("logvol"); nm.append("log volume")
S3 = S2.dropna(subset=cov)
c2 = cox(S3.time, S3.event, S3[cov], nm); print(f"Cox adjusted for {', '.join(nm[1:])} (n={len(S3)}):\n", c2.round(4).to_string(index=False))
out.append(dict(model="Cox phenotype + " + " + ".join(nm[1:]), n=len(S3), p=c2.p[0], HR=c2.HR[0], lo=c2.lo[0], hi=c2.hi[0]))
if "logvol" in S3:
    c3 = cox(S3.time, S3.event, S3[["logvol"]], ["log volume"]); print("Cox, log volume alone (n=%d):\n" % len(S3), c3.round(4).to_string(index=False))
pd.DataFrame(out).to_csv(f"{a.out}/survival_summary.csv", index=False)
print("\nNote: the TCIA 'Survival.time' is in days; deadstatus.event 1 = death. This is an exploratory, unadjusted-for-multiplicity association in a cohort in which labels are scanner- and size-associated.")
