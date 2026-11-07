import argparse, os
import numpy as np, pandas as pd
from scipy.special import erf

ap = argparse.ArgumentParser(); ap.add_argument("--nsim", type=int, default=10000)
ap.add_argument("--seed", type=int, default=20260826); ap.add_argument("--out", default="results_dea_scenarios")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

# ---- LEE histogram: 3-constraint reconstruction (P0=0.27, sum=1, mean=9 eV, bins 1..24 linear) ----
def lee_hist(p0=0.27, mean=9.0):
    k = np.arange(1, 25); c = k + 0.5
    # p_k = A - B*(k-1); solve sum=1-p0 and sum(c p)=mean-0.5*p0
    M = np.array([[24, -np.sum(k-1)], [np.sum(c), -np.sum(c*(k-1))]])
    A, B = np.linalg.solve(M, [1-p0, mean-0.5*p0])
    return np.r_[p0, A - B*(k-1)], A, B
P, A, B = lee_hist()
print(f"LEE reconstruction: bin1={P[1]:.8f} (manuscript 0.03666667), bin3={P[3]:.8f} (0.03557971), step={B:.7f}")
print(f"sum={P.sum():.9f}  mean={np.sum((np.arange(25)+0.5)*P):.6f}  min bin={P.min():.5f}")
print("=> bins 1-24 fall on a straight line: R1's observation is reproduced by 3 constraints\n")

# ---- records (Table 2) ----
rec = pd.DataFrame({"name": ["5-bromouracil","5-iodouridine","5-fluorouridine","2-nitrofuran","catechin sulfate"],
    "E": [0.0, 0.0, 0.06, 3.1, 1.1128], "sigma": [6.0e4, 2.7e4, 6800., 780., 4.61]})
rec["bin"] = np.floor(rec.E).astype(int); rec["S"] = rec.sigma*P[rec.bin]
w0 = 1.0429424357648776; rec["S_p0"] = rec.S*w0
print(rec.to_string(index=False)); rec.to_csv(f"{a.out}/deterministic_scores.csv", index=False)

# ---- R1-1: how large would a histogram change have to be to alter the order? ----
sf, nf, cat = 6800., 780., 4.61
print("\nRank-flip thresholds (all other things fixed):")
print(f"  2-NF passes 5-FUrd only if P(3-4)/P(0-1) > {sf/nf:.2f}  (actual {P[3]/P[0]:.3f})")
print(f"  catechin passes 2-NF only if P(1-2)/P(3-4) > {nf/cat:.1f}  (actual {P[1]/P[3]:.3f})")
print(f"  5-FUrd passes 5-IdU only if P(0-1)/P(0-1) changes: same bin -> never, order = sigma order (6800 vs 2.7e4)")
print("  Pure-sigma ordering (no LEE, no RED):", ", ".join(rec.sort_values('sigma',ascending=False).name))
print("  => LEE weighting and RED weighting both leave the ordering equal to the input-sigma ordering\n")

# ---- R2-8: resonance-width (peak/bin bias) sensitivity ----
# Smooth the piecewise-constant density with a unit-area Gaussian kernel (sd = FWHM/2.355) centred on E*;
# density is zero for E<0 and E>25. Width->0 recovers Eq. 5 exactly.
edges = np.arange(26.0)
def smoothed_density(Estar, fwhm):
    if fwhm <= 0: return P[int(np.floor(Estar))]
    s = fwhm/2.3548; cdf = lambda x: 0.5*(1+erf((x-Estar)/(np.sqrt(2)*s)))
    return float(np.sum(P*(cdf(edges[1:])-cdf(edges[:-1]))))   # = int rho(E) g(E) dE, rho=P per 1 eV bin
rows = []
for fw in [0, 0.1, 0.25, 0.5, 1.0, 2.0]:
    sc = {r.name: r.sigma*smoothed_density(r.E, fw) for r in rec.itertuples()}
    order = sorted(sc, key=sc.get, reverse=True)
    rows.append({"FWHM_eV": fw, **{k: round(v, 4) for k, v in sc.items()}, "order": " > ".join(order)})
wd = pd.DataFrame(rows); wd.to_csv(f"{a.out}/resonance_width_sensitivity.csv", index=False)
print("Resonance-width sensitivity (same width for every record):"); print(wd.to_string(index=False))
print("Note: near-threshold (0 eV) records lose up to half their overlap because the spectrum ends at 0 eV.\n")

# ---- R2-7: admissible LEE spectra ----
rng = np.random.default_rng(a.seed)
def draw_lee():
    while True:
        p0 = rng.uniform(0.10, 0.45)                         # wide prior on P(0-1 eV)
        x = rng.dirichlet(np.ones(24)*2.0)*(1-p0)             # arbitrary non-negative shape, not forced linear
        p = np.r_[p0, x]; m = np.sum((np.arange(25)+0.5)*p)
        if 8.0 <= m <= 10.0: return p                         # keep reported mean energy within +-1 eV
N = a.nsim; lee_ok = 0; ranks_lee = np.zeros((5,5)); orders = {}
for _ in range(N):
    p = draw_lee(); s = rec.sigma.values*p[rec.bin.values]
    o = np.argsort(-s); orders[tuple(o)] = orders.get(tuple(o), 0)+1
    for r, i in enumerate(o): ranks_lee[i, r] += 1
print("Admissible-LEE-distribution sensitivity (random spectra, P0 in [0.10,0.45], mean 8-10 eV, sigma at nominal values):")
print((pd.DataFrame(ranks_lee/N, index=rec.name, columns=[f"rank{r+1}" for r in range(5)])*100).round(2).to_string())
print("Distinct orderings:", {" > ".join(rec.name[list(k)]): v/N for k, v in sorted(orders.items(), key=lambda kv: -kv[1])[:5]}, "\n")

# ---- R1-6, R2-9, R2-10: extended Monte Carlo ----
def tnorm(mu, sd, n):
    out = rng.normal(mu, sd, n)
    while (bad := out <= 0).any(): out[bad] = rng.normal(mu, sd, bad.sum())
    return out
logu = lambda lo, hi, n: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
def mc(scn, n):
    sig = np.tile(rec.sigma.values, (n, 1)); Pm = np.tile(P, (n, 1))[:, rec.bin.values]
    sig[:, 0] *= logu(0.1, 10, n); sig[:, 3] *= logu(0.1, 10, n); sig[:, 1] = tnorm(2.7e4, 1.9e4/1.96, n)  # baseline of manuscript
    if scn == "baseline": pass
    if scn in ("FUrd_30-50pct",): sig[:, 2] *= np.exp(rng.normal(0, 0.5, n))          # source: 50% absolute accuracy
    if scn == "FUrd_1100Mb": sig[:, 2] = 1100*np.exp(rng.normal(0, 0.5, n))             # 433 K value quoted in the text of Wierzbicka
    if scn == "temp_independent":                                                         # physiological temperature factors, independent
        sig[:, 2] *= logu(0.01, 1, n); sig[:, 1] *= logu(0.1, 1, n); sig[:, 0] *= logu(0.1, 1, n)
    if scn == "temp_common":                                                              # one common factor for all gas-phase thermal records
        f = logu(0.01, 1, n); sig[:, [0, 1, 2]] *= f[:, None]
    if scn == "temp_worst_FUrd_x0.01": sig[:, 2] *= 0.01                                   # reviewer's point: 1914 -> ~19 Mb
    if scn == "catechin_x0.1-10": sig[:, 4] *= logu(0.1, 10, n)                            # catechin no longer fixed
    if scn == "LEE_random":
        for i in range(n): Pm[i] = draw_lee()[rec.bin.values]
    if scn == "all_combined":
        sig[:, 2] *= np.exp(rng.normal(0, 0.5, n))*logu(0.01, 1, n); sig[:, 1] *= logu(0.1, 1, n); sig[:, 0] *= logu(0.1, 1, n)
        sig[:, 4] *= logu(0.1, 10, n)
        for i in range(n): Pm[i] = draw_lee()[rec.bin.values]
    S = sig*Pm; order = np.argsort(-S, axis=1); R = np.zeros((5, 5))
    for r in range(5): R[:, r] = np.bincount(order[:, r], minlength=5)/n
    return R
scn_list = ["baseline","FUrd_30-50pct","FUrd_1100Mb","temp_independent","temp_common","temp_worst_FUrd_x0.01","catechin_x0.1-10","LEE_random","all_combined"]
res = []
for s in scn_list:
    n = N if s != "LEE_random" and s != "all_combined" else min(N, 5000)   # the random-LEE scenarios loop in python
    R = mc(s, n)
    for i, nm in enumerate(rec.name): res.append({"scenario": s, "compound": nm, **{f"P(rank{r+1})%": round(100*R[i, r], 2) for r in range(5)}})
res = pd.DataFrame(res); res.to_csv(f"{a.out}/extended_monte_carlo.csv", index=False)
for s in scn_list: print(f"\n== {s} =="); print(res[res.scenario == s].drop(columns="scenario").to_string(index=False))
print("\nDone. CSVs in", a.out)
