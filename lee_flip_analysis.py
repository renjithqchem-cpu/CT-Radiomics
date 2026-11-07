#!/usr/bin/env python3
import sys, os, json, itertools
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "lee_flip_out"; os.makedirs(out, exist_ok=True)
cent = np.arange(25) + 0.5
REC = {'5-bromouracil': (6.0e4, 0), '5-iodouridine': (2.70e4, 0), '5-fluorouridine': (6800., 0), '2-nitrofuran': (780., 3), 'catechin sulfate': (4.61, 1)}
names = list(REC)
def spectrum(f0, mu):
    """linear remainder p_i = a + b*(c_i-13) for bins 1..24 with sum = 1-f0 and bin-centre mean mu; None if any p<0."""
    c = cent[1:]; m = len(c)
    # unknowns a,b: sum(p)=1-f0 ; sum(p c)= mu - f0*0.5
    A = np.array([[m, (c - 13).sum()], [c.sum(), (c * (c - 13)).sum()]]); y = np.array([1 - f0, mu - 0.5 * f0])
    a, b = np.linalg.solve(A, y); p = a + b * (c - 13)
    if (p < 0).any(): return None
    return np.concatenate([[f0], p])
def order(P):
    s = np.array([REC[n][0] * P[REC[n][1]] for n in names]); return tuple(np.argsort(-s)), s
base = spectrum(0.27, 9.0); o0, s0 = order(base)
res = dict(baseline_p=[float(base[0]), float(base[1]), float(base[3])], baseline_order=[names[i] for i in o0],
           ratio_FUrd_over_2NF=float(s0[2] / s0[3]), ratio_2NF_over_Cat=float(s0[3] / s0[4]))
# flip frontier along f0 at mu=9 and over mu
f0s = np.linspace(0.0005, 0.70, 1400); mus = np.linspace(2, 16, 281)
code = np.full((len(mus), len(f0s)), -1, int); labels = {}
for i, mu in enumerate(mus):
    for j, f in enumerate(f0s):
        P = spectrum(f, mu)
        if P is None: continue
        o, _ = order(P); key = tuple(o)
        if key not in labels: labels[key] = len(labels)
        code[i, j] = labels[key]
feas = code >= 0; nflip = (feas & (code != labels[tuple(o0)])).sum()
res['feasible_cells'] = int(feas.sum()); res['cells_with_other_order'] = int(nflip)
res['other_orders'] = {' > '.join(names[k] for k in key): int((code == v).sum()) for key, v in labels.items()}
# critical f0 at mu = 9 (scan down)
i9 = int(np.argmin(abs(mus - 9.0))); row = code[i9]
crit = None
for j in range(len(f0s)):
    if row[j] >= 0 and row[j] != labels[tuple(o0)]: crit = float(f0s[j]); 
res['f0_below_which_order_changes_at_mu9'] = crit if crit is not None else None
res['f0_scan_mu9_other_order_max'] = crit
# minimal f0 over all feasible mu where order differs (the frontier)
fr = {}
for i, mu in enumerate(mus):
    js = np.where((code[i] >= 0) & (code[i] != labels[tuple(o0)]))[0]
    fr[float(round(mu, 2))] = float(f0s[js].max()) if len(js) else None
res['frontier_max_f0_with_other_order_by_mu'] = {k: v for k, v in fr.items() if abs(k * 2 - round(k * 2)) < 1e-6 and k == round(k)}
# magnitude comparison: density effect vs spectral effect
res['density_ratio_p1_over_p0'] = 1.0158 / 0.9842
res['delta_f0_equal_to_density_effect'] = float(0.27 * (1.0158 / 0.9842 - 1))
res['relative_change_in_f0_equal_to_density_effect_pct'] = 3.2
# what a 10% relative change in P(0-1 eV) would do to between-molecule score ratios and to magnitudes
for rel in (0.05, 0.10, 0.20):
    P = spectrum(0.27 * (1 + rel), 9.0)
    if P is not None:
        o, s = order(P); res[f'f0_plus_{int(rel*100)}pct'] = dict(order_same=(tuple(o) == tuple(o0)), score_ratio_FUrd_over_2NF=float(s[2] / s[3]), FUrd_score_rel_change_pct=float((s[2] / s0[2] - 1) * 100), twoNF_score_rel_change_pct=float((s[3] / s0[3] - 1) * 100))
# probability of a different order under random spectra from the published-range box and a 5x wider box
rng = np.random.default_rng(20260826)
def mc(lo_f, hi_f, lo_m, hi_m, n=20000):
    d = tot = 0
    for _ in range(n):
        P = spectrum(rng.uniform(lo_f, hi_f), rng.uniform(lo_m, hi_m))
        if P is None: continue
        tot += 1; d += tuple(order(P)[0]) != tuple(o0)
    return dict(feasible=tot, different_order_fraction=d / max(tot, 1))
res['mc_range_manuscript'] = mc(0.10, 0.45, 8, 10)
res['mc_wide'] = mc(0.01, 0.70, 3, 15)
json.dump(res, open(os.path.join(out, 'lee_flip_results.json'), 'w'), indent=1)
np.savez(os.path.join(out, 'phase.npz'), code=code, f0s=f0s, mus=mus, keys=np.array([list(k) for k in labels]))
print(json.dumps(res, indent=1)[:3500])
