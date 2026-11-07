import argparse, glob, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--labels", required=True); ap.add_argument("--spacing", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=20260826)
a = ap.parse_args()

import radiomics
from radiomics import featureextractor
import logging
logging.getLogger("radiomics").setLevel(logging.ERROR)

ext = featureextractor.RadiomicsFeatureExtractor(binWidth=25, resampledPixelSpacing=[a.spacing] * 3,
                                                 interpolator="sitkBSpline", label=1)
print("PyRadiomics version:", radiomics.__version__)

done = {}
if os.path.exists(a.out):
    old = pd.read_csv(a.out); done = {r["patient_id"]: r for r in old.to_dict("records")}
rows = list(done.values())
for pdir in sorted(glob.glob(os.path.join(a.data_dir, "LUNG1-*"))):
    pid = os.path.basename(pdir)
    if pid in done: continue
    img, msk = os.path.join(pdir, "image.nii.gz"), os.path.join(pdir, "mask.nii.gz")
    if not (os.path.exists(img) and os.path.exists(msk)): continue
    try:
        r = ext.execute(img, msk)
        feats = {k: float(v) for k, v in r.items() if not k.startswith("diagnostics_")}
        rows.append({"patient_id": pid, **feats}); print(f"[ok] {pid}: {len(feats)} features", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    except Exception as e:
        print(f"[fail] {pid}: {e}", flush=True)
pd.DataFrame(rows).to_csv(a.out, index=False)
print(f"Wrote {len(rows)} rows to {a.out}")

# ---------------- comparison with the native-geometry partition
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
F = pd.DataFrame(rows).merge(pd.read_csv(a.labels)[["patient_id", "phenotype_class"]], on="patient_id")
y = F.pop("phenotype_class").values; F.pop("patient_id")
X = F.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan); X = X.loc[:, X.nunique() > 1]
X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
for thr in (0.80, 0.85, 0.90, 0.95):
    C = np.abs(spearmanr(X.values).correlation)
    keep = X.columns[[j for j in range(C.shape[0]) if not np.any(C[:j, j] > thr)]]
    Z = StandardScaler().fit_transform(X[keep]); lab = KMeans(2, n_init=50, random_state=a.seed).fit_predict(Z)
    print(f"thr {thr:.2f}: {len(keep)} features, sizes {np.bincount(lab).tolist()}, silhouette(k=2)={silhouette_score(Z, lab):.3f}, "
          f"ARI vs reported native-geometry partition={adjusted_rand_score(y, lab):.3f}")
