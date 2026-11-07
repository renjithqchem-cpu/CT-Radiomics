import argparse, os
import numpy as np, pandas as pd, pydicom
from scipy.stats import chi2_contingency, mannwhitneyu

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", default="results_final")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 220)
if not os.path.isdir(a.raw):
    raise SystemExit(f"Raw DICOM folder not found: {a.raw!r}. From inside revision/ it is usually ../nsclc_radiomics")
L = pd.read_csv(a.labels)[["patient_id", "phenotype_class"]]


def first_ct(root):
    for dp, dn, fn in os.walk(root):
        for f in sorted(fn):
            if f.lower().endswith(".dcm"):
                try:
                    d = pydicom.dcmread(os.path.join(dp, f), stop_before_pixels=True)
                except Exception:
                    continue
                if getattr(d, "Modality", "") == "CT":
                    return d
    return None


rows = []
for pid in L.patient_id:
    d = first_ct(os.path.join(a.raw, pid))
    if d is None:
        rows.append(dict(patient_id=pid)); continue
    ps = getattr(d, "PixelSpacing", [np.nan, np.nan])
    rows.append(dict(patient_id=pid, manufacturer=str(getattr(d, "Manufacturer", "")), model=str(getattr(d, "ManufacturerModelName", "")),
                     kvp=float(getattr(d, "KVP", np.nan) or np.nan), kernel=str(getattr(d, "ConvolutionKernel", "")),
                     slice_thickness=float(getattr(d, "SliceThickness", np.nan) or np.nan), pixel_spacing=float(ps[0]),
                     tube_current=float(getattr(d, "XRayTubeCurrent", np.nan) or np.nan)))
A = pd.DataFrame(rows).merge(L, on="patient_id")
if "model" not in A.columns:
    raise SystemExit("No readable CT header found for any patient: check the --raw path (expected <raw>/LUNG1-xxx/**/*.dcm).")
A.to_csv(f"{a.out}/acquisition_parameters.csv", index=False)
print(f"patients with a readable CT header: {A.model.notna().sum()} of {len(A)}")
for c in ["model", "kvp", "kernel", "slice_thickness"]:
    print(f"\n{c} value counts:\n", A[c].value_counts(dropna=False).head(12).to_string())
print("\npixel spacing (mm): min %.3f median %.3f max %.3f" % (A.pixel_spacing.min(), A.pixel_spacing.median(), A.pixel_spacing.max()))
print("\n--- association with phenotype ---")
for c in ["model", "kernel"]:
    ct = pd.crosstab(A[c], A.phenotype_class); chi = chi2_contingency(ct)
    print(f"{c}: chi2={chi[0]:.2f}, dof={chi[2]}, p={chi[1]:.3g}  (expected counts <5 in {(chi[3] < 5).mean():.0%} of cells)")
    print(ct.to_string())
for c in ["kvp", "slice_thickness", "pixel_spacing", "tube_current"]:
    v0 = A.loc[A.phenotype_class == 0, c].dropna(); v1 = A.loc[A.phenotype_class == 1, c].dropna()
    if len(v0) and len(v1) and pd.concat([v0, v1]).nunique() > 1:
        print(f"{c}: median class0={v0.median():.3f}, class1={v1.median():.3f}, MWU p={mannwhitneyu(v0, v1).pvalue:.3g}")
    else:
        print(f"{c}: no variation or no data (values: {sorted(set(pd.concat([v0, v1])))[:5]})")
