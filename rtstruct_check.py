import argparse, glob, os, re
import numpy as np, pandas as pd, pydicom, SimpleITK as sitk
from skimage.draw import polygon

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--masks", required=True)
ap.add_argument("--out", default="rtstruct_check.csv"); ap.add_argument("--roi-regex", default=r"^GTV[-_ ]?1$")
ap.add_argument("--write-masks", default="", help="if set: write RTSTRUCT-derived masks to this folder for patients with Dice < --write-below (default 0.95)")
ap.add_argument("--write-below", type=float, default=0.95)
ap.add_argument("--id-regex", default=r"^(R01|AMC|LUNG1)-\d+$"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

def find_dcm(root):
    for dp, dn, fn_ in os.walk(root):
        for f in fn_:
            if f.lower().endswith(".dcm"): yield os.path.join(dp, f)

def find_rtstructs(root):
    out = []
    for f in find_dcm(root):
        try: d = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception: continue
        if getattr(d, "Modality", "") == "RTSTRUCT": out.append(f)
    return out

def rasterise(ds, roi_number, ref):
    """Even-odd fill of every planar contour of the ROI on the slice it lies in. Returns (array z,y,x uint8, n_contours, n_off_grid, max_dz)."""
    nx, ny, nz = ref.GetSize(); arr = np.zeros((nz, ny, nx), np.uint8); n_c = 0; off = 0; max_dz = 0.0
    rc = [r for r in ds.ROIContourSequence if int(r.ReferencedROINumber) == roi_number]
    if not rc or not hasattr(rc[0], "ContourSequence"): return arr, 0, 0, 0.0
    for c in rc[0].ContourSequence:
        if str(getattr(c, "ContourGeometricType", "CLOSED_PLANAR")) not in ("CLOSED_PLANAR", "OPEN_PLANAR"): continue
        pts = np.asarray(c.ContourData, float).reshape(-1, 3); n_c += 1
        idx = np.array([ref.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])   # (i=x, j=y, k=z)
        k = int(round(np.median(idx[:, 2]))); max_dz = max(max_dz, float(np.max(np.abs(idx[:, 2] - k))))
        if not (0 <= k < nz): off += 1; continue
        rr, cc = polygon(idx[:, 1], idx[:, 0], shape=(ny, nx))
        sl = np.zeros((ny, nx), np.uint8); sl[rr, cc] = 1; arr[k] ^= sl            # XOR gives holes where contours nest
    return arr, n_c, off, max_dz

rows, roi_rows = [], []
pdirs = [p for p in sorted(glob.glob(os.path.join(a.masks, "*"))) if re.match(a.id_regex, os.path.basename(p))]
if a.limit: pdirs = pdirs[:a.limit]
for pdir in pdirs:
    pid = os.path.basename(pdir); mfile = os.path.join(pdir, "mask.nii.gz"); rec = {"patient_id": pid, "status": "ok"}
    if not os.path.exists(mfile): rec["status"] = "no mask"; rows.append(rec); continue
    rts = find_rtstructs(os.path.join(a.raw, pid)); rec["n_rtstruct_files"] = len(rts)
    if not rts: rec["status"] = "no RTSTRUCT"; rows.append(rec); continue
    best = None
    for f in rts:
        ds = pydicom.dcmread(f, force=True)
        names = {int(r.ROINumber): str(getattr(r, "ROIName", "")) for r in ds.StructureSetROISequence}
        for n, nm in names.items(): roi_rows.append({"patient_id": pid, "file": os.path.basename(f), "roi_number": n, "roi_name": nm})
        hit = [n for n, nm in names.items() if re.match(a.roi_regex, nm.strip(), re.I)]
        if hit: best = (ds, hit[0], names[hit[0]]); break
    if best is None: rec["status"] = "no ROI matching --roi-regex"; rec["roi_names"] = "; ".join(sorted(set(names.values()))); rows.append(rec); continue
    ds, rn, rname = best; rec["roi_name_used"] = rname
    m = sitk.ReadImage(mfile); ref_arr = sitk.GetArrayFromImage(m) > 0
    arr, n_c, off, max_dz = rasterise(ds, rn, m); rt = arr > 0
    inter = int((rt & ref_arr).sum()); sr, sm = int(rt.sum()), int(ref_arr.sum()); sp = m.GetSpacing(); vox = float(np.prod(sp)) / 1000
    zr = np.where(rt.any(axis=(1, 2)))[0]; zm = np.where(ref_arr.any(axis=(1, 2)))[0]
    rec.update({"n_contours": n_c, "contours_off_grid": off, "max_contour_slice_offset_vox": round(max_dz, 3),
                "rtstruct_volume_mL": round(sr * vox, 3), "mask_volume_mL": round(sm * vox, 3),
                "volume_ratio_rt_over_mask": round(sr / sm, 4) if sm else np.nan,
                "dice": round(2 * inter / (sr + sm), 4) if (sr + sm) else np.nan,
                "rt_slices": int(len(zr)), "mask_slices": int(len(zm)),
                "slice_range_equal": bool(len(zr) == len(zm) and (len(zr) == 0 or (zr[0] == zm[0] and zr[-1] == zm[-1])))})
    if a.write_masks and rec["dice"] < a.write_below:
        od = os.path.join(a.write_masks, pid); os.makedirs(od, exist_ok=True)
        om = sitk.GetImageFromArray(rt.astype(np.uint8)); om.CopyInformation(m); sitk.WriteImage(om, os.path.join(od, "mask.nii.gz"))
        li = os.path.join(od, "image.nii.gz"); src = os.path.join(pdir, "image.nii.gz")
        if os.path.exists(src) and not os.path.lexists(li): os.symlink(os.path.realpath(src), li)
        rec["rtstruct_mask_written"] = True
    rows.append(rec)
q = pd.DataFrame(rows); q.to_csv(a.out, index=False); pd.DataFrame(roi_rows).to_csv(os.path.splitext(a.out)[0] + "_roi_names.csv", index=False)
print(q.status.value_counts().to_string())
if "dice" in q and q.dice.notna().any():
    d = q.dice.dropna(); r = q.volume_ratio_rt_over_mask.dropna()
    print("patients compared: %d" % len(d))
    print("Dice: median %.3f, IQR %.3f-%.3f, min %.3f; Dice>=0.90: %d, 0.80-0.90: %d, <0.80: %d" % (d.median(), d.quantile(.25), d.quantile(.75), d.min(), (d >= .9).sum(), ((d >= .8) & (d < .9)).sum(), (d < .8).sum()))
    print("volume ratio RTSTRUCT/mask: median %.3f, min %.3f, max %.3f; within +-10%%: %d" % (r.median(), r.min(), r.max(), ((r > .9) & (r < 1.1)).sum()))
    print("slice range identical in %d of %d" % (int(q.slice_range_equal.sum()), len(d)))
    print("\nWorst 10 by Dice:"); print(q.sort_values("dice").head(10)[["patient_id", "dice", "volume_ratio_rt_over_mask", "rtstruct_volume_mL", "mask_volume_mL"]].to_string(index=False))
