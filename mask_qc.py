import argparse, glob, os
import numpy as np, pandas as pd, pydicom, SimpleITK as sitk
ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--nifti", required=True); ap.add_argument("--out", default="mask_qc.csv")
a = ap.parse_args(); rows = []
for pdir in sorted(glob.glob(os.path.join(a.nifti, "*"))):
    pid = os.path.basename(pdir); mfile = os.path.join(pdir, "mask.nii.gz")
    if not os.path.exists(mfile): continue
    m = sitk.ReadImage(mfile); arr = sitk.GetArrayFromImage(m) > 0; sp = m.GetSpacing()
    seg = None
    for f in glob.glob(os.path.join(a.raw, pid, "**", "*.dcm"), recursive=True):
        try: d = pydicom.dcmread(f, stop_before_pixels=False)
        except Exception: continue
        if getattr(d, "Modality", "") == "SEG": seg = d; break
    rec = {"patient_id": pid, "mask_voxels": int(arr.sum()), "mask_volume_mL": float(arr.sum()*np.prod(sp)/1000), "spacing_mm": str(tuple(round(s, 3) for s in sp)),
           "slice_thickness_mm": round(sp[2], 3), "n_components_mask": m.GetNumberOfComponentsPerPixel()}
    if seg is not None:
        labels = [(s.SegmentNumber, getattr(s, "SegmentLabel", "")) for s in seg.SegmentSequence]
        frames = seg.pixel_array; nseg = len(labels)
        # map frames to segments (SEG stores frames segment-major in most TCIA files; use ReferencedSegmentNumber when present)
        per = {}
        for i, fg in enumerate(seg.PerFrameFunctionalGroupsSequence):
            sn = fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber; per[sn] = per.get(sn, 0) + int((frames[i] > 0).sum())
        rec.update({"n_segments": nseg, "segment_labels": "; ".join(f"{n}:{l}" for n, l in labels), "segment_voxels": "; ".join(f"{n}:{per.get(n,0)}" for n, _ in labels),
                    "mask_equals_union_of_all": int(sum(per.values())) == rec["mask_voxels"] and nseg > 1,
                    "gtv1_voxels": next((per.get(n, 0) for n, l in labels if "GTV-1" in str(l).upper() or "GTV1" in str(l).upper().replace("-", "")), np.nan)})
        rec["mask_matches_gtv1"] = rec["gtv1_voxels"] == rec["mask_voxels"]
    else: rec["n_segments"] = np.nan
    rows.append(rec)
q = pd.DataFrame(rows); q.to_csv(a.out, index=False)
print(f"{len(q)} patients; with >1 segment in SEG: {(q.n_segments>1).sum()}; mask == GTV-1 only: {q.mask_matches_gtv1.sum() if 'mask_matches_gtv1' in q else 'n/a'}; mask == union of all segments: {q.mask_equals_union_of_all.sum() if 'mask_equals_union_of_all' in q else 'n/a'}")
print("slice thickness (mm) value counts:\n", q.slice_thickness_mm.value_counts().head(10))
