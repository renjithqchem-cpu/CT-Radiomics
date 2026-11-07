import argparse, glob, os, shutil
import numpy as np, pandas as pd, pydicom, SimpleITK as sitk
ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--nifti", required=True)
ap.add_argument("--label", default="Neoplasm, Primary"); ap.add_argument("--replace", action="store_true"); ap.add_argument("--out", default="tumour_mask_report.csv")
a = ap.parse_args(); rows = []
for pdir in sorted(glob.glob(os.path.join(a.nifti, "*"))):
    pid = os.path.basename(pdir); img_f = os.path.join(pdir, "image.nii.gz"); rec = {"patient_id": pid, "status": "ok"}
    seg_file = None
    for f in glob.glob(os.path.join(a.raw, pid, "**", "*.dcm"), recursive=True):
        try: d = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception: continue
        if getattr(d, "Modality", "") == "SEG": seg_file = f; break
    if seg_file is None or not os.path.exists(img_f): rec["status"] = "no SEG or no image"; rows.append(rec); continue
    d = pydicom.dcmread(seg_file); labels = {int(s.SegmentNumber): str(getattr(s, "SegmentLabel", "")) for s in d.SegmentSequence}
    tum = [n for n, l in labels.items() if l.strip().lower() == a.label.lower()]
    rec.update({"segments": "; ".join(f"{n}:{l}" for n, l in labels.items()), "tumour_segments": str(tum)})
    if not tum: rec["status"] = "no tumour segment"; rows.append(rec); continue
    seg = sitk.ReadImage(seg_file); ncomp = seg.GetNumberOfComponentsPerPixel()
    parts = []
    for n in tum:
        comp = sitk.VectorIndexSelectionCast(seg, n-1) if ncomp > 1 else seg          # component index = segment number - 1 (verified by voxel counts in mask_qc.py)
        parts.append(sitk.GetArrayFromImage(comp) > 0)
    arr = np.logical_or.reduce(parts); ref = sitk.ReadImage(img_f)
    m = sitk.GetImageFromArray(arr.astype(np.uint8)); m.CopyInformation(sitk.VectorIndexSelectionCast(seg, 0) if ncomp > 1 else seg)
    same = (m.GetSize() == ref.GetSize() and np.allclose(m.GetSpacing(), ref.GetSpacing(), atol=1e-3) and np.allclose(m.GetOrigin(), ref.GetOrigin(), atol=1e-2))
    rec["geometry_matches_image"] = bool(same)
    if not same:                                                                     # nearest-neighbour onto the CT grid
        m = sitk.Resample(m, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8); rec["status"] = "resampled to CT grid"
    nv = int(sitk.GetArrayFromImage(m).sum()); rec["tumour_voxels_nifti"] = nv
    frames = d.pixel_array; cnt = 0
    for i, fg in enumerate(d.PerFrameFunctionalGroupsSequence):
        if int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber) in tum: cnt += int((frames[i] > 0).sum())
    rec["tumour_voxels_dicom"] = cnt; rec["voxel_count_agrees"] = (cnt == nv)
    rec["tumour_volume_mL"] = nv*float(np.prod(ref.GetSpacing()))/1000
    out = os.path.join(pdir, "mask_tumour.nii.gz"); sitk.WriteImage(m, out)
    if a.replace:
        old = os.path.join(pdir, "mask.nii.gz"); bak = os.path.join(pdir, "mask_first_segment_OLD.nii.gz")
        if os.path.exists(old) and not os.path.exists(bak): shutil.copy2(old, bak)
        shutil.copy2(out, old)
    rows.append(rec)
q = pd.DataFrame(rows); q.to_csv(a.out, index=False)
print(q.status.value_counts().to_string()); 
if "voxel_count_agrees" in q: print("voxel counts agree (pydicom vs SimpleITK):", int(q.voxel_count_agrees.sum()), "of", int(q.voxel_count_agrees.notna().sum()))
print("tumour volume mL: median %.1f, min %.1f, max %.1f" % (q.tumour_volume_mL.median(), q.tumour_volume_mL.min(), q.tumour_volume_mL.max()))
print("patients needing attention:\n", q[(q.status != "ok") | (q.get("voxel_count_agrees", True) == False)][["patient_id", "status"]].to_string(index=False))
