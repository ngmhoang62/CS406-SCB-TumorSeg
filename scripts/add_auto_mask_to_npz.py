"""
Fill `auto_mask` in the existing {CASE}.npz files with the real automatic masks (Synapse), without touching images.

The VOI window is recomputed exactly as in step2 (from the EXPERT mask on the 1 mm grid), the automatic mask is
resampled (nearest) on that same window, and the expert window is compared with the stored `mask` as a consistency
check. Cases without an automatic mask keep an all-zero `auto_mask` and are listed in missing_auto_mask_cases.txt.

Usage:
  python scripts/add_auto_mask_to_npz.py --data-root "G:\\My Drive\\CS406-SCB-TumorSeg\\data_real" --auto-dir D:\\scb_syn\\automatic
"""

import argparse
import itertools
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.step2_spatial_voi import (
    standardize_orientation_ras, resample_isotropic_1mm, compute_tumor_bbox_with_margin, _resample_window,
)


def one_case(case_id, raw_expert, auto_dir, npz_path, force=False, voi_size=(96, 96, 96), margin=0.25):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    try:
        z = np.load(npz_path)
        arrays = {k: z[k] for k in z.files}
        if not force and arrays.get("auto_mask") is not None and arrays["auto_mask"].any():
            return case_id, "skipped", ""
        auto_path = Path(auto_dir) / f"{case_id}.nii.gz"
        if not auto_path.exists():
            return case_id, "no_auto", ""

        exp_raw = sitk.ReadImage(str(raw_expert), sitk.sitkUInt8)
        exp_ras = standardize_orientation_ras(exp_raw, "RAS")
        exp_iso = resample_isotropic_1mm(exp_ras, (1.0, 1.0, 1.0), is_mask=True)
        exp_np = (sitk.GetArrayFromImage(exp_iso) > 0).astype(np.uint8)
        bz, by, bx = compute_tumor_bbox_with_margin(exp_np, margin_ratio=margin)
        cz, cy, cx = (bz[0] + bz[1]) // 2, (by[0] + by[1]) // 2, (bx[0] + bx[1]) // 2
        td, th, tw = (max(voi_size[0], bz[1] - bz[0]), max(voi_size[1], by[1] - by[0]), max(voi_size[2], bx[1] - bx[0]))
        start, size = (cx - tw // 2, cy - th // 2, cz - td // 2), (tw, th, td)

        exp_voi = (_resample_window(exp_ras, exp_iso, start, size, True) > 0).astype(np.uint8)
        if exp_voi.shape != arrays["mask"].shape[1:] or not np.array_equal(exp_voi, arrays["mask"][0]):
            return case_id, "window_mismatch", f"expert window != stored mask (shape {exp_voi.shape} vs {arrays['mask'].shape})"

        # Synapse masks and the XNAT NIfTIs (images + `_segmentation` = expert mask) use different axis conventions
        # (flips for DUKE/ISPY2, axis permutations for ISPY1/NACT). The Synapse EXPERT mask is the same segmentation
        # as the XNAT one, so find the exact signed axis permutation that maps it onto the XNAT array and apply that
        # same transform to the Synapse automatic mask. No transform found => the case is reported, never guessed.
        syn_exp = sitk.GetArrayFromImage(sitk.ReadImage(str(Path(auto_dir).parent / "expert" / f"{case_id}.nii.gz"))) > 0
        xnat_exp = sitk.GetArrayFromImage(exp_raw) > 0
        tf = None
        for perm in itertools.permutations(range(3)):
            sp = np.transpose(syn_exp, perm)
            if sp.shape != xnat_exp.shape or sp.sum() != xnat_exp.sum():
                continue
            for flips in itertools.product((False, True), repeat=3):
                f = sp[tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)]
                if np.array_equal(f, xnat_exp):
                    tf = (perm, flips)
                    break
            if tf:
                break
        if tf is None:
            return case_id, "failed", "no axis transform maps Synapse expert onto XNAT expert"
        perm, flips = tf
        auto_np = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(str(auto_path), sitk.sitkUInt8)), perm)
        auto_np = np.ascontiguousarray(auto_np[tuple(slice(None, None, -1) if fl else slice(None) for fl in flips)])
        if auto_np.shape != xnat_exp.shape:
            return case_id, "failed", f"automatic mask shape {auto_np.shape} != expert shape {xnat_exp.shape}"
        auto_img = sitk.GetImageFromArray(auto_np)
        auto_img.CopyInformation(exp_raw)
        auto_ras = standardize_orientation_ras(auto_img, "RAS")
        auto_voi = (_resample_window(auto_ras, exp_iso, start, size, True) > 0).astype(np.uint8)
        arrays["auto_mask"] = auto_voi[np.newaxis]
        tmp = Path(npz_path).with_name(Path(npz_path).stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(npz_path)
        inter = float((auto_voi & exp_voi).sum())
        dice = 2 * inter / max(1.0, float(auto_voi.sum() + exp_voi.sum()))
        return case_id, "updated", f"{dice:.4f}"
    except Exception as exc:
        return case_id, "failed", f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--auto-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="rewrite auto_mask even if it is already filled")
    a = ap.parse_args()
    root = Path(a.data_root)
    npz_dir = root / "processed" / "npz_96x96x96"
    cases = sorted(p.stem for p in npz_dir.glob("*.npz") if not p.stem.endswith(".tmp"))
    if a.limit:
        cases = cases[: a.limit]

    counts, dice, bad, no_auto, t0 = {}, [], [], [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one_case, c, str(root / "raw" / "segmentations" / "expert" / f"{c}.nii.gz"),
                          a.auto_dir, str(npz_dir / f"{c}.npz"), a.force) for c in cases]
        for k, f in enumerate(as_completed(futs), 1):
            cid, st, info = f.result()
            counts[st] = counts.get(st, 0) + 1
            if st == "updated":
                dice.append(float(info))
            elif st == "no_auto":
                no_auto.append(cid)
            elif st in ("failed", "window_mismatch"):
                bad.append(f"{cid}\t{st}\t{info}")
            if k % 50 == 0:
                print(f"[{k}/{len(cases)}] {counts} {(time.time() - t0) / 60:.1f} min", flush=True)
    if no_auto:
        (root / "missing_auto_mask_cases.txt").write_text("\n".join(no_auto), encoding="utf-8")
    if bad:
        (root / "add_auto_mask_problems.tsv").write_text("\n".join(bad), encoding="utf-8")
    if dice:
        d = np.array(dice)
        print(f"Dice(auto vs expert): mean={d.mean():.3f} median={np.median(d):.3f} min={d.min():.3f} "
              f"<0.5: {(d < 0.5).sum()} cases")
    print("FINISHED", counts, flush=True)


if __name__ == "__main__":
    main()
