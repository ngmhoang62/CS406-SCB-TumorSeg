"""
Step 1-2-3 on a local machine, reading raw MAMA-MIA from a mounted Google Drive (Drive for desktop) and writing
{CASE}.npz back to the same Drive. Resumable: cases whose .npz already exists are skipped.

Expected layout under --data-root (the Drive folder `data_real`):
  raw/clinical_and_imaging_info.xlsx, raw/train_test_splits.csv (optional)
  raw/images/{CASE}/{CASE}_0000.nii.gz, _0001.nii.gz
  raw/segmentations/expert/{CASE}.nii.gz [, raw/segmentations/automatic/{CASE}.nii.gz]
Outputs: processed/npz_96x96x96/{CASE}.npz, splits/splits_metadata.csv

Usage:
  python scripts/preprocess_from_drive.py --data-root "G:\\My Drive\\CS406-SCB-TumorSeg\\data_real" --workers 12
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, seed_everything
from src.preprocessing import (
    run_step1_pipeline,
    standardize_and_extract_voi_fast,
    synthesize_three_channels,
)
from scripts.download_process_xnat import make_official_style_splits

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)  # one ITK thread per worker process


def process_case(case_id, subtype_id, raw, npz_dir):
    out = Path(npz_dir) / f"{case_id}.npz"
    if out.exists():
        return case_id, "skipped", None
    try:
        raw = Path(raw)
        s2, s3 = _CFG["step2_spatial_voi"], _CFG["step3_image_processing"]
        auto = raw / "segmentations" / "automatic" / f"{case_id}.nii.gz"
        pre_voi, post1_voi, expert_voi, auto_voi, _ = standardize_and_extract_voi_fast(
            pre_sitk=sitk.ReadImage(str(raw / "images" / case_id / f"{case_id}_0000.nii.gz"), sitk.sitkFloat32),
            post1_sitk=sitk.ReadImage(str(raw / "images" / case_id / f"{case_id}_0001.nii.gz"), sitk.sitkFloat32),
            expert_mask_sitk=sitk.ReadImage(str(raw / "segmentations" / "expert" / f"{case_id}.nii.gz"), sitk.sitkUInt8),
            auto_mask_sitk=sitk.ReadImage(str(auto), sitk.sitkUInt8) if auto.exists() else None,
            target_orientation=str(s2["target_orientation"]),
            target_spacing=tuple(float(x) for x in s2["target_spacing"]),
            voi_size=tuple(int(x) for x in s2["voi_size"]),
            margin_ratio=float(s2["voi_margin_ratio"]),
            pad_mode=str(s2["pad_mode"]),
        )
        image = synthesize_three_channels(
            pre_voi=pre_voi, post1_voi=post1_voi,
            apply_n4=bool(s3["apply_n4_bias_correction"]), n4_shrink_factor=int(s3["n4_shrink_factor"]),
            apply_denoise=bool(s3["apply_edge_preserving_denoise"]), denoise_method=str(s3["denoise_method"]),
            nlmeans_patch_size=int(s3["nlmeans_patch_size"]), nlmeans_patch_distance=int(s3["nlmeans_patch_distance"]),
            nlmeans_h_factor=float(s3["nlmeans_h_factor"]),
            clip_percentiles=tuple(float(x) for x in s3["clip_percentiles"]),
            pe_tau_percentile=float(s3["pe_tau_percentile"]), clahe_clip_limit=float(s3["clahe_clip_limit"]),
            clahe_kernel_size=int(s3["clahe_kernel_size"]), zscore_eps=float(s3["zscore_eps"]),
            output_dtype=str(s3["output_dtype"]),
        )
        mask = expert_voi[np.newaxis].astype(np.uint8)
        amask = np.zeros_like(mask) if auto_voi is None else auto_voi[np.newaxis].astype(np.uint8)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, image=image, mask=mask, auto_mask=amask, subtype_id=np.int64(subtype_id))
        tmp.replace(out)
        return case_id, "done", None
    except Exception as exc:
        return case_id, "failed", f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Drive folder `data_real`")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    seed_everything(cfg["project"]["seed"])
    root = Path(a.data_root)
    raw, npz_dir = root / "raw", root / "processed" / "npz_96x96x96"
    npz_dir.mkdir(parents=True, exist_ok=True)

    xlsx, splits = raw / "clinical_and_imaging_info.xlsx", raw / "train_test_splits.csv"
    if not splits.exists():
        make_official_style_splits(str(xlsx), splits, int(cfg["project"]["seed"]))
        print("No train_test_splits.csv -> generated a stratified 20% Test split (differs from the paper's).")
    meta = run_step1_pipeline(str(xlsx), str(splits), str(root / "splits" / "splits_metadata.csv"),
                              val_ratio=float(cfg["step1_ingestion"]["val_ratio"]), seed=int(cfg["project"]["seed"]))
    print(f"Valid cases: {len(meta)}")
    if a.limit:
        meta = meta.head(a.limit)

    counts, failed, t0 = {"done": 0, "skipped": 0, "failed": 0}, [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(cfg,)) as ex:
        futs = [ex.submit(process_case, str(r.patient_id), int(r.subtype_id), str(raw), str(npz_dir)) for r in meta.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            cid, st, err = f.result()
            counts[st] += 1
            if st == "failed":
                failed.append(f"{cid}\t{err}")
            if i % 10 == 0 or st == "failed":
                print(f"[{i}/{len(futs)}] {counts} elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
    if failed:
        (root / "preprocess_failed_cases.tsv").write_text("\n".join(failed), encoding="utf-8")
        print(f"{len(failed)} failed -> {root / 'preprocess_failed_cases.tsv'} (re-run to retry)")
    print("FINISHED", counts)


if __name__ == "__main__":
    main()
