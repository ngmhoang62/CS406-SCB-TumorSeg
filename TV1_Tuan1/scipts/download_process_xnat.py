"""
Stream-download MAMA-MIA from the public Health-RI XNAT (guest access, no login) and preprocess on the fly.

For each valid case (tumor_subtype not N/A): download _0000 (pre), _0001 (post1) and the expert
segmentation, run Step 2 + Step 3, save {CASE}.npz, then DELETE the raw NIfTI files (raw set is ~150 GB).
Resumable: cases whose .npz already exists are skipped.

Limitations of the XNAT mirror: no train_test_splits.csv and no automatic masks. A stratified 20% Test split
(by tumor_subtype x dataset, seed 42) is generated instead, and auto_mask is stored as zeros.

Usage:
  python scripts/download_process_xnat.py --config configs/real_config.yaml [--workers 8] [--limit N]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, seed_everything
from src.preprocessing import run_step1_pipeline, standardize_and_extract_voi_fast as standardize_and_extract_voi, synthesize_three_channels

XNAT = "https://xnat.health-ri.nl"
PROJECT = "mama_mia"


def experiment_map() -> dict:
    r = requests.get(f"{XNAT}/data/projects/{PROJECT}/experiments", params={"format": "json", "columns": "ID,label"}, timeout=120)
    r.raise_for_status()
    return {x["label"]: x["ID"] for x in r.json()["ResultSet"]["Result"]}


def make_official_style_splits(excel_path: str, out_csv: Path, seed: int, test_frac: float = 0.2) -> None:
    from sklearn.model_selection import train_test_split

    df = pd.read_excel(excel_path, sheet_name="dataset_info")
    df = df[df["tumor_subtype"].notna()]
    key = df["tumor_subtype"].astype(str) + "__" + df["dataset"].astype(str)
    key = key.where(key.map(key.value_counts()) >= 2, df["tumor_subtype"].astype(str))
    train_ids, test_ids = train_test_split(df["patient_id"], test_size=test_frac, random_state=seed, stratify=key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"train_split": pd.Series(train_ids.values), "test_split": pd.Series(test_ids.values)}).to_csv(out_csv, index=False)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    curl = "curl.exe" if os.name == "nt" else "curl"
    for attempt in range(6):  # outer retry: XNAT occasionally drops TLS/connections
        try:
            subprocess.run(
                [curl, "-s", "-L", "--fail", "--retry", "5", "--retry-delay", "5", "--retry-all-errors",
                 "--connect-timeout", "30", "--speed-limit", "20000", "--speed-time", "120", "-o", str(dest), url],
                check=True,
            )
            return
        except subprocess.CalledProcessError:
            if attempt == 5:
                raise
            time.sleep(10 * (attempt + 1))


def process_case(args):
    case_id, subtype_id, exp_id, cfg, tmp_root, npz_dir = args
    out = Path(npz_dir) / f"{case_id}.npz"
    if out.exists():
        return case_id, "skipped", None
    tmp = Path(tmp_root) / case_id
    try:
        base = f"{XNAT}/data/experiments/{exp_id}/scans/1/resources/NIfTI/files"
        pre, post1, seg = tmp / "pre.nii.gz", tmp / "post1.nii.gz", tmp / "seg.nii.gz"
        fetch(f"{base}/{case_id}_0000.nii.gz", pre)
        fetch(f"{base}/{case_id}_0001.nii.gz", post1)
        fetch(f"{base}/{case_id}_segmentation.nii.gz", seg)

        s2, s3 = cfg["step2_spatial_voi"], cfg["step3_image_processing"]
        pre_voi, post1_voi, expert_voi, _, _ = standardize_and_extract_voi(
            pre_sitk=sitk.ReadImage(str(pre), sitk.sitkFloat32),
            post1_sitk=sitk.ReadImage(str(post1), sitk.sitkFloat32),
            expert_mask_sitk=sitk.ReadImage(str(seg), sitk.sitkUInt8),
            auto_mask_sitk=None,
            target_orientation=str(s2["target_orientation"]),
            target_spacing=tuple(float(x) for x in s2["target_spacing"]),
            voi_size=tuple(int(x) for x in s2["voi_size"]),
            margin_ratio=float(s2["voi_margin_ratio"]),
            pad_mode=str(s2["pad_mode"]),
        )
        image_3ch = synthesize_three_channels(
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
        mask = expert_voi[np.newaxis, ...].astype(np.uint8)
        tmp_out = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp_out, image=image_3ch, mask=mask, auto_mask=np.zeros_like(mask), subtype_id=np.int64(subtype_id))
        tmp_out.replace(out)
        return case_id, "done", None
    except Exception as exc:
        return case_id, "failed", f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real_config.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tmp-dir", default=None, help="local scratch dir for raw NIfTI (default: <raw_dir>/tmp)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seed_everything(cfg["project"]["seed"])
    p = cfg["paths"]
    splits_csv = Path(p["official_splits_csv"])
    if not splits_csv.exists():
        make_official_style_splits(p["clinical_excel"], splits_csv, int(cfg["project"]["seed"]))
        print(f"Generated stratified train/test split -> {splits_csv}")
    meta = run_step1_pipeline(p["clinical_excel"], splits_csv, p["splits_metadata_csv"],
                              val_ratio=float(cfg["step1_ingestion"]["val_ratio"]), seed=int(cfg["project"]["seed"]))
    print(f"Valid cases: {len(meta)}")
    if a.limit:
        meta = meta.head(a.limit)

    exp = experiment_map()
    npz_dir = Path(p["processed_npz_dir"]); npz_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(a.tmp_dir) if a.tmp_dir else Path(p["raw_dir"]) / "tmp"; tmp_root.mkdir(parents=True, exist_ok=True)
    jobs = [(str(r.patient_id), int(r.subtype_id), exp[str(r.patient_id)], cfg, str(tmp_root), str(npz_dir))
            for r in meta.itertuples() if str(r.patient_id) in exp]
    print(f"Cases to process: {len(jobs)} ({len(meta) - len(jobs)} not found on XNAT)")

    logs = Path(p["logs_dir"]); logs.mkdir(parents=True, exist_ok=True)
    counts, failed, t0 = {"done": 0, "skipped": 0, "failed": 0}, [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_case, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cid, status, err = f.result()
            counts[status] += 1
            if status == "failed":
                failed.append(f"{cid}\t{err}")
                (logs / "xnat_failed_cases.tsv").write_text("\n".join(failed), encoding="utf-8")
            if i % 10 == 0 or status == "failed":
                print(f"[{i}/{len(jobs)}] {counts} elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"FINISHED {counts}. Re-run the same command to retry failed cases.")


if __name__ == "__main__":
    main()
