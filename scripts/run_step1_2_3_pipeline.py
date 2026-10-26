"""
End-to-End Execution of Step 1, Step 2, and Step 3 Preprocessing Pipeline.

Reads raw MAMA-MIA (or synthetic raw) files from `data/raw/`, runs:
  - Step 1: Case filtering (drops N/A subtypes), canonical 4-class subtype mapping,
            Stratified Train/Val/Test splitting, and class-balanced sample weighting.
  - Step 2: RAS orientation standardization, 1.0x1.0x1.0 mm^3 isotropic resampling
            (B-Spline for MRI, Nearest-Neighbor for Masks), and 96x96x96 3D VOI extraction (+25% margin).
  - Step 3: 3D Edge-Preserving Denoising (Non-Local Means / Curvature Flow), optional N4 Bias Correction,
            Percentile Intensity Clipping [0.5%, 99.5%], and 3-channel synthesis:
              * C1: Z-scored I_post1
              * C2: Z-scored I_sub = max(0, I_post1 - I_pre)
              * C3: Z-scored (I_PE + 3D CLAHE)
Saves compressed `{CASE_ID}.npz` files to `data/processed/npz_96x96x96/` and `splits_metadata.csv` to `data/splits/`.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, seed_everything
from src.preprocessing import (
    run_step1_pipeline,
    standardize_and_extract_voi,
    synthesize_three_channels,
)


def execute_steps_1_to_3(config_path: str = "configs/default_config.yaml") -> None:
    """Run Steps 1, 2, and 3 and verify output Data Contract."""
    t_start = time.time()
    cfg = load_config(config_path)
    seed_everything(cfg["project"]["seed"])

    paths_cfg = cfg["paths"]
    s1_cfg = cfg["step1_ingestion"]
    s2_cfg = cfg["step2_spatial_voi"]
    s3_cfg = cfg["step3_image_processing"]

    # ==========================================================================
    # STEP 1: Data Ingestion, Filtering & Stratified Splitting
    # ==========================================================================
    print("=" * 80)
    print("STEP 1: DATA INGESTION, CASE FILTERING & STRATIFIED SPLITTING")
    print("=" * 80)
    metadata_df = run_step1_pipeline(
        clinical_excel_path=paths_cfg["clinical_excel"],
        splits_csv_path=paths_cfg["official_splits_csv"],
        output_metadata_csv_path=paths_cfg["splits_metadata_csv"],
        val_ratio=float(s1_cfg["val_ratio"]),
        seed=int(cfg["project"]["seed"]),
    )

    print(f"Total valid cases retained: {len(metadata_df)}")
    print("\nSplit distribution by Tumor Subtype:")
    pivot_table = pd.crosstab(metadata_df["tumor_subtype"], metadata_df["split"], margins=True)
    print(pivot_table)

    # ==========================================================================
    # STEP 2 & STEP 3: 3D VOI Standardization & Multi-Channel Image Processing
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STEP 2 & 3: 3D VOI (96x96x96) EXTRACTION & 3-CHANNEL SYNTHESIS")
    print("=" * 80)

    raw_images_dir = Path(paths_cfg["raw_images_dir"])
    raw_expert_dir = Path(paths_cfg["raw_expert_seg_dir"])
    raw_auto_dir = Path(paths_cfg["raw_auto_seg_dir"])
    out_npz_dir = Path(paths_cfg["processed_npz_dir"])
    out_npz_dir.mkdir(parents=True, exist_ok=True)

    target_orientation = str(s2_cfg["target_orientation"])
    target_spacing = tuple(float(x) for x in s2_cfg["target_spacing"])
    voi_size = tuple(int(x) for x in s2_cfg["voi_size"])
    margin_ratio = float(s2_cfg["voi_margin_ratio"])
    pad_mode = str(s2_cfg["pad_mode"])

    contrast_stats = []

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing 3D VOIs"):
        case_id = str(row["patient_id"])
        subtype_id = int(row["subtype_id"])

        pre_path = raw_images_dir / case_id / f"{case_id}_0000.nii.gz"
        post1_path = raw_images_dir / case_id / f"{case_id}_0001.nii.gz"
        expert_path = raw_expert_dir / f"{case_id}.nii.gz"
        auto_path = raw_auto_dir / f"{case_id}.nii.gz"

        pre_sitk = sitk.ReadImage(str(pre_path), sitk.sitkFloat32)
        post1_sitk = sitk.ReadImage(str(post1_path), sitk.sitkFloat32)
        expert_sitk = sitk.ReadImage(str(expert_path), sitk.sitkUInt8)
        auto_sitk = sitk.ReadImage(str(auto_path), sitk.sitkUInt8) if auto_path.exists() else None

        # Step 2: RAS + 1x1x1 mm^3 Isotropic Resampling + 96x96x96 VOI Crop/Pad
        pre_voi, post1_voi, expert_voi, auto_voi, _ = standardize_and_extract_voi(
            pre_sitk=pre_sitk,
            post1_sitk=post1_sitk,
            expert_mask_sitk=expert_sitk,
            auto_mask_sitk=auto_sitk,
            target_orientation=target_orientation,
            target_spacing=target_spacing,
            voi_size=voi_size,
            margin_ratio=margin_ratio,
            pad_mode=pad_mode,
        )

        # Step 3: Denoising + Clipping [0.5, 99.5] + 3-Channel Synthesis [C1_post1, C2_sub, C3_pe_clahe]
        image_3ch = synthesize_three_channels(
            pre_voi=pre_voi,
            post1_voi=post1_voi,
            apply_n4=bool(s3_cfg["apply_n4_bias_correction"]),
            n4_shrink_factor=int(s3_cfg["n4_shrink_factor"]),
            apply_denoise=bool(s3_cfg["apply_edge_preserving_denoise"]),
            denoise_method=str(s3_cfg["denoise_method"]),
            nlmeans_patch_size=int(s3_cfg["nlmeans_patch_size"]),
            nlmeans_patch_distance=int(s3_cfg["nlmeans_patch_distance"]),
            nlmeans_h_factor=float(s3_cfg["nlmeans_h_factor"]),
            clip_percentiles=tuple(float(x) for x in s3_cfg["clip_percentiles"]),
            pe_tau_percentile=float(s3_cfg["pe_tau_percentile"]),
            clahe_clip_limit=float(s3_cfg["clahe_clip_limit"]),
            clahe_kernel_size=int(s3_cfg["clahe_kernel_size"]),
            zscore_eps=float(s3_cfg["zscore_eps"]),
            output_dtype=str(s3_cfg["output_dtype"]),
        )

        mask_4d = expert_voi[np.newaxis, ...].astype(np.uint8)
        auto_mask_4d = auto_voi[np.newaxis, ...].astype(np.uint8)
        subtype_scalar = np.int64(subtype_id)

        # Save compressed .npz adhering strictly to the Data Contract
        npz_out_path = out_npz_dir / f"{case_id}.npz"
        np.savez_compressed(
            npz_out_path,
            image=image_3ch,
            mask=mask_4d,
            auto_mask=auto_mask_4d,
            subtype_id=subtype_scalar,
        )

        # Measure tumor-to-background Z-score contrast across C1, C2, C3
        fg = expert_voi > 0
        bg = ~fg
        if np.any(fg) and np.any(bg):
            c1_diff = float(np.mean(image_3ch[0][fg]) - np.mean(image_3ch[0][bg]))
            c2_diff = float(np.mean(image_3ch[1][fg]) - np.mean(image_3ch[1][bg]))
            c3_diff = float(np.mean(image_3ch[2][fg]) - np.mean(image_3ch[2][bg]))
            contrast_stats.append((c1_diff, c2_diff, c3_diff))

    # ==========================================================================
    # VERIFICATION OF DATA CONTRACT
    # ==========================================================================
    sample_case = str(metadata_df.iloc[0]["patient_id"])
    sample_npz = np.load(out_npz_dir / f"{sample_case}.npz")
    mean_c1, mean_c2, mean_c3 = np.mean(contrast_stats, axis=0)

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("DATA CONTRACT & QUALITY VERIFICATION REPORT")
    print("=" * 80)
    print(f"Sample file verified : {out_npz_dir / f'{sample_case}.npz'}")
    print(f"  - keys             : {list(sample_npz.keys())}")
    print(f"  - image            : shape={sample_npz['image'].shape}, dtype={sample_npz['image'].dtype}")
    print(f"  - mask             : shape={sample_npz['mask'].shape}, dtype={sample_npz['mask'].dtype}")
    print(f"  - auto_mask        : shape={sample_npz['auto_mask'].shape}, dtype={sample_npz['auto_mask'].dtype}")
    print(f"  - subtype_id       : value={sample_npz['subtype_id']}, dtype={sample_npz['subtype_id'].dtype}")
    print(f"\nMean Tumor-vs-Background Contrast (Delta Z-score):")
    print(f"  - Channel C1 (Post1)      : +{mean_c1:.3f} SD")
    print(f"  - Channel C2 (Subtraction): +{mean_c2:.3f} SD (Fat background suppressed)")
    print(f"  - Channel C3 (PE + CLAHE) : +{mean_c3:.3f} SD (Rim/NME contrast equalized)")
    print(f"\nTotal pipeline execution time: {elapsed:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    execute_steps_1_to_3()
