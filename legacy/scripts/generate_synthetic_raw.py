"""
Generate Realistic Synthetic Raw MAMA-MIA Data (.nii.gz, .xlsx, .csv) for Testing Steps 1, 2, 3.

Simulates:
  - 34 total entries in `clinical_and_imaging_info.xlsx` (sheet `dataset_info`):
      * 32 valid cases across 4 datasets (ISPY2, DUKE, ISPY1, NACT) and 4 tumor subtypes
        (including raw labels `luminal`, `luminal_a`, `luminal_b`, `triple_negative`,
         `her2_enriched`, `her2_pure`)
      * 2 invalid cases with `tumor_subtype = NaN` to test Step 1 filtering
  - `train_test_splits.csv` in official MAMA-MIA format (`train_split`, `test_split`)
  - 3D NIfTI volumes (`_0000.nii.gz` Pre-contrast, `_0001.nii.gz` Post-contrast,
    `expert/{CASE_ID}.nii.gz`, `automatic/{CASE_ID}.nii.gz`) with:
      * Multi-center anisotropic voxel spacings & Axial/Sagittal orientations
      * Bright T1 adipose tissue + fibroglandular background + smooth bias field
      * Metallic surgical clip extreme intensity spikes (>99.5th percentile)
      * Subtype-specific 3D tumor morphology:
          - Luminal: Spiculated / irregular margin
          - Triple Negative: Circumscribed rim-enhancing mass with necrotic core
          - HER2-enriched: Core tumor + weak peripheral Non-Mass Enhancement (NME)
          - HER2-pure: High-enhancement focal mass
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, seed_everything


def _create_subtype_3d_phantom(
    shape_zyx: Tuple[int, int, int],
    subtype_canonical: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize 3D Pre-contrast, Post-contrast, Expert Mask, and Preliminary Auto Mask.
    Returns (pre_vol, post1_vol, expert_mask, auto_mask).
    """
    d, h, w = shape_zyx
    zz, yy, xx = np.meshgrid(
        np.arange(d, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    cz = d / 2.0 + rng.uniform(-4.0, 4.0)
    cy = h / 2.0 + rng.uniform(-4.0, 4.0)
    cx = w / 2.0 + rng.uniform(-4.0, 4.0)

    dz = zz - cz
    dy = yy - cy
    dx = xx - cx
    radius_map = np.sqrt(dz**2 + dy**2 + dx**2) + 1e-6

    # 1. Anatomical background: breast tissue envelope + bright T1 fatty tissue lobes
    breast_env = ((zz / d - 0.5) ** 2 + (yy / h - 0.5) ** 2 + (xx / w - 0.5) ** 2) < 0.22
    fat_pattern = (
        np.sin(zz * 0.18 + rng.uniform(0, 3.14))
        * np.cos(yy * 0.18 + rng.uniform(0, 3.14))
        * np.sin(xx * 0.18 + rng.uniform(0, 3.14))
    )
    fat_mask = (fat_pattern > 0.20) & breast_env

    # Base T1 pre-contrast intensity: parenchyma ~120, bright adipose tissue ~320 (cancelable by Subtraction)
    pre_vol = np.zeros(shape_zyx, dtype=np.float32)
    pre_vol[breast_env] = 120.0 + rng.normal(0.0, 8.0, size= int(np.sum(breast_env)))
    pre_vol[fat_mask] += 210.0

    # Smooth 3D magnetic field inhomogeneity (bias field)
    bias_field = 1.0 + 0.12 * ((zz - d / 2) / d) + 0.10 * ((yy - h / 2) / h) - 0.08 * ((xx - w / 2) / w)
    pre_vol *= bias_field

    post1_vol = pre_vol.copy()
    # Slight physiological background parenchymal enhancement (BPE)
    post1_vol[breast_env] += 12.0

    base_r = float(rng.uniform(11.0, 16.0))

    # 2. Subtype-specific 3D tumor geometry and contrast enhancement kinetics
    if subtype_canonical == "Luminal":
        # Spiculated margin via angular harmonic modulation
        theta = np.arccos(np.clip(dz / radius_map, -1.0, 1.0))
        phi = np.arctan2(dy, dx)
        spiculation = 1.0 + 0.22 * np.sin(5.0 * theta) * np.cos(5.0 * phi) + 0.12 * np.cos(7.0 * theta)
        effective_r = base_r * spiculation
        expert_mask = (radius_map <= effective_r).astype(np.uint8)
        enhancement = 220.0 * np.exp(-0.5 * (radius_map / (effective_r * 0.85)) ** 2)
        post1_vol[expert_mask > 0] += enhancement[expert_mask > 0]
        auto_mask = (radius_map <= (base_r * 0.92)).astype(np.uint8)

    elif subtype_canonical == "Triple Negative":
        # Smooth oval/round rim-enhancing tumor with central unenhanced necrotic core
        ellip_r = np.sqrt((dz / 1.05) ** 2 + (dy / 0.95) ** 2 + (dx / 1.0) ** 2)
        expert_mask = (ellip_r <= base_r).astype(np.uint8)
        necrotic_core = ellip_r <= (base_r * 0.45)
        viable_rim = (expert_mask > 0) & (~necrotic_core)

        # Rim enhances strongly; necrotic center has almost no enhancement (tests Hole Filling in Step 7!)
        post1_vol[viable_rim] += 260.0
        post1_vol[necrotic_core] += 10.0
        # Preliminary auto-segmentation misses the necrotic core
        auto_mask = viable_rim.astype(np.uint8)

    elif subtype_canonical == "HER2-enriched":
        # Core mass + diffuse peripheral Non-Mass Enhancement (NME) zone (tests PE + 3D CLAHE in Step 3!)
        core_mask = radius_map <= (base_r * 0.72)
        nme_shell = (radius_map > (base_r * 0.72)) & (radius_map <= (base_r * 1.15)) & (dz > -base_r * 0.6)
        expert_mask = (core_mask | nme_shell).astype(np.uint8)

        post1_vol[core_mask] += 240.0
        # Subtle enhancement in NME shell
        post1_vol[nme_shell] += 65.0
        auto_mask = core_mask.astype(np.uint8)

    else:  # HER2-pure
        # Compact hyper-vascular tumor
        expert_mask = (radius_map <= (base_r * 0.90)).astype(np.uint8)
        post1_vol[expert_mask > 0] += 285.0
        auto_mask = (radius_map <= (base_r * 0.85)).astype(np.uint8)

    # Add a small false-positive vessel blob to automatic mask to test 3D CCA in Step 7
    fp_z = min(d - 4, max(3, int(cz + base_r + 12)))
    fp_y = min(h - 4, max(3, int(cy - 10)))
    fp_x = min(w - 4, max(3, int(cx + 8)))
    auto_mask[fp_z - 2 : fp_z + 2, fp_y - 2 : fp_y + 2, fp_x - 2 : fp_x + 2] = 1

    # 3. Add thermal MRI noise
    pre_vol += rng.normal(0.0, 9.0, size=shape_zyx).astype(np.float32)
    post1_vol += rng.normal(0.0, 9.0, size=shape_zyx).astype(np.float32)
    pre_vol = np.clip(pre_vol, 0.0, None)
    post1_vol = np.clip(post1_vol, 0.0, None)

    # 4. Inject metallic surgical clip artifacts (extreme intensity spikes > 3000)
    clip_z, clip_y, clip_x = int(cz), int(cy + base_r * 0.5), int(cx)
    pre_vol[clip_z : clip_z + 2, clip_y : clip_y + 2, clip_x : clip_x + 2] = 3500.0
    post1_vol[clip_z : clip_z + 2, clip_y : clip_y + 2, clip_x : clip_x + 2] = 3600.0

    return pre_vol, post1_vol, expert_mask, auto_mask


def _numpy_to_sitk_with_geometry(
    arr_zyx: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    view: str,
    is_mask: bool = False,
) -> sitk.Image:
    """Wrap 3D numpy array into SimpleITK Image with realistic Axial or Sagittal direction cosines."""
    img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8 if is_mask else np.float32))
    img.SetSpacing(spacing_xyz)
    img.SetOrigin((0.0, 0.0, 0.0))

    if view == "sagittal":
        # Non-RAS direction matrix (LPS / Sagittal-like) to test Step 2 RAS standardization
        direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    else:
        # Standard RAS direction matrix
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    img.SetDirection(direction)
    return img


def generate_synthetic_raw_dataset(config_path: str = "configs/default_config.yaml") -> None:
    """Generate 32 valid + 2 N/A synthetic MAMA-MIA raw cases on disk."""
    cfg = load_config(config_path)
    seed_everything(cfg["project"]["seed"])
    rng = np.random.default_rng(cfg["project"]["seed"])

    raw_images_dir = Path(cfg["paths"]["raw_images_dir"])
    raw_expert_dir = Path(cfg["paths"]["raw_expert_seg_dir"])
    raw_auto_dir = Path(cfg["paths"]["raw_auto_seg_dir"])
    clinical_excel_path = Path(cfg["paths"]["clinical_excel"])
    splits_csv_path = Path(cfg["paths"]["official_splits_csv"])

    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_expert_dir.mkdir(parents=True, exist_ok=True)
    raw_auto_dir.mkdir(parents=True, exist_ok=True)
    clinical_excel_path.parent.mkdir(parents=True, exist_ok=True)

    # 32 valid cases: 8 cases per canonical subtype (enough for 24 Train/Val + 8 Test)
    # Using authentic raw MAMA-MIA subtype strings (including luminal_a, luminal_b)
    subtype_plan = [
        ("luminal", "Luminal"),
        ("luminal_a", "Luminal"),
        ("luminal_b", "Luminal"),
        ("luminal", "Luminal"),
        ("luminal_a", "Luminal"),
        ("luminal_b", "Luminal"),
        ("luminal", "Luminal"),
        ("luminal_a", "Luminal"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("triple_negative", "Triple Negative"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_enriched", "HER2-enriched"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
        ("her2_pure", "HER2-pure"),
    ]

    datasets_cycle = ["ISPY2", "DUKE", "ISPY1", "NACT"]
    spacings_cycle = [
        (1.0, 1.0, 1.0),
        (0.85, 0.85, 1.25),
        (0.90, 0.90, 1.20),
        (1.10, 1.10, 1.00),
    ]

    clinical_rows = []
    train_ids = []
    test_ids = []

    print(f"[Synthetic Generator] Creating 32 valid 3D DCE-MRI cases + 2 N/A cases...")

    for idx, (raw_sub, canon_sub) in enumerate(subtype_plan):
        ds_name = datasets_cycle[idx % len(datasets_cycle)]
        patient_id = f"{ds_name}_{idx + 1:03d}"
        view = "sagittal" if (idx % 5 == 0) else "axial"
        spacing_xyz = spacings_cycle[idx % len(spacings_cycle)]

        # Last 2 cases of each 8-case subtype block go to official test_split (total 8 test, 24 train_val)
        within_subtype_idx = idx % 8
        if within_subtype_idx >= 6:
            test_ids.append(patient_id)
        else:
            train_ids.append(patient_id)

        # Compute raw voxel grid size so physical volume is ~84-104 mm^3 (tests both padding and cropping to 96^3)
        phys_dim = 88.0 if (idx % 2 == 0) else 104.0
        shape_zyx = (
            int(round(phys_dim / spacing_xyz[2])),
            int(round(phys_dim / spacing_xyz[1])),
            int(round(phys_dim / spacing_xyz[0])),
        )

        pre_vol, post1_vol, expert_mask, auto_mask = _create_subtype_3d_phantom(
            shape_zyx=shape_zyx,
            subtype_canonical=canon_sub,
            rng=rng,
        )

        pre_sitk = _numpy_to_sitk_with_geometry(pre_vol, spacing_xyz, view=view, is_mask=False)
        post1_sitk = _numpy_to_sitk_with_geometry(post1_vol, spacing_xyz, view=view, is_mask=False)
        expert_sitk = _numpy_to_sitk_with_geometry(expert_mask, spacing_xyz, view=view, is_mask=True)
        auto_sitk = _numpy_to_sitk_with_geometry(auto_mask, spacing_xyz, view=view, is_mask=True)

        case_img_dir = raw_images_dir / patient_id
        case_img_dir.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pre_sitk, str(case_img_dir / f"{patient_id}_0000.nii.gz"), useCompression=True)
        sitk.WriteImage(post1_sitk, str(case_img_dir / f"{patient_id}_0001.nii.gz"), useCompression=True)
        sitk.WriteImage(expert_sitk, str(raw_expert_dir / f"{patient_id}.nii.gz"), useCompression=True)
        sitk.WriteImage(auto_sitk, str(raw_auto_dir / f"{patient_id}.nii.gz"), useCompression=True)

        clinical_rows.append(
            {
                "patient_id": patient_id,
                "dataset": ds_name,
                "tumor_subtype": raw_sub,
                "age": int(rng.integers(32, 72)),
                "view": view,
                "num_phases": 4,
                "fat_suppressed": 1,
                "field_strength": 3.0 if (idx % 2 == 0) else 1.5,
                "image_rows": shape_zyx[1],
                "image_columns": shape_zyx[2],
                "num_slices": shape_zyx[0],
                "pixel_spacing": f"[{spacing_xyz[0]}, {spacing_xyz[1]}]",
                "slice_thickness": float(spacing_xyz[2]),
                "manufacturer": "SIEMENS" if (idx % 2 == 0) else "GE",
                "pcr": int(idx % 2),
            }
        )

    # Append 2 cases with missing tumor_subtype (NaN) to verify Step 1 filtering removes them
    for na_i in range(2):
        na_id = f"DUKE_99{na_i}"
        train_ids.append(na_id)
        clinical_rows.append(
            {
                "patient_id": na_id,
                "dataset": "DUKE",
                "tumor_subtype": np.nan,
                "age": 50,
                "view": "axial",
                "num_phases": 4,
                "fat_suppressed": 1,
                "field_strength": 1.5,
                "image_rows": 88,
                "image_columns": 88,
                "num_slices": 88,
                "pixel_spacing": "[1.0, 1.0]",
                "slice_thickness": 1.0,
                "manufacturer": "GE",
                "pcr": 0,
            }
        )

    clinical_df = pd.DataFrame(clinical_rows)
    with pd.ExcelWriter(clinical_excel_path, engine="openpyxl") as writer:
        clinical_df.to_excel(writer, sheet_name="dataset_info", index=False)

    # Save official train_test_splits.csv with unequal column lengths padded by NaN
    max_len = max(len(train_ids), len(test_ids))
    splits_df = pd.DataFrame(
        {
            "train_split": train_ids + [np.nan] * (max_len - len(train_ids)),
            "test_split": test_ids + [np.nan] * (max_len - len(test_ids)),
        }
    )
    splits_df.to_csv(splits_csv_path, index=False)

    print(f"[Synthetic Generator] Done!")
    print(f"  -> Clinical Excel : {clinical_excel_path} ({len(clinical_df)} rows: 32 valid + 2 NaN)")
    print(f"  -> Official Splits: {splits_csv_path} ({len(train_ids)} train_split, {len(test_ids)} test_split)")
    print(f"  -> Raw NIfTI Dir  : {raw_images_dir} (32 case folders with _0000 & _0001)")


if __name__ == "__main__":
    generate_synthetic_raw_dataset()
