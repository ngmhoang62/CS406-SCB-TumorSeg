"""Step 1: Data Ingestion, Case Filtering, Subtype Mapping & Stratified Splitting."""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SUBTYPE_TO_ID: Dict[str, int] = {
    "Luminal": 0,
    "Triple Negative": 1,
    "HER2-enriched": 2,
    "HER2-pure": 3,
}

ID_TO_SUBTYPE: Dict[int, str] = {v: k for k, v in SUBTYPE_TO_ID.items()}

RAW_SUBTYPE_MAPPING: Dict[str, str] = {
    "luminal": "Luminal",
    "luminal_a": "Luminal",
    "luminal_b": "Luminal",
    "Luminal": "Luminal",
    "triple_negative": "Triple Negative",
    "Triple Negative": "Triple Negative",
    "her2_enriched": "HER2-enriched",
    "HER2-enriched": "HER2-enriched",
    "her2_pure": "HER2-pure",
    "HER2-pure": "HER2-pure",
}


def load_and_filter_clinical_data(
    clinical_excel_path: str | Path,
    sheet_name: str = "dataset_info",
) -> pd.DataFrame:
    """Load MAMA-MIA clinical Excel file, map tumor subtypes to 4 canonical classes, and drop N/A."""
    df = pd.read_excel(clinical_excel_path, sheet_name=sheet_name)

    # Keep raw subtype for auditing
    df["raw_tumor_subtype"] = df["tumor_subtype"]

    # Map to canonical 4 classes: Luminal, Triple Negative, HER2-enriched, HER2-pure
    df["tumor_subtype"] = df["raw_tumor_subtype"].map(
        lambda x: RAW_SUBTYPE_MAPPING.get(str(x).strip(), np.nan) if pd.notna(x) else np.nan
    )

    # Filter out cases with missing/unmapped tumor_subtype (26 N/A cases in real dataset)
    filtered_df = df.dropna(subset=["tumor_subtype"]).copy()
    filtered_df["subtype_id"] = filtered_df["tumor_subtype"].map(SUBTYPE_TO_ID).astype(int)

    return filtered_df.reset_index(drop=True)


def parse_official_splits(splits_csv_path: str | Path) -> Dict[str, str]:
    """Parse official MAMA-MIA train_test_splits.csv into {patient_id: 'train' | 'test'}."""
    splits_df = pd.read_csv(splits_csv_path)
    split_map: Dict[str, str] = {}

    # Format 1: Columns ['train_split', 'test_split'] as in official MAMA-MIA repo
    if "train_split" in splits_df.columns and "test_split" in splits_df.columns:
        for pid in splits_df["train_split"].dropna().astype(str).str.strip():
            if pid:
                split_map[pid] = "train"
        for pid in splits_df["test_split"].dropna().astype(str).str.strip():
            if pid:
                split_map[pid] = "test"
    # Format 2: Columns ['patient_id', 'split']
    elif "patient_id" in splits_df.columns and "split" in splits_df.columns:
        for _, row in splits_df.iterrows():
            split_map[str(row["patient_id"]).strip()] = str(row["split"]).strip().lower()
    else:
        raise ValueError(
            f"Unrecognized columns in {splits_csv_path}: {list(splits_df.columns)}. "
            "Expected ['train_split', 'test_split'] or ['patient_id', 'split']."
        )
    return split_map


def create_stratified_splits_and_weights(
    clinical_df: pd.DataFrame,
    official_split_map: Dict[str, str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create Stratified Train (85%) / Val (15%) split from official Train set based on
    (tumor_subtype, dataset), preserve official Test set, and compute class-balanced weights.
    """
    df = clinical_df.copy()
    df["official_split"] = df["patient_id"].map(lambda pid: official_split_map.get(str(pid), "train"))

    train_val_df = df[df["official_split"] == "train"].copy()
    test_df = df[df["official_split"] == "test"].copy()
    test_df["split"] = "test"

    # Build composite stratification key: (tumor_subtype, dataset)
    strata_composite = train_val_df["tumor_subtype"].astype(str) + "__" + train_val_df["dataset"].astype(str)
    min_stratum_count = strata_composite.value_counts().min()

    if min_stratum_count >= 2:
        stratify_col = strata_composite
    else:
        # Fallback to tumor_subtype only if dataset-subtype combinations have singleton cells
        stratify_col = train_val_df["tumor_subtype"]

    train_idx, val_idx = train_test_split(
        train_val_df.index,
        test_size=val_ratio,
         random_state=seed,
        stratify=stratify_col,
    )

    train_val_df.loc[train_idx, "split"] = "train"
    train_val_df.loc[val_idx, "split"] = "val"

    merged_df = pd.concat([train_val_df, test_df], axis=0).sort_values("patient_id").reset_index(drop=True)

    # Compute gentle class-balanced sampling weights for training set (Step 1.2)
    # Using smoothed inverse frequency: w_c = (N / (K * n_c)) ** 0.5 so rare HER2-pure is boosted without overfitting
    train_subset = merged_df[merged_df["split"] == "train"]
    subtype_counts = train_subset["subtype_id"].value_counts().to_dict()
    total_train = len(train_subset)
    num_classes = len(SUBTYPE_TO_ID)

    class_weights: Dict[int, float] = {}
    for cid in SUBTYPE_TO_ID.values():
        cnt = max(subtype_counts.get(cid, 1), 1)
        class_weights[cid] = float(np.sqrt(total_train / (num_classes * cnt)))

    merged_df["sample_weight"] = merged_df.apply(
        lambda r: class_weights[int(r["subtype_id"])] if r["split"] == "train" else 1.0,
        axis=1,
    )

    # Select clean, standardized metadata columns for downstream DataLoader & FiLM conditioning
    keep_cols = [
        "patient_id",
        "dataset",
        "tumor_subtype",
        "subtype_id",
        "split",
        "sample_weight",
        "view",
        "pixel_spacing",
        "slice_thickness",
        "manufacturer",
        "field_strength",
    ]
    available_cols = [c for c in keep_cols if c in merged_df.columns]
    return merged_df[available_cols]


def run_step1_pipeline(
    clinical_excel_path: str | Path,
    splits_csv_path: str | Path,
    output_metadata_csv_path: str | Path,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Execute full Step 1 pipeline and save splits_metadata.csv."""
    filtered_df = load_and_filter_clinical_data(clinical_excel_path)
    split_map = parse_official_splits(splits_csv_path)
    metadata_df = create_stratified_splits_and_weights(
        filtered_df,
        official_split_map=split_map,
        val_ratio=val_ratio,
        seed=seed,
    )

    out_path = Path(output_metadata_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(out_path, index=False)
    return metadata_df
