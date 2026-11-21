"""Versioned full-volume NPZ data with training-only patch sampling."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose, MapTransform, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, SpatialPadd,
)

from src.preprocessing.full_volume import CONTRACT_VERSION
from .boundary import boundary_from_mask

IP_CHANNELS = {
    "IP1": [0], "IP2": [1], "IP3": [0, 1],
    "IP1_post1": [0], "IP2_sub": [1], "IP3_multi": [0, 1],
}


class LoadNPZd(MapTransform):
    def __init__(self, ip_mode, require_mask=True, boundary_radius=None):
        super().__init__(["npz_path"])
        if ip_mode not in IP_CHANNELS:
            raise ValueError(f"Unknown ip_mode {ip_mode!r}")
        self.channels = IP_CHANNELS[ip_mode]
        self.require_mask = require_mask
        self.boundary_radius = boundary_radius

    def __call__(self, item):
        result = dict(item)
        path = result.pop("npz_path")
        with np.load(path, allow_pickle=False) as data:
            if "contract_version" not in data or str(data["contract_version"]) != CONTRACT_VERSION:
                raise ValueError(f"{path}: regenerate this case using full-volume preprocessing")
            image = data["image"]
            if image.ndim != 4 or image.shape[0] != 2 or not np.isfinite(image).all():
                raise ValueError(f"{path}: expected a finite image (2,D,H,W)")
            result["image"] = torch.from_numpy(image[self.channels].astype(np.float32))
            if self.require_mask:
                mask = data["mask"]
                if mask.shape != (1, *image.shape[1:]) or not np.isin(mask, [0, 1]).all():
                    raise ValueError(f"{path}: mask must be binary and aligned with the image")
                result["mask"] = torch.from_numpy(mask.astype(np.float32))
                if self.boundary_radius is not None:
                    boundary = boundary_from_mask(mask, self.boundary_radius)
                    result["boundary"] = torch.from_numpy(boundary.astype(np.float32))
        return result


class MAMA_MIA_Dataset(CacheDataset):
    """Training draws positive/background patches. Val/test keep the full scan."""

    def __init__(self, metadata_csv, npz_dir, split="train", ip_mode="IP3",
                 patch_size=(96, 96, 96), augment=True, cache_rate=0.0, seed=42,
                 positive_patch_probability=0.5, patches_per_case=2,
                 boundary_aux=False, boundary_radius=1):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        if not 0 <= positive_patch_probability <= 1 or patches_per_case < 1:
            raise ValueError("Invalid patch sampling parameters")
        frame = pd.read_csv(metadata_csv)
        if not {"patient_id", "split"}.issubset(frame.columns):
            raise ValueError("Metadata requires patient_id and split")
        if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
            raise ValueError("patient_id must be present and unique across splits")
        if not frame.split.isin(["train", "val", "test"]).all():
            raise ValueError("Invalid split in metadata")
        frame = frame.loc[frame.split == split].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No cases for split={split}")
        records = []
        for row in frame.to_dict("records"):
            path = Path(npz_dir) / f"{row['patient_id']}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            # Clinical columns may mix strings and missing values. The baseline
            # collates only identifiers, images and labels.
            records.append({"patient_id": str(row["patient_id"]),
                            "dataset": str(row.get("dataset", "unknown")),
                            "npz_path": str(path)})
        use_boundary = boundary_aux and split == "train"
        transforms = [LoadNPZd(ip_mode, boundary_radius=boundary_radius if use_boundary else None)]
        if split == "train":
            if patch_size is None or len(patch_size) != 3 or min(patch_size) < 1:
                raise ValueError("Training patch_size must contain three positive sizes")
            keys = ["image", "mask"]
            if use_boundary:
                keys.append("boundary")
            transforms += [
                SpatialPadd(keys, spatial_size=patch_size),
                RandCropByPosNegLabeld(
                    keys, label_key="mask", spatial_size=patch_size,
                    pos=positive_patch_probability, neg=1 - positive_patch_probability,
                    num_samples=patches_per_case,
                ),
            ]
            if augment:
                transforms += [RandFlipd(keys, prob=0.5, spatial_axis=axis) for axis in range(3)]
                planes = next(((a, b) for a, b in [(0, 1), (0, 2), (1, 2)]
                               if patch_size[a] == patch_size[b]), None)
                if planes is not None:
                    transforms.append(RandRotate90d(keys, prob=0.3, spatial_axes=planes))
                transforms.append(RandScaleIntensityd(["image"], factors=0.1, prob=0.5))
        transform = Compose(transforms)
        transform.set_random_state(seed=seed)
        super().__init__(records, transform, cache_rate=cache_rate, num_workers=1,
                         progress=False, copy_cache=True)
        self.metadata = frame


def build_loaders(metadata_csv, npz_dir, ip_mode="IP3", batch_size=2,
                  patch_size=(96, 96, 96), num_workers=0, cache_rate=0.0, seed=42,
                  positive_patch_probability=0.5, patches_per_case=2,
                  boundary_aux=False, boundary_radius=1):
    datasets = {
        split: MAMA_MIA_Dataset(
            metadata_csv, npz_dir, split, ip_mode, patch_size=patch_size,
            cache_rate=cache_rate, seed=seed,
            positive_patch_probability=positive_patch_probability,
            patches_per_case=patches_per_case,
            boundary_aux=boundary_aux, boundary_radius=boundary_radius,
        )
        for split in ("train", "val")
    }
    return {
        split: DataLoader(
            dataset, batch_size=batch_size if split == "train" else 1,
            shuffle=split == "train", num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(seed),
        )
        for split, dataset in datasets.items()
    }
