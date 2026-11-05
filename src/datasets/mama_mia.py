"""TV4 week 1: cached NPZ volumes and paired, on-the-fly 3D augmentation."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import CacheDataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from monai.transforms import (
    Compose, MapTransform, RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandSpatialCropd, SpatialPadd,
)

IP_CHANNELS = {"IP1_post1": [0], "IP2_sub": [1], "IP3_multi": [0, 1, 2]}
REQUIRED_COLUMNS = {"patient_id", "dataset", "tumor_subtype", "subtype_id", "split", "sample_weight"}


class LoadNPZd(MapTransform):
    """Deterministic loading is cached; random transforms run after cache lookup."""

    def __init__(self, ip_mode):
        super().__init__(["npz_path"])
        if ip_mode not in IP_CHANNELS:
            raise ValueError(f"Unknown ip_mode {ip_mode!r}; choose {tuple(IP_CHANNELS)}")
        self.channels = IP_CHANNELS[ip_mode]

    def __call__(self, item):
        result = dict(item)
        path = result.pop("npz_path")
        with np.load(path, allow_pickle=False) as data:
            required = {"image", "mask", "auto_mask", "subtype_id"}
            if not required.issubset(data.files):
                raise ValueError(f"{path}: missing keys {required - set(data.files)}")
            image = data["image"].astype(np.float32)
            mask = data["mask"]
            if image.ndim != 4 or image.shape[0] != 3:
                raise ValueError(f"{path}: expected image (3,D,H,W), got {image.shape}")
            if mask.shape != (1, *image.shape[1:]) or data["auto_mask"].shape != mask.shape:
                raise ValueError(f"{path}: image/mask/auto_mask shape mismatch")
            if not np.isfinite(image).all() or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"{path}: image must be finite and mask binary")
            subtype = int(data["subtype_id"].item())
            if subtype not in range(4) or subtype != int(result["subtype_id"]):
                raise ValueError(f"{path}: invalid subtype or disagreement with metadata")
            result["image"] = torch.from_numpy(image[self.channels].copy())
            result["mask"] = torch.from_numpy(mask.astype(np.float32))
            result["subtype_id"] = torch.tensor(subtype, dtype=torch.long)
        return result


class MAMA_MIA_Dataset(CacheDataset):
    """Read the team NPZ contract. Validation/test retain the entire VOI.

    ``patch_size`` affects training only. Real training normally uses 96^3;
    the notebook uses smaller patches for a CPU smoke test. Cache stores only
    deterministic work, so repeated training accesses draw new augmentations.
    """

    def __init__(self, metadata_csv, npz_dir, split="train", ip_mode="IP3_multi",
                 patch_size=(96, 96, 96), augment=True, cache_rate=1.0, seed=42):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val or test")
        frame = pd.read_csv(metadata_csv)
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            raise ValueError(f"Missing metadata columns: {REQUIRED_COLUMNS - set(frame.columns)}")
        if frame.patient_id.duplicated().any():
            raise ValueError("patient_id must be unique across splits (avoid leakage)")
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
            records.append({**row, "npz_path": str(path)})
        transforms = [LoadNPZd(ip_mode)]
        if split == "train":
            keys = ["image", "mask"]
            if patch_size is not None:
                if len(patch_size) != 3 or min(patch_size) < 1:
                    raise ValueError("patch_size must contain 3 positive sizes")
                transforms += [SpatialPadd(keys, spatial_size=patch_size),
                               RandSpatialCropd(keys, roi_size=patch_size, random_size=False)]
            if augment:
                transforms += [RandFlipd(keys, prob=0.5, spatial_axis=axis) for axis in range(3)]
                # Rotate the two equal axes so non-cubic smoke patches stay batchable.
                planes = (0, 1) if patch_size is None else next(
                    ((a, b) for a, b in [(0, 1), (0, 2), (1, 2)]
                     if patch_size[a] == patch_size[b]), None)
                if planes is not None:
                    transforms.append(RandRotate90d(keys, prob=0.3, max_k=3, spatial_axes=planes))
                transforms.append(RandScaleIntensityd(["image"], factors=0.1, prob=0.5))
        transform = Compose(transforms)
        transform.set_random_state(seed=seed)
        super().__init__(records, transform=transform, cache_rate=cache_rate,
                         num_workers=1, progress=False, copy_cache=True)
        self.metadata = frame


def build_loaders(metadata_csv, npz_dir, ip_mode="IP3_multi", batch_size=2,
                  patch_size=(96, 96, 96), num_workers=0, cache_rate=1.0, seed=42,
                  use_weighted_sampler=True):
    """Use num_workers=0 in Windows notebooks; val batch=1 handles variable VOIs."""
    datasets = {
        split: MAMA_MIA_Dataset(metadata_csv, npz_dir, split, ip_mode,
                               patch_size=patch_size, cache_rate=cache_rate, seed=seed)
        for split in ("train", "val")
    }
    loaders = {}
    for split, dataset in datasets.items():
        generator = torch.Generator().manual_seed(seed)
        if split == "train" and use_weighted_sampler and "sample_weight" in dataset.metadata.columns:
            weights = torch.as_tensor(
                dataset.metadata["sample_weight"].to_numpy(dtype=np.float64),
                dtype=torch.double,
            )
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
                generator=generator,
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                generator=generator,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size if split == "train" else 1,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                generator=generator,
            )
    return loaders
