"""Generate tiny, explicitly synthetic processed volumes for TV4 week 1.

No download, SimpleITK or steps 1-3 required. Default dimensions are reduced
for CPU checks. Use --shape 96 96 96 to match the production spatial contract.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBTYPES = ("Luminal", "Triple Negative", "HER2-enriched", "HER2-pure")


def _zscore(volume):
    return ((volume - volume.mean()) / max(float(volume.std()), 1e-6)).astype(np.float16)


def generate_synthetic_tv4(output_dir=None, shape=(32, 32, 64), seed=42, cases_per_subtype=1):
    """Create independent train/val/test cases with all four subtypes per split."""
    if len(shape) != 3 or min(shape) < 8 or cases_per_subtype < 1:
        raise ValueError("Require 3 dimensions >=8 and cases_per_subtype >=1")
    output_dir = Path(output_dir or PROJECT_ROOT / "data/synthetic_tv4")
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    coords = np.indices(shape).astype(np.float32)
    rows = []
    for split in ("train", "val", "test"):
        for subtype_id, subtype_name in enumerate(SUBTYPES):
            for index in range(cases_per_subtype):
                patient_id = f"TV4_FAKE_{split}_{subtype_id}_{index:02d}"
                center = np.asarray(shape) * (0.5 + rng.uniform(-0.08, 0.08, size=3))
                radii = np.asarray(shape) * rng.uniform(0.16, 0.23, size=3)
                distance = sum(((coords[i] - center[i]) / radii[i]) ** 2 for i in range(3))
                mask = distance <= 1.0
                pre = 80 + 6 * ndimage.gaussian_filter(rng.normal(size=shape), sigma=1)
                post = pre + rng.normal(0, 0.8, shape) + (25 + 5 * subtype_id) * mask
                # Simulated non-enhancing centre: included in the expert mask.
                post[distance < 0.15] = pre[distance < 0.15]
                sub = np.maximum(post - pre, 0)
                pe_proxy = sub / (pre + 1e-6)
                # Channel 3 only mimics contrast, it does NOT implement TV2's CLAHE.
                image = np.stack([_zscore(post), _zscore(sub), _zscore(pe_proxy ** 0.7)])
                auto = ndimage.binary_erosion(mask, iterations=1)
                auto[1:3, 1:3, 1:3] = True
                np.savez_compressed(npz_dir / f"{patient_id}.npz", image=image,
                                    mask=mask[None].astype(np.uint8),
                                    auto_mask=auto[None].astype(np.uint8),
                                    subtype_id=np.int64(subtype_id))
                rows.append({"patient_id": patient_id, "dataset": "SYNTHETIC_TV4_ONLY",
                             "tumor_subtype": subtype_name, "subtype_id": subtype_id,
                             "split": split, "sample_weight": 1.0})
    metadata_csv = output_dir / "splits_metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    (output_dir / "SYNTHETIC_ONLY.json").write_text(json.dumps({
        "synthetic": True, "purpose": "TV4 week 1 code smoke test, not a benchmark",
        "seed": seed, "shape": list(shape), "cases": len(rows),
        "production_shape": list(shape) == [96, 96, 96],
        "channel_3": "contrast proxy, NOT real PE+CLAHE",
    }, indent=2), encoding="utf-8")
    return npz_dir, metadata_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/synthetic_tv4")
    parser.add_argument("--shape", type=int, nargs=3, default=(32, 32, 64))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases-per-subtype", type=int, default=1)
    args = parser.parse_args()
    paths = generate_synthetic_tv4(args.output_dir, tuple(args.shape), args.seed, args.cases_per_subtype)
    print("SYNTHETIC ONLY:", *paths, sep="\n")
