"""
Script: Overwrite C3 in all 1480 NPZ files with Core-Enhanced Hybrid Subtraction (CEHS).
- Retains C1 (Post-1), C2 (Subtraction), mask, auto_mask, subtype_id exactly as-is.
- Computes C3_CEHS using tuned parameters: sigma=1.0, thresh=0.05, alpha=3.5, power=1.4.
- Runs parallel multiprocessing across CPU cores with a clean progress bar.
"""

from __future__ import annotations

import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def compute_cehs_c3(c1: np.ndarray, c2: np.ndarray, sigma: float = 1.0, thresh: float = 0.05) -> np.ndarray:
    """Compute Core-Enhanced Hybrid Subtraction (CEHS) Channel 3."""
    c1_f = c1.astype(np.float32)
    c2_f = c2.astype(np.float32)

    c2_min = float(c2_f.min())
    e = np.maximum(0.0, c2_f - c2_min)
    e_norm = e / (e.max() + 1e-6)

    c1_pos = np.maximum(0.0, c1_f - float(c1_f.min()))
    c1_pos = c1_pos / (c1_pos.max() + 1e-6)

    # 3D Gaussian Envelope for smooth core support
    env = gaussian_filter(e_norm, sigma=sigma)
    env = env / (env.max() + 1e-6)

    # Tightly gated threshold to eliminate spillover into background and adjacent vessels
    gate = np.clip((env - thresh) / (1.0 - thresh) * 2.5, 0.0, 1.0)
    core_boost = np.power(c1_pos, 1.4) * gate

    c3 = c2_f + 3.5 * core_boost
    c3_norm = (c3 - c3.mean()) / (c3.std() + 1e-6)
    return c3_norm.astype(np.float16)


def process_single_npz(npz_path_str: str) -> Tuple[str, bool, str]:
    """Load NPZ, overwrite channel 2 of image with CEHS, and save back."""
    npz_path = Path(npz_path_str)
    try:
        with np.load(npz_path) as z:
            image = z["image"]          # (3, 96, 96, 96) float16
            mask = z["mask"]            # (1, 96, 96, 96) uint8
            auto_mask = z["auto_mask"]  # (1, 96, 96, 96) uint8
            subtype_id = z["subtype_id"]

        c1 = image[0]
        c2 = image[1]
        c3_cehs = compute_cehs_c3(c1, c2, sigma=1.0, thresh=0.05)

        # Update channel 2 (0-indexed)
        new_image = np.stack([c1, c2, c3_cehs], axis=0)

        # Atomic write to temporary file first then replace
        temp_path = npz_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temp_path,
            image=new_image,
            mask=mask,
            auto_mask=auto_mask,
            subtype_id=subtype_id,
        )
        temp_path.replace(npz_path)
        return (npz_path.stem, True, "")
    except Exception as exc:
        return (npz_path.stem, False, str(exc))


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    npz_dir = workspace / "data" / "processed" / "npz_96x96x96"
    if not npz_dir.is_dir():
        print(f"Error: Directory not found: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    npz_files = sorted(npz_dir.glob("*.npz"))
    total_files = len(npz_files)
    print(f"Found {total_files} .npz files in {npz_dir}")
    print("Beginning overwrite of Channel 3 with CEHS (sigma=1.0, thresh=0.05)...")

    num_workers = max(1, min(8, os.cpu_count() or 4))
    print(f"Using {num_workers} parallel CPU workers")

    t0 = time.time()
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_npz, str(p)) for p in npz_files]
        for i, fut in enumerate(futures, 1):
            stem, ok, err = fut.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"[FAIL] {stem}: {err}", file=sys.stderr)

            if i % 100 == 0 or i == total_files:
                elapsed = time.time() - t0
                speed = i / max(1e-5, elapsed)
                eta = (total_files - i) / max(1e-5, speed)
                print(
                    f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%) | "
                    f"Speed: {speed:.1f} files/s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s",
                    flush=True,
                )

    total_time = time.time() - t0
    print("=" * 60)
    print(f"COMPLETED in {total_time:.2f} seconds ({total_time/60:.2f} minutes)!")
    print(f"Success: {success_count}/{total_files} files, Failures: {fail_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
