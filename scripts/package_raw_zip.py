"""
Script to package data/raw/ into mama_mia_raw.zip with relative paths starting at data/raw/
so when extracted at project root, it restores data/raw/ exactly.

Uses ZIP_STORED (no recompression) for maximum speed (~2-3 mins on SSD)
since all NIfTI files are already compressed (.nii.gz).
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_ZIP = PROJECT_ROOT / "mama_mia_raw.zip"

def package_raw() -> None:
    print(f"Packaging {RAW_DIR} into {OUTPUT_ZIP}...")
    t0 = time.time()

    # Collect all files to archive
    all_files = []
    total_bytes = 0
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(PROJECT_ROOT)
            fsize = full_path.stat().st_size
            all_files.append((full_path, str(rel_path).replace("\\", "/"), fsize))
            total_bytes += fsize

    print(f"Total files to archive: {len(all_files):,}")
    print(f"Total uncompressed size: {total_bytes / (1024**3):.2f} GB")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Archiving") as pbar:
            for full_path, arcname, fsize in all_files:
                zf.write(full_path, arcname=arcname)
                pbar.update(fsize)

    elapsed = time.time() - t0
    zip_size = OUTPUT_ZIP.stat().st_size / (1024**3)
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f"Final archive size: {zip_size:.2f} GB at {OUTPUT_ZIP}")

if __name__ == "__main__":
    package_raw()