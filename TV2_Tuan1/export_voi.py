"""Recreate TV1 Step 2 VOIs and check expert masks against the saved TV1 data."""
import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from run_step3 import read_metadata, load_voi
from src.step2_spatial_voi import standardize_and_extract_voi_fast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    raw = root / 'raw'
    out = root / 'processed' / 'voi_step2'
    report = root / 'tv2_step3' / 'voi_export.csv'
    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(root / 'splits' / 'splits_metadata.csv')
    results = []
    for i, row in enumerate(rows, 1):
        case = row['patient_id']
        target = out / f'{case}.npz'
        tmp = out / f'{case}.tmp.npz'
        try:
            with np.load(root / 'processed' / 'npz_96x96x96' / f'{case}.npz') as old:
                mask = old['mask'].copy()
            if target.exists():
                load_voi(target, mask.shape[1:])
                status = 'skipped'
            else:
                images = raw / 'images' / case
                pre, post, expert, _, _ = standardize_and_extract_voi_fast(
                    pre_sitk=sitk.ReadImage(str(images / f'{case}_0000.nii.gz'), sitk.sitkFloat32),
                    post1_sitk=sitk.ReadImage(str(images / f'{case}_0001.nii.gz'), sitk.sitkFloat32),
                    expert_mask_sitk=sitk.ReadImage(str(raw / 'segmentations' / 'expert' / f'{case}.nii.gz'), sitk.sitkUInt8),
                    auto_mask_sitk=None, target_orientation='RAS', target_spacing=(1., 1., 1.),
                    voi_size=(96, 96, 96), margin_ratio=0.25, pad_mode='constant')
                if not np.array_equal(expert[np.newaxis].astype(np.uint8), mask):
                    raise ValueError('Recreated expert mask differs from TV1; cannot establish alignment')
                if pre.shape != mask.shape[1:] or post.shape != pre.shape or not (np.isfinite(pre).all() and np.isfinite(post).all()):
                    raise ValueError('Invalid VOI shape or values')
                np.savez_compressed(tmp, pre_voi=pre.astype(np.float32), post1_voi=post.astype(np.float32))
                tmp.replace(target)
                status = 'done'
            results.append((case, status, ''))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            results.append((case, 'failed', f'{type(exc).__name__}: {exc}'))
        if i % 10 == 0 or i == len(rows):
            with report.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(['patient_id', 'status', 'error']); writer.writerows(results)
            print(f'VOI {i}/{len(rows)}; lỗi: {sum(r[1] == "failed" for r in results)}', flush=True)
    if any(r[1] == 'failed' for r in results):
        raise SystemExit(f'Một số ca lỗi; xem {report} và chạy lại sau khi khắc phục.')


if __name__ == '__main__':
    main()
