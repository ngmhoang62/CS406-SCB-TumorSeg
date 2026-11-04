"""Recreate TV1 Step 2 VOIs and verify expert masks against the saved TV1 data."""

import argparse
import csv
from pathlib import Path
import sys
import traceback

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tv2_step3_qc import load_voi, read_metadata
from src.preprocessing.step2_spatial_voi import standardize_and_extract_voi_fast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--case", help="Chỉ xử lý một ca để kiểm tra")
    parser.add_argument("--metadata", type=Path, help="Danh sách ca cần xử lý")
    parser.add_argument("--report-dir", type=Path, help="Thư mục báo cáo")
    args = parser.parse_args()

    root = args.data_root
    raw = root / "raw"
    out = root / "processed" / "voi_step2"
    report_dir = args.report_dir or root / "tv2_step3"
    report = report_dir / ("voi_export_pilot.csv" if args.case else "voi_export.csv")
    out.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(args.metadata or root / "splits" / "splits_metadata.csv")
    if args.case:
        rows = [row for row in rows if row["patient_id"] == args.case]
        if not rows:
            raise SystemExit(f"Không tìm thấy ca {args.case} trong metadata")
    else:
        print("Đang kiểm tra ảnh gốc cho các ca chưa có VOI...", flush=True)
        missing = []
        for row in rows:
            case = row["patient_id"]
            if (out / f"{case}.npz").is_file():
                continue
            images = raw / "images" / case
            paths = [
                images / f"{case}_0000.nii.gz",
                images / f"{case}_0001.nii.gz",
                raw / "segmentations" / "expert" / f"{case}.nii.gz",
            ]
            missing.extend(str(path) for path in paths if not path.is_file())
        if missing:
            missing_report = report_dir / "missing_voi_inputs.txt"
            missing_report.write_text("\n".join(missing) + "\n", encoding="utf-8")
            raise SystemExit(f"Thiếu {len(missing)} file đầu vào. Xem {missing_report}; ví dụ: {missing[0]}")

    results = []
    for i, row in enumerate(rows, 1):
        case = row["patient_id"]
        target = out / f"{case}.npz"
        tmp = out / f"{case}.tmp.npz"
        try:
            with np.load(root / "processed" / "npz_96x96x96" / f"{case}.npz") as old:
                mask = old["mask"].copy()
            if target.exists():
                load_voi(target, mask.shape[1:])
                status = "skipped"
            else:
                images = raw / "images" / case
                pre, post, expert, _, _ = standardize_and_extract_voi_fast(
                    pre_sitk=sitk.ReadImage(str(images / f"{case}_0000.nii.gz"), sitk.sitkFloat32),
                    post1_sitk=sitk.ReadImage(str(images / f"{case}_0001.nii.gz"), sitk.sitkFloat32),
                    expert_mask_sitk=sitk.ReadImage(
                        str(raw / "segmentations" / "expert" / f"{case}.nii.gz"),
                        sitk.sitkUInt8,
                    ),
                    auto_mask_sitk=None,
                    target_orientation="RAS",
                    target_spacing=(1.0, 1.0, 1.0),
                    voi_size=(96, 96, 96),
                    margin_ratio=0.25,
                    pad_mode="constant",
                )
                if not np.array_equal(expert[np.newaxis].astype(np.uint8), mask):
                    raise ValueError("Recreated expert mask differs from TV1; cannot establish alignment")
                if (
                    pre.shape != mask.shape[1:]
                    or post.shape != pre.shape
                    or not (np.isfinite(pre).all() and np.isfinite(post).all())
                ):
                    raise ValueError("Invalid VOI shape or values")
                np.savez_compressed(tmp, pre_voi=pre.astype(np.float32), post1_voi=post.astype(np.float32))
                tmp.replace(target)
                status = "done"
            results.append((case, status, ""))
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            try:
                tmp.unlink(missing_ok=True)
            except OSError as cleanup_error:
                print(f"Không xóa được file tạm: {cleanup_error}", flush=True)
            results.append((case, "failed", detail))
        if i % 10 == 0 or i == len(rows) or results[-1][1] == "failed":
            with report.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["patient_id", "status", "error"])
                writer.writerows(results)
            print(f'VOI {i}/{len(rows)}; lỗi: {sum(r[1] == "failed" for r in results)}', flush=True)
        if results[-1][1] == "failed":
            break
    if any(r[1] == "failed" for r in results):
        raise SystemExit(f"Một số ca lỗi; xem {report} và chạy lại sau khi khắc phục.")


if __name__ == "__main__":
    main()
