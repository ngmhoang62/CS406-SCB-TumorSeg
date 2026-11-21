"""Unit tests for TV2 Week 1: Step 3 3-channel image synthesis and QC workflow."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_tv2_step3_qc import build_dataset, pilot, qc_dataset, read_metadata, select_pilot
import src.preprocessing.step3_image_proc as step3
from src.preprocessing.step3_image_proc import synthesize_three_channels


def test_channel_order_and_subtraction_formula(monkeypatch):
    monkeypatch.setattr(step3, "compute_3d_clahe", lambda volume_zyx, **_: volume_zyx)
    pre = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 10
    post = pre + np.linspace(0, 20, 64, dtype=np.float32).reshape(4, 4, 4)
    image = synthesize_three_channels(
        pre,
        post,
        apply_denoise=False,
        clip_percentiles=(0, 100),
        clahe_kernel_size=2,
        output_dtype="float32",
    )
    expected_c1 = (post - post.mean()) / (post.std() + 1e-8)
    sub = np.maximum(0, post - pre)
    expected_c2 = (sub - sub.mean()) / (sub.std() + 1e-8)
    assert image.shape == (3, 4, 4, 4)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[0], expected_c1, atol=1e-5)
    np.testing.assert_allclose(image[1], expected_c2, atol=1e-5)
    assert np.isfinite(image).all()


def test_constant_and_zero_pre_are_finite():
    pre = np.zeros((4, 4, 4), dtype=np.float32)
    post = np.ones_like(pre)
    image = synthesize_three_channels(
        pre,
        post,
        apply_denoise=False,
        clahe_kernel_size=2,
    )
    assert image.dtype == np.float16
    assert np.isfinite(image).all()
    assert np.count_nonzero(image) == 0


def test_mismatched_voi_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        synthesize_three_channels(
            np.ones((4, 4, 4), dtype=np.float32),
            np.ones((4, 4, 5), dtype=np.float32),
            apply_denoise=False,
        )


def test_actual_clahe_on_3d_volume_when_scipy_is_available():
    pytest.importorskip("scipy.linalg")
    volume = np.linspace(0, 1, 8**3, dtype=np.float32).reshape(8, 8, 8)
    result = step3.compute_3d_clahe(volume, kernel_size=2)
    assert result.shape == volume.shape
    assert np.isfinite(result).all()


def test_recompute_preserves_tv1_masks_and_is_resumable(tmp_path):
    root = tmp_path / "data_real"
    tv1 = root / "processed" / "npz_96x96x96"
    voi = root / "processed" / "voi_step2"
    out = root / "processed" / "npz_tv2_step3"
    report = root / "tv2_step3" / "reports"
    meta = root / "splits" / "splits_metadata.csv"
    for directory in (tv1, voi, meta.parent):
        directory.mkdir(parents=True, exist_ok=True)
    with meta.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["patient_id", "dataset", "tumor_subtype", "subtype_id", "split"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "patient_id": "CASE_1",
                "dataset": "DUKE",
                "tumor_subtype": "Luminal",
                "subtype_id": 0,
                "split": "train",
            }
        )
    shape = (96, 96, 96)
    mask = np.zeros((1,) + shape, dtype=np.uint8)
    mask[0, 48, 48, 48] = 1
    auto = np.zeros_like(mask)
    np.savez_compressed(
        tv1 / "CASE_1.npz",
        image=np.zeros((3,) + shape, dtype=np.float16),
        mask=mask,
        auto_mask=auto,
        subtype_id=np.int64(0),
    )
    np.savez_compressed(
        voi / "CASE_1.npz",
        pre_voi=np.ones(shape, dtype=np.float32),
        post1_voi=np.full(shape, 2, dtype=np.float32),
    )
    rows = read_metadata(meta)
    config = {"apply_denoise": False, "output_dtype": "float16"}
    assert build_dataset(rows, tv1, voi, out, report, config) == {"done": 1}
    assert build_dataset(rows, tv1, voi, out, report, config) == {"skipped": 1}
    summary = qc_dataset(rows, out, report)
    assert summary["valid"] == 1
    assert summary["auto_mask_empty"] == 1
    assert (report / "channel_summary_by_group.csv").is_file()
    with np.load(out / "CASE_1.npz") as data:
        np.testing.assert_array_equal(data["mask"], mask)
        np.testing.assert_array_equal(data["auto_mask"], auto)
    assert json.loads((out / "tv2_build_manifest.json").read_text())["signature"]
    pilot(rows, tv1, voi, report, config, False, 1)
    assert (report / "figures" / "CASE_1.png").is_file()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_tv2_step3_qc.py"),
        "--data-root",
        str(root),
        "--mode",
        "validate",
        "--skip-pilot",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads((report / "provenance.json").read_text())["resolved_mode"] == "validate"


def test_pilot_covers_each_available_dataset():
    rows = [
        {"patient_id": f"{dataset}_{i}", "dataset": dataset, "tumor_subtype": subtype}
        for dataset in ("DUKE", "ISPY1", "ISPY2", "NACT")
        for i, subtype in enumerate(("Luminal", "Triple Negative", "HER2-enriched", "HER2-pure"))
    ]
    selected = select_pilot(rows, 12)
    assert len(selected) == 12
    assert {r["dataset"] for r in selected} == {"DUKE", "ISPY1", "ISPY2", "NACT"}
    assert {r["tumor_subtype"] for r in selected} == {"Luminal", "Triple Negative", "HER2-enriched", "HER2-pure"}
