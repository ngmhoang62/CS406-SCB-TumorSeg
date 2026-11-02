import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_step3 import build_dataset, pilot, qc_dataset, read_metadata, select_pilot


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
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "dataset", "tumor_subtype", "subtype_id", "split"])
        writer.writeheader()
        writer.writerow({"patient_id": "CASE_1", "dataset": "DUKE", "tumor_subtype": "Luminal", "subtype_id": 0, "split": "train"})
    shape = (96, 96, 96)
    mask = np.zeros((1,) + shape, dtype=np.uint8)
    mask[0, 48, 48, 48] = 1
    auto = np.zeros_like(mask)
    np.savez_compressed(tv1 / "CASE_1.npz", image=np.zeros((3,) + shape, dtype=np.float16),
                        mask=mask, auto_mask=auto, subtype_id=np.int64(0))
    np.savez_compressed(voi / "CASE_1.npz", pre_voi=np.ones(shape, dtype=np.float32),
                        post1_voi=np.full(shape, 2, dtype=np.float32))
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
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "run_step3.py"),
               "--data-root", str(root), "--mode", "validate", "--skip-pilot"]
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
