"""Regressions for the full-volume baseline contract and selection protocol."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets import MAMA_MIA_Dataset, build_loaders
from src.datasets.mama_mia import LoadNPZd
from src.engine import LinearWarmupCosineAnnealingLR, Trainer
from src.engine.inference import predict_logits
from src.evaluation.metrics import segmentation_metrics
from src.losses import DiceBCELoss
from src.models import build_model, resolve_in_channels, validate_patch_size
from src.preprocessing.full_volume import CONTRACT_VERSION, preprocess_case, save_case, shared_zscore_channels
from scripts.run_pipeline import (data_signature, preprocessing_signature, resolve_ip,
                                  save_json, select_input, preprocess, load_run, train,
                                  evaluate, file_digest)
from src.utils.config import resolve_device

torch.set_num_threads(2)


@pytest.fixture
def cases(tmp_path):
    folder = tmp_path / "cases"
    rows = []
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for index in range(2):
            case_id = f"{split}_{index}"
            pre = rng.normal(size=(19, 23, 31)).astype(np.float32)
            post = pre + rng.normal(size=pre.shape).astype(np.float32)
            image, _ = shared_zscore_channels([pre, post])
            mask = np.zeros((1, *pre.shape), dtype=np.uint8)
            mask[:, 8:11, 10:13, 14:17] = 1
            save_case(folder / f"{case_id}.npz", {
                "image": image, "mask": mask, "contract_version": CONTRACT_VERSION,
                "spacing": np.ones(3), "origin": np.zeros(3), "direction": np.eye(3).ravel(),
            })
            rows.append({"patient_id": case_id, "split": split, "dataset": "synthetic"})
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return folder, csv


def test_shared_stats_include_later_phases_and_preserve_signed_subtraction():
    pre = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    post = np.flip(pre, axis=0).copy()
    late = np.full_like(pre, 100)
    image, stats = shared_zscore_channels([pre, post, late])
    all_values = np.stack([pre, post, late])
    assert stats["mean"] == pytest.approx(all_values.mean())
    assert stats["std"] == pytest.approx(all_values.std())
    np.testing.assert_allclose(image[0], (post - stats["mean"]) / stats["std"], atol=1e-6)
    np.testing.assert_allclose(image[1], (post - pre) / stats["std"], atol=1e-6)
    assert image[1].min() < 0 < image[1].max()
    assert not np.allclose(image, shared_zscore_channels([pre, post])[0])


def test_constant_phases_are_finite():
    image, _ = shared_zscore_channels([np.ones((2, 3, 4))] * 2)
    assert np.isfinite(image).all() and not image.any()


@pytest.mark.parametrize("phases", [[np.ones((2, 2, 2))],
                                   [np.ones((2, 2, 2)), np.ones((2, 3, 2))],
                                   [np.ones((2, 2, 2)), np.full((2, 2, 2), np.nan)]])
def test_invalid_phases_rejected(phases):
    with pytest.raises(ValueError):
        shared_zscore_channels(phases)


@pytest.mark.parametrize("mode,indices", [("IP1", [0]), ("IP2", [1]), ("IP3", [0, 1])])
def test_inputs_and_no_validation_or_test_crop(cases, mode, indices):
    folder, csv = cases
    for split in ("val", "test"):
        dataset = MAMA_MIA_Dataset(csv, folder, split, mode, patch_size=(8, 8, 8))
        item = dataset[0]
        assert item["image"].shape == (len(indices), 19, 23, 31)
        assert torch.equal(item["image"], dataset[0]["image"])
        with np.load(folder / f"{item['patient_id']}.npz") as raw:
            np.testing.assert_array_equal(item["image"].numpy(), raw["image"][indices])


def test_reject_legacy_npz(cases):
    folder, _ = cases
    path = folder / "legacy.npz"
    np.savez_compressed(path, image=np.zeros((3, 8, 8, 8)), mask=np.zeros((1, 8, 8, 8)))
    with pytest.raises(ValueError, match="regenerate"):
        LoadNPZd("IP3")({"npz_path": str(path)})


def test_inference_does_not_load_mask(cases):
    folder, _ = cases
    path = folder / "unlabelled.npz"
    np.savez_compressed(path, image=np.zeros((2, 8, 8, 8)), contract_version=CONTRACT_VERSION)
    assert "mask" not in LoadNPZd("IP3", require_mask=False)({"npz_path": str(path)})


def test_training_positive_patches_and_channel_mask_alignment(cases):
    folder, csv = cases
    for path in folder.glob("*.npz"):
        with np.load(path) as raw:
            data = {key: raw[key] for key in raw.files}
        data["image"] = np.repeat(data["mask"], 2, axis=0).astype(np.float32)
        save_case(path, data)
    dataset = MAMA_MIA_Dataset(
        csv, folder, "train", patch_size=(8, 8, 8), augment=True,
        positive_patch_probability=1.0, patches_per_case=3,
    )
    items = dataset[0]
    assert len(items) == 3
    for item in items:
        assert item["image"].shape == (2, 8, 8, 8)
        assert item["mask"].any()
        assert torch.equal(item["image"][0] > 0.5, item["mask"][0].bool())


def test_duplicate_patient_rejected(cases):
    folder, csv = cases
    frame = pd.read_csv(csv)
    frame.loc[2, "patient_id"] = frame.loc[0, "patient_id"]
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="unique"):
        MAMA_MIA_Dataset(csv, folder)


def test_optional_clinical_columns_do_not_affect_batches(cases):
    folder, csv = cases
    frame = pd.read_csv(csv)
    frame["manufacturer"] = [np.nan, "SIEMENS", "GE", np.nan, "PHILIPS", np.nan]
    frame.to_csv(csv, index=False)
    loaders = build_loaders(csv, folder, batch_size=2, patch_size=(8, 8, 8))
    batch = next(iter(loaders["train"]))
    assert batch["image"].shape[1:] == (2, 8, 8, 8)
    assert "manufacturer" not in batch


def test_sliding_window_covers_odd_full_volume_and_small_volume():
    model = torch.nn.Conv3d(2, 1, kernel_size=1)
    for shape in ((19, 23, 31), (7, 9, 13)):
        image = torch.randn(1, 2, *shape)
        with torch.no_grad():
            actual = predict_logits(model, image, "cpu", (16, 16, 16), 2, 0.5)
            np.testing.assert_allclose(actual, model(image), atol=1e-5)
        assert actual.shape == (1, 1, *shape)


def test_dice_hd95_empty_and_physical_spacing():
    truth = np.zeros((7, 8, 9), dtype=bool)
    assert segmentation_metrics(truth, truth) == {"dice": 1.0, "hd95_mm": 0.0}
    truth[2, 3, 4] = True
    prediction = np.roll(truth, 1, axis=0)
    assert segmentation_metrics(prediction, truth, (2, 1, 1)) == {"dice": 0.0, "hd95_mm": 2.0}
    assert np.isinf(segmentation_metrics(np.zeros_like(truth), truth)["hd95_mm"])


def test_model_inputs_and_patch_constraints():
    assert resolve_in_channels("IP3") == 2
    assert resolve_in_channels("IP2") == 1
    with pytest.raises(ValueError):
        resolve_in_channels("IP3", 1)
    with pytest.raises(ValueError):
        validate_patch_size("M3", (32, 32, 32))
    with pytest.raises(ValueError):
        validate_patch_size("M1", (33, 64, 64))
    with pytest.raises(ValueError):
        validate_patch_size("M2", (16, 16, 16))


def test_explicit_cuda_request_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")


def test_optimizer_must_reference_the_training_model():
    model = torch.nn.Conv3d(2, 1, 1)
    other = torch.nn.Conv3d(2, 1, 1)
    optimizer = torch.optim.AdamW(other.parameters())
    with pytest.raises(ValueError, match="before constructing"):
        Trainer(model, optimizer, DiceBCELoss(), device="cpu")


def test_m2_train_validation_checkpoint_resume_and_config_guard(cases, tmp_path):
    folder, csv = cases
    loaders = build_loaders(csv, folder, patch_size=(16, 16, 16), batch_size=1,
                            patches_per_case=1)
    model = build_model("M2", ip_mode="IP3", config={"attention_channels": [4, 8, 16]})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=0, max_epochs=1)
    trainer = Trainer(model, optimizer, DiceBCELoss(), scheduler, device="cpu",
                      val_roi_size=(16, 16, 16), sw_batch_size=2)
    directory = tmp_path / "run"
    config = {"model": "M2", "ip": "IP3"}
    history = trainer.fit(loaders["train"], loaders["val"], 1, directory, config)
    assert len(history) == 1
    for path in ("best.pt", "last.pt", "history.csv", "run_summary.json"):
        assert (directory / path).exists()
    repeat = trainer.fit(loaders["train"], loaders["val"], 1, directory, config)
    assert len(repeat) == 1
    with pytest.raises(ValueError, match="configuration"):
        trainer.fit(loaders["train"], loaders["val"], 1, directory, {"ip": "IP1"})


@pytest.mark.parametrize("threshold,expected_dice", [(0.4, 0.0), (0.6, 1.0)])
def test_validation_respects_configured_probability_threshold(threshold, expected_dice):
    model = torch.nn.Conv3d(2, 1, kernel_size=1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = Trainer(model, optimizer, DiceBCELoss(), device="cpu",
                      val_roi_size=(4, 4, 4), prob_threshold=threshold)
    batch = {"image": torch.zeros(1, 2, 4, 4, 4), "mask": torch.zeros(1, 1, 4, 4, 4)}
    assert trainer._epoch([batch], False)["dice"] == expected_dice


def test_m3_forward_backward():
    model = build_model("M3", ip_mode="IP3",
                        config={"swin_feature_size": 12, "swin_use_checkpoint": False},
                        patch_size=(32, 32, 64))
    image = torch.randn(1, 2, 32, 32, 64)
    output = model(image)
    assert output.shape == (1, 1, 32, 32, 64)
    output.mean().backward()


def test_m1_uses_plain_conv_unet_and_backpropagates():
    pytest.importorskip("dynamic_network_architectures")
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    model = build_model("M1", ip_mode="IP3",
                        config={"nnunet_features": [4, 8, 16, 24, 32, 32]})
    assert isinstance(model.network, PlainConvUNet)
    assert len(model.network.encoder.stages) == 6
    output = model(torch.randn(1, 2, 32, 32, 64))
    assert output.shape == (1, 1, 32, 32, 64)
    output.mean().backward()


def test_test_loading_requires_a_completed_matching_training_run(tmp_path):
    cfg = {"paths": {"outputs_dir": str(tmp_path)}}
    directory = tmp_path / "M2_IP3"
    directory.mkdir()
    run = {"model": "M2", "ip": "IP3", "contract": CONTRACT_VERSION,
           "config": {"training": {"epochs": 10}}}
    torch.save({"run_config": run, "val_dice": 0.7}, directory / "best.pt")
    args = Namespace(ip="IP3", model="M2", device="cpu")
    with pytest.raises(ValueError, match="completed training"):
        load_run(args, cfg, verify_dataset=False, require_completed=True)
    save_json(directory / "run_summary.json", {
        "completed": True, "last_epoch": 5, "run_config": run, "best_val_dice": 0.7,
    })
    with pytest.raises(ValueError, match="does not match"):
        load_run(args, cfg, verify_dataset=False, require_completed=True)


def test_validation_selection_and_checkpoint_lock(cases, tmp_path):
    folder, csv = cases
    cfg = {
        "paths": {"splits_metadata_csv": str(csv), "processed_npz_dir": str(folder),
                  "raw_images_dir": "images", "raw_expert_seg_dir": "masks",
                  "outputs_dir": str(tmp_path / "runs")},
        "preprocessing": {"spacing": [1, 1, 1]}, "training": {"epochs": 2},
    }
    save_json(folder / "manifest.json", {
        "signature": preprocessing_signature(cfg), "errors": {}, "complete": True,
        "processed": pd.read_csv(csv).patient_id.tolist(),
        "case_sha256": {path.stem: file_digest(path) for path in folder.glob("*.npz")},
    })
    signature = data_signature(cfg)
    for ip, score in (("IP1", 0.5), ("IP2", 0.6), ("IP3", 0.7)):
        run = {"model": "M1", "ip": ip, "contract": CONTRACT_VERSION,
               "data_signature": signature, "config": cfg}
        directory = Path(cfg["paths"]["outputs_dir"]) / f"M1_{ip}"
        directory.mkdir(parents=True)
        torch.save({"val_dice": score, "run_config": run}, directory / "best.pt")
        save_json(directory / "run_summary.json", {
            "completed": True, "last_epoch": 2, "best_val_dice": score, "run_config": run,
        })
    select_input(Namespace(), cfg)
    assert resolve_ip("selected", cfg) == "IP3"
    (Path(cfg["paths"]["outputs_dir"]) / "M1_IP3" / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        resolve_ip("selected", cfg)


def test_training_and_evaluation_entry_points(cases, tmp_path, monkeypatch):
    folder, csv = cases
    cfg = {
        "project": {"seed": 42, "num_workers": 0},
        "paths": {"splits_metadata_csv": str(csv), "processed_npz_dir": str(folder),
                  "raw_images_dir": "images", "raw_expert_seg_dir": "expert",
                  "outputs_dir": str(tmp_path / "runs")},
        "preprocessing": {"spacing": [1, 1, 1]},
        "model": {"attention_channels": [4, 8, 16]},
        "training": {"patch_size": [16, 16, 16], "batch_size": 1, "cache_rate": 0,
                     "positive_patch_probability": 0.5, "patches_per_case": 1,
                     "epochs": 1, "warmup_epochs": 0, "lr": 1e-3, "weight_decay": 1e-5,
                     "use_amp": False, "sw_batch_size": 2, "sw_overlap": 0.5},
        "inference": {"threshold": 0.5},
    }
    save_json(folder / "manifest.json", {
        "signature": preprocessing_signature(cfg), "errors": {}, "complete": True,
        "processed": pd.read_csv(csv).patient_id.tolist(),
        "case_sha256": {path.stem: file_digest(path) for path in folder.glob("*.npz")},
    })
    selected_checkpoint = tmp_path / "runs" / "M1_IP3" / "best.pt"
    selected_checkpoint.parent.mkdir(parents=True)
    selected_checkpoint.write_bytes(b"selection fixture")
    save_json(tmp_path / "runs" / "selected_input.json", {
        "ip": "IP3", "data_signature": data_signature(cfg),
        "checkpoint_sha256": file_digest(selected_checkpoint),
    })
    args = Namespace(model="M2", ip="selected", device="cpu", split="test")
    train(args, cfg)
    model, saved_cfg, ip, directory, device, _ = load_run(args, cfg, require_completed=True)
    assert saved_cfg == cfg and ip == "IP3" and device.type == "cpu"
    exported = []

    def record_export(array, geometry, path):
        assert array.shape == (19, 23, 31)
        assert set(geometry["spacing"]) == {1.0}
        exported.append(path.name)

    # Physical NIfTI serialization has its own SimpleITK integration test.
    monkeypatch.setattr("scripts.run_pipeline.write_prediction", record_export)
    evaluate(args, cfg)
    assert len(exported) == 4
    results = pd.read_csv(directory / "test" / "metrics.csv")
    assert set(results.patient_id) == {"test_0", "test_1"}
    assert results.dice.between(0, 1).all()
    assert (directory / "test" / "evaluation_config.json").exists()


def test_incomplete_preprocessing_cannot_train(cases):
    folder, csv = cases
    cfg = {"paths": {"splits_metadata_csv": str(csv), "processed_npz_dir": str(folder),
                     "raw_images_dir": "images", "raw_expert_seg_dir": "masks"},
           "preprocessing": {"spacing": [1, 1, 1]}}
    save_json(folder / "manifest.json", {
        "signature": preprocessing_signature(cfg), "processed": [], "errors": {},
        "complete": False,
    })
    with pytest.raises(ValueError, match="incomplete"):
        data_signature(cfg)


def test_preprocess_discovers_case_names_reuses_and_repairs_corrupt_cases(tmp_path, monkeypatch):
    raw = tmp_path / "images" / "duke_001"
    raw.mkdir(parents=True)
    masks = tmp_path / "expert"
    masks.mkdir()
    for index in range(3):
        (raw / f"DUKE_001_{index:04d}.nii.gz").write_bytes(b"fixture")
    (masks / "duke_001.nii.gz").write_bytes(b"fixture")
    csv = tmp_path / "metadata.csv"
    pd.DataFrame([{"patient_id": "DUKE_001", "split": "train"}]).to_csv(csv, index=False)
    cfg = {
        "paths": {"splits_metadata_csv": str(csv), "raw_images_dir": str(raw.parent),
                  "raw_expert_seg_dir": str(masks), "processed_npz_dir": str(tmp_path / "processed")},
        "preprocessing": {"spacing": [1, 1, 1], "zscore_eps": 1e-8,
                          "output_dtype": "float32"},
    }
    calls = []

    def fake_preprocess(paths, expert, *args):
        assert len(paths) == 3
        assert expert.name == "duke_001.nii.gz"
        calls.append(paths)
        return {"image": np.zeros((2, 4, 4, 4)), "mask": np.zeros((1, 4, 4, 4)),
                "contract_version": CONTRACT_VERSION}

    monkeypatch.setattr("scripts.run_pipeline.preprocess_case", fake_preprocess)
    args = Namespace(rebuild_splits=False, overwrite=False)
    preprocess(args, cfg)
    assert len(calls) == 1
    preprocess(args, cfg)
    assert len(calls) == 1
    (tmp_path / "processed" / "DUKE_001.npz").write_bytes(b"interrupted-file")
    preprocess(args, cfg)
    assert len(calls) == 2
    previous_signature = data_signature(cfg)
    (raw / "DUKE_001_0002.nii.gz").write_bytes(b"changed source phase")
    preprocess(args, cfg)
    assert len(calls) == 3
    assert data_signature(cfg) != previous_signature


def test_preprocessing_geometry_and_mask_independence(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    phase_paths = []
    for index in range(3):
        volume = sitk.GetImageFromArray(
            (np.arange(6 * 7 * 8).reshape(6, 7, 8) + index * 10).astype(np.float32))
        volume.SetSpacing((1.5, 1.2, 2.0))
        volume.SetOrigin((10.0, -5.0, 2.0))
        path = tmp_path / f"case_{index:04d}.nii.gz"
        sitk.WriteImage(volume, str(path))
        phase_paths.append(path)
    first = preprocess_case(phase_paths)
    mask = sitk.GetImageFromArray(np.ones((6, 7, 8), dtype=np.uint8))
    mask.CopyInformation(volume)
    mask_path = tmp_path / "mask.nii.gz"
    sitk.WriteImage(mask, str(mask_path))
    second = preprocess_case(phase_paths, mask_path)
    np.testing.assert_array_equal(first["image"], second["image"])
    assert second["image"].shape[1:] == second["mask"].shape[1:]
    assert tuple(second["spacing"]) == (1.0, 1.0, 1.0)
    assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        tuple(second["direction"])) == "RAS"
    assert set(np.unique(second["mask"])) <= {0, 1}
