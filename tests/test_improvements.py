"""Regression checks for separate M1 encoder-fusion and boundary experiments."""

from argparse import Namespace
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_pipeline import (
    data_signature, evaluate, file_digest, load_run, preprocessing_signature,
    run_directory, save_json, train,
)
from src.datasets import MAMA_MIA_Dataset
from src.datasets.boundary import boundary_from_mask
from src.engine.inference import predict_logits
from src.losses import BoundaryAuxiliaryLoss, DiceBCELoss
from src.models import build_model
from src.models.nnunet3d import SpatialFusion3D
from src.preprocessing import CONTRACT_VERSION, save_case

torch.set_num_threads(2)
SMALL_MODEL = {"nnunet_features": [4, 8, 16]}
SIX_STAGE_MODEL = {"nnunet_features": [4, 8, 12, 16, 24, 24]}


@pytest.fixture
def improvement_data(tmp_path):
    folder = tmp_path / "cases"
    shape = (19, 21, 23)
    z, y, x = np.indices(shape)
    mask = (((z - 9) ** 2 + (y - 10) ** 2 + (x - 11) ** 2) < 16)[None].astype(np.uint8)
    rows = []
    for split in ("train", "val", "test"):
        case_id = f"case_{split}"
        image = np.random.default_rng(7).normal(size=(2, *shape)).astype(np.float32)
        save_case(folder / f"{case_id}.npz", {
            "contract_version": CONTRACT_VERSION, "image": image, "mask": mask,
            "spacing": np.ones(3), "origin": np.zeros(3), "direction": np.eye(3).ravel(),
        })
        rows.append({"patient_id": case_id, "split": split, "dataset": "synthetic"})
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    cfg = {
        "project": {"seed": 42, "num_workers": 0},
        "paths": {"splits_metadata_csv": str(csv), "processed_npz_dir": str(folder),
                  "raw_images_dir": "images", "raw_expert_seg_dir": "expert",
                  "outputs_dir": str(tmp_path / "runs")},
        "preprocessing": {"spacing": [1, 1, 1]}, "model": SMALL_MODEL,
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
    return folder, csv, cfg


def test_boundary_band_has_inner_and_outer_voxels():
    mask = np.zeros((1, 9, 9, 9), dtype=np.uint8)
    mask[:, 2:7, 2:7, 2:7] = 1
    boundary = boundary_from_mask(mask)
    assert boundary.dtype == np.uint8 and boundary.shape == mask.shape
    assert boundary[0, 4, 4, 4] == 0
    assert boundary[0, 2, 4, 4] == 1
    assert boundary[0, 1, 4, 4] == 1
    assert boundary[0, 0, 4, 4] == 0
    assert not boundary_from_mask(np.zeros_like(mask)).any()
    assert not boundary_from_mask(np.ones_like(mask)).any()


@pytest.mark.parametrize("radius", [0, -1, 1.5, True])
def test_invalid_boundary_radius_rejected(radius):
    with pytest.raises(ValueError):
        boundary_from_mask(np.zeros((1, 4, 4, 4)), radius)


def test_boundary_labels_are_cropped_and_augmented_with_the_image(improvement_data):
    folder, csv, _ = improvement_data
    for path in folder.glob("*.npz"):
        with np.load(path) as raw:
            data = {key: raw[key] for key in raw.files}
        data["image"] = np.concatenate([boundary_from_mask(data["mask"]), data["mask"]]).astype(np.float32)
        save_case(path, data)
    dataset = MAMA_MIA_Dataset(
        csv, folder, "train", "IP3", patch_size=(12, 12, 12),
        augment=True, boundary_aux=True, patches_per_case=2,
    )
    for _ in range(5):
        for item in dataset[0]:
            assert torch.equal(item["image"][0:1] > 0.5, item["boundary"].bool())
            assert torch.equal(item["image"][1:2] > 0.5, item["mask"].bool())
    for split in ("val", "test"):
        item = MAMA_MIA_Dataset(csv, folder, split, "IP3", boundary_aux=True)[0]
        assert "boundary" not in item
        assert item["mask"].shape[1:] == (19, 21, 23)


def test_cropping_a_solid_region_does_not_create_a_boundary(improvement_data):
    folder, csv, _ = improvement_data
    path = folder / "case_train.npz"
    with np.load(path) as raw:
        data = {key: raw[key] for key in raw.files}
    data["mask"].fill(1)
    save_case(path, data)
    dataset = MAMA_MIA_Dataset(csv, folder, "train", "IP3", patch_size=(8, 8, 8),
                               augment=False, boundary_aux=True, patches_per_case=1)
    patch = dataset[0][0]
    assert patch["mask"].all()
    assert not patch["boundary"].any()


def test_spatial_gate_starts_with_equal_modality_weights():
    fusion = SpatialFusion3D(4)
    first, second = torch.randn(1, 4, 5, 6, 7), torch.randn(1, 4, 5, 6, 7)
    torch.testing.assert_close(fusion(first, second), 0.5 * (first + second))


@pytest.mark.parametrize("variant", ["dual-add", "dual-gated"])
def test_six_stage_dual_encoders_both_receive_gradients(variant):
    model = build_model("M1", ip_mode="IP3", config=SIX_STAGE_MODEL, variant=variant)
    first_parameters = {id(p) for p in model.network.encoder.parameters()}
    second_parameters = {id(p) for p in model.subtraction_encoder.parameters()}
    assert first_parameters.isdisjoint(second_parameters)
    assert len(model.network.encoder.stages) == len(model.subtraction_encoder.stages) == 6
    image = torch.randn(1, 2, 32, 32, 64, requires_grad=True)
    output = model(image)
    assert output.shape == (1, 1, 32, 32, 64)
    DiceBCELoss()(output, torch.ones_like(output)).backward()
    for encoder in (model.network.encoder, model.subtraction_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    assert image.grad[:, 0].abs().sum() > 0
    assert image.grad[:, 1].abs().sum() > 0
    if variant == "dual-gated":
        assert len(model.fusions) == 6
        assert any(f.gate.weight.grad.abs().sum() > 0 for f in model.fusions)


@pytest.mark.parametrize("variant", ["dual-add", "dual-gated"])
def test_dual_encoders_require_ip3(variant):
    with pytest.raises(ValueError, match="require IP3"):
        build_model("M1", ip_mode="IP1", config=SMALL_MODEL, variant=variant)


def test_auxiliary_head_uses_final_features_and_runs_only_during_training():
    baseline = build_model("M1", ip_mode="IP3", config=SMALL_MODEL)
    model = build_model("M1", ip_mode="IP3", config=SMALL_MODEL, variant="boundary")
    model.network.load_state_dict(baseline.network.state_dict())
    image = torch.randn(1, 2, 16, 16, 16)
    outputs = model(image)
    assert set(outputs) == {"seg", "boundary"}
    assert outputs["seg"].shape == outputs["boundary"].shape == (1, 1, 16, 16, 16)
    target = torch.zeros_like(outputs["seg"])
    loss = BoundaryAuxiliaryLoss()(outputs, target, torch.ones_like(target))
    loss.backward()
    assert model.boundary_head.weight.grad.abs().sum() > 0
    decoder_weight = next(model.network.decoder.stages[-1].parameters())
    assert decoder_weight.grad.abs().sum() > 0

    def unexpected_boundary_call(module, inputs, output):
        raise AssertionError("Boundary head ran during inference")

    handle = model.boundary_head.register_forward_hook(unexpected_boundary_call)
    model.eval()
    baseline.eval()
    with torch.no_grad():
        actual = model(image)
        torch.testing.assert_close(actual, baseline(image))
        stitched = predict_logits(model, torch.randn(1, 2, 19, 21, 23),
                                   "cpu", (16, 16, 16), 2, 0.5)
    handle.remove()
    assert isinstance(actual, torch.Tensor)
    assert stitched.shape == (1, 1, 19, 21, 23)


def test_boundary_loss_requires_explicit_full_scan_boundary_targets():
    logits = torch.zeros(1, 1, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(logits)
    criterion = BoundaryAuxiliaryLoss()
    with pytest.raises(ValueError, match="full-scan"):
        criterion({"seg": logits, "boundary": logits}, target)
    torch.testing.assert_close(criterion(logits, target), DiceBCELoss()(logits, target))
    zero_weight = BoundaryAuxiliaryLoss(boundary_weight=0)
    torch.testing.assert_close(zero_weight({"seg": logits, "boundary": logits}, target, target),
                               DiceBCELoss()(logits, target))


@pytest.mark.parametrize("variant", ["dual-add", "dual-gated", "boundary"])
def test_improvements_train_resume_reload_and_evaluate(improvement_data, variant, monkeypatch):
    _, _, cfg = improvement_data
    args = Namespace(model="M1", ip="IP3", device="cpu", split="test", variant=variant,
                     boundary_weight=0.3, boundary_radius=1)
    original_signature = data_signature(cfg)
    train(args, cfg)
    directory = run_directory(cfg, "M1", "IP3", variant)
    assert directory.name == f"M1_IP3_{variant}"
    assert not run_directory(cfg, "M1", "IP3").exists()
    checkpoint = torch.load(directory / "best.pt", weights_only=True)
    assert checkpoint["run_config"]["variant"] == variant
    if variant == "boundary":
        assert checkpoint["run_config"]["boundary_weight"] == 0.3
        assert checkpoint["run_config"]["boundary_radius"] == 1
    train(args, cfg)
    history = pd.read_csv(directory / "history.csv")
    assert len(history) == 1
    model, _, _, _, _, _ = load_run(args, cfg, require_completed=True)
    assert not model.training
    with torch.no_grad():
        assert isinstance(model(torch.zeros(1, 2, 16, 16, 16)), torch.Tensor)
    exported = []
    monkeypatch.setattr("scripts.run_pipeline.write_prediction",
                         lambda array, geometry, path: exported.append(path.name))
    evaluate(args, cfg)
    assert len(exported) == 2
    assert (directory / "test" / "metrics_by_cohort.csv").exists()
    lock = json.loads((directory / "test" / "evaluation_config.json").read_text())
    assert lock["variant"] == variant
    assert data_signature(cfg) == original_signature
    checkpoint["run_config"]["variant"] = "baseline"
    torch.save(checkpoint, directory / "best.pt")
    with pytest.raises(ValueError, match="variant differs"):
        load_run(args, cfg)


def test_improvement_variants_are_on_m1():
    with pytest.raises(ValueError, match="implemented on M1"):
        build_model("M2", ip_mode="IP3", variant="boundary")
