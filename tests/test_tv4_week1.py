"""Regression checks for TV4's week 1 data/loss/training contracts.

The notebook separately trains all three actual TV3 architectures for 2 epochs.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.generate_synthetic_tv4 import generate_synthetic_tv4
from src.datasets import MAMA_MIA_Dataset, build_loaders
from src.engine import LinearWarmupCosineAnnealingLR, Trainer
from src.losses import DiceBCELoss


@pytest.fixture
def fake_data(tmp_path):
    return generate_synthetic_tv4(tmp_path / "fake", shape=(16, 16, 32), seed=7)


def test_synthetic_contract_and_reproducibility(fake_data, tmp_path):
    folder, csv = fake_data
    frame = pd.read_csv(csv)
    assert frame.groupby("split").subtype_id.nunique().to_dict() == {"train": 4, "val": 4, "test": 4}
    assert frame.patient_id.is_unique
    other, _ = generate_synthetic_tv4(tmp_path / "repeat", shape=(16, 16, 32), seed=7)
    for path in folder.glob("*.npz"):
        with np.load(path) as a, np.load(other / path.name) as b:
            assert set(a.files) == {"image", "mask", "auto_mask", "subtype_id"}
            assert a["image"].dtype == np.float16
            assert a["image"].shape == (3, 16, 16, 32)
            assert a["mask"].dtype == np.uint8
            assert a["subtype_id"].dtype == np.int64 and a["subtype_id"].shape == ()
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("mode,channels", [("IP1_post1", [0]), ("IP2_sub", [1]), ("IP3_multi", [0, 1, 2])])
def test_ip_channels_and_validation_not_cropped(fake_data, mode, channels):
    folder, csv = fake_data
    dataset = MAMA_MIA_Dataset(csv, folder, "val", mode, patch_size=(8, 8, 8))
    a, b = dataset[0], dataset[0]
    assert torch.equal(a["image"], b["image"])
    assert a["image"].shape[1:] == (16, 16, 32)
    with np.load(folder / f"{a['patient_id']}.npz") as raw:
        np.testing.assert_array_equal(a["image"].numpy(), raw["image"][channels].astype(np.float32))


def test_augmentation_alignment_and_cache_isolation(fake_data):
    folder, csv = fake_data
    for path in folder.glob("*.npz"):
        with np.load(path) as raw:
            data = {key: raw[key].copy() for key in raw.files}
        data["image"] = np.repeat(data["mask"], 3, axis=0).astype(np.float16)
        np.savez_compressed(path, **data)
    dataset = MAMA_MIA_Dataset(csv, folder, "train", patch_size=(16, 16, 32), seed=5)
    snapshots = []
    for _ in range(12):
        item = dataset[0]
        assert torch.equal(item["image"][0:1] > 0.5, item["mask"].bool())
        assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
        snapshots.append(item["mask"].clone())
    assert any(not torch.equal(snapshots[0], value) for value in snapshots[1:])
    # Mutating a returned item must not poison the cached deterministic volume.
    item["image"].fill_(999)
    assert dataset[0]["image"].max() < 2


def test_reject_duplicate_patient_across_splits(fake_data):
    folder, csv = fake_data
    frame = pd.read_csv(csv)
    frame.loc[4, "patient_id"] = frame.loc[0, "patient_id"]
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="unique"):
        MAMA_MIA_Dataset(csv, folder)


def test_reject_bad_npz_and_unknown_mode(fake_data):
    folder, csv = fake_data
    with pytest.raises(ValueError, match="ip_mode"):
        MAMA_MIA_Dataset(csv, folder, ip_mode="unknown")
    path = next(folder.glob("*train*.npz"))
    with np.load(path) as raw:
        data = {key: raw[key].copy() for key in raw.files}
    data["mask"][0, 0, 0, 0] = 2
    np.savez_compressed(path, **data)
    # MONAI wraps transform failures in RuntimeError.
    with pytest.raises(RuntimeError):
        MAMA_MIA_Dataset(csv, folder, "train")


def test_deep_supervision_perfect_bad_empty_and_aux_gradients():
    loss = DiceBCELoss()
    target = torch.ones(2, 1, 8, 8, 8)
    perfect = [torch.full((2, 1, n, n, n), 20.0) for n in (8, 4, 2)]
    bad = [-value for value in perfect]
    assert loss(perfect, target) < 1e-6
    assert loss(bad, target) > 30
    empty = torch.zeros_like(target)
    assert torch.isfinite(loss(bad, empty))
    heads = [torch.randn(2, 1, n, n, n, requires_grad=True) for n in (8, 4, 2)]
    value = loss(heads, empty)
    value.backward()
    assert all(h.grad is not None and torch.isfinite(h.grad).all() and h.grad.abs().sum() > 0 for h in heads)
    with pytest.raises(ValueError, match="heads"):
        loss(heads[:2], target)


def test_scheduler_warmup_cosine_and_state():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.003)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=3, max_epochs=8)
    lrs = []
    for _ in range(8):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[:3] == pytest.approx([0.001, 0.002, 0.003])
    assert lrs[-1] == pytest.approx(0)
    assert all(a >= b for a, b in zip(lrs[3:], lrs[4:]))
    saved = scheduler.state_dict()
    restored = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=0, max_epochs=10)
    restored.load_state_dict(saved)
    assert restored.get_lr() == scheduler.get_lr()
    with pytest.raises(ValueError):
        LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=8, max_epochs=8)


class EngineProbe(nn.Module):
    """Small probe isolates checkpoint/optimizer behavior from architecture cost."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 1, 1)

    def forward(self, image, subtype=None):
        output = self.conv(image)
        return [output, F.avg_pool3d(output, 2), F.avg_pool3d(output, 4)] if self.training else output


def test_cpu_trainer_updates_and_checkpoint_roundtrip(fake_data, tmp_path):
    torch.set_num_threads(2)
    folder, csv = fake_data
    loaders = build_loaders(csv, folder, patch_size=(16, 16, 32))
    model = EngineProbe()
    before = model.conv.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, 1, 2)
    trainer = Trainer(model, optimizer, DiceBCELoss(), scheduler, device="cpu", use_amp=True)
    assert not trainer.amp_enabled
    history = trainer.fit(loaders["train"], loaders["val"], 2, tmp_path / "run")
    assert len(history) == 2 and not torch.equal(before, model.conv.weight)
    assert np.isfinite(history.to_numpy()).all()
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert checkpoint["val_dice"] == history.val_dice.max()
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["scheduler"]["last_epoch"] >= 1
    a, b = EngineProbe().eval(), EngineProbe().eval()
    a.load_state_dict(checkpoint["model"])
    b.load_state_dict(checkpoint["model"])
    x = next(iter(loaders["val"]))["image"]
    with torch.no_grad():
        assert torch.equal(a(x), b(x))
    assert (tmp_path / "run/history.csv").is_file()
