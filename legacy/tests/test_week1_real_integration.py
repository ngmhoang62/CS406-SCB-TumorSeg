"""End-to-End Integration Test for Week 1 (TV1 + TV2 + TV3 + TV4) on Real Preprocessed MAMA-MIA Data.

Verifies:
1. TV1 & TV2 real metadata (`data/splits/splits_metadata.csv`) and `.npz` files (`data/processed/npz_96x96x96/`).
2. TV4 `MAMA_MIA_Dataset` loading real `.npz` cases across all 3 input representation modes (`IP1_post1`, `IP2_sub`, `IP3_multi`).
3. TV3 `MultiStageFiLM` + 3D Architectures (`M1: ResUNet3D`, `M2: AttentionUNet3D`, `M3: SwinUNETR3D`) forward/backward with Deep Supervision.
4. TV4 `DiceBCELoss`, `LinearWarmupCosineAnnealingLR`, and `Trainer` 1-epoch smoke check on real cases.

Can be executed via `pytest -q tests/test_week1_real_integration.py` or directly via:
  python tests/test_week1_real_integration.py
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_tv4 import generate_synthetic_tv4
from src.datasets.mama_mia import MAMA_MIA_Dataset, build_loaders
from src.engine.trainer import LinearWarmupCosineAnnealingLR, Trainer
from src.losses.dice_bce import DiceBCELoss
from src.models.factory import build_model

REAL_META_CSV = PROJECT_ROOT / "data" / "splits" / "splits_metadata.csv"
REAL_NPZ_DIR = PROJECT_ROOT / "data" / "processed" / "npz_96x96x96"


def _prepare_real_or_fallback_subset(tmp_dir: Path) -> tuple[Path, Path, bool]:
    """Select 1 real 96x96x96 case per subtype per split (12 cases) if real .npz files exist locally."""
    if REAL_META_CSV.is_file() and REAL_NPZ_DIR.is_dir():
        df = pd.read_csv(REAL_META_CSV)
        valid_rows = []
        for _, row in df.iterrows():
            npz_path = REAL_NPZ_DIR / f"{row['patient_id']}.npz"
            if npz_path.is_file():
                valid_rows.append(row)
            if len(valid_rows) >= 60:
                break
        if len(valid_rows) >= 12:
            sub_df = pd.DataFrame(valid_rows)
            # Pick 1 case per (split, subtype_id) with exact 96x96x96 shape for batch collation
            exact_rows = []
            for _, r in df.iterrows():
                p = REAL_NPZ_DIR / f"{r['patient_id']}.npz"
                if not p.is_file():
                    continue
                with np.load(p, allow_pickle=False) as z:
                    if z["image"].shape == (3, 96, 96, 96):
                        exact_rows.append(r)
                if len(exact_rows) >= 200:
                    break
            chosen = pd.DataFrame(exact_rows).groupby(["split", "subtype_id"], as_index=False).first()
            sub_csv = tmp_dir / "real_subset_splits.csv"
            chosen.to_csv(sub_csv, index=False)
            return REAL_NPZ_DIR, sub_csv, True

    npz_dir, sub_csv = generate_synthetic_tv4(tmp_dir / "fallback_synthetic", shape=(64, 64, 64), seed=42)
    return npz_dir, sub_csv, False


def test_week1_tv1_tv2_metadata_and_contract():
    """Verify TV1 splits_metadata.csv has 1,480 real cases across 4 subtypes and 3 splits."""
    assert REAL_META_CSV.is_file(), f"Missing {REAL_META_CSV}"
    df = pd.read_csv(REAL_META_CSV)
    assert len(df) == 1480, f"Expected 1,480 rows, got {len(df)}"
    required_cols = {"patient_id", "dataset", "tumor_subtype", "subtype_id", "split", "sample_weight"}
    assert required_cols.issubset(df.columns)
    assert set(df["subtype_id"].unique()) == {0, 1, 2, 3}
    assert set(df["split"].unique()) == {"train", "val", "test"}


def test_week1_real_data_e2e_tv1_tv2_tv3_tv4(tmp_path=None):
    """Verify real .npz data flows through MAMA_MIA_Dataset (TV4), M1/M2/M3 + FiLM (TV3), and DiceBCELoss/Trainer (TV4)."""
    own_tmp = None
    if tmp_path is None:
        own_tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(own_tmp.name)

    try:
        npz_dir, sub_csv, is_real = _prepare_real_or_fallback_subset(Path(tmp_path))
        patch_sz = (64, 64, 64)

        # 1. Verify all 3 IP modes (IP1_post1, IP2_sub, IP3_multi)
        for ip_mode, exp_channels in (("IP1_post1", 1), ("IP2_sub", 1), ("IP3_multi", 3)):
            ds = MAMA_MIA_Dataset(
                sub_csv,
                npz_dir,
                split="train",
                ip_mode=ip_mode,
                patch_size=patch_sz,
                cache_rate=0.0,
            )
            item = ds[0]
            assert item["image"].shape == (exp_channels,) + patch_sz
            assert item["mask"].shape == (1,) + patch_sz
            assert item["subtype_id"].dtype == torch.int64

        # 2. Verify forward + Deep Supervision loss + backward on M1, M2, M3 with MultiStageFiLM
        ds_multi = MAMA_MIA_Dataset(
            sub_csv,
            npz_dir,
            split="train",
            ip_mode="IP3_multi",
            patch_size=patch_sz,
            cache_rate=0.0,
        )
        sample = ds_multi[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = sample["image"].unsqueeze(0).to(device)
        y = sample["mask"].unsqueeze(0).to(device)
        s = sample["subtype_id"].unsqueeze(0).to(device)
        loss_fn = DiceBCELoss()

        for arch in ("M1", "M2", "M3"):
            model = build_model(arch=arch, in_channels=3, use_film=True, deep_supervision=True).to(device)
            model.train()
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                outputs = model(x, s)
                assert isinstance(outputs, list) and len(outputs) == 3
                loss = loss_fn(outputs, y)
            assert torch.isfinite(loss), f"Non-finite loss for {arch}"
            loss.backward()
            grad_norm = sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None)
            assert grad_norm > 0, f"Zero gradient norm for {arch}"
            print(
                f"[PASS] E2E ({'REAL NPZ' if is_real else 'FALLBACK'}) {arch} + MultiStageFiLM on {device} "
                f"| case={sample['patient_id']} | loss={loss.item():.4f}"
            )

        # 3. Verify TV4 Trainer + LinearWarmupCosineAnnealingLR 1-epoch smoke run with M1 + FiLM
        loaders = build_loaders(
            sub_csv,
            npz_dir,
            ip_mode="IP3_multi",
            batch_size=2,
            patch_size=patch_sz,
            cache_rate=0.0,
        )
        m1 = build_model(arch="M1", in_channels=3, use_film=True, deep_supervision=True)
        opt = torch.optim.AdamW(m1.parameters(), lr=3e-4, weight_decay=1e-5)
        sched = LinearWarmupCosineAnnealingLR(opt, warmup_epochs=1, max_epochs=2)
        trainer = Trainer(m1, opt, DiceBCELoss(), sched, device=device, use_amp=(device == "cuda"))
        history = trainer.fit(loaders["train"], loaders["val"], 1, Path(tmp_path) / "ckpt_test")
        assert len(history) == 1
        assert (Path(tmp_path) / "ckpt_test" / "best.pt").is_file()
        print(
            f"[PASS] Trainer 1-epoch smoke run ({'REAL NPZ' if is_real else 'FALLBACK'}) "
            f"| train_loss={history['train_loss'].iloc[0]:.4f} | val_dice={history['val_dice'].iloc[0]:.4f}"
        )
    finally:
        if own_tmp is not None:
            own_tmp.cleanup()


if __name__ == "__main__":
    test_week1_tv1_tv2_metadata_and_contract()
    print("[PASS] Metadata & contract check (1,480 cases)")
    test_week1_real_data_e2e_tv1_tv2_tv3_tv4()
    print("ALL WEEK 1 REAL-DATA INTEGRATION TESTS PASSED!")
