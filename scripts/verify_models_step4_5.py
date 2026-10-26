"""
Comprehensive Verification & Smoke Test for Step 4 (`MultiStageFiLM`) and Step 5 (`M1`, `M2`, `M3`).

Verifies:
  1. Reading actual preprocessed synthetic `.npz` data in `IP-1` (1ch), `IP-2` (1ch), and `IP-3` (3ch).
  2. Strict Zero-Initialization identity property of `MultiStageFiLM` (at epoch 0, gamma=0, beta=0).
  3. Forward pass in `train()` mode returning 3-level Deep Supervision `[P0 (96^3), P1 (48^3), P2 (24^3)]`.
  4. Backward pass (gradient flow into both 3D backbone and FiLM parameters) and 1 optimizer step
     confirming FiLM modulation becomes active after step 1.
  5. Forward pass in `eval()` mode returning single full-resolution tensor `P0 (96^3)`.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_all_models() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"STEP 4 & STEP 5 MODEL ARCHITECTURE VERIFICATION (Device: {device})")
    print("=" * 88)

    # 1. Load 2 real synthetic .npz samples to form a mini-batch B=2 of (3, 96, 96, 96)
    npz_dir = PROJECT_ROOT / "data" / "processed" / "npz_96x96x96"
    npz_files = sorted(npz_dir.glob("*.npz"))[:2]
    if len(npz_files) < 2:
        raise RuntimeError(f"Expected at least 2 .npz files in {npz_dir}, found {len(npz_files)}")

    imgs, masks, subtypes = [], [], []
    for fpath in npz_files:
        data = np.load(fpath)
        imgs.append(data["image"][:, :96, :96, :96].astype(np.float32))
        masks.append(data["mask"][:, :96, :96, :96].astype(np.float32))
        subtypes.append(int(data["subtype_id"]))

    x_ip3 = torch.from_numpy(np.stack(imgs, axis=0)).to(device)      # (2, 3, 96, 96, 96)
    x_ip1 = x_ip3[:, 0:1, ...]                                       # (2, 1, 96, 96, 96)
    x_ip2 = x_ip3[:, 1:2, ...]                                       # (2, 1, 96, 96, 96)
    y_gt0 = torch.from_numpy(np.stack(masks, axis=0)).to(device)     # (2, 1, 96, 96, 96)
    y_gt1 = F.interpolate(y_gt0, scale_factor=0.5, mode="nearest")   # (2, 1, 48, 48, 48)
    y_gt2 = F.interpolate(y_gt0, scale_factor=0.25, mode="nearest")  # (2, 1, 24, 24, 24)
    subtype_ids = torch.tensor(subtypes, dtype=torch.long, device=device)

    print(f"Loaded batch from {[f.name for f in npz_files]}:")
    print(f"  - IP-1 (Post1) tensor      : {tuple(x_ip1.shape)}")
    print(f"  - IP-2 (Subtraction) tensor: {tuple(x_ip2.shape)}")
    print(f"  - IP-3 (Multi-Ch) tensor   : {tuple(x_ip3.shape)}")
    print(f"  - Subtype IDs              : {subtype_ids.tolist()}")

    architectures = [
        ("M1", "3D ResUNet (DynUNet)"),
        ("M2", "3D Attention U-Net"),
        ("M3", "3D Swin UNETR"),
    ]

    summary_rows = []

    for arch_code, arch_label in architectures:
        print("\n" + "-" * 88)
        print(f"Testing [{arch_code}] {arch_label}")
        print("-" * 88)

        # Build C0 (No FiLM) and C1 (With FiLM) for IP-3 (3 channels) and IP-1 (1 channel)
        model_c0 = build_model(arch_code, ip_mode="IP-3", use_film=False, deep_supervision=True).to(device)
        model_c1 = build_model(arch_code, ip_mode="IP-3", use_film=True, deep_supervision=True).to(device)
        model_1ch = build_model(arch_code, ip_mode="IP-1", use_film=True, deep_supervision=True).to(device)

        params_c0 = count_parameters(model_c0)
        params_c1 = count_parameters(model_c1)
        film_overhead = params_c1 - params_c0

        # Copy backbone weights from model_c0 into model_c1 to test Zero-Init equivalence at step 0
        model_c1.load_state_dict(model_c0.state_dict(), strict=False)
        model_c0.eval()
        model_c1.eval()
        with torch.no_grad():
            out_c0_eval = model_c0(x_ip3)
            out_c1_eval = model_c1(x_ip3, subtype=subtype_ids)
            zero_init_diff = float(torch.max(torch.abs(out_c0_eval - out_c1_eval)).item())

        assert zero_init_diff == 0.0, f"Zero-Init check failed for {arch_code}: diff={zero_init_diff}"
        assert tuple(out_c1_eval.shape) == (2, 1, 96, 96, 96)

        # Test 1-channel input mode (IP-1 / IP-2)
        model_1ch.eval()
        with torch.no_grad():
            out_1ch = model_1ch(x_ip1, subtype=subtype_ids)
        assert tuple(out_1ch.shape) == (2, 1, 96, 96, 96)

        # Test Training Mode with 3-Level Deep Supervision + Backward Pass + 1 Optimizer Step
        model_c1.train()
        optimizer = torch.optim.AdamW(model_c1.parameters(), lr=1e-3)
        optimizer.zero_grad()

        use_amp = device.type == "cuda"
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            ds_outputs = model_c1(x_ip3, subtype=subtype_ids)
            assert isinstance(ds_outputs, list) and len(ds_outputs) == 3
            p0, p1, p2 = ds_outputs
            assert tuple(p0.shape) == (2, 1, 96, 96, 96)
            assert tuple(p1.shape) == (2, 1, 48, 48, 48)
            assert tuple(p2.shape) == (2, 1, 24, 24, 24)

            loss = (
                1.00 * F.binary_cross_entropy_with_logits(p0, y_gt0)
                + 0.50 * F.binary_cross_entropy_with_logits(p1, y_gt1)
                + 0.25 * F.binary_cross_entropy_with_logits(p2, y_gt2)
            )

        loss.backward()

        # Verify non-zero gradients on FiLM projection heads
        film_grad_norm = sum(
            float(h.weight.grad.norm().item()) for h in model_c1.film_generator.heads if h.weight.grad is not None
        )
        assert film_grad_norm > 0.0, f"Expected non-zero FiLM gradient norm, got {film_grad_norm}"

        optimizer.step()

        # Verify that after 1 step, different subtypes produce distinct conditioned predictions
        model_c1.eval()
        with torch.no_grad():
            sub_a = torch.tensor([0, 0], dtype=torch.long, device=device)
            sub_b = torch.tensor([1, 1], dtype=torch.long, device=device)
            pred_a = model_c1(x_ip3, subtype=sub_a)
            pred_b = model_c1(x_ip3, subtype=sub_b)
            post_step_subtype_diff = float(torch.max(torch.abs(pred_a - pred_b)).item())
        assert post_step_subtype_diff > 0.0, "FiLM conditioning did not differentiate subtypes after optimizer step!"

        print(f"  [PASS] Parameters (C0 No FiLM)   : {params_c0:,}")
        print(f"  [PASS] Parameters (C1 With FiLM) : {params_c1:,} (+{film_overhead:,} FiLM params)")
        print(f"  [PASS] Zero-Init Identity Check  : max |C1 - C0| = {zero_init_diff:.1e}")
        print(f"  [PASS] Deep Supervision Shapes   : {[tuple(t.shape) for t in ds_outputs]}")
        print(f"  [PASS] Eval Output Shape         : {tuple(out_c1_eval.shape)}")
        print(f"  [PASS] Backward FiLM Grad Norm   : {film_grad_norm:.4e}")
        print(f"  [PASS] Post-Step Subtype Effect  : max |P(sub=0) - P(sub=1)| = {post_step_subtype_diff:.4e}")

        summary_rows.append((arch_code, arch_label, params_c0, params_c1, film_overhead))

    print("\n" + "=" * 88)
    print("SUMMARY OF ALL 3 ARCHITECTURES (STEP 4 & STEP 5)")
    print("=" * 88)
    print(f"{'Code':<6} | {'Architecture':<24} | {'Params (No FiLM)':>18} | {'Params (With FiLM)':>18} | {'FiLM Overhead':>14}")
    print("-" * 88)
    for code, name, p0_cnt, p1_cnt, oh_cnt in summary_rows:
        print(f"{code:<6} | {name:<24} | {p0_cnt:>18,} | {p1_cnt:>18,} | {oh_cnt:>14,}")
    print("=" * 88)
    print("ALL STEP 4 & STEP 5 CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_all_models()
