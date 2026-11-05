"""
TV3 Week 2 Training Runner: 6 Ablation Configurations for M2 (3D Attention U-Net).

Executes Runs 07 -> 12 from `docs/task_assignment.md` & `docs/detailed_pipeline.md`:
  - Run 07: M2 (Attention U-Net 3D) | IP1_post1 (C1: Post-1, 1 ch)     | C0: No FiLM
  - Run 08: M2 (Attention U-Net 3D) | IP1_post1 (C1: Post-1, 1 ch)     | C1: Multi-Stage FiLM
  - Run 09: M2 (Attention U-Net 3D) | IP2_sub   (C2: Subtraction, 1 ch)| C0: No FiLM
  - Run 10: M2 (Attention U-Net 3D) | IP2_sub   (C2: Subtraction, 1 ch)| C1: Multi-Stage FiLM
  - Run 11: M2 (Attention U-Net 3D) | IP3_multi (C1+C2+C3, 3 ch)       | C0: No FiLM
  - Run 12: M2 (Attention U-Net 3D) | IP3_multi (C1+C2+C3, 3 ch)       | C1: Multi-Stage FiLM

Features:
  - Reuses DataLoaders across (No FiLM, With FiLM) pairs sharing the same `ip_mode`.
  - Uses WeightedRandomSampler (`sample_weight`) on the training split.
  - Uses 3D Sliding Window Inference (`roi_size=(96,96,96)`, `overlap=0.5`, Gaussian blending)
    for oversized validation VOIs (> 96^3).
  - Supports automatic checkpoint resume (`last.pt` / `best.pt`), configurable `save_last_every`
    (default: every 20 epochs, overwriting a single `last.pt`), and `early_stopping_patience`
    (default: 25 epochs).
  - Exports summary table to `outputs/tables/tv3_week2_ablation_summary.csv` and learning curves
    to `outputs/figures/tv3_week2_learning_curves.png`.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from monai.utils import set_determinism

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.mama_mia import IP_CHANNELS, build_loaders
from src.engine.trainer import LinearWarmupCosineAnnealingLR, Trainer
from src.losses.dice_bce import DiceBCELoss
from src.models.factory import build_model
from src.utils.config import load_config


TV3_WEEK2_RUNS: List[Dict[str, Any]] = [
    {
        "run_id": "run07",
        "run_name": "run07_M2_IP1_post1_no_film",
        "arch": "M2",
        "ip_mode": "IP1_post1",
        "ip_label": "IP1 (Post-1)",
        "in_channels": 1,
        "use_film": False,
        "plot_label": "IP1 (Post-1) | No FiLM",
        "description": "3D Attention U-Net | IP1 (Post-1) | No FiLM",
    },
    {
        "run_id": "run08",
        "run_name": "run08_M2_IP1_post1_film",
        "arch": "M2",
        "ip_mode": "IP1_post1",
        "ip_label": "IP1 (Post-1)",
        "in_channels": 1,
        "use_film": True,
        "plot_label": "IP1 (Post-1) | With FiLM",
        "description": "3D Attention U-Net | IP1 (Post-1) | Multi-Stage FiLM",
    },
    {
        "run_id": "run09",
        "run_name": "run09_M2_IP2_sub_no_film",
        "arch": "M2",
        "ip_mode": "IP2_sub",
        "ip_label": "IP2 (Subtraction)",
        "in_channels": 1,
        "use_film": False,
        "plot_label": "IP2 (Subtraction) | No FiLM",
        "description": "3D Attention U-Net | IP2 (Subtraction) | No FiLM",
    },
    {
        "run_id": "run10",
        "run_name": "run10_M2_IP2_sub_film",
        "arch": "M2",
        "ip_mode": "IP2_sub",
        "ip_label": "IP2 (Subtraction)",
        "in_channels": 1,
        "use_film": True,
        "plot_label": "IP2 (Subtraction) | With FiLM",
        "description": "3D Attention U-Net | IP2 (Subtraction) | Multi-Stage FiLM",
    },
    {
        "run_id": "run11",
        "run_name": "run11_M2_IP3_multi_no_film",
        "arch": "M2",
        "ip_mode": "IP3_multi",
        "ip_label": "IP3 (Multi-Channel)",
        "in_channels": 3,
        "use_film": False,
        "plot_label": "IP3 (Multi-Channel) | No FiLM",
        "description": "3D Attention U-Net | IP3 (3-Channel Multi) | No FiLM",
    },
    {
        "run_id": "run12",
        "run_name": "run12_M2_IP3_multi_film",
        "arch": "M2",
        "ip_mode": "IP3_multi",
        "ip_label": "IP3 (Multi-Channel)",
        "in_channels": 3,
        "use_film": True,
        "plot_label": "IP3 (Multi-Channel) | With FiLM",
        "description": "3D Attention U-Net | IP3 (3-Channel Multi) | Multi-Stage FiLM",
    },
]


def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def update_ablation_summary_and_plots(
    checkpoints_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    logs_dir: Path | None = None,
) -> pd.DataFrame:
    """Collect metrics from all completed/in-progress M2 runs and export CSVs + learning curves."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    histories: Dict[str, pd.DataFrame] = {}
    combined_history_frames: List[pd.DataFrame] = []

    for spec in TV3_WEEK2_RUNS:
        run_dir = checkpoints_root / spec["run_name"]
        hist_path = run_dir / "history.csv"
        sum_path = run_dir / "run_summary.json"
        best_path = run_dir / "best.pt"

        if not hist_path.is_file():
            continue

        df_hist = pd.read_csv(hist_path)
        if df_hist.empty:
            continue
        histories[spec["run_id"]] = df_hist

        if logs_dir is not None:
            df_hist.to_csv(logs_dir / f"{spec['run_name']}_history.csv", index=False)

        df_tagged = df_hist.copy()
        df_tagged.insert(0, "use_film", spec["use_film"])
        df_tagged.insert(0, "ip_mode", spec["ip_mode"])
        df_tagged.insert(0, "arch", spec["arch"])
        df_tagged.insert(0, "run_name", spec["run_name"])
        df_tagged.insert(0, "run_id", spec["run_id"])
        combined_history_frames.append(df_tagged)

        best_idx = int(df_hist["val_dice"].idxmax())
        best_row = df_hist.loc[best_idx]
        completed = False
        stopped_early = False
        if sum_path.is_file():
            try:
                meta = json.loads(sum_path.read_text(encoding="utf-8"))
                completed = bool(meta.get("completed", False))
                stopped_early = bool(meta.get("stopped_early", False))
            except Exception:
                pass

        rows.append(
            {
                "run_id": spec["run_id"],
                "run_name": spec["run_name"],
                "arch": spec["arch"],
                "ip_mode": spec["ip_mode"],
                "in_channels": spec["in_channels"],
                "use_film": spec["use_film"],
                "epochs_trained": int(df_hist["epoch"].max()),
                "best_epoch": int(best_row["epoch"]),
                "best_val_dice": float(best_row["val_dice"]),
                "best_val_loss": float(best_row["val_loss"]),
                "train_dice_at_best": float(best_row["train_dice"]),
                "train_loss_at_best": float(best_row["train_loss"]),
                "final_train_loss": float(df_hist["train_loss"].iloc[-1]),
                "final_train_dice": float(df_hist["train_dice"].iloc[-1]),
                "final_val_loss": float(df_hist["val_loss"].iloc[-1]),
                "final_val_dice": float(df_hist["val_dice"].iloc[-1]),
                "completed": completed,
                "stopped_early": stopped_early,
                "best_ckpt_exists": best_path.is_file(),
            }
        )

    if combined_history_frames:
        all_hist_df = pd.concat(combined_history_frames, ignore_index=True)
        all_hist_df.to_csv(tables_dir / "tv3_week2_training_histories.csv", index=False)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_csv = tables_dir / "tv3_week2_ablation_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for spec in TV3_WEEK2_RUNS:
                rid = spec["run_id"]
                if rid not in histories:
                    continue
                h = histories[rid]
                label = spec["plot_label"]
                linestyle = "-" if spec["use_film"] else "--"
                axes[0].plot(h["epoch"], h["train_loss"], linestyle=linestyle, linewidth=1.8, label=label)
                axes[1].plot(h["epoch"], h["val_loss"], linestyle=linestyle, linewidth=1.8, label=label)
                axes[2].plot(h["epoch"], h["val_dice"], linestyle=linestyle, linewidth=1.8, label=label)

            axes[0].set_title("3D Attention U-Net — Training Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Dice + BCE Loss (Deep Supervision)")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=8)

            axes[1].set_title("3D Attention U-Net — Validation Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation Dice + BCE Loss")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)

            axes[2].set_title("3D Attention U-Net — Validation Dice Score")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Validation Dice")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(figures_dir / "tv3_week2_learning_curves.png", dpi=200)
            plt.close(fig)
        except Exception as exc:
            print(f"[WARN] Could not plot learning curves: {exc}", flush=True)

    return summary_df


def run_tv3_week2_suite(
    metadata_csv: str | Path,
    npz_dir: str | Path,
    output_root: str | Path = "outputs",
    config_path: str | Path = "configs/default_config.yaml",
    selected_runs: List[str] | None = None,
    epochs: int | None = None,
    warmup_epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 8,
    sw_overlap: float = 0.5,
    num_workers: int = 12,
    cache_rate: float = 0.0,
    use_amp: bool = True,
    use_weighted_sampler: bool = True,
    save_last_every: int = 20,
    early_stopping_patience: int | None = 25,
    resume: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Run TV3's 6 M2 (3D Attention U-Net) ablation experiments sequentially."""
    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    epochs = int(epochs if epochs is not None else train_cfg["epochs"])
    warmup_epochs = int(warmup_epochs if warmup_epochs is not None else train_cfg["warmup_epochs"])
    batch_size = int(batch_size if batch_size is not None else train_cfg["batch_size"])
    lr = float(lr if lr is not None else train_cfg["lr"])
    weight_decay = float(weight_decay if weight_decay is not None else train_cfg["weight_decay"])
    ds_weights = tuple(float(w) for w in train_cfg["deep_supervision_weights"])
    base_channels = tuple(int(c) for c in model_cfg["resunet_channels"])

    output_root = Path(output_root)
    checkpoints_root = output_root / "checkpoints" / "tv3_week2"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    logs_dir = output_root / "logs"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    if selected_runs and "all" not in [r.lower() for r in selected_runs]:
        wanted = {r.lower() for r in selected_runs}
        runs_to_execute = [
            r for r in TV3_WEEK2_RUNS
            if r["run_id"].lower() in wanted or r["run_name"].lower() in wanted
        ]
    else:
        runs_to_execute = list(TV3_WEEK2_RUNS)

    if not runs_to_execute:
        raise ValueError(f"No valid TV3 runs matched {selected_runs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"TV3 WEEK 2 — 3D ATTENTION U-NET (M2) ABLATION SUITE ({len(runs_to_execute)} RUNS)", flush=True)
    print(f"  Device                 : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"  Metadata CSV           : {metadata_csv}", flush=True)
    print(f"  NPZ Directory          : {npz_dir}", flush=True)
    print(f"  Checkpoints Root       : {checkpoints_root}", flush=True)
    print(f"  Epochs / Warmup        : {epochs} / {warmup_epochs}", flush=True)
    print(f"  Early Stopping Patience: {early_stopping_patience}", flush=True)
    print(f"  Save last.pt Every     : {save_last_every} epochs (single file overwrite)", flush=True)
    print(f"  Batch Size / LR / WD   : {batch_size} / {lr} / {weight_decay}", flush=True)
    print(f"  Patch Size / SW Batch  : {patch_size} / {sw_batch_size} (overlap={sw_overlap})", flush=True)
    print(f"  WeightedRandomSampler  : {use_weighted_sampler} | Cache Rate: {cache_rate}", flush=True)
    print("=" * 80, flush=True)

    current_ip_mode: str | None = None
    loaders: Dict[str, Any] | None = None

    for idx, spec in enumerate(runs_to_execute, 1):
        run_dir = checkpoints_root / spec["run_name"]
        ip_mode = spec["ip_mode"]

        print(f"\n[{idx}/{len(runs_to_execute)}] >>> Preparing {spec['run_name']} ({spec['description']})", flush=True)

        # Check if already completed before building dataset/loaders
        sum_path = run_dir / "run_summary.json"
        hist_path = run_dir / "history.csv"
        if resume and sum_path.is_file() and hist_path.is_file():
            try:
                summary_data = json.loads(sum_path.read_text(encoding="utf-8"))
                if summary_data.get("completed", False):
                    print(
                        f"[SKIP] {spec['run_name']} already marked completed "
                        f"(best_val_dice={summary_data.get('best_val_dice', -1.0):.4f} "
                        f"@ epoch {summary_data.get('best_epoch', 0)}).",
                        flush=True,
                    )
                    update_ablation_summary_and_plots(checkpoints_root, tables_dir, figures_dir, logs_dir)
                    continue
            except Exception:
                pass

        # Reuse loaders if consecutive runs share the same ip_mode (e.g. run07 & run08)
        if loaders is None or current_ip_mode != ip_mode:
            if loaders is not None:
                del loaders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"Building DataLoaders for ip_mode={ip_mode} (channels={IP_CHANNELS[ip_mode]})...", flush=True)
            set_all_seeds(seed)
            loaders = build_loaders(
                metadata_csv=metadata_csv,
                npz_dir=npz_dir,
                ip_mode=ip_mode,
                batch_size=batch_size,
                patch_size=patch_size,
                num_workers=num_workers,
                cache_rate=cache_rate,
                seed=seed,
                use_weighted_sampler=use_weighted_sampler,
            )
            current_ip_mode = ip_mode

        set_all_seeds(seed)
        model = build_model(
            arch=spec["arch"],
            in_channels=spec["in_channels"],
            use_film=spec["use_film"],
            num_subtypes=int(model_cfg["num_subtypes"]),
            film_embed_dim=int(model_cfg["film_embed_dim"]),
            deep_supervision=bool(model_cfg["deep_supervision"]),
            resunet_channels=base_channels,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=min(warmup_epochs, max(0, epochs - 1)),
            max_epochs=epochs,
            min_lr=1e-6,
        )
        criterion = DiceBCELoss(weights=ds_weights)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            use_amp=use_amp,
            grad_clip=1.0,
            val_roi_size=patch_size,
            sw_batch_size=sw_batch_size,
            sw_overlap=sw_overlap,
        )

        run_config = {
            "run_id": spec["run_id"],
            "run_name": spec["run_name"],
            "arch": spec["arch"],
            "ip_mode": spec["ip_mode"],
            "in_channels": int(spec["in_channels"]),
            "use_film": bool(spec["use_film"]),
            "num_subtypes": int(model_cfg["num_subtypes"]),
            "film_embed_dim": int(model_cfg["film_embed_dim"]),
            "deep_supervision": bool(model_cfg["deep_supervision"]),
            "base_channels": [int(c) for c in base_channels],
            "epochs": int(epochs),
            "warmup_epochs": int(warmup_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "patch_size": list(patch_size),
            "sw_batch_size": int(sw_batch_size),
            "sw_overlap": float(sw_overlap),
            "use_weighted_sampler": bool(use_weighted_sampler),
            "early_stopping_patience": int(early_stopping_patience) if early_stopping_patience else 0,
            "seed": int(seed),
        }

        trainer.fit(
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            epochs=epochs,
            output_dir=run_dir,
            run_config=run_config,
            resume=resume,
            save_last_every=save_last_every,
            early_stopping_patience=early_stopping_patience,
        )

        del model, optimizer, scheduler, criterion, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        update_ablation_summary_and_plots(checkpoints_root, tables_dir, figures_dir, logs_dir)

    summary_df = update_ablation_summary_and_plots(checkpoints_root, tables_dir, figures_dir, logs_dir)
    if not summary_df.empty:
        print("\n" + "=" * 80, flush=True)
        print("TV3 WEEK 2 ABLATION SUMMARY (M2: 3D Attention U-Net):", flush=True)
        print(summary_df.to_string(index=False), flush=True)
        print("=" * 80, flush=True)

    try:
        import os, time
        os.sync()
        time.sleep(3)
    except Exception:
        pass

    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TV3 Week 2 3D Attention U-Net (6 runs: run07..run12)")
    parser.add_argument("--metadata-csv", default=str(PROJECT_ROOT / "data" / "splits" / "splits_metadata.csv"))
    parser.add_argument("--npz-dir", default=str(PROJECT_ROOT / "data" / "processed" / "npz_96x96x96"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["all"],
        help="Run IDs to execute, e.g. `--runs run07 run08` or `--runs all` (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--sw-batch-size", type=int, default=8)
    parser.add_argument("--sw-overlap", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable bfloat16 AMP and run pure fp32")
    parser.add_argument("--save-last-every", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true", help="Force training from scratch (ignore existing checkpoints)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tv3_week2_suite(
        metadata_csv=args.metadata_csv,
        npz_dir=args.npz_dir,
        output_root=args.output_root,
        config_path=args.config,
        selected_runs=args.runs,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=tuple(args.patch_size),
        sw_batch_size=args.sw_batch_size,
        sw_overlap=args.sw_overlap,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        use_amp=not args.no_amp,
        save_last_every=args.save_last_every,
        early_stopping_patience=args.early_stopping_patience if args.early_stopping_patience > 0 else None,
        resume=not args.no_resume,
        seed=args.seed,
    )
