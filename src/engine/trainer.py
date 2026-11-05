"""TV4 week 1 training engine; CPU fp32 / supported CUDA bf16 AMP + Week 2 Sliding Window, Resume & Early Stopping."""

import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Epoch scheduler: warm up to base LR, then decay to min_lr.

    LR recorded for epoch e is the LR actually used for that epoch. Call step()
    once after the epoch. For a 2-epoch smoke run, warmup_epochs=1 is sufficient.
    """

    def __init__(self, optimizer, warmup_epochs=5, max_epochs=120, min_lr=0.0, last_epoch=-1):
        if max_epochs < 1 or not 0 <= warmup_epochs < max_epochs:
            raise ValueError("Require max_epochs >= 1 and 0 <= warmup_epochs < max_epochs")
        if min_lr < 0 or min_lr > min(g["lr"] for g in optimizer.param_groups):
            raise ValueError("min_lr must lie between zero and the base LR")
        self.warmup_epochs, self.max_epochs, self.min_lr = warmup_epochs, max_epochs, min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [base * (epoch + 1) / self.warmup_epochs for base in self.base_lrs]
        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs - 1))
        return [self.min_lr + (base - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base in self.base_lrs]


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None,
                 use_amp=True, grad_clip=1.0, val_roi_size=(96, 96, 96),
                 sw_batch_size=4, sw_overlap=0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer, self.criterion, self.scheduler = optimizer, criterion, scheduler
        # bf16 does not need GradScaler. Unsupported GPUs use fp32 safely.
        self.amp_enabled = bool(use_amp and self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self.grad_clip = grad_clip
        self.val_roi_size = tuple(int(x) for x in val_roi_size) if val_roi_size is not None else None
        self.sw_batch_size = int(sw_batch_size)
        self.sw_overlap = float(sw_overlap)

    def _forward_eval(self, image, subtype):
        """Run direct forward for standard VOIs or 3D Sliding Window Inference for oversized VOIs (> val_roi_size)."""
        use_sw = (
            self.val_roi_size is not None
            and any(s > r for s, r in zip(image.shape[2:], self.val_roi_size))
        )
        if not use_sw:
            return self.model(image, subtype=subtype)

        def _predictor(patch):
            sub = subtype.expand(patch.shape[0]) if subtype.numel() == 1 else subtype
            out = self.model(patch, subtype=sub)
            return out[0] if isinstance(out, (list, tuple)) else out

        return sliding_window_inference(
            inputs=image,
            roi_size=self.val_roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=_predictor,
            overlap=self.sw_overlap,
            mode="gaussian",
        )

    def _epoch(self, loader, training):
        self.model.train(training)
        loss_sum = dice_sum = count = 0
        for batch in loader:
            image = batch["image"].to(self.device, dtype=torch.float32)
            target = batch["mask"].to(self.device, dtype=torch.float32)
            subtype = batch["subtype_id"].to(self.device, dtype=torch.long)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled):
                    outputs = self.model(image, subtype=subtype) if training else self._forward_eval(image, subtype)
                    loss = self.criterion(outputs, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip, error_if_nonfinite=True)
                    self.optimizer.step()
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            pred = logits.detach().float().sigmoid() >= 0.5
            truth = target.bool()
            dims = (1, 2, 3, 4)
            intersection = (pred & truth).sum(dims).float()
            denominator = pred.sum(dims) + truth.sum(dims)
            dice = torch.where(denominator > 0, 2 * intersection / denominator.clamp_min(1), 1.0)
            batch_size = image.shape[0]
            count += batch_size
            loss_sum += loss.detach().item() * batch_size
            dice_sum += dice.sum().item()
        if count == 0:
            raise ValueError("Cannot train/evaluate an empty loader")
        return {"loss": loss_sum / count, "dice": dice_sum / count}

    def _build_ckpt(self, epoch, val_dice, best_dice, best_epoch, no_improve, run_config, completed=False, stopped_early=False):
        return {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "val_dice": float(val_dice),
            "best_dice": float(best_dice),
            "best_epoch": int(best_epoch),
            "epochs_without_improvement": int(no_improve),
            "completed": bool(completed),
            "stopped_early": bool(stopped_early),
            "run_config": run_config or {},
            "amp_bf16": bool(self.amp_enabled),
        }

    def _atomic_torch_save(self, obj, target_path: Path):
        """Save via temporary file, fsync to storage, atomic replace, and sync OS buffers."""
        import os

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        try:
            with open(tmp_path, "wb") as f:
                torch.save(obj, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            tmp_path.replace(target_path)
            try:
                os.sync()
            except Exception:
                pass
        except Exception:
            if tmp_path.is_file():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise

    def fit(self, train_loader, val_loader, epochs, output_dir, run_config=None,
            resume=True, save_last_every=20, early_stopping_patience=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / "history.csv"
        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"
        summary_path = output_dir / "run_summary.json"

        history, best_dice, best_epoch, no_improve, start_epoch = [], -1.0, 0, 0, 0

        if resume:
            candidates = []
            for p in (last_path, best_path):
                if p.is_file():
                    try:
                        ckpt = torch.load(p, map_location=self.device, weights_only=True)
                        candidates.append((int(ckpt.get("epoch", 0)), p, ckpt))
                    except Exception as exc:
                        print(f"[WARN] Could not load checkpoint {p}: {exc}", flush=True)
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                ckpt_epoch, chosen_path, ckpt = candidates[0]
                if ckpt.get("completed", False) or ckpt_epoch >= epochs:
                    print(
                        f"[SKIP] Run in {output_dir.name} already completed at epoch {ckpt_epoch} "
                        f"(best_val_dice={ckpt.get('best_dice', ckpt.get('val_dice', -1.0)):.4f}).",
                        flush=True,
                    )
                    if history_path.is_file():
                        return pd.read_csv(history_path)
                self.model.load_state_dict(ckpt["model"])
                if ckpt.get("optimizer"):
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.scheduler is not None and ckpt.get("scheduler") is not None:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt_epoch
                best_dice = float(ckpt.get("best_dice", ckpt.get("val_dice", -1.0)))
                best_epoch = int(ckpt.get("best_epoch", ckpt_epoch))
                no_improve = int(ckpt.get("epochs_without_improvement", 0))
                if history_path.is_file():
                    prev_df = pd.read_csv(history_path)
                    prev_df = prev_df[prev_df["epoch"] <= start_epoch]
                    history = prev_df.to_dict("records")
                print(
                    f"[RESUME] Resumed from {chosen_path.name} at epoch {start_epoch}/{epochs} "
                    f"(best_val_dice={best_dice:.4f} @ epoch {best_epoch})",
                    flush=True,
                )

        stopped_early = False
        for epoch in range(start_epoch, epochs):
            t_ep = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._epoch(train_loader, training=True)
            with torch.no_grad():
                val = self._epoch(val_loader, training=False)
            if self.scheduler is not None:
                self.scheduler.step()

            ep_num = epoch + 1
            elapsed = time.time() - t_ep
            row = {
                "epoch": ep_num,
                "lr": lr,
                "train_loss": train["loss"],
                "train_dice": train["dice"],
                "val_loss": val["loss"],
                "val_dice": val["dice"],
            }
            history.append(row)
            pd.DataFrame(history).to_csv(history_path, index=False)

            if val["dice"] > best_dice:
                best_dice = val["dice"]
                best_epoch = ep_num
                no_improve = 0
                best_ckpt = self._build_ckpt(
                    ep_num, best_dice, best_dice, best_epoch, no_improve, run_config,
                    completed=(ep_num == epochs), stopped_early=False,
                )
                self._atomic_torch_save(best_ckpt, best_path)
            else:
                no_improve += 1

            will_stop_early = (
                early_stopping_patience is not None
                and early_stopping_patience > 0
                and no_improve >= early_stopping_patience
            )
            is_final_epoch = (ep_num == epochs) or will_stop_early
            should_save_last = (
                save_last_every is not None
                and save_last_every > 0
                and (ep_num % save_last_every == 0 or is_final_epoch)
            )
            if should_save_last:
                last_ckpt = self._build_ckpt(
                    ep_num, val["dice"], best_dice, best_epoch, no_improve, run_config,
                    completed=is_final_epoch, stopped_early=will_stop_early,
                )
                self._atomic_torch_save(last_ckpt, last_path)

            print(
                f"Epoch {ep_num}/{epochs} ({elapsed:.1f}s) | train_loss={train['loss']:.4f} | "
                f"val_loss={val['loss']:.4f} | val_dice={val['dice']:.4f} | best={best_dice:.4f} (ep {best_epoch})",
                flush=True,
            )

            if will_stop_early:
                stopped_early = True
                print(
                    f"[EARLY STOPPING] No val_dice improvement for {no_improve} epochs "
                    f"(patience={early_stopping_patience}). Stopping at epoch {ep_num}/{epochs}.",
                    flush=True,
                )
                break

        summary = {
            "best_val_dice": float(best_dice),
            "best_epoch": int(best_epoch),
            "last_epoch": int(history[-1]["epoch"]) if history else int(start_epoch),
            "max_epochs": int(epochs),
            "stopped_early": bool(stopped_early),
            "completed": True,
            "run_config": run_config or {},
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        try:
            import os
            os.sync()
        except Exception:
            pass
        return pd.DataFrame(history)
