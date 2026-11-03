"""TV4 week 1 training engine; CPU fp32 / supported CUDA bf16 AMP."""

import math
from pathlib import Path

import pandas as pd
import torch
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
                 use_amp=True, grad_clip=1.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer, self.criterion, self.scheduler = optimizer, criterion, scheduler
        # bf16 does not need GradScaler. Unsupported GPUs use fp32 safely.
        self.amp_enabled = bool(use_amp and self.device.type == "cuda" and torch.cuda.is_bf16_supported())
        self.grad_clip = grad_clip

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
                    outputs = self.model(image, subtype=subtype)
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

    def fit(self, train_loader, val_loader, epochs, output_dir, run_config=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        history, best_dice = [], -1.0
        for epoch in range(epochs):
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._epoch(train_loader, training=True)
            with torch.no_grad():
                val = self._epoch(val_loader, training=False)
            if self.scheduler is not None:
                self.scheduler.step()
            row = {"epoch": epoch + 1, "lr": lr, "train_loss": train["loss"],
                   "train_dice": train["dice"], "val_loss": val["loss"], "val_dice": val["dice"]}
            history.append(row)
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            if val["dice"] > best_dice:
                best_dice = val["dice"]
                torch.save({"epoch": epoch + 1, "model": self.model.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                            "val_dice": best_dice, "run_config": run_config or {},
                            "amp_bf16": self.amp_enabled}, output_dir / "best.pt")
            print(f"Epoch {epoch + 1}/{epochs} | train_loss={train['loss']:.4f} | "
                  f"val_loss={val['loss']:.4f} | val_dice={val['dice']:.4f}", flush=True)
        return pd.DataFrame(history)
