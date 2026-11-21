"""Shared baseline trainer with full-scan validation and atomic checkpoints."""

import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.optim.lr_scheduler import LRScheduler

from .inference import predict_logits


class LinearWarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=120, min_lr=0.0, last_epoch=-1):
        if max_epochs < 1 or not 0 <= warmup_epochs < max_epochs:
            raise ValueError("Require max_epochs >= 1 and 0 <= warmup_epochs < max_epochs")
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            return [base * (epoch + 1) / self.warmup_epochs for base in self.base_lrs]
        progress = min(1.0, (epoch - self.warmup_epochs) /
                       max(1, self.max_epochs - self.warmup_epochs - 1))
        return [self.min_lr + (base - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                for base in self.base_lrs]


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None,
                 use_amp=True, grad_clip=1.0, val_roi_size=(96, 96, 96),
                 sw_batch_size=1, sw_overlap=0.5, prob_threshold=0.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer, self.criterion, self.scheduler = optimizer, criterion, scheduler
        model_parameters = {id(parameter) for parameter in self.model.parameters()}
        optimizer_parameters = {id(parameter) for group in optimizer.param_groups
                                for parameter in group["params"]}
        if model_parameters != optimizer_parameters:
            raise ValueError("Move the model to the training device before constructing its optimizer")
        self.amp_enabled = bool(use_amp and self.device.type == "cuda" and
                                torch.cuda.is_bf16_supported())
        self.grad_clip = grad_clip
        self.val_roi_size = tuple(val_roi_size)
        self.sw_batch_size, self.sw_overlap = sw_batch_size, sw_overlap
        if not 0 < prob_threshold < 1:
            raise ValueError("prob_threshold must be between zero and one")
        self.prob_threshold = prob_threshold

    def _epoch(self, loader, training):
        self.model.train(training)
        loss_sum, dice_sum, count = 0.0, 0.0, 0
        for batch in loader:
            # Validation inputs, targets and stitched outputs remain on CPU.
            device = self.device if training else torch.device("cpu")
            image = batch["image"].to(device, dtype=torch.float32)
            target = batch["mask"].to(device, dtype=torch.float32)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                    enabled=self.amp_enabled):
                    if training:
                        outputs = self.model(image)
                    else:
                        outputs = predict_logits(self.model, image, self.device,
                                                 self.val_roi_size, self.sw_batch_size,
                                                 self.sw_overlap)
                    if isinstance(outputs, dict):
                        if "boundary" not in batch:
                            raise ValueError("Boundary training requires a boundary-enabled training loader")
                        boundary = batch["boundary"].to(device, dtype=torch.float32)
                        loss = self.criterion(outputs, target, boundary)
                    else:
                        loss = self.criterion(outputs, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip,
                                                   error_if_nonfinite=True)
                    self.optimizer.step()
            logits = outputs["seg"] if isinstance(outputs, dict) else outputs
            pred = logits.detach().float().sigmoid() >= self.prob_threshold
            truth = target.bool()
            dims = (1, 2, 3, 4)
            denominator = pred.sum(dims) + truth.sum(dims)
            dice = torch.where(denominator > 0,
                               2 * (pred & truth).sum(dims) / denominator.clamp_min(1), 1.0)
            count += image.shape[0]
            loss_sum += loss.item() * image.shape[0]
            dice_sum += dice.sum().item()
        if not count:
            raise ValueError("Empty loader")
        return {"loss": loss_sum / count, "dice": dice_sum / count}

    @staticmethod
    def _atomic_torch_save(data, path):
        temporary = path.with_suffix(".tmp")
        try:
            torch.save(data, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def fit(self, train_loader, val_loader, epochs, output_dir, run_config=None,
            resume=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        last_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
        history_path = output_dir / "history.csv"
        history, best_dice, best_epoch, start_epoch = [], -1.0, 0, 0
        run_config = run_config or {}
        if not resume and any(p.exists() for p in (last_path, best_path, history_path)):
            raise FileExistsError(f"Existing run at {output_dir}; choose a fresh output directory")
        if resume and (last_path.exists() or best_path.exists()):
            checkpoints = [torch.load(p, map_location="cpu", weights_only=True)
                           for p in (last_path, best_path) if p.exists()]
            checkpoint = max(checkpoints, key=lambda entry: entry["epoch"])
            if checkpoint["run_config"] != run_config:
                raise ValueError("Checkpoint configuration differs; use a fresh output directory")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"]
            best_dice, best_epoch = checkpoint["best_dice"], checkpoint["best_epoch"]
            if history_path.exists():
                frame = pd.read_csv(history_path)
                history = frame.loc[frame.epoch <= start_epoch].to_dict("records")
            if start_epoch > epochs:
                raise ValueError("Checkpoint epoch exceeds configured training duration")
        for epoch in range(start_epoch, epochs):
            start = time.time()
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._epoch(train_loader, True)
            with torch.no_grad():
                val = self._epoch(val_loader, False)
            if self.scheduler is not None:
                self.scheduler.step()
            improved = val["dice"] > best_dice
            if improved:
                best_dice, best_epoch = val["dice"], epoch + 1
            history.append({
                "epoch": epoch + 1, "lr": lr,
                "train_loss": train["loss"], "train_dice": train["dice"],
                "val_loss": val["loss"], "val_dice": val["dice"],
            })
            checkpoint = {
                "epoch": epoch + 1, "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "val_dice": val["dice"], "best_dice": best_dice,
                "best_epoch": best_epoch, "run_config": run_config,
            }
            # Persist the best checkpoint before advancing last.pt.
            if improved:
                self._atomic_torch_save(checkpoint, best_path)
            pd.DataFrame(history).to_csv(history_path, index=False)
            self._atomic_torch_save(checkpoint, last_path)
            print(f"Epoch {epoch + 1}/{epochs} | {time.time() - start:.1f}s | "
                  f"train_loss={train['loss']:.4f} | val_dice={val['dice']:.4f}", flush=True)
        summary = {"best_val_dice": best_dice, "best_epoch": best_epoch,
                   "last_epoch": epochs, "completed": True, "run_config": run_config}
        temporary = output_dir / "run_summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.replace(output_dir / "run_summary.json")
        return pd.DataFrame(history)
