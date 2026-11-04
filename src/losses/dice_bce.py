"""Binary Dice + BCE on logits, with nearest-neighbour deep-supervision targets."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, weights=(1.0, 0.5, 0.25), smooth=1e-5):
        super().__init__()
        if not weights or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative with a positive sum")
        self.weights = tuple(weights)
        self.smooth = smooth

    def forward(self, outputs, target):
        heads = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
        if len(heads) not in (1, len(self.weights)):
            raise ValueError(f"Expected 1 or {len(self.weights)} heads, got {len(heads)}")
        if target.ndim != 5 or target.shape[1] != 1:
            raise ValueError("target must be (B,1,D,H,W)")
        loss = target.new_zeros((), dtype=torch.float32)
        for weight, logits in zip(self.weights, heads):
            if logits.ndim != 5 or logits.shape[:2] != target.shape[:2]:
                raise ValueError("Head batch/channel dimensions must match target")
            # Float32 reductions keep small foreground regions stable under bf16 AMP.
            logits = logits.float()
            truth = F.interpolate(target.float(), size=logits.shape[2:], mode="nearest")
            prob = logits.sigmoid()
            dims = (2, 3, 4)
            dice = (2 * (prob * truth).sum(dims) + self.smooth) / (
                prob.sum(dims) + truth.sum(dims) + self.smooth)
            loss = loss + weight * ((1 - dice).mean() + F.binary_cross_entropy_with_logits(logits, truth))
        return loss
