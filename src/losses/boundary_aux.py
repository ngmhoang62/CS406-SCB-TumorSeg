"""Segmentation Dice+BCE with a weighted training-only boundary objective."""

import math

from torch import nn

from .dice_bce import DiceBCELoss


class BoundaryAuxiliaryLoss(nn.Module):
    def __init__(self, boundary_weight=0.2):
        super().__init__()
        if not math.isfinite(boundary_weight) or boundary_weight < 0:
            raise ValueError("boundary_weight must be finite and non-negative")
        self.boundary_weight = boundary_weight
        self.segmentation_loss = DiceBCELoss()
        self.boundary_loss = DiceBCELoss()

    def forward(self, outputs, target, boundary_target=None):
        if not isinstance(outputs, dict):
            # Validation/inference use only the main segmentation head.
            return self.segmentation_loss(outputs, target)
        if set(outputs) != {"seg", "boundary"} or boundary_target is None:
            raise ValueError("Boundary training requires seg/boundary logits and full-scan boundary labels")
        if outputs["boundary"].shape != boundary_target.shape or boundary_target.shape != target.shape:
            raise ValueError("Boundary logits and labels must match the segmentation target shape")
        return (self.segmentation_loss(outputs["seg"], target)
                + self.boundary_weight * self.boundary_loss(outputs["boundary"], boundary_target))
