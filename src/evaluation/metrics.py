"""Binary Dice and symmetric 95th percentile surface Hausdorff distance."""

import numpy as np
from scipy import ndimage


def segmentation_metrics(prediction, truth, spacing_zyx=(1.0, 1.0, 1.0)):
    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape or prediction.ndim != 3:
        raise ValueError("Prediction and truth must have the same 3D shape")
    denominator = int(prediction.sum()) + int(truth.sum())
    dice = 1.0 if denominator == 0 else 2 * np.count_nonzero(prediction & truth) / denominator
    if not prediction.any() and not truth.any():
        return {"dice": dice, "hd95_mm": 0.0}
    if not prediction.any() or not truth.any():
        return {"dice": dice, "hd95_mm": float("inf")}
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure)
    true_surface = truth ^ ndimage.binary_erosion(truth, structure)
    to_truth = ndimage.distance_transform_edt(~true_surface, sampling=spacing_zyx)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing_zyx)[true_surface]
    hd95 = max(float(np.percentile(to_truth, 95)), float(np.percentile(to_pred, 95)))
    return {"dice": float(dice), "hd95_mm": hd95}
