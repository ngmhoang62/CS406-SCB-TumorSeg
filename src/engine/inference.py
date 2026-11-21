"""Full-volume sliding-window inference, stitched on CPU to limit GPU memory."""

import torch
from monai.inferers import sliding_window_inference


def predict_logits(model, image, device, roi_size, sw_batch_size=1, overlap=0.5):
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("Full-volume inference requires one case (1,C,D,H,W)")
    if not 0 <= overlap < 1 or sw_batch_size < 1:
        raise ValueError("Invalid sliding-window parameters")
    return sliding_window_inference(
        inputs=image.cpu(), roi_size=tuple(roi_size), sw_batch_size=sw_batch_size,
        predictor=model, overlap=overlap, mode="gaussian",
        sw_device=torch.device(device), device=torch.device("cpu"),
    )


def write_prediction(array, geometry, path):
    """Export D,H,W arrays with the saved Post1 physical geometry."""
    import SimpleITK as sitk

    volume = sitk.GetImageFromArray(array)
    volume.SetSpacing(tuple(float(v) for v in geometry["spacing"]))
    volume.SetOrigin(tuple(float(v) for v in geometry["origin"]))
    volume.SetDirection(tuple(float(v) for v in geometry["direction"]))
    sitk.WriteImage(volume, str(path))
