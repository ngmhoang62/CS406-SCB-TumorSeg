"""Exercise the baseline on synthetic NIfTI data using the requested device."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader

from src.datasets import MAMA_MIA_Dataset
from src.engine.inference import predict_logits, write_prediction
from src.evaluation.metrics import segmentation_metrics
from src.losses import BoundaryAuxiliaryLoss, DiceBCELoss
from src.models import build_model, validate_patch_size
from src.preprocessing import CONTRACT_VERSION, preprocess_case, save_case
from src.utils.config import PROJECT_ROOT, resolve_device

SMOKE_MODEL_CONFIG = {
    "nnunet_features": [4, 8, 16, 24, 32, 32],
    "attention_channels": [4, 8, 16, 24, 32],
    "swin_feature_size": 12,
    "swin_use_checkpoint": False,
}


def run_smoke_test(requested_device=None, improvements=False):
    import SimpleITK as sitk
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    device = resolve_device(requested_device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    patch_size = (32, 32, 64)
    temporary_root = PROJECT_ROOT / "outputs" / "baseline_smoke"
    temporary_root.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="pipeline_", dir=temporary_root) as temporary:
        root = Path(temporary)
        z, y, x = np.indices((35, 35, 67), dtype=np.float32)
        tumor = np.exp(-((z - 17) ** 2 + (y - 17) ** 2 + (x - 33) ** 2) / 25)
        pre = 20 + z + 0.5 * y + 0.25 * x
        phases = [pre, pre + 20 * tumor - 0.2, pre + 30 * tumor + 5]
        phase_paths = []
        for index, phase in enumerate(phases):
            volume = sitk.GetImageFromArray(phase.astype(np.float32))
            volume.SetOrigin((12.0, -5.0, 3.0))
            path = root / f"CASE_{index:04d}.nii.gz"
            sitk.WriteImage(volume, str(path))
            phase_paths.append(path)
        mask = sitk.GetImageFromArray((tumor > 0.2).astype(np.uint8))
        mask.CopyInformation(volume)
        expert_path = root / "expert.nii.gz"
        sitk.WriteImage(mask, str(expert_path))
        case = preprocess_case(phase_paths, expert_path)
        without_mask = preprocess_case(phase_paths)
        np.testing.assert_array_equal(case["image"], without_mask["image"])
        rows = []
        for split in ("train", "val", "test"):
            case_id = f"synthetic_{split}"
            save_case(root / f"{case_id}.npz", case)
            rows.append({"patient_id": case_id, "split": split, "dataset": "synthetic"})
        metadata = root / "metadata.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
        experiments = [("M1", "IP1", "baseline"), ("M1", "IP2", "baseline"),
                       ("M1", "IP3", "baseline"), ("M2", "IP3", "baseline"),
                       ("M3", "IP3", "baseline")]
        if improvements:
            experiments += [("M1", "IP3", variant)
                            for variant in ("dual-add", "dual-gated", "boundary")]
        for arch, ip, variant in experiments:
            validate_patch_size(arch, patch_size, SMOKE_MODEL_CONFIG)
            dataset = MAMA_MIA_Dataset(
                metadata, root, "train", ip, patch_size=patch_size,
                augment=False, patches_per_case=1,
                boundary_aux=variant == "boundary",
            )
            batch = next(iter(DataLoader(dataset, batch_size=1)))
            model = build_model(arch, ip_mode=ip, config=SMOKE_MODEL_CONFIG,
                                patch_size=patch_size, variant=variant).to(device)
            if arch == "M1" and not isinstance(model.network, PlainConvUNet):
                raise TypeError("M1 must use the nnU-Net v2 PlainConvUNet implementation")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            image, target = batch["image"].to(device), batch["mask"].to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(image)
                logits = output["seg"] if isinstance(output, dict) else output
                if logits.shape != target.shape:
                    raise ValueError(f"{arch}/{ip} output shape differs from its training label")
                if variant == "boundary":
                    loss = BoundaryAuxiliaryLoss()(output, target, batch["boundary"].to(device))
                else:
                    loss = DiceBCELoss()(output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arch}/{ip} produced non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            validation = MAMA_MIA_Dataset(metadata, root, "val", ip)[0]
            model.eval()
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    logits = predict_logits(
                        model, validation["image"].unsqueeze(0), device, patch_size, 1, 0.5)
            probability = logits.float()[0, 0].sigmoid().numpy()
            if probability.shape != case["image"].shape[1:] or not np.isfinite(probability).all():
                raise ValueError(f"{arch}/{ip} full-volume prediction failed")
            prediction = (probability >= 0.5).astype(np.uint8)
            scores = segmentation_metrics(prediction, case["mask"][0], case["spacing"][::-1])
            path = root / f"{arch}_{ip}_mask.nii.gz"
            write_prediction(prediction, case, path)
            exported = sitk.ReadImage(str(path))
            np.testing.assert_allclose(exported.GetSpacing(), case["spacing"])
            np.testing.assert_allclose(exported.GetOrigin(), case["origin"])
            np.testing.assert_allclose(exported.GetDirection(), case["direction"])
            np.testing.assert_array_equal(sitk.GetArrayFromImage(exported), prediction)
            results.append({
                "model": arch, "ip": ip, "variant": variant, "train_loss": loss.item(),
                "validation_dice": scores["dice"], "full_shape": list(probability.shape),
            })
            print(f"[PASS] {arch}/{ip}/{variant} train, full-volume validation and NIfTI export", flush=True)
            del optimizer, model, image, target, output, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "device": str(device), "bf16": use_amp, "contract": CONTRACT_VERSION,
        "torch": str(torch.__version__), "results": results,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
