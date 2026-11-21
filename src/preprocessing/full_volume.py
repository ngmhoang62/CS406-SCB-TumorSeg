"""Full Post1 field of view, shared per-case normalization and two channels."""

from pathlib import Path

import numpy as np

CONTRACT_VERSION = "full_volume_v1"


def shared_zscore_channels(phases, eps=1e-8, output_dtype="float32"):
    """Pool all voxels of every aligned phase for one population mean/std.

    Phase 0 is Pre and phase 1 is Post1. Later phases contribute only statistics.
    The signed subtraction equals (Post1 - Pre) / shared_std.
    """
    if len(phases) < 2 or eps <= 0:
        raise ValueError("Provide Pre, Post1 and a positive normalization epsilon")
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("output_dtype must be float16 or float32")
    shape = phases[0].shape
    count, mean, m2 = 0, 0.0, 0.0
    for phase in phases:
        if phase.ndim != 3 or phase.shape != shape or not phase.size or not np.isfinite(phase).all():
            raise ValueError("All phases must be finite, aligned 3D arrays")
        # Combine per-phase moments without concatenating full scans in RAM.
        phase_mean = float(np.mean(phase, dtype=np.float64))
        phase_var = float(np.var(phase, dtype=np.float64))
        new_count = count + phase.size
        delta = phase_mean - mean
        m2 += phase_var * phase.size + delta * delta * count * phase.size / new_count
        mean += delta * phase.size / new_count
        count = new_count
    std = max(float(np.sqrt(m2 / count)), eps)
    pre = (phases[0].astype(np.float32) - mean) / std
    post1 = (phases[1].astype(np.float32) - mean) / std
    image = np.stack([post1, post1 - pre]).astype(output_dtype)
    if not np.isfinite(image).all():
        raise ValueError("Normalized image overflowed the requested output dtype")
    return image, {"mean": mean, "std": std, "num_phases": len(phases)}


def preprocess_case(phase_paths, expert_path=None, spacing=(1.0, 1.0, 1.0),
                    eps=1e-8, output_dtype="float32"):
    """Reference geometry depends exclusively on the full Post1 image.

    Images use linear interpolation. The optional expert mask is resampled with
    nearest neighbour and never defines a crop or reference grid.
    """
    import SimpleITK as sitk

    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("spacing must contain three positive finite values")
    phases_by_id = {int(Path(p).name.removesuffix(".nii.gz").rsplit("_", 1)[1]): Path(p)
                    for p in phase_paths}
    if (len(phases_by_id) != len(phase_paths) or not {0, 1}.issubset(phases_by_id)
            or min(phases_by_id) < 0):
        raise ValueError("Expected unique phases including _0000 and _0001")
    post1 = sitk.DICOMOrient(sitk.ReadImage(str(phases_by_id[1]), sitk.sitkFloat32), "RAS")
    # Ceil the distance between voxel centres to cover the entire Post1 extent.
    size = [int(np.ceil((n - 1) * old / new)) + 1
            for n, old, new in zip(post1.GetSize(), post1.GetSpacing(), spacing)]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing(tuple(float(s) for s in spacing))
    reference.SetDirection(post1.GetDirection())
    reference.SetOrigin(post1.GetOrigin())
    arrays = []
    phase_ids = sorted(phases_by_id)
    for phase_id in phase_ids:
        volume = sitk.DICOMOrient(
            sitk.ReadImage(str(phases_by_id[phase_id]), sitk.sitkFloat32), "RAS")
        aligned = sitk.Resample(volume, reference, sitk.Transform(), sitk.sitkLinear,
                                0.0, sitk.sitkFloat32)
        arrays.append(sitk.GetArrayFromImage(aligned))
    image, statistics = shared_zscore_channels(arrays, eps, output_dtype)
    result = {
        "contract_version": np.array(CONTRACT_VERSION),
        "image": image,
        "spacing": np.array(reference.GetSpacing()),
        "origin": np.array(reference.GetOrigin()),
        "direction": np.array(reference.GetDirection()),
        "normalization_mean": np.float64(statistics["mean"]),
        "normalization_std": np.float64(statistics["std"]),
        "phase_ids": np.array(phase_ids, dtype=np.int64),
    }
    if expert_path is not None:
        mask = sitk.Cast(sitk.ReadImage(str(expert_path)) > 0, sitk.sitkUInt8)
        aligned = sitk.Resample(mask, reference, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        result["mask"] = sitk.GetArrayFromImage(aligned)[None]
    return result


def save_case(path, data):
    """Replace a complete case atomically, allowing preprocessing retries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
