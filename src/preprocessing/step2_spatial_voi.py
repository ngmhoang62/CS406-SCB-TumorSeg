"""Step 2: Spatial Standardization (RAS, 1x1x1 mm^3 Isotropic Resampling) & 96x96x96 3D VOI Extraction."""

from typing import Dict, List, Tuple, Any
import numpy as np
import SimpleITK as sitk


def standardize_orientation_ras(image_sitk: sitk.Image, target_orientation: str = "RAS") -> sitk.Image:
    """Reorient a 3D SimpleITK image or mask to canonical RAS coordinate system."""
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(target_orientation)
    return orient_filter.Execute(image_sitk)


def resample_isotropic_1mm(
    image_sitk: sitk.Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    is_mask: bool = False,
) -> sitk.Image:
    """
    Resample 3D volume to 1.0 x 1.0 x 1.0 mm^3 isotropic voxel spacing.
    Uses Order-3 B-Spline for MRI intensity volumes and Nearest-Neighbor for binary masks.
    """
    orig_size = np.array(image_sitk.GetSize(), dtype=np.float64)
    orig_spacing = np.array(image_sitk.GetSpacing(), dtype=np.float64)
    new_spacing = np.array(target_spacing, dtype=np.float64)

    new_size = np.round(orig_size * (orig_spacing / new_spacing)).astype(int).tolist()
    new_size = [max(1, s) for s in new_size]

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    resample_filter.SetOutputSpacing(new_spacing.tolist())
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(image_sitk.GetDirection())
    resample_filter.SetOutputOrigin(image_sitk.GetOrigin())
    resample_filter.SetOutputPixelType(sitk.sitkUInt8 if is_mask else sitk.sitkFloat32)
    resample_filter.SetTransform(sitk.Transform())

    return resample_filter.Execute(image_sitk)


def compute_tumor_bbox_with_margin(
    mask_zyx: np.ndarray,
    margin_ratio: float = 0.25,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Compute 3D bounding box (z_min..z_max, y_min..y_max, x_min..x_max) of foreground mask
    expanded by `margin_ratio` (default 25% spatial margin per MAMA-MIA paper p.5).
    Note: SimpleITK GetArrayFromImage returns array in (Z, Y, X) = (D, H, W) order.
    """
    coords = np.argwhere(mask_zyx > 0)
    if coords.size == 0:
        # Fallback to volume center if mask is empty
        d, h, w = mask_zyx.shape
        return (d // 4, 3 * d // 4), (h // 4, 3 * h // 4), (w // 4, 3 * w // 4)

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1  # exclusive upper bound

    dz = max(1, z_max - z_min)
    dy = max(1, y_max - y_min)
    dx = max(1, x_max - x_min)

    mz = int(np.ceil(dz * margin_ratio))
    my = int(np.ceil(dy * margin_ratio))
    mx = int(np.ceil(dx * margin_ratio))

    d, h, w = mask_zyx.shape
    return (
        (max(0, int(z_min - mz)), min(d, int(z_max + mz))),
        (max(0, int(y_min - my)), min(h, int(y_max + my))),
        (max(0, int(x_min - mx)), min(w, int(x_max + mx))),
    )


def extract_centered_voi_96(
    volumes_zyx: List[np.ndarray],
    mask_zyx: np.ndarray,
    target_size: Tuple[int, int, int] = (96, 96, 96),
    margin_ratio: float = 0.25,
    pad_mode: str = "constant",
) -> Tuple[List[np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Extract high-resolution 96x96x96 3D VOI centered on the tumor bounding box (+25% margin).
    Preserves true 1.0 mm^3 isotropic geometry without rescaling distortion:
      - If the volume/VOI is smaller than 96^3 along any axis, applies symmetric padding.
      - Centers the 96x96x96 window at the tumor centroid.
    """
    bbox_z, bbox_y, bbox_x = compute_tumor_bbox_with_margin(mask_zyx, margin_ratio=margin_ratio)

    cz = (bbox_z[0] + bbox_z[1]) // 2
    cy = (bbox_y[0] + bbox_y[1]) // 2
    cx = (bbox_x[0] + bbox_x[1]) // 2

    td, th, tw = target_size
    d, h, w = mask_zyx.shape

    # Desired window in original volume coordinates: [start, end)
    z0, z1 = cz - td // 2, cz - td // 2 + td
    y0, y1 = cy - th // 2, cy - th // 2 + th
    x0, x1 = cx - tw // 2, cx - tw // 2 + tw

    # Compute required padding if window extends outside [0, d), [0, h), [0, w)
    pad_z0, pad_z1 = max(0, -z0), max(0, z1 - d)
    pad_y0, pad_y1 = max(0, -y0), max(0, y1 - h)
    pad_x0, pad_x1 = max(0, -x0), max(0, x1 - w)

    # Clamped slice indices inside the valid volume
    sz0, sz1 = max(0, z0), min(d, z1)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)

    pad_width = ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1))
    needs_padding = any(p > 0 for pair in pad_width for p in pair)

    cropped_volumes: List[np.ndarray] = []
    for vol in volumes_zyx:
        sub_vol = vol[sz0:sz1, sy0:sy1, sx0:sx1]
        if needs_padding:
            mode = pad_mode if (pad_mode == "reflect" and min(sub_vol.shape) > 1) else "constant"
            sub_vol = np.pad(sub_vol, pad_width=pad_width, mode=mode)
        assert sub_vol.shape == target_size, f"Expected {target_size}, got {sub_vol.shape}"
        cropped_volumes.append(sub_vol)

    sub_mask = mask_zyx[sz0:sz1, sy0:sy1, sx0:sx1]
    if needs_padding:
        # Binary mask is always zero-padded outside image bounds
        sub_mask = np.pad(sub_mask, pad_width=pad_width, mode="constant", constant_values=0)
    assert sub_mask.shape == target_size, f"Expected {target_size}, got {sub_mask.shape}"

    voi_info = {
        "center_zyx": (int(cz), int(cy), int(cx)),
        "bbox_with_margin_zyx": (bbox_z, bbox_y, bbox_x),
        "padded_voxels": pad_width,
        "resampled_shape_zyx": (int(d), int(h), int(w)),
    }
    return cropped_volumes, sub_mask, voi_info


def standardize_and_extract_voi(
    pre_sitk: sitk.Image,
    post1_sitk: sitk.Image,
    expert_mask_sitk: sitk.Image,
    auto_mask_sitk: sitk.Image | None = None,
    target_orientation: str = "RAS",
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    voi_size: Tuple[int, int, int] = (96, 96, 96),
    margin_ratio: float = 0.25,
    pad_mode: str = "constant",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Full Step 2 execution on a single case:
      1. Reorient Pre, Post1, Expert Mask, and Auto Mask to RAS.
      2. Resample to 1x1x1 mm^3 isotropic (B-Spline for MRI, Nearest-Neighbor for Masks).
      3. Extract 96x96x96 3D VOI around primary tumor (+25% margin).
    """
    pre_ras = standardize_orientation_ras(pre_sitk, target_orientation)
    post1_ras = standardize_orientation_ras(post1_sitk, target_orientation)
    expert_ras = standardize_orientation_ras(expert_mask_sitk, target_orientation)

    pre_iso = resample_isotropic_1mm(pre_ras, target_spacing, is_mask=False)
    post1_iso = resample_isotropic_1mm(post1_ras, target_spacing, is_mask=False)
    expert_iso = resample_isotropic_1mm(expert_ras, target_spacing, is_mask=True)

    pre_np = sitk.GetArrayFromImage(pre_iso).astype(np.float32)
    post1_np = sitk.GetArrayFromImage(post1_iso).astype(np.float32)
    expert_np = (sitk.GetArrayFromImage(expert_iso) > 0).astype(np.uint8)

    vols_to_crop = [pre_np, post1_np]
    if auto_mask_sitk is not None:
        auto_ras = standardize_orientation_ras(auto_mask_sitk, target_orientation)
        auto_iso = resample_isotropic_1mm(auto_ras, target_spacing, is_mask=True)
        auto_np = (sitk.GetArrayFromImage(auto_iso) > 0).astype(np.uint8)
        vols_to_crop.append(auto_np)

    cropped_vols, expert_voi, voi_info = extract_centered_voi_96(
        volumes_zyx=vols_to_crop,
        mask_zyx=expert_np,
        target_size=voi_size,
        margin_ratio=margin_ratio,
        pad_mode=pad_mode,
    )

    pre_voi = np.clip(cropped_vols[0], a_min=0.0, a_max=None).astype(np.float32)
    post1_voi = np.clip(cropped_vols[1], a_min=0.0, a_max=None).astype(np.float32)
    if auto_mask_sitk is not None:
        auto_voi = (cropped_vols[2] > 0).astype(np.uint8)
    else:
        auto_voi = expert_voi.copy()

    return pre_voi, post1_voi, expert_voi, auto_voi, voi_info
