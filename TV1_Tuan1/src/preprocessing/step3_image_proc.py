"""Step 3: Multi-Representation Image Processing Pipeline (Denoising, Clipping, 3-Channel Synthesis)."""

from typing import Tuple
import numpy as np
import SimpleITK as sitk
from skimage import exposure, restoration


def apply_n4_bias_correction_3d(
    volume_zyx: np.ndarray,
    shrink_factor: int = 2,
) -> np.ndarray:
    """
    Optional 3D N4ITK Bias Field Correction on MRI VOI.
    Uses shrink_factor >= 2 and Otsu mask for fast execution when enabled.
    """
    raw_arr = np.maximum(volume_zyx.astype(np.float32), 1e-4)
    img_sitk = sitk.GetImageFromArray(raw_arr)
    mask_sitk = sitk.OtsuThreshold(img_sitk, 0, 1)

    if shrink_factor > 1:
        shrunk_img = sitk.Shrink(img_sitk, [shrink_factor] * 3)
        shrunk_mask = sitk.Shrink(mask_sitk, [shrink_factor] * 3)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([20, 20, 10])
        corrector.Execute(shrunk_img, shrunk_mask)
        log_bias = corrector.GetLogBiasFieldAsImage(img_sitk)
        corrected_sitk = img_sitk / sitk.Exp(log_bias)
    else:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([20, 20, 10])
        corrected_sitk = corrector.Execute(img_sitk, mask_sitk)

    return sitk.GetArrayFromImage(corrected_sitk).astype(np.float32)


def edge_preserving_denoise_3d(
    volume_zyx: np.ndarray,
    method: str = "nlmeans",
    patch_size: int = 3,
    patch_distance: int = 5,
    h_factor: float = 0.8,
) -> np.ndarray:
    """
    3D Edge-Preserving Denoising to smooth magnetic field thermal noise while preserving
    spiculated/smooth tumor margins (CS406 Chapter 2).
    """
    vol = volume_zyx.astype(np.float32)
    if method == "nlmeans":
        # Fast robust MAD noise estimator along spatial differences (equivalent to Donoho's wavelet MAD)
        diff_x = np.diff(vol, axis=-1)
        sigma_est = float(np.median(np.abs(diff_x)) / (0.6745 * np.sqrt(2.0)))
        sigma_est = max(sigma_est, 1e-4)
        h_val = max(h_factor * sigma_est, 1e-5)
        denoised = restoration.denoise_nl_means(
            vol,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h_val,
            sigma=sigma_est,
            fast_mode=True,
        )
        return denoised.astype(np.float32)
    elif method == "curvature_flow":
        img_sitk = sitk.GetImageFromArray(vol)
        cf_filter = sitk.CurvatureAnisotropicDiffusionImageFilter()
        cf_filter.SetTimeStep(0.03)
        cf_filter.SetNumberOfIterations(5)
        cf_filter.SetConductanceParameter(3.0)
        out_sitk = cf_filter.Execute(img_sitk)
        return sitk.GetArrayFromImage(out_sitk).astype(np.float32)
    else:
        raise ValueError(f"Unknown denoise_method: {method}")


def percentile_clip_3d(
    volume_zyx: np.ndarray,
    percentiles: Tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """
    Clip extreme outlier intensities (e.g., metallic surgical biopsy clips) to [P_0.5, P_99.5].
    Computes percentiles over non-zero foreground voxels when available.
    """
    fg_voxels = volume_zyx[volume_zyx > 0]
    if fg_voxels.size < 10:
        fg_voxels = volume_zyx.ravel()

    p_low, p_high = np.percentile(fg_voxels, percentiles)
    if p_high <= p_low:
        return volume_zyx.astype(np.float32)
    return np.clip(volume_zyx, p_low, p_high).astype(np.float32)


def zscore_normalize_3d(volume_zyx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize 3D volume to zero mean and unit variance."""
    mean_val = float(np.mean(volume_zyx))
    std_val = float(np.std(volume_zyx))
    return ((volume_zyx - mean_val) / (std_val + eps)).astype(np.float32)


def compute_3d_clahe(
    volume_zyx: np.ndarray,
    clip_limit: float = 0.03,
    kernel_size: int = 16,
) -> np.ndarray:
    """
    Apply 3D Contrast Limited Adaptive Histogram Equalization (3D CLAHE)
    to enhance subtle peripheral enhancement and Non-Mass Enhancement (NME).
    """
    v_min = float(np.min(volume_zyx))
    v_max = float(np.max(volume_zyx))
    if v_max - v_min < 1e-8:
        return np.zeros_like(volume_zyx, dtype=np.float32)

    vol_norm01 = (volume_zyx - v_min) / (v_max - v_min)
    clahe_vol = exposure.equalize_adapthist(
        vol_norm01,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
    )
    return clahe_vol.astype(np.float32)


def synthesize_three_channels(
    pre_voi: np.ndarray,
    post1_voi: np.ndarray,
    apply_n4: bool = False,
    n4_shrink_factor: int = 2,
    apply_denoise: bool = True,
    denoise_method: str = "nlmeans",
    nlmeans_patch_size: int = 3,
    nlmeans_patch_distance: int = 5,
    nlmeans_h_factor: float = 0.8,
    clip_percentiles: Tuple[float, float] = (0.5, 99.5),
    pe_tau_percentile: float = 5.0,
    clahe_clip_limit: float = 0.03,
    clahe_kernel_size: int = 16,
    zscore_eps: float = 1e-8,
    output_dtype: str = "float16",
) -> np.ndarray:
    """
    Execute Step 3.1 & Step 3.2 to construct the 3-channel tensor X in R^(3 x 96 x 96 x 96):
      - Channel C1 (I_post1): Z-score normalized post-contrast image
      - Channel C2 (I_sub):   Z-score normalized subtraction image max(0, I_post1 - I_pre)
      - Channel C3 (I_PE):    Percent Enhancement map + 3D CLAHE + Z-score normalization
    """
    i_pre = pre_voi.astype(np.float32)
    i_post1 = post1_voi.astype(np.float32)

    # 1. Optional N4 Bias Field Correction
    if apply_n4:
        i_pre = apply_n4_bias_correction_3d(i_pre, shrink_factor=n4_shrink_factor)
        i_post1 = apply_n4_bias_correction_3d(i_post1, shrink_factor=n4_shrink_factor)

    # 2. 3D Edge-Preserving Denoising
    if apply_denoise:
        i_pre = edge_preserving_denoise_3d(
            i_pre,
            method=denoise_method,
            patch_size=nlmeans_patch_size,
            patch_distance=nlmeans_patch_distance,
            h_factor=nlmeans_h_factor,
        )
        i_post1 = edge_preserving_denoise_3d(
            i_post1,
            method=denoise_method,
            patch_size=nlmeans_patch_size,
            patch_distance=nlmeans_patch_distance,
            h_factor=nlmeans_h_factor,
        )

    # 3. Percentile Intensity Clipping [0.5%, 99.5%]
    i_pre = percentile_clip_3d(i_pre, percentiles=clip_percentiles)
    i_post1 = percentile_clip_3d(i_post1, percentiles=clip_percentiles)

    # 4. Channel C1: Normalized First Post-Contrast (I_post1)
    c1_post1 = zscore_normalize_3d(i_post1, eps=zscore_eps)

    # 5. Channel C2: Subtraction MRI (I_sub = max(0, I_post1 - I_pre))
    i_sub_raw = np.maximum(0.0, i_post1 - i_pre)
    c2_sub = zscore_normalize_3d(i_sub_raw, eps=zscore_eps)

    # 6. Channel C3: Percent Enhancement (I_PE) + 3D CLAHE + Z-score
    fg_pre = i_pre[i_pre > 0]
    tau = float(np.percentile(fg_pre, pe_tau_percentile)) if fg_pre.size > 0 else 1.0
    tau = max(tau, 1.0)
    i_pe_raw = i_sub_raw / (i_pre + tau)
    i_pe_clahe = compute_3d_clahe(
        i_pe_raw,
        clip_limit=clahe_clip_limit,
        kernel_size=clahe_kernel_size,
    )
    c3_pe = zscore_normalize_3d(i_pe_clahe, eps=zscore_eps)

    stacked = np.stack([c1_post1, c2_sub, c3_pe], axis=0)
    return stacked.astype(np.float16 if output_dtype == "float16" else np.float32)
