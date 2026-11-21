"""Step 3: Multi-Representation Image Processing Pipeline (Denoising, Clipping, 3-Channel Synthesis)."""

from typing import Tuple
import numpy as np
from skimage import exposure, restoration


def apply_n4_bias_correction_3d(
    volume_zyx: np.ndarray,
    shrink_factor: int = 2,
) -> np.ndarray:
    """
    Optional 3D N4ITK Bias Field Correction on MRI VOI.
    Uses shrink_factor >= 2 and Otsu mask for fast execution when enabled.
    """
    import SimpleITK as sitk

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
        import SimpleITK as sitk

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


def compute_cehs_c3(
    c1: np.ndarray,
    c2: np.ndarray,
    sigma: float = 1.0,
    thresh: float = 0.05,
    alpha: float = 3.5,
    power: float = 1.4,
) -> np.ndarray:
    """
    Compute Core-Enhanced Hybrid Subtraction (CEHS) Channel 3:
      1. Uses subtraction contrast map (C2) to delineate active tumor tissue and background.
      2. Generates 3D Gaussian envelope (sigma=1.0) with tight threshold gating (thresh=0.05)
         to confine support strictly inside the tumor and eliminate spillover into background/vessels.
      3. Enhances low-perfusion necrotic/fibrotic core with anatomical tissue density from C1.
      4. Z-score standardizes output. Eliminates 100% of CLAHE 16x16 block artifacts.
    """
    from scipy.ndimage import gaussian_filter

    c1_f = c1.astype(np.float32)
    c2_f = c2.astype(np.float32)

    c2_min = float(c2_f.min())
    e = np.maximum(0.0, c2_f - c2_min)
    e_norm = e / (e.max() + 1e-6)

    c1_pos = np.maximum(0.0, c1_f - float(c1_f.min()))
    c1_pos = c1_pos / (c1_pos.max() + 1e-6)

    env = gaussian_filter(e_norm, sigma=sigma)
    env = env / (env.max() + 1e-6)

    gate = np.clip((env - thresh) / (1.0 - thresh) * 2.5, 0.0, 1.0)
    core_boost = np.power(c1_pos, power) * gate

    c3 = c2_f + alpha * core_boost
    return ((c3 - c3.mean()) / (c3.std() + 1e-6)).astype(np.float32)


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
    if pre_voi.ndim != 3 or pre_voi.shape != post1_voi.shape:
        raise ValueError("Pre and Post1 VOI must have the same 3D shape")
    if not np.isfinite(pre_voi).all() or not np.isfinite(post1_voi).all():
        raise ValueError("Pre and Post1 VOI must contain only finite values")
    if output_dtype not in {"float16", "float32"}:
        raise ValueError("output_dtype must be float16 or float32")

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

    # 6. Channel C3: Core-Enhanced Hybrid Subtraction (CEHS)
    #    Combines clean dark background of C2 with dense core illumination of C1,
    #    completely eliminating 16x16 CLAHE checkerboard noise.
    c3_cehs = compute_cehs_c3(c1_post1, c2_sub, sigma=1.0, thresh=0.05, alpha=3.5, power=1.4)

    stacked = np.stack([c1_post1, c2_sub, c3_cehs], axis=0)
    return stacked.astype(np.float16 if output_dtype == "float16" else np.float32)
