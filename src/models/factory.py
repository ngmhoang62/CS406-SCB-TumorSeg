"""
Step 5.1: Paper Baselines Helper (`B0.1 Traditional FTV Thresholding`) & Model Factory (`M1`, `M2`, `M3`).
"""

from typing import Any, Dict, Sequence
import numpy as np
from scipy import ndimage
import torch.nn as nn

from src.models.resunet3d import SCBResUNet3D
from src.models.attention_unet3d import SCBAttentionUNet3D
from src.models.swin_unetr3d import SCBSwinUNETR3D


IP_MODE_TO_CHANNELS: Dict[str, int] = {
    "IP1": 1,
    "IP1_post1": 1,
    "IP-1": 1,
    "IP2": 1,
    "IP2_sub": 1,
    "IP-2": 1,
    "IP3": 3,
    "IP3_multi": 3,
    "IP-3": 3,
}


def resolve_in_channels(ip_mode: str | None = None, in_channels: int | None = None) -> int:
    """Resolve number of input channels (1 for IP-1/IP-2, 3 for IP-3)."""
    if in_channels is not None:
        if in_channels not in (1, 3):
            raise ValueError(f"`in_channels` must be 1 or 3, got {in_channels}")
        return in_channels
    if ip_mode is not None:
        if ip_mode not in IP_MODE_TO_CHANNELS:
            raise ValueError(f"Unknown `ip_mode`: {ip_mode}. Expected one of {list(IP_MODE_TO_CHANNELS.keys())}")
        return IP_MODE_TO_CHANNELS[ip_mode]
    return 3


def build_model(
    arch: str,
    in_channels: int | None = None,
    ip_mode: str | None = None,
    use_film: bool = False,
    num_subtypes: int = 4,
    film_embed_dim: int = 128,
    deep_supervision: bool = True,
    resunet_channels: Sequence[int] = (32, 64, 128, 256, 320),
    swin_feature_size: int = 48,
    swin_use_checkpoint: bool = True,
) -> nn.Module:
    """
    Unified factory for building any of the 18 training configurations:
      - `arch`: 'M1' / 'resunet' | 'M2' / 'attention_unet' | 'M3' / 'swin_unetr'
      - `in_channels` or `ip_mode`: 1 channel ('IP-1', 'IP-2') or 3 channels ('IP-3')
      - `use_film`: False ('C0: Without FiLM') or True ('C1: With FiLM')
      - `deep_supervision`: True (returns [P0, P1, P2] in train mode, P0 in eval mode)
    """
    c_in = resolve_in_channels(ip_mode=ip_mode, in_channels=in_channels)
    arch_key = arch.strip().lower()

    if arch_key in ("m1", "resunet", "3d_resunet", "dynunet"):
        return SCBResUNet3D(
            in_channels=c_in,
            out_channels=1,
            channels=resunet_channels,
            use_film=use_film,
            num_subtypes=num_subtypes,
            film_embed_dim=film_embed_dim,
            deep_supervision=deep_supervision,
        )
    elif arch_key in ("m2", "attention_unet", "attunet", "3d_attention_unet"):
        return SCBAttentionUNet3D(
            in_channels=c_in,
            out_channels=1,
            channels=resunet_channels,
            use_film=use_film,
            num_subtypes=num_subtypes,
            film_embed_dim=film_embed_dim,
            deep_supervision=deep_supervision,
        )
    elif arch_key in ("m3", "swin_unetr", "swinunetr", "3d_swin_unetr"):
        return SCBSwinUNETR3D(
            in_channels=c_in,
            out_channels=1,
            feature_size=swin_feature_size,
            use_checkpoint=swin_use_checkpoint,
            use_film=use_film,
            num_subtypes=num_subtypes,
            film_embed_dim=film_embed_dim,
            deep_supervision=deep_supervision,
        )
    else:
        raise ValueError(
            f"Unknown model architecture '{arch}'. Supported: "
            "['M1'/'resunet', 'M2'/'attention_unet', 'M3'/'swin_unetr']."
        )


def compute_traditional_ftv_baseline(
    c2_sub_zscore: np.ndarray,
    c3_pe_zscore: np.ndarray,
    pe_z_threshold: float = 1.0,
    sub_z_threshold: float = 1.0,
    apply_morphology: bool = True,
) -> np.ndarray:
    """
    Baseline B0.1 — Traditional Functional Tumor Volume (FTV) intensity thresholding
    on Subtraction (C2) and Percent Enhancement (C3) maps, with optional 3D morphological cleanup.
    """
    raw_mask = (c2_sub_zscore >= sub_z_threshold) & (c3_pe_zscore >= pe_z_threshold)
    if not apply_morphology or not np.any(raw_mask):
        return raw_mask.astype(np.uint8)

    # Keep largest connected component and fill internal necrotic holes
    struct = ndimage.generate_binary_structure(3, 3)
    labeled, num_features = ndimage.label(raw_mask, structure=struct)
    if num_features > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        cleaned = labeled == largest_label
        cleaned = ndimage.binary_closing(cleaned, structure=ndimage.iterate_structure(struct, 2))
        cleaned = ndimage.binary_fill_holes(cleaned)
        return cleaned.astype(np.uint8)
    return raw_mask.astype(np.uint8)
