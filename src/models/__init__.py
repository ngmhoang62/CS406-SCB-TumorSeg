"""Step 4 & Step 5: 3D Segmentation Architectures (`M1`, `M2`, `M3`), `MultiStageFiLM`, and Model Factory."""

from src.models.film import MultiStageFiLM
from src.models.resunet3d import SCBResUNet3D
from src.models.attention_unet3d import AdditiveAttentionGate3D, AttentionUpBlock3D, SCBAttentionUNet3D
from src.models.swin_unetr3d import SCBSwinUNETR3D
from src.models.factory import (
    IP_MODE_TO_CHANNELS,
    resolve_in_channels,
    build_model,
    compute_traditional_ftv_baseline,
)

__all__ = [
    "MultiStageFiLM",
    "SCBResUNet3D",
    "AdditiveAttentionGate3D",
    "AttentionUpBlock3D",
    "SCBAttentionUNet3D",
    "SCBSwinUNETR3D",
    "IP_MODE_TO_CHANNELS",
    "resolve_in_channels",
    "build_model",
    "compute_traditional_ftv_baseline",
]
