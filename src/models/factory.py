"""Three baseline backbones using one fixed input configuration per run."""

from src.datasets.mama_mia import IP_CHANNELS
from .attention_unet3d import SCBAttentionUNet3D
from .nnunet3d import NNUNet3D
from .swin_unetr3d import SCBSwinUNETR3D

IP_MODE_TO_CHANNELS = {name: len(indices) for name, indices in IP_CHANNELS.items()}
MODEL_VARIANTS = ("baseline", "dual-add", "dual-gated", "boundary")


def resolve_in_channels(ip_mode=None, in_channels=None):
    if ip_mode is not None and ip_mode not in IP_MODE_TO_CHANNELS:
        raise ValueError(f"Unknown input mode {ip_mode!r}")
    expected = IP_MODE_TO_CHANNELS.get(ip_mode, 2)
    if in_channels is not None and in_channels not in (1, 2):
        raise ValueError("in_channels must be 1 or 2")
    if ip_mode is not None and in_channels is not None and in_channels != expected:
        raise ValueError("in_channels disagrees with ip_mode")
    return expected if in_channels is None else in_channels


def build_model(arch, in_channels=None, ip_mode=None, config=None, patch_size=(96, 96, 96),
                variant="baseline"):
    cfg = config or {}
    channels = resolve_in_channels(ip_mode, in_channels)
    key = arch.upper()
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant {variant!r}")
    if variant != "baseline" and key != "M1":
        raise ValueError("The improvement variants are implemented on M1")
    if key == "M1":
        fusion = {"dual-add": "add", "dual-gated": "gated"}.get(variant, "concat")
        return NNUNet3D(channels, cfg.get("nnunet_features", (32, 64, 128, 256, 320, 320)),
                        encoder_fusion=fusion, boundary_aux=variant == "boundary")
    if key == "M2":
        return SCBAttentionUNet3D(channels, cfg.get("attention_channels", (32, 64, 128, 256, 320)))
    if key == "M3":
        return SCBSwinUNETR3D(channels, cfg.get("swin_feature_size", 48),
                             cfg.get("swin_use_checkpoint", True), patch_size)
    raise ValueError("arch must be M1, M2 or M3")


def validate_patch_size(arch, patch_size, config=None):
    cfg = config or {}
    stages = {"M1": len(cfg.get("nnunet_features", [32, 64, 128, 256, 320, 320])),
              "M2": len(cfg.get("attention_channels", [32, 64, 128, 256, 320]))}
    divisor = 32 if arch == "M3" else 2 ** (stages[arch] - 1)
    if len(patch_size) != 3 or any(n <= 0 or n % divisor for n in patch_size):
        raise ValueError(f"{arch} patch dimensions must be positive multiples of {divisor}")
    if all(n == divisor for n in patch_size):
        raise ValueError(f"{arch} needs more than one spatial voxel at the bottleneck")
