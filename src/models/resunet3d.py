"""
Step 5.2 (M1): 3D ResUNet (DynUNet Backbone) with Multi-Stage FiLM & 3-Level Deep Supervision.

Architecture Specifications (`docs/detailed_pipeline.md` Step 5.2 - M1):
  - 5-stage symmetric Encoder-Decoder with MONAI Residual 3D Conv blocks (`UnetResBlock`),
    `InstanceNorm3d`, `LeakyReLU(negative_slope=0.01)`.
  - Feature channels: `[32, 64, 128, 256, 320]`.
  - Multi-Stage FiLM (`use_film=True/False`) modulating 7 stages:
      * Last 2 Encoder stages: `enc2` (128 ch, 24^3), `enc3` (256 ch, 12^3)
      * Bottleneck: `bottleneck` (320 ch, 6^3)
      * All 4 Decoder stages: `dec3` (256 ch, 12^3), `dec2` (128 ch, 24^3),
                              `dec1` (64 ch, 48^3), `dec0` (32 ch, 96^3)
  - 3-Level Deep Supervision (`deep_supervision=True/False`):
      * k=0: P0 at full resolution (B, out_channels, D, H, W)     [96^3]
      * k=1: P1 at 1/2 resolution  (B, out_channels, D/2, H/2, W/2) [48^3]
      * k=2: P2 at 1/4 resolution  (B, out_channels, D/4, H/4, W/4) [24^3]
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetResBlock, UnetUpBlock

from src.models.film import MultiStageFiLM


def _make_res_stage(
    in_channels: int,
    out_channels: int,
    stride: int,
    num_blocks: int = 2,
) -> nn.Sequential:
    """Build a stack of MONAI 3D UnetResBlocks with InstanceNorm3d and LeakyReLU(0.01)."""
    norm_name = ("INSTANCE", {"affine": True})
    act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})
    blocks = [
        UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            norm_name=norm_name,
            act_name=act_name,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            UnetResBlock(
                spatial_dims=3,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
            )
        )
    return nn.Sequential(*blocks)


class SCBResUNet3D(nn.Module):
    """
    M1: 3D Residual U-Net (nnU-Net / DynUNet core) with optional Multi-Stage FiLM and Deep Supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256, 320),
        use_film: bool = False,
        num_subtypes: int = 4,
        film_embed_dim: int = 128,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"Expected 5 channel stages for SCBResUNet3D, got {channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.use_film = use_film
        self.deep_supervision = deep_supervision

        c0, c1, c2, c3, c4 = self.channels
        norm_name = ("INSTANCE", {"affine": True})
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})

        # 5-Stage Residual 3D CNN Encoder (96^3 -> 48^3 -> 24^3 -> 12^3 -> 6^3)
        self.enc0 = _make_res_stage(in_channels, c0, stride=1, num_blocks=2)
        self.enc1 = _make_res_stage(c0, c1, stride=2, num_blocks=2)
        self.enc2 = _make_res_stage(c1, c2, stride=2, num_blocks=2)
        self.enc3 = _make_res_stage(c2, c3, stride=2, num_blocks=2)
        self.bottleneck = _make_res_stage(c3, c4, stride=2, num_blocks=2)

        # 4-Stage Residual 3D Decoder (6^3 -> 12^3 -> 24^3 -> 48^3 -> 96^3)
        self.up3 = UnetUpBlock(
            spatial_dims=3,
            in_channels=c4,
            out_channels=c3,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
        )
        self.up2 = UnetUpBlock(
            spatial_dims=3,
            in_channels=c3,
            out_channels=c2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
        )
        self.up1 = UnetUpBlock(
            spatial_dims=3,
            in_channels=c2,
            out_channels=c1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
        )
        self.up0 = UnetUpBlock(
            spatial_dims=3,
            in_channels=c1,
            out_channels=c0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name,
        )

        # Multi-Stage FiLM Generator (7 stages: enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)
        self.film_stage_channels = [c2, c3, c4, c3, c2, c1, c0]
        if self.use_film:
            self.film_generator = MultiStageFiLM(
                stage_channels=self.film_stage_channels,
                num_subtypes=num_subtypes,
                embed_dim=film_embed_dim,
            )
        else:
            self.film_generator = None

        # Output Segmentation Heads (3-Level Deep Supervision: P0 @ 96^3, P1 @ 48^3, P2 @ 24^3)
        self.out_head0 = UnetOutBlock(spatial_dims=3, in_channels=c0, out_channels=out_channels)
        self.out_head1 = UnetOutBlock(spatial_dims=3, in_channels=c1, out_channels=out_channels)
        self.out_head2 = UnetOutBlock(spatial_dims=3, in_channels=c2, out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        subtype: torch.Tensor | None = None,
        return_deep_supervision: bool | None = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass of SCBResUNet3D.
        Args:
            x: Input 3D volume tensor of shape (B, in_channels, D, H, W).
            subtype: Subtype index tensor (B,) in {0..3} or one-hot tensor (B, 4).
                     Required when `use_film=True`.
            return_deep_supervision: Override whether to return `[P0, P1, P2]` or `P0`.
                                     Defaults to `self.training and self.deep_supervision`.
        """
        if self.use_film:
            if subtype is None:
                raise ValueError("`subtype` tensor must be provided when `use_film=True`.")
            gb_list: List[Tuple[torch.Tensor, torch.Tensor] | None] = self.film_generator.compute_gammas_betas(
                subtype, dtype=x.dtype
            )
        else:
            gb_list = [None] * 7

        # Encoder pathway
        f0 = self.enc0(x)                                        # (B, 32, D, H, W)
        f1 = self.enc1(f0)                                       # (B, 64, D/2, H/2, W/2)
        f2 = MultiStageFiLM.modulate(self.enc2(f1), gb_list[0])  # (B, 128, D/4, H/4, W/4)
        f3 = MultiStageFiLM.modulate(self.enc3(f2), gb_list[1])  # (B, 256, D/8, H/8, W/8)

        # Bottleneck
        b = MultiStageFiLM.modulate(self.bottleneck(f3), gb_list[2])  # (B, 320, D/16, H/16, W/16)

        # Decoder pathway with Multi-Stage FiLM conditioning
        d3 = MultiStageFiLM.modulate(self.up3(b, f3), gb_list[3])     # (B, 256, D/8, H/8, W/8)
        d2 = MultiStageFiLM.modulate(self.up2(d3, f2), gb_list[4])    # (B, 128, D/4, H/4, W/4)
        d1 = MultiStageFiLM.modulate(self.up1(d2, f1), gb_list[5])    # (B, 64, D/2, H/2, W/2)
        d0 = MultiStageFiLM.modulate(self.up0(d1, f0), gb_list[6])    # (B, 32, D, H, W)

        p0 = self.out_head0(d0)  # (B, 1, D, H, W)

        use_ds = (self.training and self.deep_supervision) if return_deep_supervision is None else return_deep_supervision
        if use_ds:
            p1 = self.out_head1(d1)  # (B, 1, D/2, H/2, W/2)
            p2 = self.out_head2(d2)  # (B, 1, D/4, H/4, W/4)
            return [p0, p1, p2]

        return p0
