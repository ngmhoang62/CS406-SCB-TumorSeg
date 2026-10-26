"""
Step 5.2 (M2): 3D Attention U-Net with 3D Additive Spatial Attention Gates, Multi-Stage FiLM & Deep Supervision.

Architecture Specifications (`docs/detailed_pipeline.md` Step 5.2 - M2):
  - Residual 3D U-Net backbone (`[32, 64, 128, 256, 320]`) with `InstanceNorm3d` and `LeakyReLU(0.01)`.
  - 3D Additive Spatial Attention Gates integrated at all 4 skip connections (`enc3`, `enc2`, `enc1`, `enc0`)
    to suppress false-positive background parenchymal enhancement outside the tumor region.
  - Multi-Stage FiLM (`use_film=True/False`) modulating 7 stages (last 2 Encoder stages + Bottleneck + 4 Decoder stages).
  - 3-Level Deep Supervision (`[P0 @ 96^3, P1 @ 48^3, P2 @ 24^3]`).
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, UnetOutBlock, UnetResBlock

from src.models.film import MultiStageFiLM
from src.models.resunet3d import _make_res_stage


class AdditiveAttentionGate3D(nn.Module):
    """
    3D Additive Spatial Attention Gate (Oktay et al., Attention U-Net) adapted with InstanceNorm3d:
        alpha = Sigmoid(psi(ReLU(W_g(g) + W_x(x))))
        x_hat = alpha * x
    """

    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(self.relu(g1 + x1))
        return x * alpha


class AttentionUpBlock3D(nn.Module):
    """
    3D Decoder Upsampling Stage with Additive Spatial Attention Gate on the Skip Connection
    followed by a MONAI `UnetResBlock`.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        norm_name = ("INSTANCE", {"affine": True})
        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})

        self.transp_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.att_gate = AdditiveAttentionGate3D(
            F_g=out_channels,
            F_l=out_channels,
            F_int=max(out_channels // 2, 16),
        )
        self.conv_block = UnetResBlock(
            spatial_dims=3,
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.transp_conv(x)
        gated_skip = self.att_gate(g=g, x=skip)
        concat = torch.cat([g, gated_skip], dim=1)
        return self.conv_block(concat)


class SCBAttentionUNet3D(nn.Module):
    """
    M2: 3D Attention U-Net (Residual 3D U-Net + 3D Additive Spatial Attention Gates)
    with optional Multi-Stage FiLM and 3-Level Deep Supervision.
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
            raise ValueError(f"Expected 5 channel stages for SCBAttentionUNet3D, got {channels}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.use_film = use_film
        self.deep_supervision = deep_supervision

        c0, c1, c2, c3, c4 = self.channels

        # 5-Stage Residual 3D Encoder
        self.enc0 = _make_res_stage(in_channels, c0, stride=1, num_blocks=2)
        self.enc1 = _make_res_stage(c0, c1, stride=2, num_blocks=2)
        self.enc2 = _make_res_stage(c1, c2, stride=2, num_blocks=2)
        self.enc3 = _make_res_stage(c2, c3, stride=2, num_blocks=2)
        self.bottleneck = _make_res_stage(c3, c4, stride=2, num_blocks=2)

        # 4-Stage Attention-Gated 3D Decoder
        self.up3 = AttentionUpBlock3D(in_channels=c4, out_channels=c3)
        self.up2 = AttentionUpBlock3D(in_channels=c3, out_channels=c2)
        self.up1 = AttentionUpBlock3D(in_channels=c2, out_channels=c1)
        self.up0 = AttentionUpBlock3D(in_channels=c1, out_channels=c0)

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
        Forward pass of SCBAttentionUNet3D.
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

        # Attention-Gated Decoder pathway with Multi-Stage FiLM conditioning
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
