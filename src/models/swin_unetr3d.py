"""
Step 5.2 (M3): 3D Swin UNETR with Multi-Stage FiLM & 3-Level Deep Supervision.

Architecture Specifications (`docs/detailed_pipeline.md` Step 5.2 - M3):
  - 3D Shifted-Window Vision Transformer (`SwinTransformer`) Encoder:
      * `feature_size = 48`
      * `depths = (2, 2, 2, 2)`
      * `num_heads = (3, 6, 12, 24)`
      * `use_checkpoint = True` (optional gradient checkpointing)
  - CNN Residual Decoder (`UnetrBasicBlock` & `UnetrUpBlock` with `InstanceNorm3d`).
  - Multi-Stage FiLM (`use_film=True/False`) modulating 7 stages:
      * Last 2 Encoder skip stages: `enc2` (96 ch, 24^3), `enc3` (192 ch, 12^3)
      * Bottleneck: `dec4` (768 ch, 3^3)
      * 4 Decoder stages: `d3` (192 ch, 12^3), `d2` (96 ch, 24^3),
                          `d1` (48 ch, 48^3), `d0` (48 ch, 96^3)
  - 3-Level Deep Supervision (`[P0 @ 96^3, P1 @ 48^3, P2 @ 24^3]`).
"""

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer

from src.models.film import MultiStageFiLM


class SCBSwinUNETR3D(nn.Module):
    """
    M3: 3D Swin UNETR with optional Multi-Stage FiLM and 3-Level Deep Supervision.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = True,
        use_film: bool = False,
        num_subtypes: int = 4,
        film_embed_dim: int = 128,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if feature_size % 12 != 0:
            raise ValueError(f"`feature_size` must be divisible by 12, got {feature_size}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.normalize = normalize
        self.use_film = use_film
        self.deep_supervision = deep_supervision

        window_size = (7, 7, 7)
        patch_size = (2, 2, 2)
        norm_name = "instance"

        # 1. 3D Shifted-Window Transformer Encoder (MONAI SwinTransformer)
        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

        c = feature_size  # 48

        # 2. Encoder Residual Projection Blocks (matching MONAI SwinUNETR)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=c,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=2 * c,
            out_channels=2 * c,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=4 * c,
            out_channels=4 * c,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=16 * c,
            out_channels=16 * c,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        # 3. CNN Residual Decoder Stages (3^3 -> 6^3 -> 12^3 -> 24^3 -> 48^3 -> 96^3)
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=16 * c,
            out_channels=8 * c,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=8 * c,
            out_channels=4 * c,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=4 * c,
            out_channels=2 * c,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=2 * c,
            out_channels=c,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=c,
            out_channels=c,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        # 4. Multi-Stage FiLM Generator (7 stages: enc2, enc3, bottleneck, dec3, dec2, dec1, dec0)
        self.film_stage_channels = [2 * c, 4 * c, 16 * c, 4 * c, 2 * c, c, c]
        if self.use_film:
            self.film_generator = MultiStageFiLM(
                stage_channels=self.film_stage_channels,
                num_subtypes=num_subtypes,
                embed_dim=film_embed_dim,
            )
        else:
            self.film_generator = None

        # 5. Output Segmentation Heads (3-Level Deep Supervision: P0 @ 96^3, P1 @ 48^3, P2 @ 24^3)
        self.out_head0 = UnetOutBlock(spatial_dims=3, in_channels=c, out_channels=out_channels)
        self.out_head1 = UnetOutBlock(spatial_dims=3, in_channels=c, out_channels=out_channels)
        self.out_head2 = UnetOutBlock(spatial_dims=3, in_channels=2 * c, out_channels=out_channels)

    def forward(
        self,
        x: torch.Tensor,
        subtype: torch.Tensor | None = None,
        return_deep_supervision: bool | None = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        Forward pass of SCBSwinUNETR3D.
        Note: Spatial dimensions (D, H, W) must be divisible by 32 (e.g., 96x96x96).
        """
        if self.use_film:
            if subtype is None:
                raise ValueError("`subtype` tensor must be provided when `use_film=True`.")
            gb_list: List[Tuple[torch.Tensor, torch.Tensor] | None] = self.film_generator.compute_gammas_betas(
                subtype, dtype=x.dtype
            )
        else:
            gb_list = [None] * 7

        hidden_states_out = self.swinViT(x, self.normalize)

        enc0 = self.encoder1(x)                                                        # (B, 48, 96, 96, 96)
        enc1 = self.encoder2(hidden_states_out[0])                                     # (B, 48, 48, 48, 48)
        enc2 = MultiStageFiLM.modulate(self.encoder3(hidden_states_out[1]), gb_list[0])  # (B, 96, 24, 24, 24)
        enc3 = MultiStageFiLM.modulate(self.encoder4(hidden_states_out[2]), gb_list[1])  # (B, 192, 12, 12, 12)

        # Bottleneck (3^3 for 96^3 input)
        dec4 = MultiStageFiLM.modulate(self.encoder10(hidden_states_out[4]), gb_list[2])  # (B, 768, 3, 3, 3)

        # Decoder pathway
        d4 = self.decoder5(dec4, hidden_states_out[3])                                 # (B, 384, 6, 6, 6)
        d3 = MultiStageFiLM.modulate(self.decoder4(d4, enc3), gb_list[3])              # (B, 192, 12, 12, 12)
        d2 = MultiStageFiLM.modulate(self.decoder3(d3, enc2), gb_list[4])              # (B, 96, 24, 24, 24)
        d1 = MultiStageFiLM.modulate(self.decoder2(d2, enc1), gb_list[5])              # (B, 48, 48, 48, 48)
        d0 = MultiStageFiLM.modulate(self.decoder1(d1, enc0), gb_list[6])              # (B, 48, 96, 96, 96)

        p0 = self.out_head0(d0)  # (B, 1, 96, 96, 96)

        use_ds = (self.training and self.deep_supervision) if return_deep_supervision is None else return_deep_supervision
        if use_ds:
            p1 = self.out_head1(d1)  # (B, 1, 48, 48, 48)
            p2 = self.out_head2(d2)  # (B, 1, 24, 24, 24)
            return [p0, p1, p2]

        return p0
