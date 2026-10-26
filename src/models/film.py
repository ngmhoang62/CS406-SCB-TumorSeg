"""
Step 4: Multi-Stage Feature-wise Linear Modulation (FiLM) for Subtype-Conditioned 3D Segmentation.

Mathematical formulation (see `docs/detailed_pipeline.md` Step 4):
  1. Subtype representation:
     s in {0, 1}^4 (Luminal, Triple Negative, HER2-enriched, HER2-pure)
  2. Shared Embedding MLP:
     e = SiLU(LayerNorm(W_1 s + b_1)) in R^128
  3. Stage-specific Linear Projection Heads (for each modulated stage l in {1..L}):
     [gamma^(l), beta^(l)] = W_2^(l) e + b_2^(l) in R^(2 * C_l)
     Initialized with Zero-Init: W_2^(l) = 0, b_2^(l) = 0 so gamma^(l) = 0, beta^(l) = 0 at epoch 0.
  4. Residual 3D Feature Modulation:
     F_tilde^(l)_{b, c, d, h, w} = (1 + gamma^(l)_{b, c}) * F^(l)_{b, c, d, h, w} + beta^(l)_{b, c}
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStageFiLM(nn.Module):
    """
    Multi-Stage FiLM Generator that modulates multiple 3D feature stages
    (last 2 Encoder stages, Bottleneck, and 4 Decoder stages) from a shared subtype embedding.
    """

    def __init__(
        self,
        stage_channels: List[int] | Tuple[int, ...],
        num_subtypes: int = 4,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        self.stage_channels = list(stage_channels)
        self.num_subtypes = num_subtypes
        self.embed_dim = embed_dim

        # Shared Embedding MLP: s (4) -> Linear(4, 128) -> LayerNorm(128) -> SiLU
        self.shared_mlp = nn.Sequential(
            nn.Linear(num_subtypes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(inplace=True),
        )

        # Stage-specific Linear Projection Heads: e (128) -> [gamma^(l), beta^(l)] (2 * C_l)
        self.heads = nn.ModuleList([nn.Linear(embed_dim, 2 * c) for c in self.stage_channels])

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming initialization for shared MLP and strict Zero-Initialization for all stage heads
        so that at epoch 0: gamma^(l) = 0 and beta^(l) = 0 (exact identity mapping).
        """
        nn.init.kaiming_normal_(self.shared_mlp[0].weight, nonlinearity="relu")
        if self.shared_mlp[0].bias is not None:
            nn.init.zeros_(self.shared_mlp[0].bias)

        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def encode_subtype_onehot(self, subtype: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert `subtype` tensor into a 2D float one-hot tensor of shape (B, num_subtypes).
        Accepts either:
          - 1D integer class indices of shape (B,) or (B, 1) in {0..num_subtypes-1}
          - 2D one-hot / soft probability tensor of shape (B, num_subtypes)
        """
        if subtype.dim() == 2 and subtype.size(-1) == self.num_subtypes:
            return subtype.to(dtype=dtype)

        if subtype.dim() == 2 and subtype.size(-1) == 1:
            subtype = subtype.squeeze(-1)

        if subtype.dim() == 1:
            return F.one_hot(subtype.long(), num_classes=self.num_subtypes).to(dtype=dtype)

        raise ValueError(
            f"Invalid subtype shape {tuple(subtype.shape)}. "
            f"Expected (B,) class indices or (B, {self.num_subtypes}) one-hot vector."
        )

    def compute_gammas_betas(
        self,
        subtype: torch.Tensor,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute list of (gamma^(l), beta^(l)) broadcastable to 3D feature maps (B, C_l, 1, 1, 1).
        """
        s_onehot = self.encode_subtype_onehot(subtype, dtype=dtype)
        embed = self.shared_mlp(s_onehot)  # (B, embed_dim)

        params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for head, c_l in zip(self.heads, self.stage_channels):
            gb = head(embed)  # (B, 2 * C_l)
            gamma, beta = torch.split(gb, c_l, dim=-1)
            params.append((gamma.view(-1, c_l, 1, 1, 1), beta.view(-1, c_l, 1, 1, 1)))
        return params

    @staticmethod
    def modulate(
        feature_map: torch.Tensor,
        gamma_beta: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        Apply Residual FiLM modulation on a 3D feature map:
            F_tilde = (1 + gamma) * F + beta
        """
        if gamma_beta is None:
            return feature_map
        gamma, beta = gamma_beta
        return (1.0 + gamma.to(feature_map.dtype)) * feature_map + beta.to(feature_map.dtype)
