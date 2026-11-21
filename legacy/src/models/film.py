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
        subtype_dropout_prob: float = 0.15,
        max_modulation: float = 0.25,
    ) -> None:
        super().__init__()
        self.stage_channels = list(stage_channels)
        self.num_subtypes = num_subtypes
        self.embed_dim = embed_dim
        self.subtype_dropout_prob = subtype_dropout_prob
        self.max_modulation = max_modulation

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
        Orthogonal initialization for shared MLP (maximally separating the 4 subtype rays into a
        regular simplex with pairwise cosine -1/3 after centering) and strict Zero-Initialization
        for all stage heads so that at epoch 0: gamma^(l) = 0 and beta^(l) = 0 (exact identity).
        """
        nn.init.orthogonal_(self.shared_mlp[0].weight)
        if self.shared_mlp[0].bias is not None:
            nn.init.zeros_(self.shared_mlp[0].bias)

        for head in self.heads:
            nn.init.zeros_(head.weight)
            if head.bias is not None:
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
        dtype: torch.dtype = torch.float32,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute list of (gamma^(l), beta^(l)) broadcastable to 3D feature maps (B, C_l, 1, 1, 1).
        Key Mathematical Guarantees:
          1. Zero-Mean Subtype Centering across all K=4 subtypes:
             sum_{k=0}^{K-1} gamma^(l)(k) = 0, sum_{k=0}^{K-1} beta^(l)(k) = 0.
             Eliminates 100% of the Global Bias Shortcut so FiLM cannot starve the 3D conv
             backbone or collapse all 4 subtype embeddings into a constant vector.
          2. True-Zero Subtype Conditioning Dropout (p=0.15 during training):
             Dropped samples receive exact zero modulation (gamma=0, beta=0), training the
             3D backbone to remain self-sufficient on unconditioned features.
          3. Smooth Bounded Tanh Modulation with Re-Centering:
             Prevents channel saturation while preserving exact zero-sum contrast across subtypes.
        """
        s_onehot = self.encode_subtype_onehot(subtype, dtype=dtype)
        dev = s_onehot.device

        # 1. Compute all K=4 canonical subtype embeddings and zero-center across subtypes
        eye = torch.eye(self.num_subtypes, device=dev, dtype=dtype)  # (K, K)
        all_embeds = self.shared_mlp(eye)  # (K, embed_dim)
        centered_embeds = all_embeds - all_embeds.mean(dim=0, keepdim=True)  # (K, embed_dim)

        # 2. Classifier-free subtype dropout mask (post-projection true-zero identity when dropped)
        if self.training and self.subtype_dropout_prob > 0.0:
            keep_mask = (torch.rand(s_onehot.size(0), 1, device=dev) > self.subtype_dropout_prob).to(dtype=dtype)
            s_onehot = s_onehot * keep_mask
        else:
            keep_mask = (s_onehot.abs().sum(dim=-1, keepdim=True) > 0).to(dtype=dtype)

        params: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for head, c_l in zip(self.heads, self.stage_channels):
            gb_all = head(centered_embeds)  # (K, 2 * C_l)
            gb_all = gb_all - gb_all.mean(dim=0, keepdim=True)
            if self.max_modulation > 0.0:
                gb_all = self.max_modulation * torch.tanh(gb_all)
                gb_all = gb_all - gb_all.mean(dim=0, keepdim=True)
            gb = (s_onehot @ gb_all) * keep_mask  # (B, 2 * C_l)
            gamma, beta = torch.split(gb, c_l, dim=-1)
            params.append((gamma.view(-1, c_l, 1, 1, 1), beta.view(-1, c_l, 1, 1, 1)))
        return params

    def forward(
        self,
        subtype: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return self.compute_gammas_betas(subtype, dtype=dtype)

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
