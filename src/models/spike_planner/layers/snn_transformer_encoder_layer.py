"""
SNN version of Transformer Encoder Layer

This module provides an SNN-compatible transformer encoder layer
using spiking attention mechanisms.
"""

from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch import Tensor

from .snn_attention import SNNMultiheadAttention
from .snn_common_layers import SNNMlp


class SNNTransformerEncoderLayer(nn.Module):
    """
    SNN version of TransformerEncoderLayer

    Input/Output shape: [T, B, L, C] where T is time steps
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        tau=2.0,
        v_threshold=0.5,
        scale=0.25,
        backend='torch',
    ):
        super().__init__()

        self.dim = dim
        # Note: LayerNorm is kept for pre-normalization
        # In SNN context, we apply it before attention/MLP
        self.norm1 = nn.LayerNorm(dim)

        self.attn = SNNMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            qkv_bias=qkv_bias,
            tau=tau,
            v_threshold=v_threshold,
            scale=scale,
            backend=backend,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SNNMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            tau=tau,
            drop=drop,
            backend=backend,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            src: Input tensor of shape [T, B, L, C]
            mask: Attention mask (optional)
            key_padding_mask: Key padding mask of shape [B, L] (optional)

        Returns:
            Output tensor of shape [T, B, L, C]
        """
        T, B, L, C = src.shape

        # Attention block with residual connection
        # Apply LayerNorm across time dimension
        src_normalized = torch.stack([self.norm1(src[t]) for t in range(T)], dim=0)

        src2 = self.attn(
            src_normalized,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.drop_path1(src2)

        # MLP block with residual connection
        src_normalized = torch.stack([self.norm2(src[t]) for t in range(T)], dim=0)
        src = src + self.drop_path2(self.mlp(src_normalized))

        return src
