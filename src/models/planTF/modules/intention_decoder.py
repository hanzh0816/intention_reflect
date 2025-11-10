"""
Intention Decoder Module

This module decodes ego features into intention-aware representations using
transformer decoder layers.
"""

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class IntentionDecoder(nn.Module):
    """
    Transformer-based intention decoder.

    Takes ego feature and produces intention feature through self-attention.
    The intention feature captures the vehicle's intended behavior.
    """

    def __init__(
        self,
        dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Initialize Intention Decoder.

        Args:
            dim: Feature dimension
            depth: Number of transformer decoder layers
            num_heads: Number of attention heads
            mlp_ratio: Ratio of mlp hidden dim to embedding dim
            qkv_bias: Enable bias for qkv if True
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
        """
        super().__init__()

        self.dim = dim
        self.depth = depth

        # Import here to avoid circular dependency
        from src.models.planTF.layers.transformer_encoder_layer import TransformerEncoderLayer

        # Stack of transformer layers for intention decoding
        # We use TransformerEncoderLayer but it acts as a decoder through self-attention
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
            )
            for _ in range(depth)
        ])

        # Layer norm for output
        self.norm = nn.LayerNorm(dim)

    def forward(self, ego_feature: Tensor) -> Tensor:
        """
        Forward pass of intention decoder.

        Args:
            ego_feature: Ego vehicle feature tensor [B, dim]

        Returns:
            intention_feature: Decoded intention feature [B, dim]
        """
        # Expand ego feature to sequence format for transformer
        # [B, dim] -> [B, 1, dim]
        x = ego_feature.unsqueeze(1)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Apply final normalization
        x = self.norm(x)

        # Remove sequence dimension: [B, 1, dim] -> [B, dim]
        intention_feature = x.squeeze(1)

        return intention_feature
