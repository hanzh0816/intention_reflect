"""
SNN Attention Modules for PlanTF

This module implements Spiking Neural Network (SNN) versions of:
1. MultiheadAttention (corresponding to torch.nn.MultiheadAttention)
2. NeighborhoodAttention1D (corresponding to natten.NeighborhoodAttention1D)

Input shape: [T, B, L, C] where:
- T: time steps
- B: batch size
- L: sequence length
- C: channels/embedding dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .snn_neuron import LIFNeuron


class SNNMultiheadAttention(nn.Module):
    """
    SNN version of torch.nn.MultiheadAttention

    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability
        qkv_bias: If True, add bias to Q, K, V projections
        scale: Scaling factor for attention scores (default: 0.25)
        neuron_cfg: Dictionary containing neuron configuration (spike_mode, tau,
                   v_threshold, v_reset, detach_reset, backend, etc.)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        scale: float = 0.25,
        neuron_cfg: dict = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = scale

        if neuron_cfg is None:
            neuron_cfg = {} 

        # Q projection
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = LIFNeuron(**neuron_cfg)

        # K projection
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = LIFNeuron(**neuron_cfg)

        # V projection
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = LIFNeuron(**neuron_cfg)

        # Attention dropout
        self.attn_drop = nn.Dropout(dropout)

        # Attention output LIF
        self.attn_lif = LIFNeuron(**neuron_cfg)

        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        self.out_bn = nn.BatchNorm1d(embed_dim)
        self.out_lif = LIFNeuron(**neuron_cfg)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [T, B, L, C]
            attn_mask: Attention mask (not implemented yet)
            key_padding_mask: Key padding mask (not implemented yet)

        Returns:
            Output tensor of shape [T, B, L, C]
        """
        T, B, L, C = x.shape

        # Reshape for linear layers: [T, B, L, C] -> [T*B*L, C]
        x_flat = x.reshape(T * B * L, C)

        # Q projection
        q = self.q_linear(x_flat)  # [T*B*L, C]
        q = self.q_bn(q.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)  # BatchNorm over L dimension
        q = q.reshape(T, B, L, C)
        q = self.q_lif(q)
        q = q.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # K projection
        k = self.k_linear(x_flat)
        k = self.k_bn(k.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        k = k.reshape(T, B, L, C)
        k = self.k_lif(k)
        k = k.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # V projection
        v = self.v_linear(x_flat)
        v = self.v_bn(v.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        v = v.reshape(T, B, L, C)
        v = self.v_lif(v)
        v = v.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # Attention computation: Q @ K^T @ V
        attn = q @ k.transpose(-2, -1)  # [T, B, num_heads, L, L]

        # Apply attention mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask shape: [B, L] where True indicates padding
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3),  # [1, B, 1, 1, L]
                float('-inf')
            )

        # Apply attention to values
        out = (attn @ v) * self.scale  # [T, B, num_heads, L, head_dim]

        # Reshape back to [T, B, L, C]
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, L, C)
        out = self.attn_lif(out)

        # Output projection
        out_flat = out.reshape(T * B * L, C)
        out = self.out_linear(out_flat)
        out = self.out_bn(out.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        out = out.reshape(T, B, L, C)
        out = self.out_lif(out)

        return out


class SNNNeighborhoodAttention1D(nn.Module):
    """
    SNN version of NeighborhoodAttention1D

    This implements neighborhood attention where each position only attends to
    a local window of neighbors around it.

    Args:
        dim: Dimension of the model
        kernel_size: Size of the local attention window
        num_heads: Number of parallel attention heads
        dilation: Dilation for the neighborhood (default: 1)
        qkv_bias: If True, add bias to Q, K, V projections
        dropout: Dropout probability
        scale: Scaling factor for attention scores (default: 0.25)
        neuron_cfg: Dictionary containing neuron configuration (spike_mode, tau,
                   v_threshold, v_reset, detach_reset, backend, etc.)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        num_heads: int = 8,
        dilation: int = 1,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        scale: float = 0.25,
        neuron_cfg: dict = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.scale = scale

        # Calculate the window size (how many neighbors on each side)
        self.window_size = (kernel_size - 1) // 2

        if neuron_cfg is None:
            neuron_cfg = {}

        # Q projection
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNeuron(**neuron_cfg)

        # K projection
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNeuron(**neuron_cfg)

        # V projection
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNeuron(**neuron_cfg)

        # Attention dropout
        self.attn_drop = nn.Dropout(dropout)

        # Attention output LIF
        self.attn_lif = LIFNeuron(**neuron_cfg)

        # Output projection
        self.out_linear = nn.Linear(dim, dim)
        self.out_bn = nn.BatchNorm1d(dim)
        self.out_lif = LIFNeuron(**neuron_cfg)

    def _get_local_indices(self, L: int, device: torch.device):
        """
        Generate indices for local attention window

        Returns:
            indices: [L, kernel_size] containing the indices of neighbors for each position
        """
        # For each position i, we want indices [i-window, i-window+1, ..., i+window]
        positions = torch.arange(L, device=device)
        offsets = torch.arange(-self.window_size, self.window_size + 1, device=device) * self.dilation

        # Broadcasting: [L, 1] + [1, kernel_size] -> [L, kernel_size]
        indices = positions.unsqueeze(1) + offsets.unsqueeze(0)

        # Clamp to valid range [0, L-1]
        indices = torch.clamp(indices, 0, L - 1)

        return indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [T, B, L, C]

        Returns:
            Output tensor of shape [T, B, L, C]
        """
        T, B, L, C = x.shape

        # Reshape for linear layers: [T, B, L, C] -> [T*B*L, C]
        x_flat = x.reshape(T * B * L, C)

        # Q projection
        q = self.q_linear(x_flat)
        q = self.q_bn(q.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        q = q.reshape(T, B, L, C)
        q = self.q_lif(q)
        q = q.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # K projection
        k = self.k_linear(x_flat)
        k = self.k_bn(k.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        k = k.reshape(T, B, L, C)
        k = self.k_lif(k)
        k = k.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # V projection
        v = self.v_linear(x_flat)
        v = self.v_bn(v.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        v = v.reshape(T, B, L, C)
        v = self.v_lif(v)
        v = v.reshape(T, B, L, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # Get local neighborhood indices
        indices = self._get_local_indices(L, x.device)  # [L, kernel_size]

        # Gather local K and V values
        # k: [T, B, num_heads, L, head_dim] -> [T, B, num_heads, L, kernel_size, head_dim]
        k_local = k[:, :, :, indices, :]  # [T, B, num_heads, L, kernel_size, head_dim]
        v_local = v[:, :, :, indices, :]  # [T, B, num_heads, L, kernel_size, head_dim]

        # Compute local attention
        # q: [T, B, num_heads, L, head_dim] -> [T, B, num_heads, L, 1, head_dim]
        q_expanded = q.unsqueeze(4)  # [T, B, num_heads, L, 1, head_dim]

        # Attention scores: [T, B, num_heads, L, 1, head_dim] @ [T, B, num_heads, L, head_dim, kernel_size]
        # -> [T, B, num_heads, L, 1, kernel_size]
        attn = q_expanded @ k_local.transpose(-2, -1)  # [T, B, num_heads, L, 1, kernel_size]
        attn = attn.squeeze(4)  # [T, B, num_heads, L, kernel_size]

        # Apply attention to local values
        # attn: [T, B, num_heads, L, 1, kernel_size] @ v_local: [T, B, num_heads, L, kernel_size, head_dim]
        # -> [T, B, num_heads, L, 1, head_dim]
        out = attn.unsqueeze(-2) @ v_local  # [T, B, num_heads, L, 1, head_dim]
        out = out.squeeze(-2) * self.scale  # [T, B, num_heads, L, head_dim]

        # Reshape back to [T, B, L, C]
        out = out.permute(0, 1, 3, 2, 4).reshape(T, B, L, C)
        out = self.attn_lif(out)

        # Output projection
        out_flat = out.reshape(T * B * L, C)
        out = self.out_linear(out_flat)
        out = self.out_bn(out.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        out = out.reshape(T, B, L, C)
        out = self.out_lif(out)

        return out
