"""
SNN version of embedding layers for sequence and point cloud encoding

This module provides SNN-compatible versions of:
1. NATSequenceEncoder - for encoding temporal sequences with neighborhood attention
2. PointsEncoder - for encoding point cloud data (map polygons)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import DropPath

from .snn_attention import SNNNeighborhoodAttention1D


class SNNConvTokenizer(nn.Module):
    """
    SNN version of ConvTokenizer for initial sequence embedding

    Input: [T, B, C_in, L] where T is time steps
    Output: [T, B, L, C_out]
    """

    def __init__(self, in_chans=3, embed_dim=32, tau=2.0, backend='torch'):
        super().__init__()
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

    def forward(self, x):
        """
        Args:
            x: [T, B, C, L]
        Returns:
            [T, B, L, C]
        """
        T, B, C, L = x.shape
        x = x.reshape(T * B, C, L)

        x = self.proj(x)  # [T*B, embed_dim, L]
        x = self.bn(x)
        x = x.reshape(T, B, -1, L)
        x = self.lif(x)

        # Permute to [T, B, L, C]
        x = x.permute(0, 1, 3, 2)
        return x


class SNNConvDownsampler(nn.Module):
    """
    SNN version of ConvDownsampler

    Input: [T, B, L, C]
    Output: [T, B, L//2, C*2]
    """

    def __init__(self, dim, tau=2.0, backend='torch'):
        super().__init__()
        self.reduction = nn.Conv1d(dim, 2 * dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(2 * dim)
        self.lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

    def forward(self, x):
        """
        Args:
            x: [T, B, L, C]
        Returns:
            [T, B, L//2, C*2]
        """
        T, B, L, C = x.shape

        # [T, B, L, C] -> [T*B, C, L]
        x = x.permute(0, 1, 3, 2).reshape(T * B, C, L)
        x = self.reduction(x)  # [T*B, 2C, L//2]
        x = self.bn(x)

        # Reshape back to [T, B, L//2, 2C]
        _, C_new, L_new = x.shape
        x = x.reshape(T, B, C_new, L_new).permute(0, 1, 3, 2)
        x = self.lif(x)

        return x


class SNNNATLayer(nn.Module):
    """
    SNN version of NAT Layer (Neighborhood Attention Layer)

    Input/Output: [T, B, L, C]
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
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
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Note: We keep LayerNorm for pre-normalization
        self.norm1 = nn.LayerNorm(dim)

        self.attn = SNNNeighborhoodAttention1D(
            dim=dim,
            kernel_size=kernel_size,
            num_heads=num_heads,
            dilation=dilation if dilation is not None else 1,
            qkv_bias=qkv_bias,
            dropout=attn_drop,
            tau=tau,
            v_threshold=v_threshold,
            scale=scale,
            backend=backend,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        # MLP using SNN components
        hidden_features = int(dim * mlp_ratio)
        self.mlp_fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.mlp_bn1 = nn.BatchNorm1d(hidden_features)
        self.mlp_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.mlp_drop1 = nn.Dropout(drop)

        self.mlp_fc2 = nn.Linear(hidden_features, dim, bias=False)
        self.mlp_bn2 = nn.BatchNorm1d(dim)
        self.mlp_lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.mlp_drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: [T, B, L, C]
        Returns:
            [T, B, L, C]
        """
        T, B, L, C = x.shape

        # Attention block
        shortcut = x
        x_norm = torch.stack([self.norm1(x[t]) for t in range(T)], dim=0)
        x = self.attn(x_norm)
        x = shortcut + self.drop_path(x)

        # MLP block
        shortcut = x
        x_norm = torch.stack([self.norm2(x[t]) for t in range(T)], dim=0)

        # Apply MLP
        x_flat = x_norm.reshape(T * B * L, C)
        x_mlp = self.mlp_fc1(x_flat)
        x_mlp = self.mlp_bn1(x_mlp.reshape(T * B, L, -1).transpose(1, 2)).transpose(1, 2)
        x_mlp = x_mlp.reshape(T, B, L, -1)
        x_mlp = self.mlp_lif1(x_mlp)
        x_mlp = self.mlp_drop1(x_mlp)

        x_flat = x_mlp.reshape(T * B * L, -1)
        x_mlp = self.mlp_fc2(x_flat)
        x_mlp = self.mlp_bn2(x_mlp.reshape(T * B, L, C).transpose(1, 2)).transpose(1, 2)
        x_mlp = x_mlp.reshape(T, B, L, C)
        x_mlp = self.mlp_lif2(x_mlp)
        x_mlp = self.mlp_drop2(x_mlp)

        x = shortcut + self.drop_path(x_mlp)
        return x


class SNNNATBlock(nn.Module):
    """
    SNN version of NAT Block (contains multiple NAT layers)

    Input: [T, B, L, C]
    Output: [T, B, L//2, C*2] (if downsample) or [T, B, L, C]
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
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
        self.depth = depth

        self.blocks = nn.ModuleList([
            SNNNATLayer(
                dim=dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=None if dilations is None else dilations[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                tau=tau,
                v_threshold=v_threshold,
                scale=scale,
                backend=backend,
            )
            for i in range(depth)
        ])

        self.downsample = (
            None if not downsample else SNNConvDownsampler(dim=dim, tau=tau, backend=backend)
        )

    def forward(self, x):
        """
        Args:
            x: [T, B, L, C]
        Returns:
            downsampled_x, x_before_downsample
        """
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is None:
            return x, x
        return self.downsample(x), x


class SNNNATSequenceEncoder(nn.Module):
    """
    SNN version of NATSequenceEncoder for temporal sequence encoding

    Input: [T, B, C_in, L] where T is time steps
    Output: [T, B, C_out] - last temporal feature
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        mlp_ratio=3,
        kernel_size=[3, 3, 5],
        depths=[2, 2, 2],
        num_heads=[2, 4, 8],
        out_indices=[0, 1, 2],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        tau=2.0,
        v_threshold=0.5,
        scale=0.25,
        backend='torch',
    ):
        super().__init__()

        self.embed = SNNConvTokenizer(in_chans, embed_dim, tau=tau, backend=backend)
        self.num_levels = len(depths)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()

        for i in range(self.num_levels):
            level = SNNNATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size[i],
                dilations=None,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]): sum(depths[:i + 1])],
                downsample=(i < self.num_levels - 1),
                tau=tau,
                v_threshold=v_threshold,
                scale=scale,
                backend=backend,
            )
            self.levels.append(level)

        # Output normalization layers
        for i_layer in self.out_indices:
            layer = nn.LayerNorm(self.num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # FPN lateral convolutions
        n = self.num_features[-1]
        self.lateral_convs = nn.ModuleList()
        for i_layer in self.out_indices:
            self.lateral_convs.append(
                nn.Conv1d(self.num_features[i_layer], n, 3, padding=1, bias=False)
            )

        self.fpn_conv = nn.Conv1d(n, n, 3, padding=1, bias=False)
        self.fpn_bn = nn.BatchNorm1d(n)
        self.fpn_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

    def forward(self, x):
        """
        Args:
            x: [T, B, C, L]
        Returns:
            [T, B, C_out] - last temporal position feature
        """
        T = x.shape[0]
        x = self.embed(x)  # [T, B, L, C]

        out = []
        for idx, level in enumerate(self.levels):
            x, xo = level(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                # Apply norm to each time step
                x_out = torch.stack([norm_layer(xo[t]) for t in range(T)], dim=0)
                # [T, B, L, C] -> [T, B, C, L]
                out.append(x_out.permute(0, 1, 3, 2).contiguous())

        # FPN fusion
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            T, B, C, L = out[i].shape
            out_flat = out[i].reshape(T * B, C, L)
            lateral = lateral_conv(out_flat)
            _, C_new, L_new = lateral.shape
            lateral = lateral.reshape(T, B, C_new, L_new)
            laterals.append(lateral)

        # Top-down fusion
        for i in range(len(out) - 1, 0, -1):
            T, B, C, L_target = laterals[i - 1].shape
            _, _, _, L_src = laterals[i].shape
            laterals_i_flat = laterals[i].reshape(T * B, C, L_src)

            upsampled = F.interpolate(
                laterals_i_flat,
                size=L_target,
                mode="linear",
                align_corners=False,
            )
            upsampled = upsampled.reshape(T, B, C, L_target)
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Final convolution
        T, B, C, L = laterals[0].shape
        out_flat = laterals[0].reshape(T * B, C, L)
        out = self.fpn_conv(out_flat)
        out = self.fpn_bn(out)
        _, C_new, L_new = out.shape
        out = out.reshape(T, B, C_new, L_new)
        out = self.fpn_lif(out)

        # Return last temporal feature: [T, B, C, L] -> [T, B, C]
        return out[:, :, :, -1]


class SNNPointsEncoder(nn.Module):
    """
    SNN version of PointsEncoder for point cloud (map polygon) encoding

    Input: [T, B, M, N, C] where M is number of polygons, N is points per polygon
    Output: [T, B, M, C_out]
    """

    def __init__(self, feat_channel, encoder_channel, tau=2.0, backend='torch'):
        super().__init__()
        self.encoder_channel = encoder_channel

        # First MLP
        self.fc1_1 = nn.Linear(feat_channel, 128, bias=False)
        self.bn1_1 = nn.BatchNorm1d(128)
        self.lif1_1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        self.fc1_2 = nn.Linear(128, 256, bias=False)
        self.bn1_2 = nn.BatchNorm1d(256)
        self.lif1_2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        # Second MLP
        self.fc2_1 = nn.Linear(512, 256, bias=False)
        self.bn2_1 = nn.BatchNorm1d(256)
        self.lif2_1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        self.fc2_2 = nn.Linear(256, self.encoder_channel, bias=False)
        self.bn2_2 = nn.BatchNorm1d(self.encoder_channel)
        self.lif2_2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, M, N, C] - batch, polygons, points, features
            mask: [B, M, N] - valid point mask

        Note: This function assumes input is NOT batched in time dimension yet.
        For SNN planning model, we'll expand time dimension in the encoder.

        Returns:
            [B, M, C_out] - encoded polygon features
        """
        bs, m, n, c = x.shape
        device = x.device

        # First MLP: process valid points
        x_flat = x[mask]  # [num_valid_points, C]

        x_valid = self.fc1_1(x_flat)
        x_valid = self.bn1_1(x_valid)
        x_valid = self.lif1_1(x_valid.unsqueeze(0)).squeeze(0)  # Add/remove time dim

        x_valid = self.fc1_2(x_valid)
        x_valid = self.bn1_2(x_valid)
        x_valid = self.lif1_2(x_valid.unsqueeze(0)).squeeze(0)

        # Scatter back to full shape
        x_features = torch.zeros(bs, m, n, 256, device=device)
        x_features[mask] = x_valid

        # Max pooling over points
        pooled_feature = x_features.max(dim=2)[0]  # [B, M, 256]

        # Concatenate with broadcasted pooled feature
        x_features = torch.cat(
            [x_features, pooled_feature.unsqueeze(2).repeat(1, 1, n, 1)], dim=-1
        )  # [B, M, N, 512]

        # Second MLP
        x_flat = x_features[mask]  # [num_valid_points, 512]

        x_valid = self.fc2_1(x_flat)
        x_valid = self.bn2_1(x_valid)
        x_valid = self.lif2_1(x_valid.unsqueeze(0)).squeeze(0)

        x_valid = self.fc2_2(x_valid)
        x_valid = self.bn2_2(x_valid)
        x_valid = self.lif2_2(x_valid.unsqueeze(0)).squeeze(0)

        # Scatter and max pool
        res = torch.zeros(bs, m, n, self.encoder_channel, device=device)
        res[mask] = x_valid
        res = res.max(dim=2)[0]  # [B, M, C_out]

        return res
