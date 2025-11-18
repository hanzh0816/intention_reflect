"""
SNN Attention Modules for PlanTF Intention Decoder
适配意图解码器的SNN注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .snn_utils import LIFNeuron


class SNNMultiheadAttention(nn.Module):
    """
    SNN版本的MultiheadAttention，适配意图解码器使用

    Args:
        embed_dim: 模型总维度
        num_heads: 注意力头数
        dropout: dropout概率
        qkv_bias: Q, K, V投影是否使用偏置
        scale: 注意力分数缩放因子 (默认: 0.25)
        neuron_cfg: 神经元配置字典
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

        # Q投影
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = LIFNeuron(**neuron_cfg)

        # K投影
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = LIFNeuron(**neuron_cfg)

        # V投影
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = LIFNeuron(**neuron_cfg)

        # 注意力dropout
        self.attn_drop = nn.Dropout(dropout)

        # 注意力输出LIF
        self.attn_lif = LIFNeuron(**neuron_cfg)

        # 输出投影
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
            x: 输入张量 [T, B, L, C]
            attn_mask: 注意力掩码（暂未实现）
            key_padding_mask: 键填充掩码（暂未实现）

        Returns:
            输出张量 [T, B, L, C]
        """
        T, B, L, C = x.shape

        # 为线性层重塑形状: [T, B, L, C] -> [T*B*L, C]
        x_flat = x.reshape(T * B * L, C)

        # Q, K, V投影
        q = self.q_linear(x_flat)  # [T*B*L, C]
        q = self.q_bn(q).reshape(T, B, L, C)
        q = self.q_lif(q)  # [T, B, L, C]

        k = self.k_linear(x_flat)  # [T*B*L, C]
        k = self.k_bn(k).reshape(T, B, L, C)
        k = self.k_lif(k)  # [T, B, L, C]

        v = self.v_linear(x_flat)  # [T*B*L, C]
        v = self.v_bn(v).reshape(T, B, L, C)
        v = self.v_lif(v)  # [T, B, L, C]

        # 重塑为多头: [T, B, L, C] -> [T, B, L, num_heads, head_dim]
        q = q.reshape(T, B, L, self.num_heads, self.head_dim)
        k = k.reshape(T, B, L, self.num_heads, self.head_dim)
        v = v.reshape(T, B, L, self.num_heads, self.head_dim)

        # 注意力计算: [T, B, num_heads, L, L]
        # q, k调整为 [T, B, num_heads, L, head_dim]
        q = q.permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]
        k = k.permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # 注意力分数: q @ k^T -> [T, B, num_heads, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用掩码（如果提供）
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # 应用softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 注意力应用到v
        # v调整为 [T, B, num_heads, L, head_dim]
        v = v.permute(0, 1, 3, 2, 4)  # [T, B, num_heads, L, head_dim]

        # attn @ v -> [T, B, num_heads, L, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # 重塑回原始维度: [T, B, num_heads, L, head_dim] -> [T, B, L, C]
        attn_output = attn_output.permute(0, 1, 3, 2, 4).reshape(T, B, L, C)

        # 注意力输出LIF
        attn_output = self.attn_lif(attn_output)

        # 输出投影
        attn_output_flat = attn_output.reshape(T * B * L, C)
        output = self.out_linear(attn_output_flat)  # [T*B*L, C]
        output = self.out_bn(output).reshape(T, B, L, C)
        output = self.out_lif(output)  # [T, B, L, C]

        return output


class SNNTransformerEncoderLayer(nn.Module):
    """
    SNN版本的Transformer编码器层

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
        scale=0.25,
        neuron_cfg=None,
    ):
        super().__init__()

        self.dim = dim
        # 注意：在SNN上下文中，LayerNorm在注意力/MLP之前应用
        self.norm1 = nn.LayerNorm(dim)

        self.attn = SNNMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            qkv_bias=qkv_bias,
            scale=scale,
            neuron_cfg=neuron_cfg,
        )
        self.drop_path1 = nn.Identity()  # 简化版本，不使用DropPath

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SNNMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            neuron_cfg=neuron_cfg,
        )
        self.drop_path2 = nn.Identity()  # 简化版本，不使用DropPath

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src: 输入张量 [T, B, L, C]
            mask: 注意力掩码（可选）
            key_padding_mask: 键填充掩码 [B, L]（可选）

        Returns:
            输出张量 [T, B, L, C]
        """
        T, B, L, C = src.shape

        # 注意块 + 残差连接
        # 在时间维度上应用LayerNorm
        src_normalized = torch.stack([self.norm1(src[t]) for t in range(T)], dim=0)
        src2 = self.attn(src_normalized, attn_mask=mask, key_padding_mask=key_padding_mask)
        src = src + self.drop_path1(src2)

        # MLP块 + 残差连接
        src_normalized = torch.stack([self.norm2(src[t]) for t in range(T)], dim=0)
        src = src + self.drop_path2(self.mlp(src_normalized))

        return src


class SNNMlp(nn.Module):
    """
    SNN版本的Transformer MLP前馈网络

    Input shape: [T, B, L, C] where T is time steps
    Output shape: [T, B, L, C]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        neuron_cfg=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if neuron_cfg is None:
            neuron_cfg = {}

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = LIFNeuron(**neuron_cfg)
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = LIFNeuron(**neuron_cfg)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: 输入张量 [T, B, L, C_in]
        Returns:
            输出张量 [T, B, L, C_out]
        """
        T, B, L, C_in = x.shape

        # 第一层: Linear -> BN -> LIF -> Dropout
        x_flat = x.reshape(T * B * L, C_in)  # [T*B*L, C_in]
        x = self.fc1(x_flat)  # [T*B*L, hidden_features]

        # BatchNorm处理：需要正确的维度顺序
        # 重塑为 [T*B, L, hidden_features] 用于 BatchNorm1d
        x = x.reshape(T * B, L, -1)  # [T*B, L, hidden_features]
        x = x.transpose(1, 2)  # [T*B, hidden_features, L]
        x = self.bn1(x)  # [T*B, hidden_features, L]
        x = x.transpose(1, 2)  # [T*B, L, hidden_features]

        # 重塑回时间维度并应用LIF
        x = x.reshape(T, B, L, -1)  # [T, B, L, hidden_features]
        x = self.lif1(x)  # [T, B, L, hidden_features]

        x_flat = x.reshape(T * B * L, -1)  # [T*B*L, hidden_features]
        x = self.drop1(x)

        # 第二层: Linear -> BN -> LIF -> Dropout
        x = self.fc2(x)  # [T*B*L, out_features]

        # BatchNorm处理：需要正确的维度顺序
        # 重塑为 [T*B, L, out_features] 用于 BatchNorm1d
        x = x.reshape(T * B, L, -1)  # [T*B, L, out_features]
        x = x.transpose(1, 2)  # [T*B, out_features, L]
        x = self.bn2(x)  # [T*B, out_features, L]
        x = x.transpose(1, 2)  # [T*B, L, out_features]

        # 重塑回时间维度并应用LIF
        x = x.reshape(T, B, L, -1)  # [T, B, L, out_features]
        x = self.lif2(x)  # [T, B, L, out_features]

        x_flat = x.reshape(T * B * L, -1)  # [T*B*L, out_features]
        x = self.drop2(x)

        # 最终重塑
        x = x.reshape(T, B, L, -1)  # [T, B, L, C_out]

        return x