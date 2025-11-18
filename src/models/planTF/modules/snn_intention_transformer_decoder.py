"""
基于SNN的Intention解码器 - Transformer版本
使用SNN Transformer编码器层和LIF神经元处理意图信息
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .snn_utils import TimeDimAverage, TimeDimExpander, get_default_neuron_config
from .snn_attention import SNNTransformerEncoderLayer


class SNNIntentionTransformerDecoder(nn.Module):
    """
    基于SNN的意图解码器 - Transformer版本

    使用SNN Transformer编码器层处理ego车辆特征，
    通过自注意力机制捕获意图信息

    Args:
        dim: 特征维度
        depth: Transformer层数（默认2）
        num_heads: 注意力头数（默认8）
        mlp_ratio: MLP扩展比例（默认4.0）
        qkv_bias: QKV投影是否使用偏置
        drop: dropout比例
        attn_drop: 注意力dropout比例
        drop_path: DropPath比例
        neuron_cfg: LIF神经元配置
        time_steps: 时间步数
    """

    def __init__(self, dim: int, depth: int = 2, num_heads: int = 8,
                 mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, neuron_cfg: Optional[Dict] = None,
                 time_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.time_steps = time_steps

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        # 更新神经元配置中的时间步
        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # Transformer编码器层
        self.layers = nn.ModuleList()
        for i in range(depth):
            # 支持递增的drop_path
            current_drop_path = drop_path * i / (depth - 1) if depth > 1 and drop_path > 0 else 0.0

            layer = SNNTransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=current_drop_path,
                neuron_cfg=self.neuron_cfg,
            )
            self.layers.append(layer)

        # LayerNorm（在最后应用）
        self.norm = nn.LayerNorm(dim)

        # 时间维度平均
        self.time_average = TimeDimAverage()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, ego_feature: torch.Tensor):
        """
        前向传播

        Args:
            ego_feature: [B, dim] ego车辆特征

        Returns:
            intention_feature: [B, dim] 意图特征
        """
        B, C = ego_feature.shape

        # 扩展时间维度: [B, C] -> [T, B, 1, C]
        x = ego_feature.unsqueeze(1)  # [B, 1, C]
        x = self.time_expander(x)  # [T, B, 1, C]

        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)  # [T, B, 1, C]

        # 移除序列维度: [T, B, 1, C] -> [T, B, C]
        x = x.squeeze(2)

        # LayerNorm（在时间维度上应用）
        T = x.shape[0]
        x_normalized = torch.stack([self.norm(x[t]) for t in range(T)], dim=0)

        # 时间维度平均: [T, B, C] -> [B, C]
        intention_feature = self.time_average(x_normalized)

        return intention_feature

    def reset_neurons(self):
        """重置所有LIF神经元的状态"""
        for layer in self.layers:
            # 重置注意力机制中的神经元
            if hasattr(layer.attn, 'q_lif') and hasattr(layer.attn.q_lif, 'reset'):
                layer.attn.q_lif.reset()
            if hasattr(layer.attn, 'k_lif') and hasattr(layer.attn.k_lif, 'reset'):
                layer.attn.k_lif.reset()
            if hasattr(layer.attn, 'v_lif') and hasattr(layer.attn.v_lif, 'reset'):
                layer.attn.v_lif.reset()
            if hasattr(layer.attn, 'attn_lif') and hasattr(layer.attn.attn_lif, 'reset'):
                layer.attn.attn_lif.reset()
            if hasattr(layer.attn, 'out_lif') and hasattr(layer.attn.out_lif, 'reset'):
                layer.attn.out_lif.reset()

            # 重置MLP中的神经元
            if hasattr(layer.mlp, 'lif1') and hasattr(layer.mlp.lif1, 'reset'):
                layer.mlp.lif1.reset()
            if hasattr(layer.mlp, 'lif2') and hasattr(layer.mlp.lif2, 'reset'):
                layer.mlp.lif2.reset()

    def get_attention_maps(self, ego_feature: torch.Tensor):
        """
        获取注意力图用于分析

        Args:
            ego_feature: [B, dim] ego车辆特征

        Returns:
            attention_maps: 每层每头的注意力权重列表
        """
        B, C = ego_feature.shape

        # 扩展时间维度
        x = ego_feature.unsqueeze(1)  # [B, 1, C]
        x = self.time_expander(x)  # [T, B, 1, C]

        attention_maps = []

        # 通过Transformer层并收集注意力图
        for layer in self.layers:
            # 保存当前特征用于计算注意力
            T = x.shape[0]
            src_normalized = torch.stack([layer.norm1(x[t]) for t in range(T)], dim=0)

            # 手动计算注意力以获取注意力权重
            with torch.no_grad():
                # 这只是一个简化版本，实际需要修改SNNMultiheadAttention来返回注意力权重
                pass

            x = layer(x)

        return attention_maps


class SNNIntentionLightTransformerDecoder(nn.Module):
    """
    轻量级SNN Transformer意图解码器
    减少参数数量，适用于资源受限场景
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 qkv_bias: bool = False, drop: float = 0.0,
                 neuron_cfg: Optional[Dict] = None, time_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.time_steps = time_steps

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # 轻量级单层Transformer
        self.transformer_layer = SNNTransformerEncoderLayer(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=drop,
            neuron_cfg=self.neuron_cfg,
        )

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

        # 时间维度平均
        self.time_average = TimeDimAverage()

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, ego_feature: torch.Tensor):
        """
        前向传播
        """
        # 扩展时间维度
        x = ego_feature.unsqueeze(1)  # [B, 1, C]
        x = self.time_expander(x)  # [T, B, 1, C]

        # 通过Transformer层
        x = self.transformer_layer(x)  # [T, B, 1, C]

        # 移除序列维度
        x = x.squeeze(2)  # [T, B, C]

        # LayerNorm
        T = x.shape[0]
        x_normalized = torch.stack([self.norm(x[t]) for t in range(T)], dim=0)

        # 时间维度平均
        intention_feature = self.time_average(x_normalized)

        return intention_feature

    def reset_neurons(self):
        """重置神经元状态"""
        # 重置注意力机制中的神经元
        if hasattr(self.transformer_layer.attn, 'q_lif'):
            self.transformer_layer.attn.q_lif.reset()
        if hasattr(self.transformer_layer.attn, 'k_lif'):
            self.transformer_layer.attn.k_lif.reset()
        if hasattr(self.transformer_layer.attn, 'v_lif'):
            self.transformer_layer.attn.v_lif.reset()
        if hasattr(self.transformer_layer.attn, 'attn_lif'):
            self.transformer_layer.attn.attn_lif.reset()
        if hasattr(self.transformer_layer.attn, 'out_lif'):
            self.transformer_layer.attn.out_lif.reset()

        # 重置MLP中的神经元
        if hasattr(self.transformer_layer.mlp, 'lif1'):
            self.transformer_layer.mlp.lif1.reset()
        if hasattr(self.transformer_layer.mlp, 'lif2'):
            self.transformer_layer.mlp.lif2.reset()


# 预定义配置
SNN_INTENTION_TRANSFORMER_CONFIGS = {
    "tiny": {
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2.0,
        "time_steps": 4
    },
    "small": {
        "depth": 1,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "time_steps": 4
    },
    "standard": {
        "depth": 2,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "time_steps": 4
    },
    "large": {
        "depth": 3,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "time_steps": 4
    },
    "huge": {
        "depth": 4,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "time_steps": 4
    }
}


def create_snn_intention_transformer_decoder(decoder_type: str = "standard",
                                           size: str = "standard",
                                           **kwargs):
    """
    工厂函数：创建不同类型的SNN Transformer意图解码器

    Args:
        decoder_type: "standard" 或 "light"
        size: "tiny", "small", "standard", "large", "huge"
        **kwargs: 额外的参数

    Returns:
        SNNIntentionTransformerDecoder 或 SNNIntentionLightTransformerDecoder 实例
    """
    if size not in SNN_INTENTION_TRANSFORMER_CONFIGS:
        raise ValueError(f"Unsupported size: {size}. Available: {list(SNN_INTENTION_TRANSFORMER_CONFIGS.keys())}")

    config = SNN_INTENTION_TRANSFORMER_CONFIGS[size].copy()
    config.update(kwargs)

    if decoder_type == "standard":
        return SNNIntentionTransformerDecoder(**config)
    elif decoder_type == "light":
        # 轻量级版本
        light_config = {
            "dim": config.get("dim"),
            "num_heads": min(config.get("num_heads", 4), 4),
            "mlp_ratio": min(config.get("mlp_ratio", 2.0), 2.0),
            "drop": config.get("drop", 0.0),
            "neuron_cfg": config.get("neuron_cfg"),
            "time_steps": config.get("time_steps", 4)
        }
        return SNNIntentionLightTransformerDecoder(**light_config)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")