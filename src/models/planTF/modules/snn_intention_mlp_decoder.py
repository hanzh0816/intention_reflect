"""
基于SNN的Intention解码器 - MLP版本
使用多层感知机和LIF神经元处理意图信息
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

from .snn_utils import (
    SNNLinearBlock, LIFNeuron, TimeDimAverage, TimeDimExpander,
    get_default_neuron_config
)


class SNNIntentionMLPDecoder(nn.Module):
    """
    基于SNN的意图解码器 - MLP版本

    使用多层感知机 + LIF神经元处理ego车辆特征，
    输出意图感知的特征表示

    Args:
        dim: 特征维度
        depth: MLP层数（默认2）
        hidden_dim: 隐藏层维度（默认与输入相同）
        neuron_cfg: LIF神经元配置
        dropout: dropout比例
        time_steps: 时间步数
    """

    def __init__(self, dim: int, depth: int = 2, hidden_dim: Optional[int] = None,
                 neuron_cfg: Optional[Dict] = None, dropout: float = 0.0,
                 time_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.time_steps = time_steps

        if hidden_dim is None:
            hidden_dim = dim

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        # 更新神经元配置中的时间步
        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # MLP层
        self.mlp_layers = nn.ModuleList()

        # 输入层
        self.mlp_layers.append(
            SNNLinearBlock(dim, hidden_dim, self.neuron_cfg, dropout=dropout)
        )

        # 隐藏层
        for i in range(depth - 1):
            self.mlp_layers.append(
                SNNLinearBlock(hidden_dim, hidden_dim, self.neuron_cfg, dropout=dropout)
            )

        # 输出层（无LIF激活，保持特征空间）
        self.output_linear = nn.Linear(hidden_dim, dim, bias=True)

        # 时间维度平均
        self.time_average = TimeDimAverage()

        # LayerNorm（在最后应用）
        self.norm = nn.LayerNorm(dim)

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
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, ego_feature: torch.Tensor):
        """
        前向传播

        Args:
            ego_feature: [B, dim] ego车辆特征

        Returns:
            intention_feature: [B, dim] 意图特征
        """
        B, C = ego_feature.shape

        # 扩展时间维度
        x = self.time_expander(ego_feature)  # [T, B, C]

        # 通过MLP层
        for layer in self.mlp_layers:
            x = layer(x)  # [T, B, hidden_dim]

        # 输出线性层（无LIF激活）
        T = x.shape[0]
        x_flat = x.reshape(T * B, -1)  # [T*B, hidden_dim]
        x = self.output_linear(x_flat)  # [T*B, dim]
        x = x.reshape(T, B, C)  # [T, B, dim]

        # 时间维度平均
        x = self.time_average(x)  # [B, dim]

        # LayerNorm
        intention_feature = self.norm(x)

        return intention_feature

    def get_spike_rates(self):
        """获取各层的脉冲发放率（用于分析）"""
        spike_rates = {}

        # 从LIF神经元获取脉冲率
        for i, layer in enumerate(self.mlp_layers):
            if hasattr(layer, 'lif') and hasattr(layer.lif, 'get_spike_rate'):
                spike_rates[f'mlp_layer_{i}'] = layer.lif.get_spike_rate()

        return spike_rates

    def reset_neurons(self):
        """重置神经元状态"""
        for layer in self.mlp_layers:
            if hasattr(layer, 'lif') and hasattr(layer.lif, 'reset'):
                layer.lif.reset()


class SNNIntentionShallowDecoder(nn.Module):
    """
    简化的SNN意图解码器 - 单层版本
    适用于轻量级应用
    """

    def __init__(self, dim: int, neuron_cfg: Optional[Dict] = None,
                 time_steps: int = 4):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # 单层处理
        self.process_layer = SNNLinearBlock(dim, dim, self.neuron_cfg)

        # 时间维度平均
        self.time_average = TimeDimAverage()

        # LayerNorm
        self.norm = nn.LayerNorm(dim)

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
        # 扩展时间维度
        x = self.time_expander(ego_feature)  # [T, B, C]

        # 处理层
        x = self.process_layer(x)  # [T, B, C]

        # 时间维度平均
        x = self.time_average(x)  # [B, C]

        # LayerNorm
        intention_feature = self.norm(x)

        return intention_feature

    def reset_neurons(self):
        """重置神经元状态"""
        if hasattr(self.process_layer, 'lif') and hasattr(self.process_layer.lif, 'reset'):
            self.process_layer.lif.reset()


def create_snn_intention_mlp_decoder(decoder_type: str = "standard", **kwargs):
    """
    工厂函数：创建不同类型的SNN MLP意图解码器

    Args:
        decoder_type: "standard" 或 "shallow"
        **kwargs: 传递给解码器的参数

    Returns:
        SNNIntentionMLPDecoder 或 SNNIntentionShallowDecoder 实例
    """
    if decoder_type == "standard":
        return SNNIntentionMLPDecoder(**kwargs)
    elif decoder_type == "shallow":
        return SNNIntentionShallowDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")


# 预定义配置
SNN_INTENTION_MLP_CONFIGS = {
    "small": {
        "depth": 1,
        "hidden_dim": 64,
        "dropout": 0.1,
        "time_steps": 4
    },
    "standard": {
        "depth": 2,
        "hidden_dim": 128,
        "dropout": 0.1,
        "time_steps": 4
    },
    "large": {
        "depth": 3,
        "hidden_dim": 256,
        "dropout": 0.15,
        "time_steps": 4
    }
}