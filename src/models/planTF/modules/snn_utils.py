"""
SNN基础工具类和组件，支持意图解码器的SNN实现
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Union

try:
    # 优先尝试clock_driven模块（MultiStep支持）
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode, MultiStepParametricLIFNode, MultiStepIFNode
    )
    from spikingjelly.clock_driven import functional
    SPIKING_JELLY_AVAILABLE = True
    SPIKING_MODE = "clock_driven"
    print("Using SpikingJelly clock_driven module with MultiStep neurons")
except ImportError:
    try:
        # 备用activation_based模块
        from spikingjelly.activation_based import neuron, layer
        from spikingjelly.activation_based import functional
        SPIKING_JELLY_AVAILABLE = True
        SPIKING_MODE = "activation_based"
        print("Using SpikingJelly activation_based module")
    except ImportError:
        SPIKING_JELLY_AVAILABLE = False
        SPIKING_MODE = "mock"
        print("Warning: SpikingJelly not available. Using mock SNN implementation.")


class MockLIFNode(nn.Module):
    """当SpikingJelly不可用时的Mock LIF神经元"""
    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, v_reset: float = 0.0,
                 detach_reset: bool = True, backend: str = "torch"):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.backend = backend

    def forward(self, x: torch.Tensor):
        # 简化的mock实现：直接返回输入
        return torch.relu(x)


class LIFNeuron(nn.Module):
    """统一的LIF神经元封装，支持多种神经元类型"""

    def __init__(self, spike_mode: str = "lif", tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, detach_reset: bool = True, backend: str = "torch",
                 time_steps: int = 4):
        super().__init__()
        self.spike_mode = spike_mode
        self.time_steps = time_steps

        if SPIKING_JELLY_AVAILABLE:
            # 使用SpikingJelly的神经元
            if SPIKING_MODE == "clock_driven":
                # clock_driven模块 - 使用MultiStep神经元
                if spike_mode == "lif":
                    self.lif_neuron = MultiStepLIFNode(
                        tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                elif spike_mode == "plif":
                    self.lif_neuron = MultiStepParametricLIFNode(
                        init_tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                elif spike_mode == "if":
                    self.lif_neuron = MultiStepIFNode(
                        v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset,
                        backend=backend
                    )
                elif spike_mode == "ilif":
                    self.lif_neuron = MultiStepLIFNode(
                        tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                else:
                    raise ValueError(f"Unsupported spike mode: {spike_mode}")

            elif SPIKING_MODE == "activation_based":
                # activation_based模块 - 使用简化命名
                if spike_mode == "lif":
                    self.lif_neuron = neuron.LIFNode(
                        tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                elif spike_mode == "plif":
                    self.lif_neuron = neuron.ParametricLIFNode(
                        init_tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                elif spike_mode == "if":
                    self.lif_neuron = neuron.IFNode(
                        v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset,
                        backend=backend
                    )
                elif spike_mode == "ilif":
                    self.lif_neuron = neuron.LIFNode(
                        tau=tau, v_threshold=v_threshold, detach_reset=detach_reset,
                        v_reset=v_reset, backend=backend
                    )
                else:
                    raise ValueError(f"Unsupported spike mode: {spike_mode}")

            else:
                # Mock模式
                self.lif_neuron = MockLIFNode(
                    tau=tau, v_threshold=v_threshold, v_reset=v_reset,
                    detach_reset=detach_reset, backend=backend
                )
        else:
            # 使用mock实现
            self.lif_neuron = MockLIFNode(
                tau=tau, v_threshold=v_threshold, v_reset=v_reset,
                detach_reset=detach_reset, backend=backend
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, ...] 时间步×批次×其他维度
        Returns:
            [T, B, ...] 脉冲输出
        """
        return self.lif_neuron(x)


class SNNLinearBlock(nn.Module):
    """SNN线性块：Linear -> BatchNorm -> LIF"""

    def __init__(self, in_features: int, out_features: int, neuron_cfg: Dict,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron_cfg = neuron_cfg

        # 线性层（无偏置，因为后面有BatchNorm）
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # BatchNorm（在通道维度上）
        self.bn = nn.BatchNorm1d(out_features)

        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # LIF神经元
        self.lif = LIFNeuron(**neuron_cfg)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, L, C_in] 或 [T, B, C_in]
        Returns:
            [T, B, L, C_out] 或 [T, B, C_out]
        """
        original_shape = x.shape
        T = x.shape[0]

        # 处理不同输入形状
        if len(original_shape) == 4:
            # [T, B, L, C_in]
            T, B, L, C_in = x.shape
            x_flat = x.reshape(T * B * L, C_in)
            x = self.linear(x_flat)  # [T*B*L, C_out]

            # BatchNorm：需要 [N, C] 格式
            x = self.bn(x)  # [T*B*L, C_out]

            # 恢复形状
            x = x.reshape(T, B, L, self.out_features)

        elif len(original_shape) == 3:
            # [T, B, C_in]
            T, B, C_in = x.shape
            x_flat = x.reshape(T * B, C_in)
            x = self.linear(x_flat)  # [T*B, C_out]
            x = self.bn(x)  # [T*B, C_out]
            x = x.reshape(T, B, self.out_features)

        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        # Dropout（在LIF之前）
        if self.dropout is not None:
            x = self.dropout(x)

        # LIF神经元
        x = self.lif(x)

        return x


class SNNClassifier(nn.Module):
    """SNN分类器：用于意图分类任务"""

    def __init__(self, in_features: int, num_classes: int, hidden_dims: list,
                 neuron_cfg: Dict, dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # 构建隐藏层
        layers = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(SNNLinearBlock(
                prev_dim, hidden_dim, neuron_cfg, dropout=dropout
            ))
            prev_dim = hidden_dim

        # 输出层（无LIF激活）
        self.output_linear = nn.Linear(prev_dim, num_classes, bias=True)

        self.hidden_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, C_in] 时间步×批次×特征
        Returns:
            [B, num_classes] 分类logits
        """
        # 通过隐藏层
        for layer in self.hidden_layers:
            x = layer(x)

        # 最终线性层（无LIF激活）
        T, B, C = x.shape
        x_flat = x.reshape(T * B, C)
        x = self.output_linear(x_flat)  # [T*B, num_classes]
        x = x.reshape(T, B, self.num_classes)

        # 时间维度平均得到最终logits
        x = x.mean(dim=0)  # [B, num_classes]

        return x


class TimeDimAverage(nn.Module):
    """时间维度平均池化"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, ...] 时间步×批次×其他维度
        Returns:
            [B, ...] 时间平均后的输出
        """
        return x.mean(dim=0)


class TimeDimExpander(nn.Module):
    """扩展时间维度"""

    def __init__(self, time_steps: int = 4):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, ...] 输入张量
        Returns:
            [T, B, ...] 扩展时间维度后的张量
        """
        return x.unsqueeze(0).repeat(self.time_steps, 1, *([1] * (x.dim() - 1)))


def get_default_neuron_config():
    """获取默认神经元配置"""
    return {
        'spike_mode': 'lif',
        'tau': 2.0,
        'v_threshold': 1.0,
        'v_reset': 0.0,
        'detach_reset': True,
        'backend': 'torch',
        'time_steps': 4
    }


def check_spiking_jelly_available():
    """检查SpikingJelly是否可用"""
    return SPIKING_JELLY_AVAILABLE