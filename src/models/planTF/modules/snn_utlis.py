"""
SNN基础工具类和组件，支持意图解码器的SNN实现
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union
from spikingjelly.activation_based import neuron, layer
from spikingjelly.activation_based import functional

print("Using SpikingJelly activation_based module")


class MockLIFNode(nn.Module):
    """当SpikingJelly不可用时的Mock LIF神经元"""

    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        backend: str = "torch",
    ):
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

    def __init__(
        self,
        spike_mode: str = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = True,
        backend: str = "torch",
    ):
        super().__init__()
        self.spike_mode = spike_mode

        if spike_mode == "lif":
            self.lif_neuron = neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                step_mode="m",  # multi-step模式
            )
        elif spike_mode == "plif":
            self.lif_neuron = neuron.ParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                step_mode="m",  # multi-step模式
            )
        elif spike_mode == "if":
            self.lif_neuron = neuron.IFNode(
                v_threshold=v_threshold,
                v_reset=v_reset,
                detach_reset=detach_reset,
                step_mode="m",  # multi-step模式
            )
        elif spike_mode == "ilif":
            self.lif_neuron = neuron.LIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                step_mode="m",  # multi-step模式
            )
        else:
            raise ValueError(f"Unsupported spike mode: {spike_mode}")

    def forward(self, x: torch.Tensor, return_v: bool = False):
        """
        Args:
            x: [T, B, ...] 时间步×批次×其他维度
            return_v: 是否返回膜电位
        Returns:
            如果return_v=False: [T, B, ...] 脉冲输出
            如果return_v=True: (spikes, membrane_potential) 元组
        """
        spikes = self.lif_neuron(x)
        if return_v:
            # 获取膜电位
            if hasattr(self.lif_neuron, "v"):
                # SpikingJelly神经元有v属性
                v = self.lif_neuron.v
            else:
                # Mock实现，使用输入作为膜电位近似
                v = x
            return spikes, v
        return spikes


def get_default_snn_config():
    """获取默认SNN配置（统一配置对象）

    Returns:
        dict: 包含以下键的SNN配置字典：
            - neuron_cfg: 神经元配置
            - time_steps: 时间步数
            - use_stdp: 是否使用STDP学习
            - stdp_cfg: STDP参数配置
    """
    return {
        "neuron_cfg": {
            "spike_mode": "lif",
            "tau": 2.0,
            "v_threshold": 1.0,
            "v_reset": 0.0,
            "detach_reset": True,
            "backend": "torch",
        },
        "time_steps": 8,
        "use_stdp": False,
        "stdp_cfg": {
            "learning_rate": 0.001,
            "A_pre": 0.01,
            "A_post": -0.01,
            "tau_pre": 10.0,
            "tau_post": 10.0,
        },
    }


def check_spiking_jelly_available():
    """检查SpikingJelly是否可用"""
    return SPIKING_JELLY_AVAILABLE
