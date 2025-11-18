"""
SNN基础工具类和组件，支持意图解码器的SNN实现
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

try:
    # 优先尝试clock_driven模块（MultiStep支持）
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode,
        MultiStepParametricLIFNode,
        MultiStepIFNode,
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
        time_steps: int = 4,
    ):
        super().__init__()
        self.spike_mode = spike_mode
        self.time_steps = time_steps

        if SPIKING_JELLY_AVAILABLE:
            # 使用SpikingJelly的神经元
            if SPIKING_MODE == "clock_driven":
                # clock_driven模块 - 使用MultiStep神经元
                if spike_mode == "lif":
                    self.lif_neuron = MultiStepLIFNode(
                        tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                elif spike_mode == "plif":
                    self.lif_neuron = MultiStepParametricLIFNode(
                        init_tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                elif spike_mode == "if":
                    self.lif_neuron = MultiStepIFNode(
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        detach_reset=detach_reset,
                        backend=backend,
                    )
                elif spike_mode == "ilif":
                    self.lif_neuron = MultiStepLIFNode(
                        tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                else:
                    raise ValueError(f"Unsupported spike mode: {spike_mode}")

            elif SPIKING_MODE == "activation_based":
                # activation_based模块 - 使用简化命名
                if spike_mode == "lif":
                    self.lif_neuron = neuron.LIFNode(
                        tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                elif spike_mode == "plif":
                    self.lif_neuron = neuron.ParametricLIFNode(
                        init_tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                elif spike_mode == "if":
                    self.lif_neuron = neuron.IFNode(
                        v_threshold=v_threshold,
                        v_reset=v_reset,
                        detach_reset=detach_reset,
                        backend=backend,
                    )
                elif spike_mode == "ilif":
                    self.lif_neuron = neuron.LIFNode(
                        tau=tau,
                        v_threshold=v_threshold,
                        detach_reset=detach_reset,
                        v_reset=v_reset,
                        backend=backend,
                    )
                else:
                    raise ValueError(f"Unsupported spike mode: {spike_mode}")

            else:
                # Mock模式
                self.lif_neuron = MockLIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    v_reset=v_reset,
                    detach_reset=detach_reset,
                    backend=backend,
                )
        else:
            # 使用mock实现
            self.lif_neuron = MockLIFNode(
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                detach_reset=detach_reset,
                backend=backend,
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, ...] 时间步×批次×其他维度
        Returns:
            [T, B, ...] 脉冲输出
        """
        return self.lif_neuron(x)



def get_default_neuron_config():
    """获取默认神经元配置"""
    return {
        "spike_mode": "lif",
        "tau": 2.0,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "detach_reset": True,
        "backend": "torch",
        "time_steps": 4,
    }


def check_spiking_jelly_available():
    """检查SpikingJelly是否可用"""
    return SPIKING_JELLY_AVAILABLE
