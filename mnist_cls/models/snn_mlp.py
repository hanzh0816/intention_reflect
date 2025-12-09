import sys

sys.path.append("/home/hzh/code/planning/planTF")

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.planTF.modules.snn_layers import SNNLinearBlock, TimeDimExpander
from src.models.planTF.modules.snn_utlis import LIFNeuron, get_default_snn_config


class SNNMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_dim: int = 512,
        num_classes: int = 10,
        snn_cfg: Optional[Dict] = None,
        use_stdp: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_stdp = use_stdp

        if snn_cfg is None:
            snn_cfg = get_default_snn_config()

        self.neuron_cfg = snn_cfg["neuron_cfg"].copy()
        self.time_steps = snn_cfg["time_steps"]

        self.input_norm = nn.BatchNorm1d(input_size)
        self.time_expander = TimeDimExpander(time_steps=self.time_steps)
        self.hidden = SNNLinearBlock(input_size, hidden_dim, self.neuron_cfg)

        self.output_linear = nn.Linear(hidden_dim, num_classes, bias=True)

        self.output_lif = LIFNeuron(**self.neuron_cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)

        B = x.size(0)
        x = self.input_norm(x)
        x = self.time_expander(x)
        x = self.hidden(x)

        hidden_output = x
        T, B, C = x.shape

        # 通过输出层的Linear
        x_flat = x.reshape(T * B, C)
        x = self.output_linear(x_flat)
        x = x.reshape(T, B, self.num_classes)

        # 通过LIF神经元，获取脉冲和膜电位
        spike_trains, membrane_potential = self.output_lif(x, return_v=True)
        logits = membrane_potential

        # 计算脉冲计数
        spike_counts = spike_trains.sum(dim=0)

        if self.use_stdp is False:
            return logits

        return {
            "logits": logits,
            "spike_trains": spike_trains,
            "spike_counts": spike_counts,
            "hidden_output": hidden_output,
            "membrane_potential": membrane_potential,
        }
