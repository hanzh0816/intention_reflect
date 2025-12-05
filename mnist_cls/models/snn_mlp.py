import sys
sys.path.append('/home/hzh/code/planning/planTF')

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.planTF.modules.snn_layers import SNNLinearBlock, TimeDimExpander
from src.models.planTF.modules.snn_utlis import LIFNeuron, get_default_snn_config


class SNNMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 784,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        num_classes: int = 10,
        snn_cfg: Optional[Dict] = None,
        dropout: float = 0.2,
        use_stdp: bool = False,
        population_size: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_stdp = use_stdp
        self.population_size = population_size

        if snn_cfg is None:
            snn_cfg = get_default_snn_config()

        self.neuron_cfg = snn_cfg["neuron_cfg"].copy()
        self.time_steps = snn_cfg["time_steps"]

        self.input_norm = nn.LayerNorm(input_size)
        self.time_expander = TimeDimExpander(time_steps=self.time_steps)
        self.hidden1 = SNNLinearBlock(input_size, hidden_dim1, self.neuron_cfg, dropout)
        self.hidden2 = SNNLinearBlock(hidden_dim1, hidden_dim2, self.neuron_cfg, dropout)

        self.output_neurons = num_classes * population_size
        self.output_linear = nn.Linear(hidden_dim2, self.output_neurons, bias=True)

        if use_stdp or population_size > 1:
            self.output_lif = LIFNeuron(**self.neuron_cfg)
        else:
            self.output_lif = None

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
        x = self.hidden1(x)
        x = self.hidden2(x)

        hidden_output = x
        T, B, C = x.shape

        x_flat = x.reshape(T * B, C)
        x = self.output_linear(x_flat)
        x = x.reshape(T, B, self.output_neurons)

        if self.population_size == 1 and not self.use_stdp:
            return x.mean(dim=0)

        x = x.reshape(T, B, self.num_classes, self.population_size)
        x_flat = x.reshape(T, B, self.output_neurons)
        spike_trains_flat = self.output_lif(x_flat)
        spike_trains = spike_trains_flat.reshape(T, B, self.num_classes, self.population_size)

        logits = x.mean(dim=3).mean(dim=0)
        spike_counts = spike_trains.sum(dim=(0, 3))

        return {
            "logits": logits,
            "spike_trains": spike_trains,
            "spike_counts": spike_counts,
            "hidden_output": hidden_output,
        }
