import sys
from tkinter import X

sys.path.append("/home/hzh/code/planning/planTF")

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.planTF.modules.snn_layers import SNNLinearBlock, TimeDimExpander
from src.models.planTF.modules.snn_utlis import LIFNeuron, get_default_snn_config

from spikingjelly.activation_based import neuron, layer


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

        self.time_expander = TimeDimExpander(time_steps=self.time_steps)
        # self.hidden_linear = layer.Linear(input_size, hidden_dim, bias=False)
        # self.hidden_lif = neuron.IFNode()

        self.output_linear = layer.Linear(input_size, num_classes, bias=False)
        self.output_lif = neuron.IFNode()

        # self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.hidden_linear.weight.data, 0.4)
        nn.init.constant_(self.output_linear.weight.data, 0.4)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)

        x = self.time_expander(x)
        # T, B, L = x.shape
        # x = self.hidden_linear(x)
        # x = self.hidden_lif(x)

        x = self.output_linear(x)
        # logits = x.mean(0)
        x = self.output_lif(x)
        logits = x.mean(0)

        return {"logits": logits, "spike_train": x}
