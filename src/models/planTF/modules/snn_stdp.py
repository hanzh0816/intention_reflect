"""
Spike-Timing-Dependent Plasticity (STDP) with Reward Modulation
基于SpikingJelly STDPLearner的实现,去除群体编码,简化为MLP网络STDP
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable
from spikingjelly.activation_based.learning import STDPLearner


from spikingjelly.activation_based.learning import (
    stdp_linear_single_step,
    stdp_conv2d_single_step,
    stdp_conv1d_single_step,
    stdp_multi_step,
)


class RSTDPLearner(STDPLearner):
    def step(self, on_grad=True, scale=1, reward=None, neuron_index=None):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == "s":
            if isinstance(self.synapse, nn.Linear):
                stdp_f = stdp_linear_single_step
            elif isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Conv1d):
                stdp_f = stdp_conv1d_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == "m":
            if isinstance(self.synapse, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                stdp_f = stdp_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)  # [batch_size, N_in]
            out_spike = self.out_spike_monitor.records.pop(0)  # [batch_size, N_out]

            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse,
                in_spike,
                out_spike,
                self.trace_pre,
                self.trace_post,
                self.tau_pre,
                self.tau_post,
                self.f_pre,
                self.f_post,
            )
            if scale != 1.0:
                dw *= scale

            delta_w = dw if (delta_w is None) else (delta_w + dw)

        # 如果指定了neuron_index，则只保留对应行的delta_w，其余置0
        if neuron_index is not None:
            mask = torch.zeros_like(delta_w)
            for i in range(neuron_index.shape[0]):
                mask[neuron_index[i], :] += delta_w[neuron_index[i], :]
            delta_w = mask

        if on_grad:

            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
            return delta_w
        else:
            # 如果指定了neuron_index，则只保留对应行的delta_w，其余置0
            if neuron_index is not None:
                mask = torch.zeros_like(delta_w)
                mask[neuron_index, :] = delta_w[neuron_index, :]
                delta_w = mask
            return delta_w

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
            return delta_w
        else:
            return delta_w
