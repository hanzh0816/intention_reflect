"""
SNN version of common layers for PlanTF

This module provides SNN-compatible versions of common neural network layers
using Spiking Neural Network components from spikingjelly.
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


def build_snn_mlp(c_in, channels, tau=2.0, backend='torch'):
    """
    Build a multi-layer perceptron with SNN components.

    Args:
        c_in: Input channel dimension
        channels: List of output dimensions for each layer
        tau: Time constant for LIF neurons
        backend: Backend for LIF neurons ('torch' or 'cupy')

    Returns:
        nn.Sequential: SNN MLP module
    """
    layers = []
    num_layers = len(channels)

    for k in range(num_layers):
        if k == num_layers - 1:
            # Last layer: Linear + BN + LIF (no intermediate activation)
            layers.extend([
                nn.Linear(c_in, channels[k], bias=False),
                nn.BatchNorm1d(channels[k]),
                MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
            ])
        else:
            # Intermediate layers: Linear + BN + LIF
            layers.extend([
                nn.Linear(c_in, channels[k], bias=False),
                nn.BatchNorm1d(channels[k]),
                MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
            ])
            c_in = channels[k]

    return nn.Sequential(*layers)


class SNNMlp(nn.Module):
    """
    SNN version of MLP used in transformer blocks.

    Input shape: [T, B, L, C] where T is time steps
    Output shape: [T, B, L, C]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        tau=2.0,
        drop=0.0,
        backend='torch',
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.drop1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [T, B, L, C]

        Returns:
            Output tensor of shape [T, B, L, C]
        """
        T, B, L, C = x.shape

        # Reshape for linear and batch norm: [T, B, L, C] -> [T*B*L, C]
        x_flat = x.reshape(T * B * L, C)

        # First layer
        x = self.fc1(x_flat)
        # BatchNorm over the feature dimension
        x = self.bn1(x.reshape(T * B, L, -1).transpose(1, 2)).transpose(1, 2)
        x = x.reshape(T, B, L, -1)
        x = self.lif1(x)
        x = self.drop1(x)

        # Second layer
        x_flat = x.reshape(T * B * L, -1)
        x = self.fc2(x_flat)
        x = self.bn2(x.reshape(T * B, L, -1).transpose(1, 2)).transpose(1, 2)
        x = x.reshape(T, B, L, -1)
        x = self.lif2(x)
        x = self.drop2(x)

        return x
