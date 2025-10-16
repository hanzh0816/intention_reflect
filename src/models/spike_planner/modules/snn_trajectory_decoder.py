"""
SNN version of TrajectoryDecoder module

This module decodes the encoded features into multimodal trajectory predictions
using SNN components.
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode


class SNNTrajectoryDecoder(nn.Module):
    """
    SNN version of TrajectoryDecoder

    Generates multimodal trajectory predictions from encoded features.
    """

    def __init__(
        self,
        embed_dim,
        num_modes,
        future_steps,
        out_channels,
        tau=2.0,
        backend='torch',
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.out_channels = out_channels

        # Multimodal projection
        self.multimodal_proj = nn.Linear(embed_dim, num_modes * embed_dim, bias=False)
        self.multimodal_bn = nn.BatchNorm1d(num_modes * embed_dim)
        self.multimodal_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        # Trajectory location prediction branch
        hidden = 2 * embed_dim

        self.loc_fc1 = nn.Linear(embed_dim, hidden, bias=False)
        self.loc_bn1 = nn.BatchNorm1d(hidden)
        self.loc_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        # Output layer (no activation)
        self.loc_fc2 = nn.Linear(hidden, future_steps * out_channels)

        # Mode probability prediction branch
        self.pi_fc1 = nn.Linear(embed_dim, hidden, bias=False)
        self.pi_bn1 = nn.BatchNorm1d(hidden)
        self.pi_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

        # Output layer (no activation, will apply softmax later)
        self.pi_fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [T, B, C] where T is time steps

        Returns:
            loc: [B, num_modes, future_steps, out_channels] - trajectory predictions
            pi: [B, num_modes] - mode probabilities (logits)
        """
        T, B, C = x.shape

        # Multimodal projection: [T, B, C] -> [T, B, num_modes * C]
        x_flat = x.reshape(T * B, C)
        x = self.multimodal_proj(x_flat)
        x = self.multimodal_bn(x)
        x = x.reshape(T, B, -1)
        x = self.multimodal_lif(x)

        # Average over time dimension to get single representation
        x = x.mean(dim=0)  # [B, num_modes * embed_dim]

        # Reshape to separate modes: [B, num_modes, embed_dim]
        x = x.view(B, self.num_modes, self.embed_dim)
        x_flat = x.reshape(B * self.num_modes, self.embed_dim)

        # Location prediction branch
        loc = self.loc_fc1(x_flat)
        loc = self.loc_bn1(loc)
        # Add time dimension for LIF
        loc = loc.unsqueeze(0).repeat(T, 1, 1)
        loc = self.loc_lif1(loc)
        # Average over time
        loc = loc.mean(dim=0)  # [B * num_modes, hidden]

        loc = self.loc_fc2(loc)
        loc = loc.view(B, self.num_modes, self.future_steps, self.out_channels)

        # Mode probability branch
        pi = self.pi_fc1(x_flat)
        pi = self.pi_bn1(pi)
        # Add time dimension for LIF
        pi = pi.unsqueeze(0).repeat(T, 1, 1)
        pi = self.pi_lif1(pi)
        # Average over time
        pi = pi.mean(dim=0)  # [B * num_modes, hidden]

        pi = self.pi_fc2(pi)
        pi = pi.view(B, self.num_modes)

        return loc, pi
