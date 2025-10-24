"""
Unified LIF Neuron Wrapper for Spike Planner

This module provides a unified interface for different types of spiking neurons
(LIF, PLIF, IF, ILIF) used in the spike planner model.
"""

from typing import Literal

import torch.nn as nn
from spikingjelly.clock_driven.neuron import (
    MultiStepIFNode,
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)
from torch import Tensor


class LIFNeuron(nn.Module):
    """
    Unified wrapper for LIF neuron node interface.

    This wrapper provides a consistent interface for different types of
    spiking neurons, making it easy to switch between LIF, PLIF, IF, and ILIF
    neurons with the same API.

    Args:
        spike_mode: Type of spiking neuron ('lif', 'plif', 'if', 'ilif')
        tau: Time constant for LIF neurons (default: 2.0)
        v_threshold: Voltage threshold for spiking (default: 1.0)
        v_reset: Reset voltage after spike (default: 0.0)
        detach_reset: Whether to detach reset in gradient computation (default: False)
        backend: Computation backend ('torch' or 'cupy') (default: 'torch')
        **kwargs: Additional keyword arguments passed to the neuron constructor
    """

    def __init__(
        self,
        spike_mode: Literal["lif", "plif", "if", "ilif"] = "lif",
        tau: float = 2.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        detach_reset: bool = False,
        backend: Literal["torch", "cupy"] = "torch",
        **kwargs,
    ):
        super().__init__()

        if spike_mode == "lif":
            self.lif_neuron = MultiStepLIFNode(
                tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
                **kwargs,
            )
        elif spike_mode == "plif":
            self.lif_neuron = MultiStepParametricLIFNode(
                init_tau=tau,
                v_threshold=v_threshold,
                detach_reset=detach_reset,
                v_reset=v_reset,
                backend=backend,
                **kwargs,
            )
        elif spike_mode == "if":
            self.lif_neuron = MultiStepIFNode(
                v_threshold=v_threshold,
                v_reset=v_reset,
                detach_reset=detach_reset,
                backend=backend,
                **kwargs,
            )
        elif spike_mode == "ilif":
            # ILIF mode is a special variant of LIF
            # Check if MultiStepLIFNode supports ilif parameter
            try:
                self.lif_neuron = MultiStepLIFNode(
                    tau=tau,
                    v_threshold=v_threshold,
                    detach_reset=detach_reset,
                    v_reset=v_reset,
                    backend=backend,
                    ilif=True,
                    **kwargs,
                )
            except TypeError:
                # Fallback if ilif parameter is not supported
                raise NotImplementedError(
                    "ILIF mode is not supported by the current spikingjelly version. "
                    "Please use 'lif', 'plif', or 'if' mode instead."
                )
        else:
            raise NotImplementedError(
                f"Only 'lif', 'plif', 'if', and 'ilif' spiking neuron modes are supported. "
                f"Got: {spike_mode}"
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the spiking neuron.

        Args:
            x: Input tensor

        Returns:
            Output spike tensor
        """
        return self.lif_neuron(x)
