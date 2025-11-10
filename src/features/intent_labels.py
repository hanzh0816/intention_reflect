"""
Intent Labels Feature

This module defines the IntentLabels feature class for storing intent classification labels
as training targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
)


@dataclass
class IntentLabels(AbstractModelFeature):
    """
    Intent classification labels for ego vehicle trajectory.

    Attributes:
        lateral_intent: Lateral intent class index (0-4)
            0: turn_left
            1: turn_right
            2: shift_left
            3: shift_right
            4: stay_in_lane

        longitudinal_intent: Longitudinal intent class index (0-3)
            0: accelerate
            1: maintain_speed
            2: decelerate
            3: stop
    """

    lateral_intent: torch.Tensor  # [B] or scalar
    longitudinal_intent: torch.Tensor  # [B] or scalar

    @classmethod
    def collate(cls, batch: List[IntentLabels]) -> IntentLabels:
        """
        Batch features together.

        Args:
            batch: List of IntentLabels to be batched

        Returns:
            Batched IntentLabels
        """
        lateral_intents = torch.stack([sample.lateral_intent for sample in batch], dim=0)
        longitudinal_intents = torch.stack([sample.longitudinal_intent for sample in batch], dim=0)

        return IntentLabels(
            lateral_intent=lateral_intents,
            longitudinal_intent=longitudinal_intents
        )

    def to_feature_tensor(self) -> IntentLabels:
        """
        Convert to tensor representation.

        Returns:
            IntentLabels with tensor data
        """
        if not isinstance(self.lateral_intent, torch.Tensor):
            lateral_tensor = torch.tensor(self.lateral_intent, dtype=torch.long)
        else:
            lateral_tensor = self.lateral_intent

        if not isinstance(self.longitudinal_intent, torch.Tensor):
            longitudinal_tensor = torch.tensor(self.longitudinal_intent, dtype=torch.long)
        else:
            longitudinal_tensor = self.longitudinal_intent

        return IntentLabels(
            lateral_intent=lateral_tensor,
            longitudinal_intent=longitudinal_tensor
        )

    def to_device(self, device: torch.device) -> IntentLabels:
        """
        Move feature to a device.

        Args:
            device: Target device

        Returns:
            IntentLabels on the specified device
        """
        return IntentLabels(
            lateral_intent=self.lateral_intent.to(device),
            longitudinal_intent=self.longitudinal_intent.to(device)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize to dictionary for caching.

        Returns:
            Dictionary of serialized data
        """
        return {
            "lateral_intent": self.lateral_intent.cpu().numpy() if isinstance(self.lateral_intent, torch.Tensor)
                              else self.lateral_intent,
            "longitudinal_intent": self.longitudinal_intent.cpu().numpy() if isinstance(self.longitudinal_intent, torch.Tensor)
                                   else self.longitudinal_intent,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> IntentLabels:
        """
        Deserialize from dictionary.

        Args:
            data: Dictionary of serialized data

        Returns:
            IntentLabels instance
        """
        return IntentLabels(
            lateral_intent=torch.tensor(data["lateral_intent"], dtype=torch.long),
            longitudinal_intent=torch.tensor(data["longitudinal_intent"], dtype=torch.long)
        )

    def unpack(self) -> List[IntentLabels]:
        """
        Unpack a batched feature to a list of features.

        Returns:
            List of IntentLabels
        """
        batch_size = self.lateral_intent.shape[0] if len(self.lateral_intent.shape) > 0 else 1

        if batch_size == 1 or len(self.lateral_intent.shape) == 0:
            return [self]

        return [
            IntentLabels(
                lateral_intent=self.lateral_intent[i],
                longitudinal_intent=self.longitudinal_intent[i]
            )
            for i in range(batch_size)
        ]

    @property
    def is_valid(self) -> bool:
        """
        Check if the feature is valid.

        Returns:
            Always True for intent labels
        """
        return True

    @property
    def num_lateral_classes(self) -> int:
        """Number of lateral intent classes."""
        return 5

    @property
    def num_longitudinal_classes(self) -> int:
        """Number of longitudinal intent classes."""
        return 4
