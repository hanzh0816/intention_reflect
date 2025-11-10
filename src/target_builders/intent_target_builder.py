"""
Intent Target Builder

This module defines the IntentTargetBuilder that computes intent labels from expert trajectories
during feature caching.
"""

from __future__ import annotations

from typing import Type, List

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from src.features.intent_labels import IntentLabels
from src.utils.intent_classification import (
    classify_intent_from_states,
    get_lateral_intent_index,
    get_longitudinal_intent_index,
    IntentClassificationConfig,
)


class IntentTargetBuilder(AbstractTargetBuilder):
    """
    Target builder that computes intent classification labels from expert trajectories.

    This builder extracts the ego vehicle's future trajectory from a scenario and
    classifies the short-term intent in both lateral and longitudinal dimensions.
    """

    def __init__(
        self,
        time_horizon: float = 2.0,
        sample_interval: float = 0.1,
        config: IntentClassificationConfig = None
    ) -> None:
        """
        Initialize the IntentTargetBuilder.

        Args:
            time_horizon: Time window for intent classification (seconds)
            sample_interval: Time interval between trajectory samples (seconds)
            config: Configuration for intent classification thresholds
        """
        self._time_horizon = time_horizon
        self._sample_interval = sample_interval

        if config is None:
            config = IntentClassificationConfig(sample_interval=sample_interval)
        self._config = config

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """
        Get unique identifier for this target.

        Returns:
            Unique name string
        """
        return "intent"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """
        Get the feature type produced by this builder.

        Returns:
            IntentLabels class
        """
        return IntentLabels  # type: ignore

    def get_targets(self, scenario: AbstractScenario) -> IntentLabels:
        """
        Compute intent labels from scenario's expert trajectory.

        Args:
            scenario: The scenario containing expert trajectory

        Returns:
            IntentLabels containing lateral and longitudinal intent indices
        """
        # Extract ego states
        ego_states = self._extract_ego_states(scenario)

        # Classify intent from current state (index where past meets future)
        # For scenarios, initial_ego_state is the "present" moment
        current_idx = 0  # We start from initial state and look into future

        lateral_intent_str, longitudinal_intent_str = classify_intent_from_states(
            ego_states=ego_states,
            current_idx=current_idx,
            time_horizon=self._time_horizon,
            config=self._config
        )

        # Convert to class indices
        lateral_idx = get_lateral_intent_index(lateral_intent_str)
        longitudinal_idx = get_longitudinal_intent_index(longitudinal_intent_str)

        return IntentLabels(
            lateral_intent=lateral_idx,
            longitudinal_intent=longitudinal_idx
        )

    def _extract_ego_states(self, scenario: AbstractScenario) -> List[EgoState]:
        """
        Extract ego states from scenario.

        Args:
            scenario: The scenario to extract states from

        Returns:
            List of EgoState objects starting from initial state
        """
        # Number of samples needed for intent classification
        num_samples = int(self._time_horizon / self._sample_interval)

        # Get current state
        ego_current_state = scenario.initial_ego_state

        # Get future trajectory
        future_ego_trajectory = scenario.get_ego_future_trajectory(
            iteration=0,
            time_horizon=self._time_horizon,
            num_samples=num_samples,
        )

        # Combine into list: [current, future...]
        ego_state_list = [ego_current_state] + list(future_ego_trajectory)

        return ego_state_list
