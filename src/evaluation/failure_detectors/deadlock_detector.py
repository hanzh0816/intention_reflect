"""
Deadlock detector.
"""
from typing import Optional
import logging
import numpy as np

from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from src.evaluation.failure_detectors.detection_engine import DeadlockEvent

logger = logging.getLogger(__name__)


class DeadlockDetector:
    """
    Detects deadlock/stalled situations.

    Criteria:
    - Simulation duration > threshold (e.g., 10s)
    - Progress ratio < threshold (e.g., < 10% of expected)
    - Final speed very low
    """

    def __init__(
        self,
        duration_threshold_s: float = 10.0,
        min_progress_ratio: float = 0.1,
    ):
        self._duration_threshold = duration_threshold_s
        self._min_progress_ratio = min_progress_ratio

    def detect(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario,
    ) -> Optional[DeadlockEvent]:
        """
        Detect if ego is deadlocked/stalled.

        Returns DeadlockEvent if deadlocked, None otherwise.
        """
        if len(history) < 2:
            return None

        # Calculate simulation duration
        duration_s = (len(history) - 1) * history.interval_seconds

        # Calculate actual progress
        ego_states = history.extract_ego_state
        start_pos = ego_states[0].rear_axle.point
        end_pos = ego_states[-1].rear_axle.point

        actual_progress = np.sqrt(
            (end_pos.x - start_pos.x)**2 + (end_pos.y - start_pos.y)**2
        )

        # Calculate expected progress (based on scenario duration and typical speeds)
        expected_progress = self._compute_expected_progress(scenario, duration_s)

        # Calculate progress ratio
        progress_ratio = actual_progress / expected_progress if expected_progress > 0 else 1.0

        # Get final speed
        final_speed = ego_states[-1].dynamic_car_state.speed

        # Determine if deadlocked
        is_deadlocked = (
            duration_s > self._duration_threshold
            and progress_ratio < self._min_progress_ratio
        )

        if is_deadlocked:
            logger.info(
                f"Deadlock detected: duration={duration_s:.1f}s, "
                f"progress={actual_progress:.1f}m, "
                f"ratio={progress_ratio:.2%}"
            )

            return DeadlockEvent(
                start_timestamp_us=ego_states[0].time_point.time_us,
                end_timestamp_us=ego_states[-1].time_point.time_us,
                duration_seconds=duration_s,
                total_progress_meters=float(actual_progress),
                expected_progress_meters=float(expected_progress),
                progress_ratio=float(progress_ratio),
                final_speed_mps=float(final_speed),
            )

        return None

    def _compute_expected_progress(self, scenario, duration_s: float) -> float:
        """
        Compute expected progress based on scenario.

        Simplified: assume average speed of 15 m/s
        """
        # Could use expert trajectory length from scenario
        average_speed = 15.0  # m/s
        return average_speed * duration_s
