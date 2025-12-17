"""
Speed violation detector.
"""
from typing import List, Optional
import logging
import statistics

from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from src.evaluation.failure_detectors.detection_engine import SpeedViolationEvent

logger = logging.getLogger(__name__)


class SpeedViolationDetector:
    """
    Detects speed limit violations.

    Reuses logic from:
    - nuplan.planning.metrics.evaluation_metrics.common.speed_limit_compliance
    """

    def __init__(self, max_overspeed_threshold_mps: float = 5.0):
        self._max_overspeed_threshold = max_overspeed_threshold_mps

    def detect(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario,
    ) -> List[SpeedViolationEvent]:
        """
        Detect speed violations exceeding threshold.

        Returns list of SpeedViolationEvent.
        """
        violations: List[SpeedViolationEvent] = []

        # State machine for violation tracking
        current_violation: Optional[dict] = None

        for sample in history.data:
            ego_state = sample.ego_state
            timestamp = ego_state.time_point.time_us

            # Get speed limit (simplified - use fixed value)
            speed_limit = self._get_speed_limit(scenario)
            if speed_limit is None:
                continue

            # Calculate overspeed
            overspeed = ego_state.dynamic_car_state.speed - speed_limit

            # Check if violating threshold
            if overspeed > self._max_overspeed_threshold:
                if current_violation is None:
                    # Start new violation
                    current_violation = {
                        'start_timestamp': timestamp,
                        'speed_limit': speed_limit,
                        'overspeed_values': [overspeed],
                    }
                else:
                    # Continue violation
                    current_violation['overspeed_values'].append(overspeed)
            else:
                # End violation if exists
                if current_violation:
                    violations.append(self._finalize_violation(current_violation, timestamp))
                    current_violation = None

        # Finalize any open violation
        if current_violation:
            last_timestamp = history.data[-1].ego_state.time_point.time_us
            violations.append(self._finalize_violation(current_violation, last_timestamp))

        logger.debug(f"Detected {len(violations)} speed violations")
        return violations

    def _finalize_violation(self, violation_data: dict, end_timestamp: int) -> SpeedViolationEvent:
        """Convert violation tracking data to event."""
        start_ts = violation_data['start_timestamp']
        overspeed_values = violation_data['overspeed_values']

        return SpeedViolationEvent(
            start_timestamp_us=start_ts,
            end_timestamp_us=end_timestamp,
            duration_us=end_timestamp - start_ts,
            max_overspeed_mps=max(overspeed_values),
            mean_overspeed_mps=statistics.mean(overspeed_values),
            speed_limit_mps=violation_data['speed_limit'],
        )

    def _get_speed_limit(self, scenario) -> Optional[float]:
        """Extract speed limit from scenario (simplified)."""
        # Simplified implementation: return default 25 m/s
        return 25.0
