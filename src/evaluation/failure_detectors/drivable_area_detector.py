"""
Drivable area detector (simplified implementation).
"""
from typing import List
import logging

from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

from src.evaluation.failure_detectors.detection_engine import DrivableAreaViolationEvent

logger = logging.getLogger(__name__)


class DrivableAreaDetector:
    """
    Detects drivable area violations.

    This is a simplified implementation that returns empty list.
    Full implementation should check ego corners against drivable area polygons.
    """

    def __init__(self, tolerance_m: float = 0.5):
        self._tolerance = tolerance_m

    def detect(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario,
    ) -> List[DrivableAreaViolationEvent]:
        """
        Detect drivable area violations.

        Returns list of DrivableAreaViolationEvent.
        """
        violations: List[DrivableAreaViolationEvent] = []

        # Simplified implementation - return empty list
        # Full implementation should:
        # 1. Extract ego corners from each state
        # 2. Check if corners are in drivable area using map API
        # 3. Calculate distance to drivable area boundary
        # 4. Create violation events when distance exceeds tolerance

        logger.debug(f"Detected {len(violations)} drivable area violations")
        return violations
