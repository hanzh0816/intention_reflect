"""
Failure detection engine and data classes.
"""
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from nuplan.planning.simulation.history.simulation_history import SimulationHistory
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)


@dataclass
class CollisionEvent:
    """Collision event details."""

    timestamp_us: int
    collision_type: str
    tracked_object_type: str
    collision_energy: float
    is_at_fault: bool
    ego_speed: float
    object_speed: float


@dataclass
class SpeedViolationEvent:
    """Speed violation event details."""

    start_timestamp_us: int
    end_timestamp_us: int
    duration_us: int
    max_overspeed_mps: float
    mean_overspeed_mps: float
    speed_limit_mps: Optional[float]


@dataclass
class DeadlockEvent:
    """Deadlock/stalled event details."""

    start_timestamp_us: int
    end_timestamp_us: int
    duration_seconds: float
    total_progress_meters: float
    expected_progress_meters: float
    progress_ratio: float
    final_speed_mps: float


@dataclass
class DrivableAreaViolationEvent:
    """Drivable area violation event details."""

    timestamp_us: int
    max_distance_to_drivable_area: float
    violation_duration_us: int


@dataclass
class FailureDetectionResult:
    """Aggregated failure detection results."""

    scenario_type: str
    scenario_name: str
    log_name: str
    planner_name: str
    duration_seconds: float

    # Failure events
    collision_events: List[CollisionEvent] = field(default_factory=list)
    speed_violations: List[SpeedViolationEvent] = field(default_factory=list)
    deadlock_event: Optional[DeadlockEvent] = None
    drivable_area_violations: List[DrivableAreaViolationEvent] = field(default_factory=list)

    def has_any_failure(self) -> bool:
        """Check if any failure detected."""
        return (
            len(self.collision_events) > 0
            or len(self.speed_violations) > 0
            or self.deadlock_event is not None
            or len(self.drivable_area_violations) > 0
        )

    def get_failure_severity(self) -> str:
        """Determine overall severity: CRITICAL, HIGH, MEDIUM."""
        # CRITICAL: Any at-fault collision
        if any(c.is_at_fault for c in self.collision_events):
            return "CRITICAL"

        # HIGH: Deadlock or severe speed violation
        if self.deadlock_event or any(v.max_overspeed_mps > 10 for v in self.speed_violations):
            return "HIGH"

        # MEDIUM: Other violations
        if self.has_any_failure():
            return "MEDIUM"

        return "NONE"

    def get_failure_summary(self) -> str:
        """Human-readable summary."""
        parts = []
        if self.collision_events:
            at_fault = sum(1 for c in self.collision_events if c.is_at_fault)
            parts.append(f"{len(self.collision_events)} collisions ({at_fault} at-fault)")
        if self.speed_violations:
            parts.append(f"{len(self.speed_violations)} speed violations")
        if self.deadlock_event:
            parts.append(f"deadlock ({self.deadlock_event.duration_seconds:.1f}s)")
        if self.drivable_area_violations:
            parts.append(f"{len(self.drivable_area_violations)} drivable area violations")
        return ", ".join(parts) if parts else "no failures"


class FailureDetectionEngine:
    """
    Orchestrates all failure detectors.
    """

    def __init__(
        self,
        enable_collision: bool = True,
        enable_speed: bool = True,
        enable_deadlock: bool = True,
        enable_drivable_area: bool = True,
        max_overspeed_threshold: float = 5.0,
        deadlock_duration_threshold: float = 10.0,
        deadlock_min_progress_ratio: float = 0.1,
        drivable_area_tolerance: float = 0.5,
    ):
        # Initialize detectors (imported here to avoid circular imports)
        self._collision_detector = None
        self._speed_detector = None
        self._deadlock_detector = None
        self._drivable_area_detector = None

        if enable_collision:
            from src.evaluation.failure_detectors.collision_detector import CollisionDetector
            self._collision_detector = CollisionDetector()

        if enable_speed:
            from src.evaluation.failure_detectors.speed_violation_detector import SpeedViolationDetector
            self._speed_detector = SpeedViolationDetector(max_overspeed_threshold)

        if enable_deadlock:
            from src.evaluation.failure_detectors.deadlock_detector import DeadlockDetector
            self._deadlock_detector = DeadlockDetector(
                deadlock_duration_threshold,
                deadlock_min_progress_ratio
            )

        if enable_drivable_area:
            from src.evaluation.failure_detectors.drivable_area_detector import DrivableAreaDetector
            self._drivable_area_detector = DrivableAreaDetector(drivable_area_tolerance)

    def detect_failures(
        self,
        history: "SimulationHistory",
        scenario: "AbstractScenario",
        planner_name: str,
    ) -> FailureDetectionResult:
        """
        Run all enabled detectors and aggregate results.

        Args:
            history: Complete simulation history
            scenario: Scenario object
            planner_name: Name of the planner

        Returns:
            FailureDetectionResult with all detected failures
        """
        result = FailureDetectionResult(
            scenario_type=scenario.scenario_type,
            scenario_name=scenario.scenario_name,
            log_name=scenario.log_name,
            planner_name=planner_name,
            duration_seconds=len(history) * history.interval_seconds if len(history) > 1 else 0.0,
        )

        # Run collision detection
        if self._collision_detector:
            result.collision_events = self._collision_detector.detect(history, scenario)

        # Run speed violation detection
        if self._speed_detector:
            result.speed_violations = self._speed_detector.detect(history, scenario)

        # Run deadlock detection
        if self._deadlock_detector:
            result.deadlock_event = self._deadlock_detector.detect(history, scenario)

        # Run drivable area detection
        if self._drivable_area_detector:
            result.drivable_area_violations = self._drivable_area_detector.detect(history, scenario)

        return result
