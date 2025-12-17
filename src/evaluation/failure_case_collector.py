"""
Failure case collector callback for simulation.
"""
from pathlib import Path
import logging

from nuplan.planning.simulation.callback.abstract_callback import AbstractCallback
from nuplan.planning.simulation.history.simulation_history import SimulationHistory, SimulationHistorySample
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory

from src.evaluation.failure_detectors.detection_engine import FailureDetectionEngine
from src.evaluation.database_manager import DatabaseManager
from src.evaluation.history_serializer import HistorySerializer

logger = logging.getLogger(__name__)


class FailureCaseCollectorCallback(AbstractCallback):
    """
    Callback to collect and store failure cases during simulation.

    Failure types detected:
    - At-fault collisions
    - Speed limit violations (>5 m/s over limit)
    - Deadlock/stalled (>10s with minimal progress)
    - Drivable area violations
    """

    def __init__(
        self,
        database_path: str,
        detection_mode: str = "standard",  # standard, strict, permissive
        enable_collision_detection: bool = True,
        enable_speed_detection: bool = True,
        enable_deadlock_detection: bool = True,
        enable_drivable_area_detection: bool = True,
        # Thresholds
        max_overspeed_threshold_mps: float = 5.0,
        deadlock_duration_threshold_s: float = 10.0,
        deadlock_min_progress_ratio: float = 0.1,
        drivable_area_tolerance_m: float = 0.5,
    ):
        """
        Initialize the failure case collector.

        Args:
            database_path: Path to SQLite database
            detection_mode: Detection strictness level
            enable_*_detection: Toggle individual detectors
            *_threshold: Configurable thresholds
        """
        self._database_path = Path(database_path)
        self._detection_mode = detection_mode

        # Initialize components
        self._db_manager = DatabaseManager(self._database_path)
        self._serializer = HistorySerializer()
        self._detection_engine = FailureDetectionEngine(
            enable_collision=enable_collision_detection,
            enable_speed=enable_speed_detection,
            enable_deadlock=enable_deadlock_detection,
            enable_drivable_area=enable_drivable_area_detection,
            max_overspeed_threshold=max_overspeed_threshold_mps,
            deadlock_duration_threshold=deadlock_duration_threshold_s,
            deadlock_min_progress_ratio=deadlock_min_progress_ratio,
            drivable_area_tolerance=drivable_area_tolerance_m,
        )

        logger.info(
            f"FailureCaseCollector initialized: "
            f"database={database_path}, mode={detection_mode}"
        )

    def on_initialization_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_initialization_end(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_step_end(self, setup: SimulationSetup, planner: AbstractPlanner, sample: SimulationHistorySample) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_start(self, setup: SimulationSetup, planner: AbstractPlanner) -> None:
        """Inherited, see superclass."""
        pass

    def on_planner_end(
        self, setup: SimulationSetup, planner: AbstractPlanner, trajectory: AbstractTrajectory
    ) -> None:
        """Inherited, see superclass."""
        pass

    def on_simulation_start(self, setup: SimulationSetup) -> None:
        """
        Ensure database is initialized.

        :param setup: simulation setup
        """
        self._db_manager.initialize_database()

    def on_simulation_end(
        self,
        setup: SimulationSetup,
        planner: AbstractPlanner,
        history: SimulationHistory
    ) -> None:
        """
        Main detection and collection logic.

        This is where all failure detection happens after simulation completes.

        :param setup: simulation setup
        :param planner: planner when simulation ends
        :param history: resulting from simulation
        """
        try:
            # Run failure detection
            detection_result = self._detection_engine.detect_failures(
                history=history,
                scenario=setup.scenario,
                planner_name=planner.name()
            )

            # Only store if failures detected
            if detection_result.has_any_failure():
                logger.info(
                    f"Failures detected in {setup.scenario.scenario_name}: "
                    f"{detection_result.get_failure_summary()}"
                )

                # Serialize history
                serialized_history = self._serializer.serialize(history)

                # Store in database
                self._db_manager.save_failure_case(
                    detection_result=detection_result,
                    serialized_history=serialized_history,
                    scenario=setup.scenario,
                    planner_name=planner.name(),
                )

                logger.info(
                    f"Failure case saved: {setup.scenario.scenario_name}"
                )
            else:
                logger.debug(
                    f"No failures detected in {setup.scenario.scenario_name}"
                )

        except Exception as e:
            logger.error(
                f"Error collecting failure case for {setup.scenario.scenario_name}: {e}",
                exc_info=True
            )
