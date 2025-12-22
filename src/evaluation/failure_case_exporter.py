"""
Failure case exporter for nuBoard visualization.

Converts failure cases from SQLite database to nuBoard-compatible format.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator, Any
from dataclasses import dataclass

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.simulation_log import SimulationLog
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Sensors
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

from src.evaluation.database_manager import DatabaseManager
from src.evaluation.history_serializer import HistorySerializer

logger = logging.getLogger(__name__)


class StubScenario(AbstractScenario):
    """
    Lightweight scenario stub for nuBoard visualization.

    Creates a minimal scenario object from database metadata.
    The actual simulation data is contained in SimulationHistory.
    """

    def __init__(
        self,
        scenario_type: str,
        scenario_name: str,
        log_name: str,
        map_api: AbstractMap,
        database_interval: float = 0.05,
    ):
        """
        Initialize stub scenario.

        Args:
            scenario_type: Type of scenario (e.g., 'highway', 'urban')
            scenario_name: Unique scenario name
            log_name: Log file name
            map_api: Map API from SimulationHistory
            database_interval: Time interval between samples
        """
        self._scenario_type = scenario_type
        self._scenario_name = scenario_name
        self._log_name = log_name
        self._map_api = map_api
        self._database_interval = database_interval

    @property
    def token(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_name

    @property
    def log_name(self) -> str:
        """Inherited, see superclass."""
        return self._log_name

    @property
    def scenario_name(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_name

    @property
    def scenario_type(self) -> str:
        """Inherited, see superclass."""
        return self._scenario_type

    @property
    def map_api(self) -> AbstractMap:
        """Inherited, see superclass."""
        return self._map_api

    @property
    def database_interval(self) -> float:
        """Inherited, see superclass."""
        return self._database_interval

    def get_number_of_iterations(self) -> int:
        """Inherited, see superclass."""
        return 0

    def get_time_point(self, iteration: int) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubScenario does not support get_time_point")

    def get_lidar_to_ego_transform(self) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubScenario does not support get_lidar_to_ego_transform")

    def get_mission_goal(self) -> Any:
        """Inherited, see superclass."""
        return None

    def get_route_roadblock_ids(self) -> List[str]:
        """Inherited, see superclass."""
        return []

    def get_expert_goal_state(self) -> Any:
        """Inherited, see superclass."""
        return None

    def get_tracked_objects_at_iteration(self, iteration: int) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubScenario does not support get_tracked_objects_at_iteration")

    def get_sensors_at_iteration(self, iteration: int) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubScenario does not support get_sensors_at_iteration")

    def get_ego_state_at_iteration(self, iteration: int) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubScenario does not support get_ego_state_at_iteration")

    def get_tracked_objects_within_time_window_at_iteration(
        self, iteration: int, past_time_horizon: float, future_time_horizon: float
    ) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError(
            "StubScenario does not support get_tracked_objects_within_time_window_at_iteration"
        )


class StubPlanner(AbstractPlanner):
    """
    Lightweight planner stub for nuBoard visualization.

    Creates a minimal planner object with just the name.
    """

    def __init__(self, name: str):
        """
        Initialize stub planner.

        Args:
            name: Planner name
        """
        self._name = name

    def name(self) -> str:
        """Inherited, see superclass."""
        return self._name

    def observation_type(self) -> Any:
        """Inherited, see superclass."""
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        """Inherited, see superclass."""
        pass

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        """Inherited, see superclass."""
        raise NotImplementedError("StubPlanner does not support compute_planner_trajectory")


class FailureCaseExporter:
    """
    Exports failure cases from SQLite database to nuBoard-compatible format.

    Converts stored simulation histories into SimulationLog files that
    nuBoard can load and visualize.
    """

    def __init__(
        self,
        database_path: Path,
        output_base_dir: Path,
        scenario_builder: Optional[AbstractScenarioBuilder] = None,
    ):
        """
        Initialize exporter.

        Args:
            database_path: Path to failure cases SQLite database
            output_base_dir: Base directory for exported nuBoard files
            scenario_builder: Optional scenario builder to retrieve real scenarios.
                            If None, will use lightweight StubScenario objects.
        """
        self._database_path = Path(database_path)
        self._output_base = Path(output_base_dir)
        self._scenario_builder = scenario_builder
        self._db_manager = DatabaseManager(self._database_path)
        self._serializer = HistorySerializer()
        self._worker_pool = WorkerPool(num_workers=1) if scenario_builder else None

        # Initialize database connection
        if not self._database_path.exists():
            raise FileNotFoundError(f"Database not found: {self._database_path}")

        self._db_manager.initialize_database()

    def list_failure_cases(self) -> List[Dict]:
        """
        List all failure cases in database.

        Returns:
            List of dicts with failure case summary information
        """
        cases = self._db_manager.list_all_failure_cases()
        logger.info(f"Found {len(cases)} failure cases in database")
        return cases

    def _get_scenario_from_builder(
        self,
        scenario_name: str,
        log_name: str,
    ) -> Optional[AbstractScenario]:
        """
        Retrieve real scenario from scenario_builder using scenario_name (token).

        Args:
            scenario_name: Scenario name/token (e.g., '00cca24d240f5980')
            log_name: Log name for the scenario

        Returns:
            AbstractScenario object, or None if not found
        """
        if self._scenario_builder is None:
            return None

        logger.debug(f"Querying scenario_builder for token: {scenario_name}")

        # Create scenario filter to query by token
        scenario_filter = ScenarioFilter(
            scenario_types=None,
            scenario_tokens=[scenario_name],  # Query by token
            log_names=[log_name],  # Also filter by log name
            map_names=None,
            num_scenarios_per_type=None,
            limit_total_scenarios=None,
            expand_scenarios=False,
            remove_invalid_goals=True,
            shuffle=False,
        )

        try:
            scenarios = self._scenario_builder.get_scenarios(
                scenario_filter=scenario_filter,
                worker=self._worker_pool
            )

            if not scenarios:
                logger.warning(
                    f"Scenario not found in scenario_builder: token={scenario_name}, log={log_name}"
                )
                return None

            if len(scenarios) > 1:
                logger.warning(
                    f"Multiple scenarios found for token={scenario_name}, using first one"
                )

            scenario = scenarios[0]
            logger.info(
                f"Retrieved real scenario: {scenario.scenario_name} "
                f"(type: {scenario.scenario_type}, log: {scenario.log_name})"
            )
            return scenario

        except Exception as e:
            logger.error(f"Failed to retrieve scenario from scenario_builder: {e}", exc_info=True)
            return None

    def export_failure_case(self, scenario_name: str) -> Path:
        """
        Export a single failure case to nuBoard format.

        Args:
            scenario_name: Scenario name to export

        Returns:
            Path to the exported .pkl.xz file

        Raises:
            ValueError: If scenario_name not found in database
            RuntimeError: If export fails
        """
        logger.info(f"Exporting failure case: {scenario_name}")

        # Step 1: Query database for failure case metadata
        logger.debug(f"Querying database for scenario: {scenario_name}")
        failure_case = self._db_manager.query_failure_case_by_scenario(scenario_name)

        if failure_case is None:
            # List available scenarios for user
            all_cases = self.list_failure_cases()
            available_names = [case['scenario_name'] for case in all_cases]
            raise ValueError(
                f"Scenario '{scenario_name}' not found in database.\n"
                f"Available scenarios ({len(available_names)}):\n"
                + "\n".join(f"  - {name}" for name in available_names[:10])
                + (f"\n  ... and {len(available_names) - 10} more" if len(available_names) > 10 else "")
            )

        logger.info(
            f"Found failure case: {failure_case['scenario_name']} "
            f"(planner: {failure_case['planner_name']}, "
            f"severity: {failure_case['failure_severity']})"
        )

        # Step 2: Get simulation history BLOB
        logger.debug(f"Retrieving simulation history for failure case ID: {failure_case['id']}")
        history_blob = self._db_manager.get_simulation_history_blob(failure_case['id'])

        if history_blob is None:
            raise RuntimeError(
                f"No simulation history found for failure case {failure_case['id']}. "
                f"Database may be corrupted."
            )

        # Step 3: Deserialize simulation history
        logger.debug("Deserializing simulation history...")
        try:
            simulation_history = self._serializer.deserialize_from_bytes(history_blob)
        except Exception as e:
            raise RuntimeError(
                f"Failed to deserialize simulation history: {e}. "
                f"The history data may be corrupted."
            ) from e

        logger.info(f"Deserialized history with {len(simulation_history)} samples")

        # Step 4: Get scenario object (real or stub)
        logger.debug("Retrieving scenario object...")

        # Try to get real scenario from scenario_builder first
        scenario = self._get_scenario_from_builder(
            scenario_name=failure_case['scenario_name'],
            log_name=failure_case['log_name'],
        )

        if scenario is not None:
            logger.info("Using real scenario from scenario_builder")
        else:
            # Fall back to creating stub scenario
            logger.info("scenario_builder not available or scenario not found, using StubScenario")

            # Get map_api from simulation history
            if len(simulation_history) == 0:
                raise RuntimeError("Simulation history is empty, cannot create scenario")

            map_api = simulation_history.data[0].observation.map_api

            scenario = StubScenario(
                scenario_type=failure_case['scenario_type'],
                scenario_name=failure_case['scenario_name'],
                log_name=failure_case['log_name'],
                map_api=map_api,
            )

        # Step 5: Create stub planner
        stub_planner = StubPlanner(name=failure_case['planner_name'])

        # Step 6: Construct output path following nuBoard structure
        # Structure: <output_base>/<planner>/<scenario_type>/<log_name>/<scenario_name>/<scenario_name>.pkl.xz
        output_dir = (
            self._output_base
            / failure_case['planner_name']
            / failure_case['scenario_type']
            / failure_case['log_name']
            / failure_case['scenario_name']
        )

        output_file = output_dir / f"{failure_case['scenario_name']}.pkl.xz"

        # Create output directory
        logger.debug(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 7: Create and save SimulationLog
        logger.debug(f"Creating SimulationLog at: {output_file}")
        simulation_log = SimulationLog(
            file_path=output_file,
            scenario=scenario,  # Use real scenario or stub
            planner=stub_planner,
            simulation_history=simulation_history,
        )

        logger.info(f"Saving SimulationLog to: {output_file}")
        simulation_log.save_to_file()

        logger.info(f"✓ Successfully exported failure case to: {output_file}")
        return output_file

    def export_multiple(self, scenario_names: List[str]) -> List[Path]:
        """
        Export multiple failure cases.

        Args:
            scenario_names: List of scenario names to export

        Returns:
            List of paths to exported files
        """
        exported_files = []
        failed_exports = []

        for scenario_name in scenario_names:
            try:
                output_file = self.export_failure_case(scenario_name)
                exported_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to export {scenario_name}: {e}")
                failed_exports.append((scenario_name, str(e)))

        logger.info(
            f"Export summary: {len(exported_files)} succeeded, {len(failed_exports)} failed"
        )

        if failed_exports:
            logger.warning("Failed exports:")
            for name, error in failed_exports:
                logger.warning(f"  - {name}: {error}")

        return exported_files
