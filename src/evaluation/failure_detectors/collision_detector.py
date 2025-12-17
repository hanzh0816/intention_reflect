"""
Collision detector using existing nuplan collision detection logic.
"""
from typing import List, Set
import logging

from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.oriented_box import in_collision
from nuplan.planning.metrics.utils.collision_utils import ego_delta_v_collision, CollisionType

from src.evaluation.failure_detectors.detection_engine import CollisionEvent

logger = logging.getLogger(__name__)


class CollisionDetector:
    """
    Detects collisions using existing nuplan collision detection logic.

    Reuses:
    - nuplan.common.actor_state.oriented_box.in_collision
    - nuplan.planning.metrics.utils.collision_utils.ego_delta_v_collision
    """

    def detect(
        self,
        history: SimulationHistory,
        scenario: AbstractScenario,
    ) -> List[CollisionEvent]:
        """
        Detect all collision events in simulation history.

        Returns list of CollisionEvent with details for database storage.
        """
        collision_events: List[CollisionEvent] = []
        collided_track_ids: Set[str] = set()

        # Compute timestamps in route (simplified - assume all timestamps valid)
        timestamps_in_route = self._compute_timestamps_in_route(history, scenario)

        for sample in history.data:
            ego_state = sample.ego_state
            observation = sample.observation
            timestamp = ego_state.time_point.time_us

            if not isinstance(observation, DetectionsTracks):
                continue

            # Find new collisions
            for tracked_object in observation.tracked_objects:
                if tracked_object.track_token in collided_track_ids:
                    continue

                if in_collision(ego_state.car_footprint.oriented_box, tracked_object.box):
                    # Mark as collided
                    collided_track_ids.add(tracked_object.track_token)

                    # Calculate collision energy
                    collision_energy = ego_delta_v_collision(ego_state, tracked_object)

                    # Classify collision type
                    collision_type = self._classify_collision_type(ego_state, tracked_object)

                    # Determine if at-fault
                    is_at_fault = self._is_at_fault_collision(
                        collision_type,
                        timestamp,
                        timestamps_in_route
                    )

                    # Create event
                    event = CollisionEvent(
                        timestamp_us=timestamp,
                        collision_type=collision_type.name,
                        tracked_object_type=tracked_object.tracked_object_type.name,
                        collision_energy=collision_energy,
                        is_at_fault=is_at_fault,
                        ego_speed=ego_state.dynamic_car_state.speed,
                        object_speed=tracked_object.velocity.magnitude(),
                    )

                    collision_events.append(event)

                    logger.debug(
                        f"Collision detected at {timestamp}us: "
                        f"type={collision_type.name}, at_fault={is_at_fault}"
                    )

        return collision_events

    def _classify_collision_type(self, ego_state, tracked_object) -> CollisionType:
        """Classify collision type (reuse from no_ego_at_fault_collisions.py)."""
        from nuplan.planning.metrics.evaluation_metrics.common.no_ego_at_fault_collisions import (
            _get_collision_type
        )
        return _get_collision_type(ego_state, tracked_object)

    def _is_at_fault_collision(self, collision_type: CollisionType, timestamp: int, timestamps_in_route: Set[int]) -> bool:
        """Determine if collision is at-fault (simplified logic)."""
        # Front collisions and stopped track collisions are at-fault
        if collision_type in [
            CollisionType.ACTIVE_FRONT_COLLISION,
            CollisionType.STOPPED_TRACK_COLLISION,
        ]:
            return True

        # Lateral collisions at-fault if not in proper lane
        if collision_type == CollisionType.ACTIVE_LATERAL_COLLISION:
            return timestamp not in timestamps_in_route

        return False

    def _compute_timestamps_in_route(self, history, scenario) -> Set[int]:
        """
        Compute timestamps where ego is in proper route.

        Simplified - real implementation should use EgoLaneChangeStatistics
        """
        # Placeholder: assume all timestamps valid for now
        return set(sample.ego_state.time_point.time_us for sample in history.data)
