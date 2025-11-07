"""
Short-Term Intent Classification and Visualization (v3.0)

This script classifies ego vehicle's short-term intent (2 seconds) based on
trajectory geometry features, without relying on lane IDs.

Intent Classification:
- Lateral Intent: turn_left, turn_right, shift_left, shift_right, stay_in_lane
- Longitudinal Intent: accelerate, maintain_speed, decelerate, stop

Features used:
- Trajectory curvature
- Heading angle change
- Lateral displacement
- Velocity and acceleration
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import hydra
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from omegaconf import DictConfig
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.ego_state import EgoState

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

set_default_path()

# ==================== Configuration Parameters ====================

# Time window for short-term intent (seconds)
SHORT_TERM_HORIZON = 2.0
SAMPLE_INTERVAL = 0.1

# Lateral intent thresholds
TURN_ANGLE_THRESHOLD = 15.0  # degrees
SHIFT_ANGLE_THRESHOLD = 5.0   # degrees
TURN_CURVATURE_THRESHOLD = 0.05  # 1/meters (higher = sharper turn)
LATERAL_SHIFT_THRESHOLD = 1.5  # meters

# Longitudinal intent thresholds
ACCEL_THRESHOLD = 1.0  # m/s²
DECEL_THRESHOLD = -1.0  # m/s²
VELOCITY_CHANGE_RATIO = 0.20  # 20%
STOP_VELOCITY_THRESHOLD = 0.5  # m/s

# Visualization parameters
DEFAULT_HISTORY_HORIZON = 2.0
DEFAULT_FUTURE_HORIZON = 8.0


# ==================== Utility Functions ====================

def extract_ego_states(
    scenario: AbstractScenario,
    history_horizon: float = DEFAULT_HISTORY_HORIZON,
    future_horizon: float = DEFAULT_FUTURE_HORIZON,
    sample_interval: float = SAMPLE_INTERVAL
) -> List[EgoState]:
    """
    Extract full ego states from a scenario.
    """
    history_samples = int(history_horizon / sample_interval)
    future_samples = int(future_horizon / sample_interval)

    ego_cur_state = scenario.initial_ego_state

    past_ego_trajectory = scenario.get_ego_past_trajectory(
        iteration=0,
        time_horizon=history_horizon,
        num_samples=history_samples,
    )

    future_ego_trajectory = scenario.get_ego_future_trajectory(
        iteration=0,
        time_horizon=future_horizon,
        num_samples=future_samples,
    )

    ego_state_list = (
        list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
    )

    return ego_state_list


def extract_trajectory_coordinates(ego_states: List[EgoState]) -> np.ndarray:
    """
    Extract (x, y) coordinates from ego states.
    """
    trajectory_coords = []
    for ego_state in ego_states:
        x = ego_state.rear_axle.x
        y = ego_state.rear_axle.y
        trajectory_coords.append([x, y])

    return np.array(trajectory_coords)


# ==================== Feature Calculation Functions ====================

def compute_trajectory_curvature(positions: np.ndarray) -> float:
    """
    Compute average curvature of a trajectory.

    Curvature at point i is computed using three consecutive points.
    Higher curvature = sharper turn.

    :param positions: Array of shape (N, 2) with (x, y) positions
    :return: Average absolute curvature (1/meters)
    """
    if len(positions) < 3:
        return 0.0

    curvatures = []

    for i in range(1, len(positions) - 1):
        p1 = positions[i - 1]
        p2 = positions[i]
        p3 = positions[i + 1]

        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Lengths
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)

        if l1 < 1e-6 or l2 < 1e-6:
            continue

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (l1 * l2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Curvature approximation: angle / average_length
        avg_length = (l1 + l2) / 2.0
        curvature = angle / avg_length if avg_length > 0 else 0.0

        curvatures.append(abs(curvature))

    return np.mean(curvatures) if curvatures else 0.0


def compute_heading_change(ego_states: List[EgoState], start_idx: int, end_idx: int) -> float:
    """
    Compute heading angle change between two states.

    :return: Heading change in radians (normalized to [-π, π])
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_heading = ego_states[start_idx].rear_axle.heading
    end_heading = ego_states[end_idx].rear_axle.heading

    # Normalize to [-π, π]
    delta = end_heading - start_heading
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi

    return delta


def compute_lateral_displacement(positions: np.ndarray) -> float:
    """
    Compute lateral displacement relative to initial direction.

    Positive = left displacement, Negative = right displacement

    :param positions: Array of shape (N, 2) with (x, y) positions
    :return: Lateral displacement in meters
    """
    if len(positions) < 2:
        return 0.0

    # Initial direction vector
    initial_direction = positions[-1] - positions[0]
    initial_length = np.linalg.norm(initial_direction)

    if initial_length < 1e-6:
        return 0.0

    initial_direction = initial_direction / initial_length

    # Perpendicular direction (left is positive)
    perpendicular = np.array([-initial_direction[1], initial_direction[0]])

    # Vector from start to end
    displacement_vector = positions[-1] - positions[0]

    # Lateral component
    lateral_disp = np.dot(displacement_vector, perpendicular)

    return lateral_disp


def compute_acceleration(ego_states: List[EgoState], start_idx: int, end_idx: int) -> float:
    """
    Compute average acceleration.

    :return: Acceleration in m/s²
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_state = ego_states[start_idx]
    end_state = ego_states[end_idx]

    start_vel = np.linalg.norm([
        start_state.dynamic_car_state.rear_axle_velocity_2d.x,
        start_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    time_diff = (end_idx - start_idx) * SAMPLE_INTERVAL

    if time_diff < 1e-6:
        return 0.0

    acceleration = (end_vel - start_vel) / time_diff

    return acceleration


def compute_velocity_change_ratio(ego_states: List[EgoState], start_idx: int, end_idx: int) -> float:
    """
    Compute velocity change ratio.

    :return: Ratio of velocity change (positive = acceleration, negative = deceleration)
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_state = ego_states[start_idx]
    end_state = ego_states[end_idx]

    start_vel = np.linalg.norm([
        start_state.dynamic_car_state.rear_axle_velocity_2d.x,
        start_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    if start_vel < 0.1:  # Avoid division by very small numbers
        return 0.0

    ratio = (end_vel - start_vel) / start_vel

    return ratio


# ==================== Intent Classification Functions ====================

def classify_lateral_intent(
    ego_states: List[EgoState],
    positions: np.ndarray,
    start_idx: int,
    end_idx: int
) -> str:
    """
    Classify lateral (horizontal) intent based on trajectory geometry.

    Intent categories:
    - turn_left: Large left turn (heading change > 15° and high curvature)
    - turn_right: Large right turn (heading change < -15° and high curvature)
    - shift_left: Slight left maneuver (5° < heading < 15° or low curvature with left displacement)
    - shift_right: Slight right maneuver (-15° < heading < -5° or low curvature with right displacement)
    - stay_in_lane: Minimal lateral movement

    :return: Lateral intent string
    """
    # Compute features
    heading_change = compute_heading_change(ego_states, start_idx, end_idx)
    heading_change_deg = np.degrees(heading_change)

    curvature = compute_trajectory_curvature(positions)
    lateral_disp = compute_lateral_displacement(positions)

    # Classification logic
    # Turn detection: significant heading change AND high curvature
    if heading_change_deg > TURN_ANGLE_THRESHOLD and curvature > TURN_CURVATURE_THRESHOLD:
        return "turn_left"

    if heading_change_deg < -TURN_ANGLE_THRESHOLD and curvature > TURN_CURVATURE_THRESHOLD:
        return "turn_right"

    # Shift detection: moderate heading change OR low curvature with displacement
    if SHIFT_ANGLE_THRESHOLD < heading_change_deg <= TURN_ANGLE_THRESHOLD:
        return "shift_left"

    if -TURN_ANGLE_THRESHOLD <= heading_change_deg < -SHIFT_ANGLE_THRESHOLD:
        return "shift_right"

    # Check lateral displacement for low-curvature shifts
    if curvature <= TURN_CURVATURE_THRESHOLD:
        if lateral_disp > LATERAL_SHIFT_THRESHOLD:
            return "shift_left"
        if lateral_disp < -LATERAL_SHIFT_THRESHOLD:
            return "shift_right"

    # Default: staying in lane
    return "stay_in_lane"


def classify_longitudinal_intent(
    ego_states: List[EgoState],
    start_idx: int,
    end_idx: int
) -> str:
    """
    Classify longitudinal (speed) intent based on velocity and acceleration.

    Intent categories:
    - stop: Vehicle comes to a stop (end velocity < 0.5 m/s)
    - accelerate: Significant acceleration (accel > 1.0 m/s² or speed increase > 20%)
    - decelerate: Significant deceleration (accel < -1.0 m/s² or speed decrease > 20%)
    - maintain_speed: Minimal speed change

    :return: Longitudinal intent string
    """
    if end_idx >= len(ego_states):
        return "maintain_speed"

    # Check for stop
    end_state = ego_states[end_idx]
    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    if end_vel < STOP_VELOCITY_THRESHOLD:
        return "stop"

    # Compute acceleration and velocity change ratio
    acceleration = compute_acceleration(ego_states, start_idx, end_idx)
    vel_change_ratio = compute_velocity_change_ratio(ego_states, start_idx, end_idx)

    # Accelerate: high positive acceleration OR significant speed increase
    if acceleration > ACCEL_THRESHOLD or vel_change_ratio > VELOCITY_CHANGE_RATIO:
        return "accelerate"

    # Decelerate: high negative acceleration OR significant speed decrease
    if acceleration < DECEL_THRESHOLD or vel_change_ratio < -VELOCITY_CHANGE_RATIO:
        return "decelerate"

    # Default: maintaining speed
    return "maintain_speed"


def classify_short_term_intent(
    scenario: AbstractScenario,
    ego_states: List[EgoState],
    current_idx: int
) -> Tuple[str, str]:
    """
    Classify short-term (2-second) intent in both lateral and longitudinal dimensions.

    :param scenario: The scenario object
    :param ego_states: List of all ego states
    :param current_idx: Index of the current state
    :return: Tuple of (lateral_intent, longitudinal_intent)
    """
    # Define short-term window (2 seconds)
    window_size = int(SHORT_TERM_HORIZON / SAMPLE_INTERVAL)
    end_idx = min(current_idx + window_size, len(ego_states) - 1)

    # Extract positions for this window
    positions = []
    for i in range(current_idx, end_idx + 1):
        if i < len(ego_states):
            state = ego_states[i]
            positions.append([state.rear_axle.x, state.rear_axle.y])

    positions = np.array(positions)

    # Classify lateral and longitudinal intents independently
    lateral_intent = classify_lateral_intent(ego_states, positions, current_idx, end_idx)
    longitudinal_intent = classify_longitudinal_intent(ego_states, current_idx, end_idx)

    return lateral_intent, longitudinal_intent


# ==================== Visualization Functions ====================

def plot_trajectory_with_dual_intent(
    scenario: AbstractScenario,
    ego_states: List[EgoState],
    trajectory: np.ndarray,
    lateral_intent: str,
    longitudinal_intent: str,
    scenario_name: str,
    save_path: str,
    map_radius: float = 80.0
) -> None:
    """
    Plot trajectory with dual-dimension intent classification.
    """
    map_api = scenario.map_api
    ego_state = scenario.initial_ego_state

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot nearby lanes for context
    try:
        proximal_lanes = map_api.get_proximal_map_objects(
            ego_state.center, map_radius, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )
        lanes = proximal_lanes[SemanticMapLayer.LANE] + proximal_lanes[SemanticMapLayer.LANE_CONNECTOR]

        lane_colors = ['#E8F4F8', '#D0E8F0']

        for idx, lane in enumerate(lanes):
            try:
                left_boundary_points = lane.left_boundary.discrete_path
                right_boundary_points = lane.right_boundary.discrete_path

                left_coords = [(p.x, p.y) for p in left_boundary_points]
                right_coords = [(p.x, p.y) for p in reversed(right_boundary_points)]

                if len(left_coords) > 0 and len(right_coords) > 0:
                    lane_polygon = mpatches.Polygon(
                        left_coords + right_coords,
                        closed=True,
                        facecolor=lane_colors[idx % len(lane_colors)],
                        edgecolor='none',
                        alpha=0.4,
                        zorder=1
                    )
                    ax.add_patch(lane_polygon)

                    lx = [p.x for p in left_boundary_points]
                    ly = [p.y for p in left_boundary_points]
                    ax.plot(lx, ly, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    rx = [p.x for p in right_boundary_points]
                    ry = [p.y for p in right_boundary_points]
                    ax.plot(rx, ry, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    centerline_points = lane.baseline_path.discrete_path
                    cx = [p.x for p in centerline_points]
                    cy = [p.y for p in centerline_points]
                    ax.plot(cx, cy, color='#808080', linewidth=1.0, alpha=0.5,
                           linestyle='--', zorder=2)
            except Exception as e:
                logger.debug(f"Could not plot lane: {e}")
                continue
    except Exception as e:
        logger.warning(f"Failed to plot lanes: {e}")

    # Plot the trajectory
    history_samples = int(DEFAULT_HISTORY_HORIZON / SAMPLE_INTERVAL)

    if len(trajectory) > 0:
        # Plot current position marker
        current_pos = trajectory[history_samples]
        ax.plot(current_pos[0], current_pos[1], 'go', markersize=15,
               label='Current', zorder=12, markeredgewidth=2, markeredgecolor='white')

        past_traj = trajectory[:history_samples+1]
        if len(past_traj) > 0:
            ax.plot(past_traj[:, 0], past_traj[:, 1],
                   color='#1E90FF', linewidth=4, alpha=0.9,
                   label='Past Trajectory', marker='o', markersize=4, zorder=10)

        # Short-term future (2 seconds)
        short_term_idx = history_samples + int(SHORT_TERM_HORIZON / SAMPLE_INTERVAL)
        short_term_traj = trajectory[history_samples+1:min(short_term_idx+1, len(trajectory))]
        if len(short_term_traj) > 0:
            ax.plot(short_term_traj[:, 0], short_term_traj[:, 1],
                   color='#FF4500', linewidth=5, alpha=0.95,
                   label='Short-term (2s)', marker='o', markersize=5, zorder=10)

        # Remaining future (lighter color)
        remaining_traj = trajectory[min(short_term_idx, len(trajectory)):]
        if len(remaining_traj) > 0:
            ax.plot(remaining_traj[:, 0], remaining_traj[:, 1],
                   color='#FFA07A', linewidth=3, alpha=0.6,
                   label='Extended Future', marker='o', markersize=3, zorder=9)

        ax.plot(trajectory[0, 0], trajectory[0, 1], 'bs', markersize=12,
               label='Start', zorder=11, markeredgewidth=2, markeredgecolor='white')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=12,
               label='End', zorder=11, markeredgewidth=2, markeredgecolor='white')

    # Intent color mapping
    lateral_colors = {
        'turn_left': '#9C27B0',
        'turn_right': '#673AB7',
        'shift_left': '#2196F3',
        'shift_right': '#03A9F4',
        'stay_in_lane': '#4CAF50'
    }

    longitudinal_colors = {
        'accelerate': '#4CAF50',
        'maintain_speed': '#FFC107',
        'decelerate': '#FF9800',
        'stop': '#F44336'
    }

    # Add dual intent labels
    lateral_color = lateral_colors.get(lateral_intent, '#757575')
    longitudinal_color = longitudinal_colors.get(longitudinal_intent, '#757575')

    # Lateral intent label (top)
    lateral_label = f"Lateral: {lateral_intent.replace('_', ' ').upper()}"
    ax.text(0.02, 0.98, lateral_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=lateral_color, alpha=0.85,
                     edgecolor='white', linewidth=2),
            color='white',
            zorder=20)

    # Longitudinal intent label (below lateral)
    longitudinal_label = f"Longitudinal: {longitudinal_intent.replace('_', ' ').upper()}"
    ax.text(0.02, 0.92, longitudinal_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=longitudinal_color, alpha=0.85,
                     edgecolor='white', linewidth=2),
            color='white',
            zorder=20)

    # Set axis properties
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Short-Term Intent Classification (2-second horizon)\n{scenario_name}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    # Zoom to show trajectory
    if len(trajectory) > 0:
        x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
        y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
        margin = 30
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

    plt.tight_layout()

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(save_path_obj), dpi=150, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to {save_path_obj}")
    plt.close()


# ==================== Main Function ====================

@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Main function to visualize ego trajectories with short-term intent classification.
    """
    original_cwd = hydra.utils.get_original_cwd()
    current_cwd = os.getcwd()

    # Configuration parameters
    num_scenarios_to_sample = cfg.get('num_scenarios_to_visualize', 10)
    output_dir = cfg.get('trajectory_output_dir', 'work_dirs/short_term_intent_visualizations')
    map_radius = cfg.get('map_radius', 80.0)
    scenario_type_filter = cfg.get('scenario_type', None)

    # Add timestamp to output directory
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if scenario_type_filter:
        output_dir_with_timestamp = f"{output_dir}_{scenario_type_filter}_{current_timestamp}"
    else:
        output_dir_with_timestamp = f"{output_dir}_{current_timestamp}"

    if not Path(output_dir_with_timestamp).is_absolute():
        output_dir_with_timestamp = str(Path(original_cwd) / output_dir_with_timestamp)

    output_dir_path = Path(output_dir_with_timestamp)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting short-term intent visualization (v3.0)...")
    logger.info(f"Output directory: {output_dir_path}")
    logger.info(f"Will sample {num_scenarios_to_sample} scenarios")
    logger.info(f"Short-term horizon: {SHORT_TERM_HORIZON}s")
    if scenario_type_filter:
        logger.info(f"Scenario type filter: {scenario_type_filter}")

    # Build worker pool
    worker = build_worker(cfg)

    # Load scenarios
    logger.info("Loading scenarios from dataset...")
    all_scenarios = build_scenarios(cfg, worker, model=None)
    logger.info(f"Total scenarios available: {len(all_scenarios)}")

    # Filter by scenario type if specified
    if scenario_type_filter:
        logger.info(f"Filtering scenarios by type: {scenario_type_filter}")
        filtered_scenarios = [s for s in all_scenarios if s.scenario_type == scenario_type_filter]
        logger.info(f"Found {len(filtered_scenarios)} scenarios matching type '{scenario_type_filter}'")

        if len(filtered_scenarios) == 0:
            logger.error(f"No scenarios found with type '{scenario_type_filter}'")
            return

        scenarios_to_sample = filtered_scenarios
    else:
        scenarios_to_sample = all_scenarios

    # Sample scenarios
    num_to_sample = min(num_scenarios_to_sample, len(scenarios_to_sample))
    sampled_scenarios = random.sample(scenarios_to_sample, num_to_sample)
    logger.info(f"Sampled {num_to_sample} scenarios")

    # Statistics tracking
    successful_count = 0
    intent_stats = {}

    # Process each scenario
    for idx, scenario in enumerate(sampled_scenarios):
        scenario_name = f"{scenario.scenario_type}_{scenario.token[:8]}"
        logger.info(f"Processing scenario {idx + 1}/{num_to_sample}: {scenario_name}")

        try:
            # Extract ego states
            ego_states = extract_ego_states(scenario)
            traj = extract_trajectory_coordinates(ego_states)
            logger.info(f"  Extracted trajectory with {len(traj)} points")

            # Classify intent at current state
            current_idx = int(DEFAULT_HISTORY_HORIZON / SAMPLE_INTERVAL)
            lateral_intent, longitudinal_intent = classify_short_term_intent(
                scenario, ego_states, current_idx
            )
            logger.info(f"  Classified intent: lateral={lateral_intent}, longitudinal={longitudinal_intent}")

            # Track statistics
            intent_key = f"{lateral_intent} + {longitudinal_intent}"
            intent_stats[intent_key] = intent_stats.get(intent_key, 0) + 1

            # Create output path
            output_path = output_dir_path / f"trajectory_{idx:03d}_{scenario_name}_{lateral_intent}_{longitudinal_intent}.png"

            # Plot trajectory with intent
            plot_trajectory_with_dual_intent(
                scenario=scenario,
                ego_states=ego_states,
                trajectory=traj,
                lateral_intent=lateral_intent,
                longitudinal_intent=longitudinal_intent,
                scenario_name=scenario_name,
                save_path=str(output_path),
                map_radius=map_radius
            )

            successful_count += 1
            logger.info(f"  Successfully saved to {output_path.name}")

        except Exception as e:
            logger.warning(f"  Failed to process scenario: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue

    # Summary
    logger.info(f"\nVisualization complete!")
    logger.info(f"Successfully processed {successful_count}/{num_to_sample} scenarios")
    logger.info(f"\nIntent Statistics:")
    for intent_key, count in sorted(intent_stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {intent_key}: {count} scenarios")
    logger.info(f"\nAll visualizations saved to: {output_dir_path}")

    print(f"\nSuccessfully created {successful_count} short-term intent visualizations")
    print(f"\nIntent Statistics:")
    for intent_key, count in sorted(intent_stats.items(), key=lambda x: -x[1]):
        print(f"  {intent_key}: {count}")
    print(f"\nOutput directory: {output_dir_path}\n")


if __name__ == "__main__":
    main()
