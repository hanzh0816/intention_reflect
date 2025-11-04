"""
Visualize ego vehicle trajectories from multiple scenarios in the dataset.
Each trajectory is plotted separately with the scene's map context.
Trajectories are NOT normalized - they use original coordinates to match the map.
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path

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

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


def extract_ego_trajectory_coordinates(
    scenario: AbstractScenario,
    history_horizon: float = 2.0,
    future_horizon: float = 8.0,
    sample_interval: float = 0.1
) -> np.ndarray:
    """
    Extract ego trajectory coordinates (x, y) from a scenario.
    This method follows the implementation in nuplan_feature_builder.py.

    :param scenario: The scenario object containing ego trajectory data
    :param history_horizon: Time horizon for past trajectory in seconds (default: 2.0s)
    :param future_horizon: Time horizon for future trajectory in seconds (default: 8.0s)
    :param sample_interval: Time interval between samples in seconds (default: 0.1s)
    :return: Numpy array of shape (N, 2) containing (x, y) coordinates
    """
    # Calculate number of samples
    history_samples = int(history_horizon / sample_interval)
    future_samples = int(future_horizon / sample_interval)

    # Get current ego state
    ego_cur_state = scenario.initial_ego_state

    # Get past ego trajectory
    past_ego_trajectory = scenario.get_ego_past_trajectory(
        iteration=0,
        time_horizon=history_horizon,
        num_samples=history_samples,
    )

    # Get future ego trajectory
    future_ego_trajectory = scenario.get_ego_future_trajectory(
        iteration=0,
        time_horizon=future_horizon,
        num_samples=future_samples,
    )

    # Combine past, present, and future ego states
    ego_state_list = (
        list(past_ego_trajectory) + [ego_cur_state] + list(future_ego_trajectory)
    )

    # Extract x, y coordinates from ego states
    trajectory_coords = []
    for ego_state in ego_state_list:
        x = ego_state.rear_axle.x
        y = ego_state.rear_axle.y
        trajectory_coords.append([x, y])

    return np.array(trajectory_coords)


def plot_single_trajectory_with_map(
    scenario: AbstractScenario,
    trajectory: np.ndarray,
    scenario_name: str,
    save_path: str,
    map_radius: float = 80.0
) -> None:
    """
    Plot a single ego trajectory with the scene's map context.

    :param scenario: The scenario object containing map information
    :param trajectory: Trajectory array of shape (N, 2) with original coordinates
    :param scenario_name: Name of the scenario for the title
    :param save_path: Path to save the figure
    :param map_radius: Radius around ego vehicle to show map elements (meters)
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

        # Use alternating colors to distinguish lanes
        lane_colors = ['#E8F4F8', '#D0E8F0']  # Light blue shades for alternating lanes

        for idx, lane in enumerate(lanes):
            try:
                # Get lane boundaries
                left_boundary_points = lane.left_boundary.discrete_path
                right_boundary_points = lane.right_boundary.discrete_path

                # Create polygon for lane fill
                left_coords = [(p.x, p.y) for p in left_boundary_points]
                right_coords = [(p.x, p.y) for p in reversed(right_boundary_points)]

                if len(left_coords) > 0 and len(right_coords) > 0:
                    # Fill lane with alternating color
                    lane_polygon = mpatches.Polygon(
                        left_coords + right_coords,
                        closed=True,
                        facecolor=lane_colors[idx % len(lane_colors)],
                        edgecolor='none',
                        alpha=0.4,
                        zorder=1
                    )
                    ax.add_patch(lane_polygon)

                    # Draw lane boundaries with more visible lines
                    lx = [p.x for p in left_boundary_points]
                    ly = [p.y for p in left_boundary_points]
                    ax.plot(lx, ly, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    rx = [p.x for p in right_boundary_points]
                    ry = [p.y for p in right_boundary_points]
                    ax.plot(rx, ry, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    # Draw centerline with dashed style
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

    # Plot route roadblocks
    try:
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        for roadblock_id in route_roadblock_ids:
            try:
                roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
                if not roadblock:
                    roadblock = map_api.get_map_object(
                        roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
                    )
                if roadblock:
                    ax.plot(
                        *roadblock.polygon.exterior.xy,
                        color='#90CAF9',
                        linewidth=2.0,
                        alpha=0.5,
                        label='Route Roadblock' if 'Route Roadblock' not in [l.get_label() for l in ax.get_lines()] else '',
                        zorder=3
                    )
            except Exception as e:
                logger.debug(f"Could not plot roadblock {roadblock_id}: {e}")
    except Exception as e:
        logger.warning(f"Failed to plot roadblocks: {e}")


    # Plot the trajectory
    # Split into history and future
    history_samples = int(2.0 / 0.1)  # Default values

    # Past trajectory (blue)
    if len(trajectory) > 0:
        past_traj = trajectory[:history_samples+1]
        if len(past_traj) > 0:
            ax.plot(past_traj[:, 0], past_traj[:, 1],
                   color='#1E90FF', linewidth=4, alpha=0.9,
                   label='Past Trajectory', marker='o', markersize=4, zorder=10)

        # Future trajectory (red)
        future_traj = trajectory[history_samples+1:]
        if len(future_traj) > 0:
            ax.plot(future_traj[:, 0], future_traj[:, 1],
                   color='#FF4500', linewidth=4, alpha=0.9,
                   label='Future Trajectory', marker='o', markersize=4, zorder=10)

        # Mark start and end
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'bs', markersize=12,
               label='Start', zorder=11, markeredgewidth=2, markeredgecolor='white')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r^', markersize=12,
               label='End', zorder=11, markeredgewidth=2, markeredgecolor='white')

    # Set axis properties
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Ego Trajectory with Map Context\n{scenario_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)

    # Zoom to show trajectory with some margin
    if len(trajectory) > 0:
        x_min, x_max = trajectory[:, 0].min(), trajectory[:, 0].max()
        y_min, y_max = trajectory[:, 1].min(), trajectory[:, 1].max()
        margin = 30  # meters
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
    else:
        # Fallback to ego position
        ax.set_xlim(ego_state.center.x - map_radius, ego_state.center.x + map_radius)
        ax.set_ylim(ego_state.center.y - map_radius, ego_state.center.y + map_radius)

    plt.tight_layout()

    # Ensure the save path directory exists
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(str(save_path_obj), dpi=150, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to {save_path_obj}")
    plt.close()


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Main function to visualize ego trajectories from sampled scenarios.
    Each trajectory is saved as a separate image with map context.

    :param cfg: Hydra configuration
    """
    # Get the original working directory (before Hydra changes it)
    original_cwd = hydra.utils.get_original_cwd()
    current_cwd = os.getcwd()

    # Configuration parameters
    num_scenarios_to_sample = cfg.get('num_scenarios_to_visualize', 10)
    output_dir = cfg.get('trajectory_output_dir', 'work_dirs/trajectory_visualizations')
    map_radius = cfg.get('map_radius', 80.0)
    scenario_type_filter = cfg.get('scenario_type', None)  # Optional scenario type filter

    # Add timestamp to output directory
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if scenario_type_filter:
        output_dir_with_timestamp = f"{output_dir}_{scenario_type_filter}_{current_timestamp}"
    else:
        output_dir_with_timestamp = f"{output_dir}_{current_timestamp}"

    # If output_dir is relative, make it relative to the original working directory
    if not Path(output_dir_with_timestamp).is_absolute():
        output_dir_with_timestamp = str(Path(original_cwd) / output_dir_with_timestamp)

    # Create output directory
    output_dir_path = Path(output_dir_with_timestamp)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Trajectory extraction parameters (following nuplan_feature_builder defaults)
    history_horizon = cfg.get('history_horizon', 2.0)
    future_horizon = cfg.get('future_horizon', 8.0)
    sample_interval = cfg.get('sample_interval', 0.1)

    logger.info(f"Starting ego trajectory visualization with map context...")
    logger.info(f"Original working directory: {original_cwd}")
    logger.info(f"Current working directory: {current_cwd}")
    logger.info(f"Output directory: {output_dir_path}")
    logger.info(f"Timestamp: {current_timestamp}")
    logger.info(f"Will sample {num_scenarios_to_sample} scenarios")
    if scenario_type_filter:
        logger.info(f"Scenario type filter: {scenario_type_filter}")
    logger.info(f"Trajectory config: history={history_horizon}s, future={future_horizon}s, interval={sample_interval}s")
    logger.info(f"Map radius: {map_radius}m")

    # Build worker pool
    worker = build_worker(cfg)

    # Load scenarios from the dataset
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
            logger.info("Available scenario types in dataset:")
            scenario_types = set(s.scenario_type for s in all_scenarios)
            for st in sorted(scenario_types):
                count = sum(1 for s in all_scenarios if s.scenario_type == st)
                logger.info(f"  - {st}: {count} scenarios")
            return

        scenarios_to_sample = filtered_scenarios
    else:
        scenarios_to_sample = all_scenarios

    # Sample a subset of scenarios
    num_to_sample = min(num_scenarios_to_sample, len(scenarios_to_sample))
    sampled_scenarios = random.sample(scenarios_to_sample, num_to_sample)

    logger.info(f"Sampled {num_to_sample} scenarios")

    # Process each scenario separately
    successful_count = 0
    for idx, scenario in enumerate(sampled_scenarios):
        scenario_name = f"{scenario.scenario_type}_{scenario.token[:8]}"
        logger.info(f"Processing scenario {idx + 1}/{num_to_sample}: {scenario_name}")

        try:
            # Extract trajectory
            traj = extract_ego_trajectory_coordinates(
                scenario,
                history_horizon=history_horizon,
                future_horizon=future_horizon,
                sample_interval=sample_interval
            )
            logger.info(f"  Extracted trajectory with {len(traj)} points")

            # Create output path for this trajectory
            output_path = output_dir_path / f"trajectory_{idx:03d}_{scenario_name}.png"

            # Plot trajectory with map
            plot_single_trajectory_with_map(
                scenario=scenario,
                trajectory=traj,
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
    logger.info(f"All visualizations saved to: {output_dir_path}")
    print(f"\n✓ Successfully created {successful_count} trajectory visualizations")
    print(f"✓ Output directory: {output_dir_path}\n")


if __name__ == "__main__":
    main()
