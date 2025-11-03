"""
Visualize ego vehicle trajectories from multiple scenarios in the dataset.
This script loads scenarios from the dataset, samples a number of them,
and plots all ego trajectories on the same coordinate system.
"""

import logging
import os
import random
from pathlib import Path
from typing import List

import hydra
import matplotlib.pyplot as plt
import numpy as np
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from omegaconf import DictConfig

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


def plot_trajectories(
    trajectories: List[np.ndarray],
    scenario_names: List[str],
    save_path: str = "ego_trajectories.png",
    num_points_to_plot: int = None
) -> None:
    """
    Plot multiple ego trajectories on the same coordinate system.
    Each trajectory is normalized to start at (0, 0) for easier comparison.

    :param trajectories: List of trajectory arrays, each of shape (N, 2)
    :param scenario_names: List of scenario names for the legend
    :param save_path: Path to save the figure
    :param num_points_to_plot: If specified, only plot first N points of each trajectory
    """
    plt.figure(figsize=(12, 10))

    # Use a color map to distinguish different trajectories
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))

    for idx, (traj, name) in enumerate(zip(trajectories, scenario_names)):
        # Optionally limit the number of points to plot
        if num_points_to_plot is not None and len(traj) > num_points_to_plot:
            traj = traj[:num_points_to_plot]

        # Normalize trajectory to start at (0, 0)
        if len(traj) > 0:
            first_point = traj[0].copy()
            traj = traj - first_point

        # Plot trajectory
        plt.plot(traj[:, 0], traj[:, 1],
                color=colors[idx],
                linewidth=2,
                alpha=0.7,
                label=f"{name[:30]}...")  # Truncate long names

    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title('Ego Vehicle Trajectories (Normalized to Starting Point)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    # Add legend
    plt.legend(loc='best', fontsize=8, ncol=2)

    plt.tight_layout()

    # Ensure the save path is absolute
    save_path_abs = Path(save_path).resolve()

    # Create parent directory if it doesn't exist
    save_path_abs.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(str(save_path_abs), dpi=300, bbox_inches='tight')
    logger.info(f"Saved trajectory plot to {save_path_abs}")
    print(f"\n✓ Plot saved successfully to: {save_path_abs}\n")
    plt.close()


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Main function to visualize ego trajectories from sampled scenarios.

    :param cfg: Hydra configuration
    """
    # Get the original working directory (before Hydra changes it)
    # Hydra changes the working directory to its output directory
    original_cwd = hydra.utils.get_original_cwd()
    current_cwd = os.getcwd()

    # Configuration parameters
    num_scenarios_to_sample = cfg.get('num_scenarios_to_visualize', 10)
    num_points_per_trajectory = cfg.get('num_points_per_trajectory', None)
    output_path = cfg.get('trajectory_plot_path', 'ego_trajectories.png')

    # If output_path is relative, make it relative to the original working directory
    if not Path(output_path).is_absolute():
        output_path = str(Path(original_cwd) / output_path)

    # Trajectory extraction parameters (following nuplan_feature_builder defaults)
    history_horizon = cfg.get('history_horizon', 2.0)
    future_horizon = cfg.get('future_horizon', 8.0)
    sample_interval = cfg.get('sample_interval', 0.1)

    logger.info(f"Starting ego trajectory visualization...")
    logger.info(f"Original working directory: {original_cwd}")
    logger.info(f"Current working directory: {current_cwd}")
    logger.info(f"Output will be saved to: {output_path}")
    logger.info(f"Will sample {num_scenarios_to_sample} scenarios")
    logger.info(f"Trajectory config: history={history_horizon}s, future={future_horizon}s, interval={sample_interval}s")
    logger.info(f"Trajectories will be normalized to start at (0, 0)")

    # Build worker pool
    worker = build_worker(cfg)

    # Load scenarios from the dataset
    logger.info("Loading scenarios from dataset...")
    all_scenarios = build_scenarios(cfg, worker, model=None)

    logger.info(f"Total scenarios available: {len(all_scenarios)}")

    # Sample a subset of scenarios
    num_to_sample = min(num_scenarios_to_sample, len(all_scenarios))
    sampled_scenarios = random.sample(all_scenarios, num_to_sample)

    logger.info(f"Sampled {num_to_sample} scenarios")

    # Extract trajectories from sampled scenarios
    trajectories = []
    scenario_names = []

    for idx, scenario in enumerate(sampled_scenarios):
        logger.info(f"Processing scenario {idx + 1}/{num_to_sample}: {scenario.scenario_name}")
        try:
            traj = extract_ego_trajectory_coordinates(
                scenario,
                history_horizon=history_horizon,
                future_horizon=future_horizon,
                sample_interval=sample_interval
            )
            trajectories.append(traj)
            scenario_names.append(f"{scenario.scenario_type}_{scenario.token[:8]}")
            logger.info(f"  Extracted trajectory with {len(traj)} points")
        except Exception as e:
            logger.warning(f"  Failed to extract trajectory: {e}")
            continue

    # Plot all trajectories
    if len(trajectories) > 0:
        logger.info(f"Plotting {len(trajectories)} trajectories...")
        plot_trajectories(
            trajectories,
            scenario_names,
            save_path=output_path,
            num_points_to_plot=num_points_per_trajectory
        )
        logger.info("Visualization complete!")
    else:
        logger.error("No trajectories were extracted successfully!")


if __name__ == "__main__":
    main()
