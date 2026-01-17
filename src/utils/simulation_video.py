"""
Simulation video generation module.

Generates ego-centric videos from simulation history logs with agent IDs.
"""

import json
import logging
import lzma
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import msgpack
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.simulation_log import SimulationLog

from src.utils.visualization_utils import (
    plot_map_context_ego_centric,
    plot_tracked_objects_ego_centric,
    plot_vehicle_box_ego_centric,
    plot_planned_trajectory_ego_centric,
)

logger = logging.getLogger(__name__)


class SimulationVideoGenerator:
    """Generate ego-centric video from simulation history."""

    def __init__(
        self,
        simulation_history: SimulationHistory,
        map_api: AbstractMap,
        config: Optional[Dict] = None,
    ):
        """
        Initialize video generator.

        Args:
            simulation_history: Simulation history object
            map_api: Map API for rendering
            config: Optional configuration dict with keys:
                - fps: Video FPS (default: 10)
                - resolution: (width, height) tuple (default: (1920, 1080))
                - map_radius: Map display radius in meters (default: 50)
                - dpi: Figure DPI (default: 100)
        """
        self.history = simulation_history
        self.map_api = map_api

        # Parse config
        if config is None:
            config = {}

        self.fps = config.get("fps", 10)
        self.resolution = config.get("resolution", (1920, 1080))
        self.map_radius = config.get("map_radius", 50.0)
        self.dpi = config.get("dpi", 100)

        # Calculate figure size from resolution and DPI
        self.fig_width = self.resolution[0] / self.dpi
        self.fig_height = self.resolution[1] / self.dpi

        logger.info(
            f"Video generator initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps} FPS"
        )

    def _build_agent_id_mapping(self) -> Dict[str, int]:
        """
        Build mapping from track_token to sequential ID.

        Scans all timesteps and assigns sequential IDs based on first appearance.
        Only includes agents that appear within map_radius of ego vehicle.

        Returns:
            Dict mapping track_token (str) to sequential ID (int)
        """
        logger.info("Building agent ID mapping...")

        token_to_id = {}
        next_id = 1

        # Iterate through all samples in history
        for sample in self.history.data:
            ego_state = sample.ego_state
            tracked_objects = sample.observation.tracked_objects

            if not hasattr(tracked_objects, "tracked_objects"):
                continue

            for obj in tracked_objects.tracked_objects:
                # Calculate distance from ego to agent
                dx = obj.center.x - ego_state.rear_axle.x
                dy = obj.center.y - ego_state.rear_axle.y
                distance = np.sqrt(dx * dx + dy * dy)

                # Skip agents outside map_radius
                if distance > self.map_radius:
                    continue

                # Get track token
                if hasattr(obj.metadata, "track_token"):
                    track_token = obj.metadata.track_token
                    if isinstance(track_token, str):
                        token_str = track_token
                    elif hasattr(track_token, "hex"):
                        token_str = track_token.hex
                    else:
                        token_str = str(track_token)

                    # Assign ID if not seen before
                    if token_str and token_str not in token_to_id:
                        token_to_id[token_str] = next_id
                        next_id += 1

        logger.info(f"Found {len(token_to_id)} unique agents within {self.map_radius}m radius")
        return token_to_id

    def _render_frame(
        self,
        ego_state: EgoState,
        tracked_objects: TrackedObjects,
        trajectory,
        agent_id_map: Dict[str, int],
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray:
        """
        Render a single frame in ego-centric view.

        Args:
            ego_state: Ego vehicle state
            tracked_objects: Tracked objects at this timestep
            trajectory: Planned trajectory at this timestep
            agent_id_map: Agent ID mapping
            frame_idx: Current frame index
            total_frames: Total number of frames

        Returns:
            Frame as numpy array (H, W, 3) in RGB format
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)

        # 1. Render map in ego frame
        plot_map_context_ego_centric(ax, self.map_api, ego_state, self.map_radius)

        # 2. Render other vehicles with IDs (only within map_radius)
        plot_tracked_objects_ego_centric(
            ax, tracked_objects, ego_state, agent_id_map, show_ids=True, map_radius=self.map_radius
        )

        # 3. Render planned trajectory
        if trajectory is not None:
            plot_planned_trajectory_ego_centric(
                ax, trajectory, ego_state,
                color='#FF00FF',  # Magenta color for trajectory
                linewidth=2.5,
                alpha=0.8,
                label='Planned Trajectory'
            )

        # 4. Render ego vehicle at center (0, 0) heading up
        plot_vehicle_box_ego_centric(
            ax,
            ego_state.rear_axle.x,
            ego_state.rear_axle.y,
            ego_state.rear_axle.heading,
            ego_state.car_footprint.length,
            ego_state.car_footprint.width,
            ego_state,
            color="#00FF00",
            label="Ego Vehicle",
            alpha=0.7,
            linewidth=2.5,
        )

        # 5. Configure axes
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-self.map_radius, self.map_radius)
        ax.set_ylim(-self.map_radius, self.map_radius)
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_title(
            f"Ego-Centric View - Frame {frame_idx + 1}/{total_frames}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10)

        plt.tight_layout()

        # Convert figure to numpy array
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        plt.close(fig)

        return frame

    def generate_video(
        self, output_path: str, agent_id_mapping_path: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Generate video from simulation history.

        Args:
            output_path: Path to save video file (e.g., 'output.mp4')
            agent_id_mapping_path: Path to save agent ID mapping JSON
                                   (default: same dir as video with '_agent_ids.json' suffix)

        Returns:
            Agent ID mapping dict
        """
        logger.info(f"Generating video: {output_path}")

        # Build agent ID mapping
        agent_id_map = self._build_agent_id_mapping()

        # Determine mapping file path
        if agent_id_mapping_path is None:
            video_path = Path(output_path)
            agent_id_mapping_path = str(
                video_path.parent / f"{video_path.stem}_agent_ids.json"
            )

        # Save agent ID mapping
        with open(agent_id_mapping_path, "w") as f:
            json.dump(agent_id_map, f, indent=2)
        logger.info(f"Saved agent ID mapping to: {agent_id_mapping_path}")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.resolution)

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        # Generate frames
        total_frames = len(self.history.data)
        logger.info(f"Rendering {total_frames} frames...")

        for frame_idx, sample in enumerate(self.history.data):
            if frame_idx % 10 == 0:
                logger.info(f"Rendering frame {frame_idx + 1}/{total_frames}")

            # Get ego state, tracked objects, and trajectory
            ego_state = sample.ego_state
            tracked_objects = sample.observation.tracked_objects
            trajectory = sample.trajectory if hasattr(sample, 'trajectory') else None

            # Render frame
            frame_rgb = self._render_frame(
                ego_state, tracked_objects, trajectory, agent_id_map, frame_idx, total_frames
            )

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Write frame
            video_writer.write(frame_bgr)

        # Release video writer
        video_writer.release()

        logger.info(f"Video saved to: {output_path}")
        logger.info(
            f"Total frames: {total_frames}, Duration: {total_frames / self.fps:.2f}s"
        )

        return agent_id_map


def load_simulation_history(log_path: str) -> Tuple[SimulationHistory, AbstractMap]:
    """
    Load simulation history from log file.

    Args:
        log_path: Path to simulation log directory or file (.pkl or .msgpack.xz)

    Returns:
        (SimulationHistory, AbstractMap) tuple
    """
    log_path = Path(log_path)

    # Find simulation history file
    if log_path.is_dir():
        # Look for msgpack.xz files first (lzma compressed pickle)
        msgpack_files = list(log_path.glob("**/*.msgpack.xz"))
        if msgpack_files:
            log_file = msgpack_files[0]
            logger.info(f"Found simulation log: {log_file}")
        else:
            # Fallback to .pkl files
            pickle_files = list(log_path.glob("**/simulation_log*.pkl"))
            if not pickle_files:
                pickle_files = list(log_path.glob("**/*.pkl"))

            if not pickle_files:
                raise FileNotFoundError(f"No simulation log files found in {log_path}")

            log_file = pickle_files[0]
            logger.info(f"Found simulation log: {log_file}")
    else:
        log_file = log_path

    # Load file based on extension
    if log_file.suffix == ".xz" or str(log_file).endswith((".msgpack.xz", ".pkl.xz")):
        # Load compressed file using nuplan's SimulationLog loader
        # This handles both .msgpack.xz and .pkl.xz formats correctly
        logger.info(f"Loading compressed simulation log: {log_file}")
        simulation_log = SimulationLog.load_data(log_file)
    else:
        # Load regular pickle file
        logger.info(f"Loading pickle file: {log_file}")
        with open(log_file, "rb") as f:
            simulation_log = pickle.load(f)

    # Extract history and map
    if hasattr(simulation_log, "simulation_history"):
        history = simulation_log.simulation_history
    elif isinstance(simulation_log, SimulationHistory):
        history = simulation_log
    else:
        raise ValueError(f"Unexpected simulation log type: {type(simulation_log)}")

    # Get map API
    if hasattr(history, "map_api"):
        map_api = history.map_api
    else:
        raise ValueError("Simulation history missing map_api")

    logger.info(f"Loaded simulation history with {len(history.data)} samples")

    return history, map_api


def generate_video_from_log(
    log_path: str, output_video_path: str, config: Optional[Dict] = None
) -> Dict[str, int]:
    """
    Convenience function to generate video from simulation log.

    Args:
        log_path: Path to simulation log directory or pickle file
        output_video_path: Path to save video file
        config: Optional video generation config

    Returns:
        Agent ID mapping dict
    """
    # Load simulation history
    history, map_api = load_simulation_history(log_path)

    # Create video generator
    generator = SimulationVideoGenerator(history, map_api, config)

    # Generate video
    agent_id_map = generator.generate_video(output_video_path)

    return agent_id_map
