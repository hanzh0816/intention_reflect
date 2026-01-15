#!/usr/bin/env python3
"""
Simulate a specific scenario and generate ego-centric video visualization.

This script uses local config files (like eval.sh) to get all configuration
including planner, checkpoint, paths, etc., then runs simulation for a specific
scenario and generates a video.

Usage:
    python scripts/simulate_and_visualize.py \\
        --config 251218_eval-plantf \\
        --scenario_token <token> \\
        --simulation_mode closed_loop_nonreactive

Simulation Modes:
    - open_loop: Agents follow logged trajectories (open_loop_boxes)
    - closed_loop_nonreactive: Agents use IDM model (closed_loop_nonreactive_agents)
    - closed_loop_reactive: Agents use IDM model with reactive metrics (closed_loop_reactive_agents)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.config_utils import load_yaml, flatten_dict, format_value
from scripts.load_config import build_hydra_params
from src.utils.simulation_video import generate_video_from_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run simulation for a specific scenario and generate video visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Local config name (e.g., 251218_eval-plantf) from config/local/",
    )

    parser.add_argument(
        "--scenario_token", type=str, required=True, help="Scenario token to simulate"
    )

    parser.add_argument(
        "--simulation_mode",
        type=str,
        required=True,
        choices=[
            "open_loop_boxes",
            "closed_loop_nonreactive_agents",
            "closed_loop_reactive_agents",
        ],
        help="Simulation evaluation mode",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-generated based on config)",
    )

    parser.add_argument(
        "--video_fps", type=int, default=10, help="Video FPS (default: 10)"
    )

    parser.add_argument(
        "--video_resolution",
        type=str,
        default="1920x1080",
        help="Video resolution as WIDTHxHEIGHT (default: 1920x1080)",
    )

    parser.add_argument(
        "--map_radius",
        type=float,
        default=50.0,
        help="Map display radius in meters (default: 50)",
    )

    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip video generation (only run simulation)",
    )

    return parser.parse_args()


def load_local_config(config_name: str) -> Dict:
    """
    Load local config file.

    Args:
        config_name: Config name (without .yaml extension)

    Returns:
        Config dictionary
    """
    config_file = project_root / "config" / "local" / f"{config_name}.yaml"

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        logger.error("\nAvailable configs:")
        local_dir = project_root / "config" / "local"
        if local_dir.exists():
            for f in sorted(local_dir.glob("*.yaml")):
                logger.error(f"  - {f.stem}")
        sys.exit(1)

    logger.info(f"Loading config from: {config_file}")
    config = load_yaml(str(config_file))

    return config


def setup_environment(config: Dict):
    """
    Setup environment variables from config.

    Args:
        config: Config dictionary
    """
    paths = config.get("paths", {})

    if "nuplan_data_root" in paths:
        os.environ["NUPLAN_DATA_ROOT"] = paths["nuplan_data_root"]
        logger.info(f"NUPLAN_DATA_ROOT: {paths['nuplan_data_root']}")

    if "nuplan_maps_root" in paths:
        os.environ["NUPLAN_MAPS_ROOT"] = paths["nuplan_maps_root"]
        logger.info(f"NUPLAN_MAPS_ROOT: {paths['nuplan_maps_root']}")

    if "nuplan_exp_root" in paths:
        os.environ["NUPLAN_EXP_ROOT"] = paths["nuplan_exp_root"]
        logger.info(f"NUPLAN_EXP_ROOT: {paths['nuplan_exp_root']}")

    if "nuplan_db_files" in paths:
        os.environ["NUPLAN_DB_FILES"] = paths["nuplan_db_files"]
        logger.info(f"NUPLAN_DB_FILES: {paths['nuplan_db_files']}")

    # Setup GPU
    gpu_devices = config.get("gpu", {}).get("devices", "")
    if gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_devices)
        logger.info(f"CUDA_VISIBLE_DEVICES: {gpu_devices}")


def build_simulation_command(config: Dict, args) -> List[str]:
    """
    Build simulation command with Hydra parameters.

    Args:
        config: Config dictionary
        args: Command-line arguments

    Returns:
        Command as list of strings
    """
    # Get experiment name based on simulation mode
    experiment = args.simulation_mode

    # Build base Hydra params from config
    hydra_params = build_hydra_params(config, mode="eval")

    # Add simulation experiment
    hydra_params.append(f"+simulation={experiment}")

    # Override scenario filter to specific token
    hydra_params.append(f"scenario_filter.scenario_tokens=[{args.scenario_token}]")
    hydra_params.append("scenario_filter.limit_total_scenarios=1")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Auto-generate output dir
        exp_root = config.get("paths", {}).get("nuplan_exp_root", "work_dirs")
        output_dir = (
            f"{exp_root}/scenario_sim/{args.scenario_token[:8]}_{args.simulation_mode}"
        )

    hydra_params.append(f"output_dir={output_dir}")

    # Enable simulation log callback to save history
    # Use ++ to override existing config, or no prefix if it exists
    hydra_params.append(
        "++callback.simulation_log_callback.simulation_log_dir=${output_dir}/simulation_logs"
    )

    # Build command
    cmd = [sys.executable, str(project_root / "run_simulation.py")] + hydra_params

    return cmd, output_dir


def run_simulation(cmd: List[str], output_dir: str):
    """
    Run simulation command.

    Args:
        cmd: Command as list of strings
        output_dir: Output directory path
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Running Simulation")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("")

    # Run simulation
    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        logger.error(f"Simulation failed with exit code: {result.returncode}")
        sys.exit(result.returncode)

    logger.info("Simulation completed successfully!")


def find_simulation_log(output_dir: str) -> Optional[Path]:
    """
    Find simulation log file in output directory.

    Args:
        output_dir: Output directory path

    Returns:
        Path to simulation log file, or None if not found
    """
    output_path = Path(output_dir)

    # Look for simulation logs
    log_dirs = list(output_path.glob("**/simulation_logs"))

    if not log_dirs:
        logger.warning(f"No simulation_logs directory found in {output_dir}")
        return None

    # Find log files (try msgpack.xz first, then pkl as fallback)
    for log_dir in log_dirs:
        # Try msgpack.xz files (lzma compressed pickle)
        msgpack_files = list(log_dir.glob("**/*.msgpack.xz"))
        if msgpack_files:
            return msgpack_files[0]

        # Fallback to .pkl files
        pickle_files = list(log_dir.glob("**/*.pkl"))
        if pickle_files:
            return pickle_files[0]

    logger.warning(f"No simulation log files found in {output_dir}")
    return None


def generate_video(args, output_dir: str, simulation_log_path: Path):
    """
    Generate video from simulation log.

    Args:
        args: Command-line arguments
        output_dir: Output directory path
        simulation_log_path: Path to simulation log file
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Generating Video")
    logger.info("=" * 80)
    logger.info(f"Simulation log: {simulation_log_path}")

    # Parse video resolution
    width, height = map(int, args.video_resolution.split("x"))

    # Video config
    video_config = {
        "fps": args.video_fps,
        "resolution": (width, height),
        "map_radius": args.map_radius,
        "dpi": 100,
    }

    # Output video path
    output_video_path = (
        Path(output_dir)
        / f"simulation_{args.scenario_token[:8]}_{args.simulation_mode}.mp4"
    )
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate video
    agent_id_map = generate_video_from_log(
        str(simulation_log_path), str(output_video_path), video_config
    )

    logger.info("=" * 80)
    logger.info("Video generation completed!")
    logger.info(f"Video saved to: {output_video_path}")
    logger.info(
        f"Agent ID mapping saved to: {output_video_path.parent / f'{output_video_path.stem}_agent_ids.json'}"
    )
    logger.info(f"Total agents: {len(agent_id_map)}")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    try:
        # Load local config
        config = load_local_config(args.config)

        # Setup environment
        setup_environment(config)

        # Build simulation command
        cmd, output_dir = build_simulation_command(config, args)

        # Step 1: Run simulation
        # run_simulation(cmd, output_dir)

        # Step 2: Generate video (unless skipped)
        if not args.skip_video:
            # Find simulation log
            simulation_log_path = find_simulation_log(output_dir)

            if simulation_log_path is None:
                logger.error("Failed to find simulation log. Cannot generate video.")
                logger.error(
                    "You can try running with --skip_video to only run simulation."
                )
                sys.exit(1)

            # Generate video
            generate_video(args, output_dir, simulation_log_path)

        logger.info("=" * 80)
        logger.info("All tasks completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
