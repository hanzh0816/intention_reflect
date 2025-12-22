#!/usr/bin/env python3
"""
Create .nuboard metadata file for exported failure cases.

This script creates a .nuboard file for directories that were exported
before v1.2 and don't have the required metadata file.

Usage:
    python scripts/create_nuboard_file.py <output_dir>

Example:
    python scripts/create_nuboard_file.py work_dirs/failure_viz
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from nuplan.planning.nuboard.base.data_class import NuBoardFile

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_nuboard_file(output_dir: Path) -> Path:
    """
    Create a .nuboard metadata file for the given directory.

    Args:
        output_dir: Directory containing exported failure cases

    Returns:
        Path to the created .nuboard file
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {output_dir}")

    # Check if there are already .nuboard files
    existing_nuboard_files = list(output_dir.glob("*.nuboard"))
    if existing_nuboard_files:
        logger.warning(f"Found existing .nuboard files: {len(existing_nuboard_files)}")
        for f in existing_nuboard_files:
            logger.warning(f"  - {f.name}")

        response = input("Do you want to create a new .nuboard file anyway? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return existing_nuboard_files[0]

    # Create .nuboard filename with timestamp
    nuboard_filename = output_dir / f"nuboard_{int(time.time())}{NuBoardFile.extension()}"

    # Create NuBoardFile object
    # simulation_folder is "simulation" to match the standard nuBoard structure
    # Directory structure: <output_dir>/simulation/<planner>/<scenario_type>/<log>/<scenario>.pkl.xz
    # This separates .nuboard files from simulation logs
    nuboard_file = NuBoardFile(
        simulation_main_path=str(output_dir),
        simulation_folder="simulation",  # Simulation logs are in "simulation" subfolder
        metric_main_path=str(output_dir),
        metric_folder="metrics",  # Empty folder (no metrics for failure cases)
        aggregator_metric_folder="aggregator_metric",  # Empty folder
    )

    # Save .nuboard file
    logger.info(f"Creating .nuboard file: {nuboard_filename}")
    nuboard_file.save_nuboard_file(nuboard_filename)

    logger.info(f"✓ Successfully created .nuboard file: {nuboard_filename}")
    return nuboard_filename


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create .nuboard metadata file for exported failure cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'output_dir',
        type=Path,
        help='Directory containing exported failure cases (e.g., work_dirs/failure_viz)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        nuboard_file = create_nuboard_file(args.output_dir)

        print("\n" + "=" * 80)
        print("Success! You can now use nuBoard to visualize your failure cases:")
        print("=" * 80)
        print(f"\n  python run_nuboard.py \\")
        print(f"    simulation_path={args.output_dir} \\")
        print(f"    port_number=5006")
        print(f"\n  Then open: http://localhost:5006\n")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Failed to create .nuboard file: {e}")
        if args.verbose:
            logger.exception("Detailed error:")
        sys.exit(1)


if __name__ == '__main__':
    main()
