#!/usr/bin/env python3
"""
Failure case visualization export tool.

Exports failure cases from SQLite database to nuBoard-compatible format
for interactive visualization.

Usage examples:
    # List all available failure cases
    python scripts/export_failure_cases.py \\
        --list \\
        --database-path work_dirs/exp/failure_cases.db

    # Export a specific scenario (with real scenario from nuplan database)
    python scripts/export_failure_cases.py \\
        --scenario-name "00cca24d240f5980" \\
        --database-path work_dirs/exp/failure_cases.db \\
        --output-dir work_dirs/failure_viz \\
        --data-root /data/sets/nuplan \\
        --map-root /data/sets/nuplan/maps \\
        --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db

    # Export without scenario_builder (uses stub scenario)
    python scripts/export_failure_cases.py \\
        --scenario-name "00cca24d240f5980" \\
        --database-path work_dirs/exp/failure_cases.db \\
        --output-dir work_dirs/failure_viz

    # Then view with nuBoard
    python run_nuboard.py \\
        simulation_path=work_dirs/failure_viz \\
        port_number=5006
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.failure_case_exporter import FailureCaseExporter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_failure_type(case: dict) -> str:
    """Format failure types as a comma-separated string."""
    types = []
    if case.get('has_collision'):
        types.append('collision')
    if case.get('has_speed_violation'):
        types.append('speed_violation')
    if case.get('has_deadlock'):
        types.append('deadlock')
    if case.get('has_drivable_area_violation'):
        types.append('drivable_area')

    return ', '.join(types) if types else 'unknown'


def build_scenario_builder(
    data_root: Optional[str],
    map_root: Optional[str],
    db_files: Optional[str],
    sensor_root: Optional[str] = None,
    map_version: str = "nuplan-maps-v1.0",
) -> Optional[NuPlanScenarioBuilder]:
    """
    Build NuPlanScenarioBuilder if parameters are provided.

    Args:
        data_root: Path to nuplan dataset root
        map_root: Path to map files
        db_files: Path to database file(s), can be comma-separated
        sensor_root: Optional path to sensor data
        map_version: Map version string

    Returns:
        NuPlanScenarioBuilder instance, or None if parameters are missing
    """
    if not all([data_root, map_root, db_files]):
        logger.info("Scenario builder parameters not provided, will use StubScenario")
        return None

    logger.info("Building scenario_builder with nuplan database...")

    # Parse db_files (can be comma-separated)
    if ',' in db_files:
        db_file_list = [f.strip() for f in db_files.split(',')]
    else:
        db_file_list = db_files

    try:
        scenario_builder = NuPlanScenarioBuilder(
            data_root=data_root,
            map_root=map_root,
            sensor_root=sensor_root,
            db_files=db_file_list,
            map_version=map_version,
            vehicle_parameters=get_pacifica_parameters(),
        )
        logger.info("✓ Scenario builder created successfully")
        return scenario_builder

    except Exception as e:
        logger.error(f"Failed to create scenario_builder: {e}")
        logger.warning("Will fall back to using StubScenario")
        return None


def list_failure_cases(exporter: FailureCaseExporter, verbose: bool = False) -> None:
    """
    List all failure cases in the database.

    Args:
        exporter: FailureCaseExporter instance
        verbose: If True, show detailed information including log_name
    """
    cases = exporter.list_failure_cases()

    if not cases:
        print("\nNo failure cases found in database.")
        print("Make sure you have run simulation with failure_case_collector callback enabled.")
        return

    print(f"\nFound {len(cases)} failure case(s):\n")

    if verbose:
        # Detailed view with log_name
        print("-" * 150)
        print(f"{'#':<4} {'Scenario Name':<18} {'Log Name':<50} {'Planner':<12} {'Type':<20} {'Severity':<10}")
        print("-" * 150)

        for idx, case in enumerate(cases, 1):
            scenario_name = case['scenario_name']
            log_name = case['log_name']
            planner_name = case['planner_name']
            severity = case['failure_severity']
            failure_type = format_failure_type(case)

            # Truncate long names
            if len(log_name) > 47:
                log_name = log_name[:44] + "..."

            print(f"{idx:<4} {scenario_name:<18} {log_name:<50} {planner_name:<12} {failure_type:<20} {severity:<10}")

        print("-" * 150)
        print(f"\nTotal: {len(cases)} failure case(s)")
        print("\nTip: The log_name corresponds to the database file:")
        print("     <data-root>/nuplan-v1.1/<split>/<log_name>.db")
        print("     Example: /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db")
    else:
        # Compact view
        print("-" * 120)
        print(f"{'#':<4} {'Scenario Name':<40} {'Planner':<15} {'Severity':<10} {'Type':<25} {'Duration':<8}")
        print("-" * 120)

        for idx, case in enumerate(cases, 1):
            scenario_name = case['scenario_name']
            planner_name = case['planner_name']
            severity = case['failure_severity']
            failure_type = format_failure_type(case)
            duration = f"{case['duration_seconds']:.1f}s"

            # Truncate long scenario names
            if len(scenario_name) > 37:
                scenario_name = scenario_name[:34] + "..."

            print(f"{idx:<4} {scenario_name:<40} {planner_name:<15} {severity:<10} {failure_type:<25} {duration:<8}")

        print("-" * 120)
        print(f"\nTotal: {len(cases)} failure case(s)")
        print("\nTip: Use --verbose to see log_name and database file information")


def show_scenario_info(exporter: FailureCaseExporter, scenario_name: str) -> bool:
    """
    Show detailed information about a specific scenario.

    Args:
        exporter: FailureCaseExporter instance
        scenario_name: Scenario name to query

    Returns:
        True if found, False otherwise
    """
    # Query the failure case
    failure_case = exporter._db_manager.query_failure_case_by_scenario(scenario_name)

    if failure_case is None:
        print(f"\n✗ Scenario '{scenario_name}' not found in database.")
        print("\nTip: Use --list to see all available scenarios")
        return False

    print(f"\n{'='*80}")
    print(f"Scenario Information: {scenario_name}")
    print(f"{'='*80}\n")

    print(f"Basic Information:")
    print(f"  Scenario Name (Token): {failure_case['scenario_name']}")
    print(f"  Scenario Type:         {failure_case['scenario_type']}")
    print(f"  Log Name:              {failure_case['log_name']}")
    print(f"  Planner:               {failure_case['planner_name']}")
    print(f"  Duration:              {failure_case['duration_seconds']:.2f} seconds")
    print(f"  Severity:              {failure_case['failure_severity']}")

    print(f"\nFailure Types:")
    print(f"  Collision:             {'Yes' if failure_case['has_collision'] else 'No'}")
    print(f"  Speed Violation:       {'Yes' if failure_case['has_speed_violation'] else 'No'}")
    print(f"  Deadlock:              {'Yes' if failure_case['has_deadlock'] else 'No'}")
    print(f"  Drivable Area:         {'Yes' if failure_case['has_drivable_area_violation'] else 'No'}")

    print(f"\nDatabase File Mapping:")
    log_name = failure_case['log_name']
    print(f"  Log Name:    {log_name}")
    print(f"  DB File:     <data-root>/nuplan-v1.1/<split>/{log_name}.db")
    print(f"\n  Common paths:")
    print(f"    Mini split:      /data/sets/nuplan/nuplan-v1.1/mini/{log_name}.db")
    print(f"    Trainval split:  /data/sets/nuplan/nuplan-v1.1/trainval/{log_name}.db")
    print(f"    Test split:      /data/sets/nuplan/nuplan-v1.1/test/{log_name}.db")

    print(f"\nExport Command (with real scenario):")
    print(f"  python scripts/export_failure_cases.py \\")
    print(f"    --scenario-name \"{scenario_name}\" \\")
    print(f"    --database-path work_dirs/exp/failure_cases.db \\")
    print(f"    --output-dir work_dirs/failure_viz \\")
    print(f"    --data-root /data/sets/nuplan \\")
    print(f"    --map-root /data/sets/nuplan/maps \\")
    print(f"    --db-files /data/sets/nuplan/nuplan-v1.1/mini/{log_name}.db")

    print(f"\nExport Command (simple, using StubScenario):")
    print(f"  python scripts/export_failure_cases.py \\")
    print(f"    --scenario-name \"{scenario_name}\" \\")
    print(f"    --database-path work_dirs/exp/failure_cases.db \\")
    print(f"    --output-dir work_dirs/failure_viz")

    print(f"\n{'='*80}\n")

    return True


def export_scenario(
    exporter: FailureCaseExporter,
    scenario_name: str,
    output_dir: Path
) -> bool:
    """
    Export a single scenario.

    Args:
        exporter: FailureCaseExporter instance
        scenario_name: Scenario name to export
        output_dir: Output directory

    Returns:
        True if successful, False otherwise
    """
    print(f"\nExporting failure case: {scenario_name}")
    print("-" * 80)

    try:
        # Export SimulationLog
        output_file = exporter.export_failure_case(scenario_name)

        # Create .nuboard metadata file
        nuboard_file = exporter.create_nuboard_file()

        print("\n✓ Export completed successfully!")
        print(f"\nSimulationLog file: {output_file}")
        print(f"NuBoard file: {nuboard_file}")
        print(f"Output directory: {output_dir}")

        # Print nuBoard command
        print("\n" + "=" * 80)
        print("To visualize with nuBoard, run:")
        print("=" * 80)
        print(f"\n  python run_nuboard.py \\")
        print(f"    simulation_path={output_dir} \\")
        print(f"    port_number=5006")
        print(f"\n  Then open: http://localhost:5006\n")
        print("=" * 80)

        return True

    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nTip: Use --list to see all available failure cases")
        return False

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        logger.error("Export failed", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export failure cases to nuBoard visualization format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Database argument
    parser.add_argument(
        '--database-path',
        type=Path,
        default=Path('work_dirs/exp/failure_cases.db'),
        help='Path to failure cases SQLite database (default: work_dirs/exp/failure_cases.db)'
    )

    # Action: list or export
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--list',
        action='store_true',
        help='List all failure cases in database'
    )
    action_group.add_argument(
        '--info',
        type=str,
        metavar='SCENARIO_NAME',
        help='Show detailed information about a specific scenario'
    )
    action_group.add_argument(
        '--scenario-name',
        type=str,
        help='Scenario name/token to export (e.g., "00cca24d240f5980")'
    )

    # Output directory (required for export)
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for nuBoard files (required when exporting)'
    )

    # Scenario builder parameters (optional)
    parser.add_argument(
        '--data-root',
        type=str,
        help='Path to nuplan dataset root (optional, enables real scenario loading)'
    )
    parser.add_argument(
        '--map-root',
        type=str,
        help='Path to nuplan map files (optional, required if data-root is provided)'
    )
    parser.add_argument(
        '--db-files',
        type=str,
        help='Path to nuplan database file(s), comma-separated for multiple files '
             '(optional, required if data-root is provided)'
    )
    parser.add_argument(
        '--sensor-root',
        type=str,
        help='Path to sensor data (optional)'
    )
    parser.add_argument(
        '--map-version',
        type=str,
        default='nuplan-maps-v1.0',
        help='Map version (default: nuplan-maps-v1.0)'
    )

    # Logging level
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('src.evaluation').setLevel(logging.DEBUG)

    # Validate arguments
    if args.scenario_name and not args.output_dir:
        parser.error("--output-dir is required when exporting a scenario")

    # Check database exists
    if not args.database_path.exists():
        print(f"\n✗ Error: Database not found: {args.database_path}")
        print("\nMake sure you have:")
        print("  1. Run simulation with failure_case_collector callback enabled")
        print("  2. Specified the correct database path")
        print("\nExample configuration in your eval config:")
        print("  callback:")
        print("    - failure_case_collector")
        sys.exit(1)

    # Build scenario_builder (if parameters provided)
    scenario_builder = build_scenario_builder(
        data_root=args.data_root,
        map_root=args.map_root,
        db_files=args.db_files,
        sensor_root=args.sensor_root,
        map_version=args.map_version,
    )

    # Create exporter
    try:
        exporter = FailureCaseExporter(
            database_path=args.database_path,
            output_base_dir=args.output_dir if args.output_dir else Path('.'),
            scenario_builder=scenario_builder,
        )
    except Exception as e:
        print(f"\n✗ Error initializing exporter: {e}")
        sys.exit(1)

    # Execute action
    if args.list:
        list_failure_cases(exporter, verbose=args.verbose)
        sys.exit(0)

    elif args.info:
        success = show_scenario_info(exporter, args.info)
        sys.exit(0 if success else 1)

    elif args.scenario_name:
        success = export_scenario(exporter, args.scenario_name, args.output_dir)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
