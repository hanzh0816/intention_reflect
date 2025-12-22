#!/usr/bin/env python3
"""
Verify nuBoard directory structure and .nuboard file.

This script checks if the exported failure case directory has the correct
structure for nuBoard visualization.

Usage:
    python scripts/verify_nuboard_structure.py <output_dir>

Example:
    python scripts/verify_nuboard_structure.py work_dirs/failure_viz
"""
import argparse
import sys
from pathlib import Path
import pickle

from nuplan.planning.nuboard.base.data_class import NuBoardFile


def verify_structure(output_dir: Path) -> bool:
    """
    Verify the directory structure.

    Args:
        output_dir: Output directory to verify

    Returns:
        True if structure is valid, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Verifying nuBoard structure: {output_dir}")
    print(f"{'='*80}\n")

    errors = []
    warnings = []

    # Check 1: Output directory exists
    if not output_dir.exists():
        errors.append(f"Output directory does not exist: {output_dir}")
        return False

    print("✓ Output directory exists")

    # Check 2: .nuboard file exists
    nuboard_files = list(output_dir.glob("*.nuboard"))
    if not nuboard_files:
        errors.append("No .nuboard file found")
    elif len(nuboard_files) > 1:
        warnings.append(f"Multiple .nuboard files found: {len(nuboard_files)}")
        print(f"⚠ Multiple .nuboard files found ({len(nuboard_files)})")
    else:
        print(f"✓ .nuboard file found: {nuboard_files[0].name}")

    # Check 3: Read .nuboard file
    if nuboard_files:
        try:
            nuboard_file = NuBoardFile.load_nuboard_file(nuboard_files[0])
            print(f"✓ .nuboard file is readable")

            # Check 4: Verify .nuboard configuration
            print(f"\n  NuBoard Configuration:")
            print(f"    simulation_main_path: {nuboard_file.simulation_main_path}")
            print(f"    simulation_folder: {nuboard_file.simulation_folder}")
            print(f"    metric_main_path: {nuboard_file.metric_main_path}")
            print(f"    metric_folder: {nuboard_file.metric_folder}")

            # Check if simulation_folder is correct
            if nuboard_file.simulation_folder == "simulation":
                print(f"  ✓ simulation_folder is set to 'simulation' (v1.2.1+)")
            elif nuboard_file.simulation_folder == ".":
                warnings.append("simulation_folder is set to '.' (v1.2, may cause errors)")
                print(f"  ⚠ simulation_folder is '.' (old version, may cause NotADirectoryError)")
            else:
                warnings.append(f"Unexpected simulation_folder: {nuboard_file.simulation_folder}")
                print(f"  ⚠ Unexpected simulation_folder: {nuboard_file.simulation_folder}")

        except Exception as e:
            errors.append(f"Failed to read .nuboard file: {e}")
            print(f"✗ Failed to read .nuboard file: {e}")

    # Check 5: simulation directory exists
    simulation_dir = output_dir / "simulation"
    if simulation_dir.exists() and simulation_dir.is_dir():
        print(f"✓ simulation/ directory exists")

        # Check 6: Planner directories
        planner_dirs = [d for d in simulation_dir.iterdir() if d.is_dir()]
        if planner_dirs:
            print(f"✓ Found {len(planner_dirs)} planner(s): {[d.name for d in planner_dirs]}")

            # Check 7: .pkl.xz files
            pkl_files = list(simulation_dir.rglob("*.pkl.xz"))
            if pkl_files:
                print(f"✓ Found {len(pkl_files)} SimulationLog file(s)")
                print(f"\n  SimulationLog files:")
                for pkl in pkl_files[:5]:  # Show first 5
                    rel_path = pkl.relative_to(simulation_dir)
                    print(f"    - {rel_path}")
                if len(pkl_files) > 5:
                    print(f"    ... and {len(pkl_files) - 5} more")
            else:
                errors.append("No .pkl.xz files found in simulation/")
        else:
            errors.append("No planner directories found in simulation/")
    else:
        # Check for old structure (planner dirs at root)
        old_planner_dirs = [
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name not in ['simulation', 'metrics', 'aggregator_metric']
        ]
        if old_planner_dirs:
            errors.append(
                f"Planner directories found at root level (old structure): {[d.name for d in old_planner_dirs]}\n"
                f"  Expected: planner directories should be in simulation/ subdirectory\n"
                f"  Fix: Move planner directories into simulation/ or re-export"
            )
            print(f"✗ Old directory structure detected (v1.0-v1.2)")
            print(f"  Planner directories at root: {[d.name for d in old_planner_dirs]}")
            print(f"  These should be moved to simulation/ subdirectory")
        else:
            errors.append("simulation/ directory not found")

    # Summary
    print(f"\n{'='*80}")
    print("Verification Summary")
    print(f"{'='*80}\n")

    if not errors and not warnings:
        print("✅ All checks passed! Directory structure is correct.")
        print("\nYou can now run nuBoard:")
        print(f"  python run_nuboard.py \\")
        print(f"    simulation_path={output_dir} \\")
        print(f"    port_number=5006")
        return True
    else:
        if warnings:
            print(f"⚠️  Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"  - {w}")
            print()

        if errors:
            print(f"❌ Errors ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
            print()
            print("Recommendations:")
            print("  1. Re-export using the latest version:")
            print(f"     python scripts/export_failure_cases.py ...")
            print("  2. Or create .nuboard file:")
            print(f"     python scripts/create_nuboard_file.py {output_dir}")
            return False
        else:
            # Only warnings
            print("\n⚠️  Structure has warnings but may still work.")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify nuBoard directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'output_dir',
        type=Path,
        help='Directory to verify (e.g., work_dirs/failure_viz)'
    )

    args = parser.parse_args()

    try:
        success = verify_structure(args.output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
