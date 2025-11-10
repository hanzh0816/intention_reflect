"""
Check if cached features contain intent labels (v2 - supports .gz files).

This script inspects cached .gz files to verify whether intent labels
are present in the cached data.
"""

import os
import pickle
import gzip
from pathlib import Path
from typing import Dict, Any, List


def check_cache_directory(cache_dir: str) -> Dict[str, Any]:
    """
    Check a cache directory for intent.gz files.

    Args:
        cache_dir: Path to a scenario cache directory (containing feature.gz, trajectory.gz, etc.)

    Returns:
        Dictionary with check results
    """
    result = {
        'dir': cache_dir,
        'has_feature': False,
        'has_trajectory': False,
        'has_intent': False,
        'intent_data': None,
        'error': None
    }

    cache_path = Path(cache_dir)

    # Check for required files
    feature_file = cache_path / 'feature.gz'
    trajectory_file = cache_path / 'trajectory.gz'
    intent_file = cache_path / 'intent.gz'

    result['has_feature'] = feature_file.exists()
    result['has_trajectory'] = trajectory_file.exists()
    result['has_intent'] = intent_file.exists()

    # If intent file exists, try to read it
    if result['has_intent']:
        try:
            with gzip.open(intent_file, 'rb') as f:
                intent_data = pickle.load(f)

            result['intent_data'] = {
                'type': str(type(intent_data)),
                'class_name': intent_data.__class__.__name__ if hasattr(intent_data, '__class__') else 'Unknown'
            }

            # Check if it's IntentLabels
            if hasattr(intent_data, 'lateral_intent'):
                lateral = intent_data.lateral_intent
                result['intent_data']['lateral_intent'] = {
                    'value': int(lateral) if hasattr(lateral, '__int__') else str(lateral),
                    'type': str(type(lateral))
                }

            if hasattr(intent_data, 'longitudinal_intent'):
                longitudinal = intent_data.longitudinal_intent
                result['intent_data']['longitudinal_intent'] = {
                    'value': int(longitudinal) if hasattr(longitudinal, '__int__') else str(longitudinal),
                    'type': str(type(longitudinal))
                }

        except Exception as e:
            result['error'] = str(e)

    return result


def find_cache_directories(cache_path: str, max_samples: int = 20) -> List[str]:
    """
    Find scenario cache directories.

    Args:
        cache_path: Root cache path
        max_samples: Maximum number of samples to find

    Returns:
        List of cache directory paths
    """
    cache_dirs = []
    cache_root = Path(cache_path)

    # Cache structure: <cache_root>/<scenario>/<scenario_type>/<hash>/
    # We need to find directories containing feature.gz files

    for scenario_dir in cache_root.iterdir():
        if not scenario_dir.is_dir():
            continue

        for scenario_type_dir in scenario_dir.iterdir():
            if not scenario_type_dir.is_dir():
                continue

            for sample_dir in scenario_type_dir.iterdir():
                if not sample_dir.is_dir():
                    continue

                # Check if this is a cache directory
                if (sample_dir / 'feature.gz').exists():
                    cache_dirs.append(str(sample_dir))

                    if len(cache_dirs) >= max_samples:
                        return cache_dirs

    return cache_dirs


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check if cached features contain intent labels (v2)"
    )
    parser.add_argument(
        '--cache_path',
        type=str,
        default='/data2/hzh/nuplan/exp/cache_plantf_1M',
        help='Path to cache directory'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=20,
        help='Maximum number of samples to check'
    )

    args = parser.parse_args()

    print(f"📁 Scanning cache directory: {args.cache_path}\n")

    # Find cache directories
    print(f"🔍 Finding cache samples...")
    cache_dirs = find_cache_directories(args.cache_path, args.max_samples)

    if not cache_dirs:
        print(f"❌ No cache directories found in {args.cache_path}")
        print("\n   Cache might be empty or have a different structure.")
        return

    print(f"✓ Found {len(cache_dirs)} cache samples\n")

    # Check each directory
    samples_with_intent = 0
    samples_without_intent = 0
    samples_with_error = 0

    intent_labels_map = {
        'lateral': {
            0: 'turn_left',
            1: 'turn_right',
            2: 'shift_left',
            3: 'shift_right',
            4: 'stay_in_lane'
        },
        'longitudinal': {
            0: 'accelerate',
            1: 'maintain_speed',
            2: 'decelerate',
            3: 'stop'
        }
    }

    for i, cache_dir in enumerate(cache_dirs):
        result = check_cache_directory(cache_dir)

        # Get relative path for display
        rel_path = Path(cache_dir).relative_to(args.cache_path)

        print(f"Sample {i+1}/{len(cache_dirs)}: {rel_path}")
        print(f"  feature.gz:    {'✓' if result['has_feature'] else '✗'}")
        print(f"  trajectory.gz: {'✓' if result['has_trajectory'] else '✗'}")
        print(f"  intent.gz:     {'✓' if result['has_intent'] else '✗'}", end='')

        if result['error']:
            print(f" (Error: {result['error']})")
            samples_with_error += 1
        elif result['has_intent']:
            print()
            samples_with_intent += 1

            if result['intent_data']:
                print(f"     Type: {result['intent_data']['class_name']}")

                if 'lateral_intent' in result['intent_data']:
                    lat_val = result['intent_data']['lateral_intent']['value']
                    lat_name = intent_labels_map['lateral'].get(lat_val, f'unknown({lat_val})')
                    print(f"     Lateral: {lat_val} ({lat_name})")

                if 'longitudinal_intent' in result['intent_data']:
                    long_val = result['intent_data']['longitudinal_intent']['value']
                    long_name = intent_labels_map['longitudinal'].get(long_val, f'unknown({long_val})')
                    print(f"     Longitudinal: {long_val} ({long_name})")
        else:
            print()
            samples_without_intent += 1

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples checked: {len(cache_dirs)}")
    print(f"Samples WITH intent.gz: {samples_with_intent}")
    print(f"Samples WITHOUT intent.gz: {samples_without_intent}")
    print(f"Samples with errors: {samples_with_error}")
    print()

    if samples_with_intent > 0:
        percentage = (samples_with_intent / len(cache_dirs)) * 100
        print(f"✅ RESULT: {percentage:.1f}% of samples have intent labels!")
        print()

        if percentage == 100.0:
            print("   ✓ All checked samples have intent labels")
            print("   ✓ Cache is ready for intent-conditioned training")
        else:
            print(f"   ⚠️  Only {percentage:.1f}% of samples have intent labels")
            print("   ⚠️  Cache might be incomplete or partially generated")
            print()
            print("   Recommendation:")
            print("   1. Check if cache generation completed successfully")
            print("   2. Consider regenerating cache with:")
            print("      ./cache.sh")
    else:
        print("❌ RESULT: NO samples have intent labels!")
        print()
        print("   The cache was generated without intent support.")
        print()
        print("   To fix this:")
        print("   1. Ensure model.intent_enabled=true in cache.sh")
        print("   2. Run: ./cache.sh")


if __name__ == '__main__':
    main()
