#!/usr/bin/env python3
"""
Query detailed failure information by scenario token.

This script queries the failure case database and displays:
- Failure types (collision, speed violation, deadlock, drivable area)
- Frame numbers where failures occur
- Timestamps in microseconds
- Responsibility determination (for collisions)
- Detailed event information

Usage:
    python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"
    python scripts/query_failure_details.py --scenario-name "00cca24d240f5980" --show-frames
"""
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.database_manager import DatabaseManager
from src.evaluation.history_serializer import HistorySerializer


class FailureDetailsQuery:
    """Query detailed failure information from database."""

    def __init__(self, database_path: Path):
        self._database_path = Path(database_path)
        self._db_manager = DatabaseManager(self._database_path)
        self._serializer = HistorySerializer()

        if not self._database_path.exists():
            raise FileNotFoundError(f"Database not found: {self._database_path}")

        self._db_manager.initialize_database()

    def query_failure_details(
        self,
        scenario_name: str,
        load_history: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Query complete failure details for a scenario.

        Args:
            scenario_name: Scenario token to query
            load_history: If True, load and analyze SimulationHistory to get frame numbers

        Returns:
            Dict containing all failure details, or None if not found
        """
        # Get basic failure case info
        failure_case = self._db_manager.query_failure_case_by_scenario(scenario_name)
        if failure_case is None:
            return None

        failure_id = failure_case['id']

        # Query all failure details
        details = {
            'basic_info': failure_case,
            'collisions': self._query_collisions(failure_id),
            'speed_violations': self._query_speed_violations(failure_id),
            'deadlock': self._query_deadlock(failure_id),
            'drivable_area_violations': self._query_drivable_area_violations(failure_id),
        }

        # Load history to get frame numbers if requested
        if load_history:
            history = self._load_simulation_history(failure_id)
            if history:
                details['frame_mapping'] = self._compute_frame_mapping(history)
                details['traffic_light_info'] = self._extract_traffic_light_info(history)
            else:
                details['frame_mapping'] = None
                details['traffic_light_info'] = None
        else:
            details['frame_mapping'] = None
            details['traffic_light_info'] = None

        return details

    def _query_collisions(self, failure_case_id: int) -> List[Dict]:
        """Query collision details."""
        with self._db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM collision_details
                WHERE failure_case_id = ?
                ORDER BY timestamp_us
            """, (failure_case_id,))
            return [dict(row) for row in cursor.fetchall()]

    def _query_speed_violations(self, failure_case_id: int) -> List[Dict]:
        """Query speed violation details."""
        with self._db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM speed_violation_details
                WHERE failure_case_id = ?
                ORDER BY start_timestamp_us
            """, (failure_case_id,))
            return [dict(row) for row in cursor.fetchall()]

    def _query_deadlock(self, failure_case_id: int) -> Optional[Dict]:
        """Query deadlock details."""
        with self._db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM deadlock_details
                WHERE failure_case_id = ?
            """, (failure_case_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def _query_drivable_area_violations(self, failure_case_id: int) -> List[Dict]:
        """Query drivable area violation details."""
        with self._db_manager._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM drivable_area_violation_details
                WHERE failure_case_id = ?
                ORDER BY timestamp_us
            """, (failure_case_id,))
            return [dict(row) for row in cursor.fetchall()]

    def _load_simulation_history(self, failure_case_id: int):
        """Load simulation history from database."""
        try:
            history_blob = self._db_manager.get_simulation_history_blob(failure_case_id)
            if history_blob:
                return self._serializer.deserialize_from_bytes(history_blob)
        except Exception as e:
            print(f"Warning: Failed to load simulation history: {e}")
        return None

    def _compute_frame_mapping(self, history) -> Dict[int, int]:
        """
        Compute mapping from timestamp (us) to frame number.

        Args:
            history: SimulationHistory object

        Returns:
            Dict mapping timestamp_us -> frame_number
        """
        mapping = {}
        for idx, sample in enumerate(history.data):
            timestamp_us = sample.ego_state.time_point.time_us
            mapping[timestamp_us] = idx
        return mapping

    def _extract_traffic_light_info(self, history) -> Dict[int, List]:
        """
        Extract traffic light status for each frame.

        Args:
            history: SimulationHistory object

        Returns:
            Dict mapping timestamp_us -> list of traffic light status
        """
        traffic_light_info = {}
        try:
            for sample in history.data:
                timestamp_us = sample.ego_state.time_point.time_us
                # Access traffic_light_status from SimulationHistorySample
                if hasattr(sample, 'traffic_light_status') and sample.traffic_light_status:
                    # Convert to list of dicts for easier display
                    tl_list = []
                    for tl in sample.traffic_light_status:
                        tl_dict = {
                            'lane_connector_id': tl.lane_connector_id,
                            'status': tl.status.name if hasattr(tl.status, 'name') else str(tl.status)
                        }
                        tl_list.append(tl_dict)
                    traffic_light_info[timestamp_us] = tl_list
        except Exception as e:
            print(f"Warning: Failed to extract traffic light info: {e}")

        return traffic_light_info


def format_basic_info(details: Dict) -> None:
    """Format and print basic scenario information."""
    basic = details['basic_info']

    print(f"\n{'='*80}")
    print(f"Failure Case Details: {basic['scenario_name']}")
    print(f"{'='*80}\n")

    print("Basic Information:")
    print(f"  Scenario Name (Token): {basic['scenario_name']}")
    print(f"  Scenario Type:         {basic['scenario_type']}")
    print(f"  Log Name:              {basic['log_name']}")
    print(f"  Planner:               {basic['planner_name']}")
    print(f"  Duration:              {basic['duration_seconds']:.2f} seconds")
    print(f"  Severity:              {basic['failure_severity']}")

    print(f"\nFailure Types Summary:")
    print(f"  Collision:             {'Yes' if basic['has_collision'] else 'No'}")
    print(f"  Speed Violation:       {'Yes' if basic['has_speed_violation'] else 'No'}")
    print(f"  Deadlock:              {'Yes' if basic['has_deadlock'] else 'No'}")
    print(f"  Drivable Area:         {'Yes' if basic['has_drivable_area_violation'] else 'No'}")


def format_collisions(collisions: List[Dict], frame_mapping: Optional[Dict], traffic_light_info: Optional[Dict] = None) -> None:
    """Format and print collision details."""
    if not collisions:
        return

    print(f"\n{'-'*80}")
    print(f"Collision Events ({len(collisions)} total)")
    print(f"{'-'*80}\n")

    for idx, collision in enumerate(collisions, 1):
        timestamp = collision['timestamp_us']
        frame_num = frame_mapping.get(timestamp, '?') if frame_mapping else '?'

        print(f"Collision #{idx}:")
        print(f"  Timestamp:       {timestamp} us")
        print(f"  Frame Number:    {frame_num}")
        print(f"  Collision Type:  {collision['collision_type']}")
        print(f"  Object Type:     {collision['tracked_object_type']}")
        print(f"  At Fault:        {'YES ⚠️' if collision['is_at_fault'] else 'NO'}")
        print(f"  Collision Energy: {collision['collision_energy']:.2f}")
        print(f"  Ego Speed:       {collision['ego_speed']:.2f} m/s")
        print(f"  Object Speed:    {collision['object_speed']:.2f} m/s")

        # Display traffic light info if available
        if traffic_light_info and timestamp in traffic_light_info:
            tl_status_list = traffic_light_info[timestamp]
            if tl_status_list:
                print(f"  Traffic Lights:  {len(tl_status_list)} light(s) detected")
                for tl_idx, tl in enumerate(tl_status_list, 1):
                    # Format status with color indicators
                    status = tl['status']
                    if status == 'RED':
                        status_display = f"🔴 {status}"
                    elif status == 'GREEN':
                        status_display = f"🟢 {status}"
                    elif status == 'YELLOW':
                        status_display = f"🟡 {status}"
                    else:
                        status_display = f"⚪ {status}"

                    print(f"    Light {tl_idx}: {status_display} (Lane: {tl['lane_connector_id']})")
        print()


def format_speed_violations(violations: List[Dict], frame_mapping: Optional[Dict]) -> None:
    """Format and print speed violation details."""
    if not violations:
        return

    print(f"\n{'-'*80}")
    print(f"Speed Violations ({len(violations)} total)")
    print(f"{'-'*80}\n")

    for idx, violation in enumerate(violations, 1):
        start_ts = violation['start_timestamp_us']
        end_ts = violation['end_timestamp_us']

        if frame_mapping:
            start_frame = frame_mapping.get(start_ts, '?')
            end_frame = frame_mapping.get(end_ts, '?')
            frame_info = f"Frame {start_frame} to {end_frame}"
        else:
            frame_info = "Frame info not available"

        duration_s = violation['duration_us'] / 1e6

        print(f"Violation #{idx}:")
        print(f"  Start Time:      {start_ts} us")
        print(f"  End Time:        {end_ts} us")
        print(f"  {frame_info}")
        print(f"  Duration:        {duration_s:.2f} seconds")
        print(f"  Max Overspeed:   {violation['max_overspeed_mps']:.2f} m/s")
        print(f"  Mean Overspeed:  {violation['mean_overspeed_mps']:.2f} m/s")
        if violation['speed_limit_mps']:
            print(f"  Speed Limit:     {violation['speed_limit_mps']:.2f} m/s")
        print()


def format_deadlock(deadlock: Optional[Dict], frame_mapping: Optional[Dict]) -> None:
    """Format and print deadlock details."""
    if not deadlock:
        return

    print(f"\n{'-'*80}")
    print(f"Deadlock Event")
    print(f"{'-'*80}\n")

    start_ts = deadlock['start_timestamp_us']
    end_ts = deadlock['end_timestamp_us']

    if frame_mapping:
        start_frame = frame_mapping.get(start_ts, '?')
        end_frame = frame_mapping.get(end_ts, '?')
        frame_info = f"Frame {start_frame} to {end_frame}"
    else:
        frame_info = "Frame info not available"

    print(f"  Start Time:          {start_ts} us")
    print(f"  End Time:            {end_ts} us")
    print(f"  {frame_info}")
    print(f"  Duration:            {deadlock['duration_seconds']:.2f} seconds")
    print(f"  Total Progress:      {deadlock['total_progress_meters']:.2f} meters")
    print(f"  Expected Progress:   {deadlock['expected_progress_meters']:.2f} meters")
    print(f"  Progress Ratio:      {deadlock['progress_ratio']:.2%}")
    print(f"  Final Speed:         {deadlock['final_speed_mps']:.2f} m/s")
    print()


def format_drivable_area_violations(violations: List[Dict], frame_mapping: Optional[Dict]) -> None:
    """Format and print drivable area violation details."""
    if not violations:
        return

    print(f"\n{'-'*80}")
    print(f"Drivable Area Violations ({len(violations)} total)")
    print(f"{'-'*80}\n")

    for idx, violation in enumerate(violations, 1):
        timestamp = violation['timestamp_us']
        frame_num = frame_mapping.get(timestamp, '?') if frame_mapping else '?'
        duration_s = violation['violation_duration_us'] / 1e6

        print(f"Violation #{idx}:")
        print(f"  Timestamp:       {timestamp} us")
        print(f"  Frame Number:    {frame_num}")
        print(f"  Max Distance:    {violation['max_distance_to_drivable_area']:.2f} meters")
        print(f"  Duration:        {duration_s:.2f} seconds")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Query detailed failure information by scenario token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--scenario-name',
        type=str,
        required=True,
        help='Scenario name/token to query (e.g., "00cca24d240f5980")'
    )

    parser.add_argument(
        '--database-path',
        type=Path,
        default=Path('work_dirs/exp/failure_cases.db'),
        help='Path to failure cases database (default: work_dirs/exp/failure_cases.db)'
    )

    parser.add_argument(
        '--show-frames',
        action='store_true',
        help='Load simulation history to show frame numbers (slower but more informative)'
    )

    args = parser.parse_args()

    # Check database exists
    if not args.database_path.exists():
        print(f"\n✗ Error: Database not found: {args.database_path}")
        print("\nMake sure you have run simulation with failure_case_collector callback enabled.")
        sys.exit(1)

    # Create query object
    try:
        query = FailureDetailsQuery(args.database_path)
    except Exception as e:
        print(f"\n✗ Error initializing query: {e}")
        sys.exit(1)

    # Query failure details
    print(f"\nQuerying failure details for: {args.scenario_name}")
    if args.show_frames:
        print("(Loading simulation history to compute frame numbers...)")

    try:
        details = query.query_failure_details(
            args.scenario_name,
            load_history=args.show_frames
        )
    except Exception as e:
        print(f"\n✗ Error querying database: {e}")
        sys.exit(1)

    if details is None:
        print(f"\n✗ Scenario '{args.scenario_name}' not found in database.")
        print("\nTip: Use 'python scripts/export_failure_cases.py --list' to see all available scenarios")
        sys.exit(1)

    # Format and display results
    format_basic_info(details)

    frame_mapping = details.get('frame_mapping')
    traffic_light_info = details.get('traffic_light_info')

    format_collisions(details['collisions'], frame_mapping, traffic_light_info)
    format_speed_violations(details['speed_violations'], frame_mapping)
    format_deadlock(details['deadlock'], frame_mapping)
    format_drivable_area_violations(details['drivable_area_violations'], frame_mapping)

    # Footer
    print(f"{'='*80}\n")

    if not args.show_frames and (details['collisions'] or details['speed_violations'] or
                                  details['deadlock'] or details['drivable_area_violations']):
        print("Tip: Use --show-frames to see frame numbers and traffic light status (requires loading simulation history)")
        print()


if __name__ == '__main__':
    main()
