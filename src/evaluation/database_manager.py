"""
SQLite database manager for failure case storage.
"""
import sqlite3
import logging
import time
from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.failure_detectors.detection_engine import FailureDetectionResult
    from src.evaluation.history_serializer import SerializedHistory
    from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database operations for failure case storage.

    Thread-safe with context managers for concurrent access.
    """

    # SQL schema definition
    SCHEMA_SQL = """
    -- Failure cases main table
    CREATE TABLE IF NOT EXISTS failure_cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scenario_type TEXT NOT NULL,
        scenario_name TEXT NOT NULL,
        log_name TEXT NOT NULL,
        planner_name TEXT NOT NULL,
        simulation_timestamp INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        failure_severity TEXT NOT NULL,
        has_collision INTEGER DEFAULT 0,
        has_speed_violation INTEGER DEFAULT 0,
        has_deadlock INTEGER DEFAULT 0,
        has_drivable_area_violation INTEGER DEFAULT 0,
        UNIQUE(scenario_name, planner_name, simulation_timestamp)
    );

    CREATE INDEX IF NOT EXISTS idx_failure_type
        ON failure_cases(scenario_type, planner_name);
    CREATE INDEX IF NOT EXISTS idx_severity
        ON failure_cases(failure_severity);
    CREATE INDEX IF NOT EXISTS idx_collision
        ON failure_cases(has_collision);

    -- Collision details table
    CREATE TABLE IF NOT EXISTS collision_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        failure_case_id INTEGER NOT NULL,
        timestamp_us INTEGER NOT NULL,
        collision_type TEXT NOT NULL,
        tracked_object_type TEXT NOT NULL,
        collision_energy REAL NOT NULL,
        is_at_fault INTEGER NOT NULL,
        ego_speed REAL,
        object_speed REAL,
        FOREIGN KEY(failure_case_id) REFERENCES failure_cases(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_collision_timestamp
        ON collision_details(failure_case_id, timestamp_us);

    -- Speed violation details table
    CREATE TABLE IF NOT EXISTS speed_violation_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        failure_case_id INTEGER NOT NULL,
        start_timestamp_us INTEGER NOT NULL,
        end_timestamp_us INTEGER NOT NULL,
        duration_us INTEGER NOT NULL,
        max_overspeed_mps REAL NOT NULL,
        mean_overspeed_mps REAL NOT NULL,
        speed_limit_mps REAL,
        FOREIGN KEY(failure_case_id) REFERENCES failure_cases(id) ON DELETE CASCADE
    );

    -- Deadlock details table
    CREATE TABLE IF NOT EXISTS deadlock_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        failure_case_id INTEGER NOT NULL,
        start_timestamp_us INTEGER NOT NULL,
        end_timestamp_us INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        total_progress_meters REAL NOT NULL,
        expected_progress_meters REAL NOT NULL,
        progress_ratio REAL NOT NULL,
        final_speed_mps REAL,
        FOREIGN KEY(failure_case_id) REFERENCES failure_cases(id) ON DELETE CASCADE
    );

    -- Drivable area violation details table
    CREATE TABLE IF NOT EXISTS drivable_area_violation_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        failure_case_id INTEGER NOT NULL,
        timestamp_us INTEGER NOT NULL,
        max_distance_to_drivable_area REAL NOT NULL,
        violation_duration_us INTEGER NOT NULL,
        FOREIGN KEY(failure_case_id) REFERENCES failure_cases(id) ON DELETE CASCADE
    );

    -- Simulation histories storage table
    CREATE TABLE IF NOT EXISTS simulation_histories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        failure_case_id INTEGER NOT NULL UNIQUE,
        history_blob BLOB NOT NULL,
        serialization_format TEXT DEFAULT 'pickle_lzma',
        blob_size_bytes INTEGER NOT NULL,
        num_samples INTEGER NOT NULL,
        FOREIGN KEY(failure_case_id) REFERENCES failure_cases(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_history_size
        ON simulation_histories(blob_size_bytes);
    """

    def __init__(self, database_path: Path):
        self._database_path = Path(database_path)
        self._initialized = False

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(
            str(self._database_path),
            timeout=30.0,  # Wait up to 30s for lock
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_database(self):
        """Create database schema if not exists."""
        if self._initialized:
            return

        # Create parent directory
        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(self.SCHEMA_SQL)

        self._initialized = True
        logger.info(f"Database initialized: {self._database_path}")

    def save_failure_case(
        self,
        detection_result: "FailureDetectionResult",
        serialized_history: "SerializedHistory",
        scenario: "AbstractScenario",
        planner_name: str,
    ):
        """
        Save complete failure case to database.

        Inserts into multiple tables atomically.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert main failure case
            cursor.execute("""
                INSERT INTO failure_cases (
                    scenario_type, scenario_name, log_name, planner_name,
                    simulation_timestamp, duration_seconds, failure_severity,
                    has_collision, has_speed_violation, has_deadlock,
                    has_drivable_area_violation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                detection_result.scenario_type,
                detection_result.scenario_name,
                detection_result.log_name,
                planner_name,
                int(time.time() * 1e6),  # Current timestamp in microseconds
                detection_result.duration_seconds,
                detection_result.get_failure_severity(),
                len(detection_result.collision_events) > 0,
                len(detection_result.speed_violations) > 0,
                detection_result.deadlock_event is not None,
                len(detection_result.drivable_area_violations) > 0,
            ))

            failure_case_id = cursor.lastrowid

            # Insert collision details
            for collision in detection_result.collision_events:
                cursor.execute("""
                    INSERT INTO collision_details (
                        failure_case_id, timestamp_us, collision_type,
                        tracked_object_type, collision_energy, is_at_fault,
                        ego_speed, object_speed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    failure_case_id,
                    collision.timestamp_us,
                    collision.collision_type,
                    collision.tracked_object_type,
                    collision.collision_energy,
                    collision.is_at_fault,
                    collision.ego_speed,
                    collision.object_speed,
                ))

            # Insert speed violations
            for violation in detection_result.speed_violations:
                cursor.execute("""
                    INSERT INTO speed_violation_details (
                        failure_case_id, start_timestamp_us, end_timestamp_us,
                        duration_us, max_overspeed_mps, mean_overspeed_mps,
                        speed_limit_mps
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    failure_case_id,
                    violation.start_timestamp_us,
                    violation.end_timestamp_us,
                    violation.duration_us,
                    violation.max_overspeed_mps,
                    violation.mean_overspeed_mps,
                    violation.speed_limit_mps,
                ))

            # Insert deadlock if exists
            if detection_result.deadlock_event:
                deadlock = detection_result.deadlock_event
                cursor.execute("""
                    INSERT INTO deadlock_details (
                        failure_case_id, start_timestamp_us, end_timestamp_us,
                        duration_seconds, total_progress_meters,
                        expected_progress_meters, progress_ratio, final_speed_mps
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    failure_case_id,
                    deadlock.start_timestamp_us,
                    deadlock.end_timestamp_us,
                    deadlock.duration_seconds,
                    deadlock.total_progress_meters,
                    deadlock.expected_progress_meters,
                    deadlock.progress_ratio,
                    deadlock.final_speed_mps,
                ))

            # Insert drivable area violations
            for violation in detection_result.drivable_area_violations:
                cursor.execute("""
                    INSERT INTO drivable_area_violation_details (
                        failure_case_id, timestamp_us, max_distance_to_drivable_area,
                        violation_duration_us
                    ) VALUES (?, ?, ?, ?)
                """, (
                    failure_case_id,
                    violation.timestamp_us,
                    violation.max_distance_to_drivable_area,
                    violation.violation_duration_us,
                ))

            # Insert simulation history blob
            cursor.execute("""
                INSERT INTO simulation_histories (
                    failure_case_id, history_blob, serialization_format,
                    blob_size_bytes, num_samples
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                failure_case_id,
                serialized_history.data,
                serialized_history.format,
                serialized_history.size_bytes,
                serialized_history.num_samples,
            ))

        logger.info(f"Saved failure case {failure_case_id} to database")

    def query_failure_case_by_scenario(self, scenario_name: str):
        """
        Query failure case by scenario_name.

        Args:
            scenario_name: Scenario name to query

        Returns:
            Dict with failure case data, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM failure_cases
                WHERE scenario_name = ?
                LIMIT 1
            """, (scenario_name,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_simulation_history_blob(self, failure_case_id: int):
        """
        Get simulation history BLOB for a failure case.

        Args:
            failure_case_id: Failure case ID

        Returns:
            Compressed history BLOB bytes, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT history_blob FROM simulation_histories
                WHERE failure_case_id = ?
            """, (failure_case_id,))
            row = cursor.fetchone()
            if row:
                return row['history_blob']
            return None

    def list_all_failure_cases(self):
        """
        List all failure cases with summary information.

        Returns:
            List of dicts with failure case summary data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    id, scenario_name, scenario_type, log_name, planner_name,
                    failure_severity, duration_seconds,
                    has_collision, has_speed_violation, has_deadlock,
                    has_drivable_area_violation
                FROM failure_cases
                ORDER BY simulation_timestamp DESC
            """)
            return [dict(row) for row in cursor.fetchall()]
