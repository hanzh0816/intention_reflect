"""
Intent Classification Utilities

This module provides functions to classify ego vehicle intent from trajectory data.
Extracted from visualize_short_term_intent.py for reuse in training pipeline.

Intent Categories:
- Lateral: turn_left, turn_right, shift_left, shift_right, stay_in_lane
- Longitudinal: accelerate, maintain_speed, decelerate, stop

Author: Adapted for PlanTF training pipeline
"""

from typing import List, Tuple
import numpy as np
from nuplan.common.actor_state.ego_state import EgoState


# ==================== Intent Category Mappings ====================

LATERAL_INTENT_CLASSES = {
    "turn_left": 0,
    "turn_right": 1,
    "shift_left": 2,
    "shift_right": 3,
    "stay_in_lane": 4,
}

LONGITUDINAL_INTENT_CLASSES = {
    "accelerate": 0,
    "maintain_speed": 1,
    "decelerate": 2,
    "stop": 3,
}


# ==================== Configuration Parameters ====================

class IntentClassificationConfig:
    """Configuration for intent classification thresholds."""

    def __init__(
        self,
        # Time parameters
        sample_interval: float = 0.1,  # seconds

        # Lateral thresholds
        turn_angle_threshold: float = 15.0,  # degrees
        shift_angle_threshold: float = 5.0,   # degrees
        turn_curvature_threshold: float = 0.05,  # 1/meters
        lateral_shift_threshold: float = 1.5,  # meters

        # Longitudinal thresholds
        accel_threshold: float = 1.0,  # m/s²
        decel_threshold: float = -1.0,  # m/s²
        velocity_change_ratio: float = 0.20,  # 20%
        stop_velocity_threshold: float = 0.5,  # m/s
    ):
        self.sample_interval = sample_interval

        # Lateral
        self.turn_angle_threshold = turn_angle_threshold
        self.shift_angle_threshold = shift_angle_threshold
        self.turn_curvature_threshold = turn_curvature_threshold
        self.lateral_shift_threshold = lateral_shift_threshold

        # Longitudinal
        self.accel_threshold = accel_threshold
        self.decel_threshold = decel_threshold
        self.velocity_change_ratio = velocity_change_ratio
        self.stop_velocity_threshold = stop_velocity_threshold


# Default configuration
DEFAULT_CONFIG = IntentClassificationConfig()


# ==================== Feature Computation Functions ====================

def compute_trajectory_curvature(positions: np.ndarray) -> float:
    """
    Compute average curvature of a trajectory.

    Curvature at point i is computed using three consecutive points.
    Higher curvature indicates sharper turns.

    Args:
        positions: Array of shape (N, 2) with (x, y) positions

    Returns:
        Average absolute curvature (1/meters)
    """
    if len(positions) < 3:
        return 0.0

    curvatures = []

    for i in range(1, len(positions) - 1):
        p1 = positions[i - 1]
        p2 = positions[i]
        p3 = positions[i + 1]

        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2

        # Lengths
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)

        if l1 < 1e-6 or l2 < 1e-6:
            continue

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (l1 * l2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # Curvature approximation: angle / average_length
        avg_length = (l1 + l2) / 2.0
        curvature = angle / avg_length if avg_length > 0 else 0.0

        curvatures.append(abs(curvature))

    return np.mean(curvatures) if curvatures else 0.0


def compute_heading_change(ego_states: List[EgoState], start_idx: int, end_idx: int) -> float:
    """
    Compute heading angle change between two states.

    Args:
        ego_states: List of ego states
        start_idx: Starting index
        end_idx: Ending index

    Returns:
        Heading change in radians (normalized to [-π, π])
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_heading = ego_states[start_idx].rear_axle.heading
    end_heading = ego_states[end_idx].rear_axle.heading

    # Normalize to [-π, π]
    delta = end_heading - start_heading
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi

    return delta


def compute_lateral_displacement(positions: np.ndarray) -> float:
    """
    Compute lateral displacement relative to initial direction.

    Positive = left displacement, Negative = right displacement

    Args:
        positions: Array of shape (N, 2) with (x, y) positions

    Returns:
        Lateral displacement in meters
    """
    if len(positions) < 2:
        return 0.0

    # Initial direction vector
    initial_direction = positions[-1] - positions[0]
    initial_length = np.linalg.norm(initial_direction)

    if initial_length < 1e-6:
        return 0.0

    initial_direction = initial_direction / initial_length

    # Perpendicular direction (left is positive)
    perpendicular = np.array([-initial_direction[1], initial_direction[0]])

    # Vector from start to end
    displacement_vector = positions[-1] - positions[0]

    # Lateral component
    lateral_disp = np.dot(displacement_vector, perpendicular)

    return lateral_disp


def compute_acceleration(
    ego_states: List[EgoState],
    start_idx: int,
    end_idx: int,
    sample_interval: float = 0.1
) -> float:
    """
    Compute average acceleration.

    Args:
        ego_states: List of ego states
        start_idx: Starting index
        end_idx: Ending index
        sample_interval: Time interval between samples (seconds)

    Returns:
        Acceleration in m/s²
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_state = ego_states[start_idx]
    end_state = ego_states[end_idx]

    start_vel = np.linalg.norm([
        start_state.dynamic_car_state.rear_axle_velocity_2d.x,
        start_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    time_diff = (end_idx - start_idx) * sample_interval

    if time_diff < 1e-6:
        return 0.0

    acceleration = (end_vel - start_vel) / time_diff

    return acceleration


def compute_velocity_change_ratio(ego_states: List[EgoState], start_idx: int, end_idx: int) -> float:
    """
    Compute velocity change ratio.

    Args:
        ego_states: List of ego states
        start_idx: Starting index
        end_idx: Ending index

    Returns:
        Ratio of velocity change (positive = acceleration, negative = deceleration)
    """
    if start_idx >= len(ego_states) or end_idx >= len(ego_states):
        return 0.0

    start_state = ego_states[start_idx]
    end_state = ego_states[end_idx]

    start_vel = np.linalg.norm([
        start_state.dynamic_car_state.rear_axle_velocity_2d.x,
        start_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    if start_vel < 0.1:  # Avoid division by very small numbers
        return 0.0

    ratio = (end_vel - start_vel) / start_vel

    return ratio


# ==================== Intent Classification Functions ====================

def classify_lateral_intent(
    ego_states: List[EgoState],
    positions: np.ndarray,
    start_idx: int,
    end_idx: int,
    config: IntentClassificationConfig = DEFAULT_CONFIG
) -> str:
    """
    Classify lateral (horizontal) intent based on trajectory geometry.

    Intent categories:
    - turn_left: Large left turn (heading change > threshold and high curvature)
    - turn_right: Large right turn (heading change < -threshold and high curvature)
    - shift_left: Slight left maneuver
    - shift_right: Slight right maneuver
    - stay_in_lane: Minimal lateral movement

    Args:
        ego_states: List of ego states
        positions: Array of (x, y) positions for the window
        start_idx: Starting index in ego_states
        end_idx: Ending index in ego_states
        config: Configuration object with thresholds

    Returns:
        Lateral intent string
    """
    # Compute features
    heading_change = compute_heading_change(ego_states, start_idx, end_idx)
    heading_change_deg = np.degrees(heading_change)

    curvature = compute_trajectory_curvature(positions)
    lateral_disp = compute_lateral_displacement(positions)

    # Classification logic
    # Turn detection: significant heading change AND high curvature
    if heading_change_deg > config.turn_angle_threshold and curvature > config.turn_curvature_threshold:
        return "turn_left"

    if heading_change_deg < -config.turn_angle_threshold and curvature > config.turn_curvature_threshold:
        return "turn_right"

    # Shift detection: moderate heading change OR low curvature with displacement
    if config.shift_angle_threshold < heading_change_deg <= config.turn_angle_threshold:
        return "shift_left"

    if -config.turn_angle_threshold <= heading_change_deg < -config.shift_angle_threshold:
        return "shift_right"

    # Check lateral displacement for low-curvature shifts
    if curvature <= config.turn_curvature_threshold:
        if lateral_disp > config.lateral_shift_threshold:
            return "shift_left"
        if lateral_disp < -config.lateral_shift_threshold:
            return "shift_right"

    # Default: staying in lane
    return "stay_in_lane"


def classify_longitudinal_intent(
    ego_states: List[EgoState],
    start_idx: int,
    end_idx: int,
    config: IntentClassificationConfig = DEFAULT_CONFIG
) -> str:
    """
    Classify longitudinal (speed) intent based on velocity and acceleration.

    Intent categories:
    - stop: Vehicle comes to a stop (end velocity < threshold)
    - accelerate: Significant acceleration
    - decelerate: Significant deceleration
    - maintain_speed: Minimal speed change

    Args:
        ego_states: List of ego states
        start_idx: Starting index
        end_idx: Ending index
        config: Configuration object with thresholds

    Returns:
        Longitudinal intent string
    """
    if end_idx >= len(ego_states):
        return "maintain_speed"

    # Check for stop
    end_state = ego_states[end_idx]
    end_vel = np.linalg.norm([
        end_state.dynamic_car_state.rear_axle_velocity_2d.x,
        end_state.dynamic_car_state.rear_axle_velocity_2d.y
    ])

    if end_vel < config.stop_velocity_threshold:
        return "stop"

    # Compute acceleration and velocity change ratio
    acceleration = compute_acceleration(ego_states, start_idx, end_idx, config.sample_interval)
    vel_change_ratio = compute_velocity_change_ratio(ego_states, start_idx, end_idx)

    # Accelerate: high positive acceleration OR significant speed increase
    if acceleration > config.accel_threshold or vel_change_ratio > config.velocity_change_ratio:
        return "accelerate"

    # Decelerate: high negative acceleration OR significant speed decrease
    if acceleration < config.decel_threshold or vel_change_ratio < -config.velocity_change_ratio:
        return "decelerate"

    # Default: maintaining speed
    return "maintain_speed"


def classify_intent_from_states(
    ego_states: List[EgoState],
    current_idx: int,
    time_horizon: float = 2.0,
    config: IntentClassificationConfig = DEFAULT_CONFIG
) -> Tuple[str, str]:
    """
    Classify intent in both lateral and longitudinal dimensions.

    Args:
        ego_states: List of all ego states
        current_idx: Index of the current state
        time_horizon: Time window for intent classification (seconds)
        config: Configuration object with thresholds

    Returns:
        Tuple of (lateral_intent, longitudinal_intent)
    """
    # Define window
    window_size = int(time_horizon / config.sample_interval)
    end_idx = min(current_idx + window_size, len(ego_states) - 1)

    # Extract positions for this window
    positions = []
    for i in range(current_idx, end_idx + 1):
        if i < len(ego_states):
            state = ego_states[i]
            positions.append([state.rear_axle.x, state.rear_axle.y])

    positions = np.array(positions)

    # Classify lateral and longitudinal intents independently
    lateral_intent = classify_lateral_intent(ego_states, positions, current_idx, end_idx, config)
    longitudinal_intent = classify_longitudinal_intent(ego_states, current_idx, end_idx, config)

    return lateral_intent, longitudinal_intent


def get_lateral_intent_index(lateral_intent: str) -> int:
    """Convert lateral intent string to class index."""
    return LATERAL_INTENT_CLASSES.get(lateral_intent, LATERAL_INTENT_CLASSES["stay_in_lane"])


def get_longitudinal_intent_index(longitudinal_intent: str) -> int:
    """Convert longitudinal intent string to class index."""
    return LONGITUDINAL_INTENT_CLASSES.get(longitudinal_intent, LONGITUDINAL_INTENT_CLASSES["maintain_speed"])


# ==================== Trajectory Tensor Classification ====================

def classify_intent_from_trajectory(
    trajectory: np.ndarray,
    dt: float = 0.1,
    time_horizon: float = 2.0,
    config: IntentClassificationConfig = None
) -> Tuple[int, int]:
    """
    Classify intent from a predicted trajectory tensor.

    This function is designed to work with model predictions during training/inference.
    It converts trajectory tensors into the format expected by the classification logic.

    Args:
        trajectory: Trajectory array of shape [T, 4] or [T, 2]
                   If shape is [T, 4]: contains (x, y, cos(heading), sin(heading))
                   If shape is [T, 2]: contains (x, y) only
        dt: Time interval between trajectory steps (seconds)
        time_horizon: Time window for intent classification (seconds)
        config: Configuration object with thresholds. If None, uses DEFAULT_CONFIG

    Returns:
        Tuple of (lateral_intent_idx, longitudinal_intent_idx)
    """
    if config is None:
        config = IntentClassificationConfig(sample_interval=dt)

    # Extract positions
    if trajectory.shape[-1] >= 2:
        positions = trajectory[:, :2]  # [T, 2]
    else:
        raise ValueError(f"Trajectory must have at least 2 channels (x, y), got shape {trajectory.shape}")

    # Determine time window
    window_size = int(time_horizon / dt)
    end_idx = min(window_size, len(positions) - 1)
    positions_window = positions[:end_idx + 1]

    # === Lateral Intent Classification ===

    # Compute heading change
    if trajectory.shape[-1] >= 4:
        # Extract heading from cos/sin representation
        cos_heading = trajectory[:, 2]
        sin_heading = trajectory[:, 3]
        headings = np.arctan2(sin_heading, cos_heading)

        start_heading = headings[0]
        end_heading = headings[end_idx]

        # Normalize to [-π, π]
        heading_change = end_heading - start_heading
        while heading_change > np.pi:
            heading_change -= 2 * np.pi
        while heading_change < -np.pi:
            heading_change += 2 * np.pi
    else:
        # Estimate heading from position changes
        heading_change = 0.0
        if len(positions_window) >= 2:
            start_direction = positions_window[1] - positions_window[0]
            end_direction = positions_window[-1] - positions_window[-2]

            start_heading = np.arctan2(start_direction[1], start_direction[0])
            end_heading = np.arctan2(end_direction[1], end_direction[0])

            heading_change = end_heading - start_heading
            while heading_change > np.pi:
                heading_change -= 2 * np.pi
            while heading_change < -np.pi:
                heading_change += 2 * np.pi

    heading_change_deg = np.degrees(heading_change)

    # Compute curvature and lateral displacement
    curvature = compute_trajectory_curvature(positions_window)
    lateral_disp = compute_lateral_displacement(positions_window)

    # Classify lateral intent
    if heading_change_deg > config.turn_angle_threshold and curvature > config.turn_curvature_threshold:
        lateral_intent = "turn_left"
    elif heading_change_deg < -config.turn_angle_threshold and curvature > config.turn_curvature_threshold:
        lateral_intent = "turn_right"
    elif config.shift_angle_threshold < heading_change_deg <= config.turn_angle_threshold:
        lateral_intent = "shift_left"
    elif -config.turn_angle_threshold <= heading_change_deg < -config.shift_angle_threshold:
        lateral_intent = "shift_right"
    elif curvature <= config.turn_curvature_threshold:
        if lateral_disp > config.lateral_shift_threshold:
            lateral_intent = "shift_left"
        elif lateral_disp < -config.lateral_shift_threshold:
            lateral_intent = "shift_right"
        else:
            lateral_intent = "stay_in_lane"
    else:
        lateral_intent = "stay_in_lane"

    # === Longitudinal Intent Classification ===

    # Compute velocities from positions
    velocities = []
    for i in range(len(positions_window) - 1):
        displacement = positions_window[i + 1] - positions_window[i]
        velocity = np.linalg.norm(displacement) / dt
        velocities.append(velocity)

    if len(velocities) == 0:
        longitudinal_intent = "maintain_speed"
    else:
        start_vel = velocities[0] if len(velocities) > 0 else 0.0
        end_vel = velocities[-1] if len(velocities) > 0 else 0.0

        # Check for stop
        if end_vel < config.stop_velocity_threshold:
            longitudinal_intent = "stop"
        else:
            # Compute acceleration
            time_diff = len(velocities) * dt
            if time_diff > 1e-6:
                acceleration = (end_vel - start_vel) / time_diff
            else:
                acceleration = 0.0

            # Compute velocity change ratio
            if start_vel > 0.1:
                vel_change_ratio = (end_vel - start_vel) / start_vel
            else:
                vel_change_ratio = 0.0

            # Classify
            if acceleration > config.accel_threshold or vel_change_ratio > config.velocity_change_ratio:
                longitudinal_intent = "accelerate"
            elif acceleration < config.decel_threshold or vel_change_ratio < -config.velocity_change_ratio:
                longitudinal_intent = "decelerate"
            else:
                longitudinal_intent = "maintain_speed"

    # Convert to indices
    lateral_idx = get_lateral_intent_index(lateral_intent)
    longitudinal_idx = get_longitudinal_intent_index(longitudinal_intent)

    return lateral_idx, longitudinal_idx
