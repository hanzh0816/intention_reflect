"""
Visualization utilities for ego-centric scene rendering.

Adapted from Intent_label branch visualization module for video generation.
"""

import logging
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.transforms import Affine2D

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

logger = logging.getLogger(__name__)

# 为不同agent分配的颜色列表（使用区分度高的颜色）
AGENT_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#FFA07A',  # 浅橙色
    '#98D8C8',  # 薄荷绿
    '#F7DC6F',  # 黄色
    '#BB8FCE',  # 紫色
    '#85C1E2',  # 天蓝色
    '#F8B88B',  # 桃色
    '#52B788',  # 绿色
    '#E76F51',  # 橙红色
    '#2A9D8F',  # 深青色
    '#E9C46A',  # 金黄色
    '#F4A261',  # 橙色
    '#264653',  # 深蓝灰
]

# 为不同类型的交通参与者分配颜色
OBJECT_TYPE_COLORS = {
    TrackedObjectType.VEHICLE: '#4169E1',  # 蓝色 - 车辆
    TrackedObjectType.PEDESTRIAN: '#FF1493',  # 深粉色 - 行人
    TrackedObjectType.BICYCLE: '#32CD32',  # 绿色 - 自行车
    TrackedObjectType.GENERIC_OBJECT: '#FFD700',  # 金色 - 通用物体
    TrackedObjectType.EGO: '#00FF00',  # 绿色 - 自车
    TrackedObjectType.TRAFFIC_CONE: '#FF8C00',  # 橙色 - 交通锥
    TrackedObjectType.BARRIER: '#8B4513',  # 棕色 - 障碍物
    TrackedObjectType.CZONE_SIGN: '#FF0000',  # 红色 - 施工区标志
}

# 类型名称映射（用于图例）
OBJECT_TYPE_LABELS = {
    TrackedObjectType.VEHICLE: 'Other Vehicles',
    TrackedObjectType.PEDESTRIAN: 'Pedestrians',
    TrackedObjectType.BICYCLE: 'Bicycles',
    TrackedObjectType.GENERIC_OBJECT: 'Generic Objects',
    TrackedObjectType.TRAFFIC_CONE: 'Traffic Cones',
    TrackedObjectType.BARRIER: 'Barriers',
    TrackedObjectType.CZONE_SIGN: 'Construction Signs',
}


def transform_to_ego_frame(
    world_x: float,
    world_y: float,
    ego_state: EgoState
) -> Tuple[float, float]:
    """
    Transform world coordinates to ego-centric frame.

    In ego frame:
    - Ego is at (0, 0)
    - Ego heading is 0 (pointing up)

    Args:
        world_x: X coordinate in world frame
        world_y: Y coordinate in world frame
        ego_state: Ego vehicle state

    Returns:
        (x, y) in ego-centric frame
    """
    # Get ego position and heading
    ego_x = ego_state.rear_axle.x
    ego_y = ego_state.rear_axle.y
    ego_heading = ego_state.rear_axle.heading

    # Translate to ego position
    dx = world_x - ego_x
    dy = world_y - ego_y

    # Rotate to ego heading
    cos_h = np.cos(-ego_heading)
    sin_h = np.sin(-ego_heading)

    ego_frame_x = dx * cos_h - dy * sin_h
    ego_frame_y = dx * sin_h + dy * cos_h

    return ego_frame_x, ego_frame_y


def plot_map_context_ego_centric(
    ax: plt.Axes,
    map_api: AbstractMap,
    ego_state: EgoState,
    map_radius: float = 80.0
) -> None:
    """
    Render map context (lanes, boundaries, centerlines) in ego-centric frame.

    Args:
        ax: Matplotlib axes object
        map_api: Map API
        ego_state: Ego vehicle state (for coordinate transformation)
        map_radius: Map display radius in meters
    """
    try:
        # Get proximal lanes in world frame
        proximal_lanes = map_api.get_proximal_map_objects(
            ego_state.center,
            map_radius,
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        )

        lanes = (
            proximal_lanes[SemanticMapLayer.LANE] +
            proximal_lanes[SemanticMapLayer.LANE_CONNECTOR]
        )

        # Alternate lane colors
        lane_colors = ['#E8F4F8', '#D0E8F0']

        for idx, lane in enumerate(lanes):
            try:
                # Get lane boundaries in world frame
                left_boundary_points = lane.left_boundary.discrete_path
                right_boundary_points = lane.right_boundary.discrete_path

                # Transform to ego frame
                left_coords_ego = [
                    transform_to_ego_frame(p.x, p.y, ego_state)
                    for p in left_boundary_points
                ]
                right_coords_ego = [
                    transform_to_ego_frame(p.x, p.y, ego_state)
                    for p in reversed(right_boundary_points)
                ]

                if len(left_coords_ego) > 0 and len(right_coords_ego) > 0:
                    # Fill lane polygon
                    lane_polygon = mpatches.Polygon(
                        left_coords_ego + right_coords_ego,
                        closed=True,
                        facecolor=lane_colors[idx % len(lane_colors)],
                        edgecolor='none',
                        alpha=0.4,
                        zorder=1
                    )
                    ax.add_patch(lane_polygon)

                    # Draw lane boundaries
                    lx, ly = zip(*left_coords_ego)
                    ax.plot(lx, ly, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    rx, ry = zip(*right_coords_ego[::-1])  # Reverse back for boundary
                    ax.plot(rx, ry, color='#4A4A4A', linewidth=1.5, alpha=0.7, zorder=2)

                    # Draw centerline (dashed)
                    centerline_points = lane.baseline_path.discrete_path
                    center_coords_ego = [
                        transform_to_ego_frame(p.x, p.y, ego_state)
                        for p in centerline_points
                    ]
                    if len(center_coords_ego) > 0:
                        cx, cy = zip(*center_coords_ego)
                        ax.plot(cx, cy, color='#808080', linewidth=1.0, alpha=0.5,
                               linestyle='--', zorder=2)

            except Exception as e:
                logger.debug(f"Failed to render lane: {e}")
                continue

    except Exception as e:
        logger.warning(f"Map rendering failed: {e}")


def get_agent_color(agent_id: int) -> str:
    """
    根据agent ID获取对应的颜色

    Args:
        agent_id: Agent的序号ID

    Returns:
        颜色的十六进制字符串
    """
    return AGENT_COLORS[(agent_id - 1) % len(AGENT_COLORS)]


def plot_vehicle_box_ego_centric(
    ax: plt.Axes,
    world_x: float,
    world_y: float,
    world_heading: float,
    length: float,
    width: float,
    ego_state: EgoState,
    color: str,
    label: Optional[str] = None,
    alpha: float = 0.8,
    linewidth: float = 2.0
) -> None:
    """
    Draw vehicle oriented box in ego-centric frame.

    Args:
        ax: Matplotlib axes object
        world_x, world_y: Vehicle center position in world frame
        world_heading: Vehicle heading in world frame (radians)
        length, width: Vehicle dimensions (meters)
        ego_state: Ego vehicle state (for coordinate transformation)
        color: Fill color
        label: Legend label
        alpha: Transparency
        linewidth: Border line width
    """
    # Transform vehicle position to ego frame
    ego_x, ego_y = transform_to_ego_frame(world_x, world_y, ego_state)

    # Transform heading to ego frame
    ego_heading = world_heading - ego_state.rear_axle.heading

    # Create rectangle (centered at origin)
    rect = mpatches.Rectangle(
        (-length / 2, -width / 2),
        length,
        width,
        facecolor=color,
        edgecolor='black',
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=8
    )

    # Apply rotation and translation
    transform = Affine2D().rotate(ego_heading).translate(ego_x, ego_y) + ax.transData
    rect.set_transform(transform)

    ax.add_patch(rect)


def add_text_with_outline(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    fontsize: int = 10,
    color: str = 'white',
    outline_color: str = 'black',
    outline_width: float = 3.0
) -> None:
    """
    Add text with outline for better visibility.

    Args:
        ax: Matplotlib axes object
        x, y: Text position
        text: Text content
        fontsize: Font size
        color: Text color
        outline_color: Outline color
        outline_width: Outline width
    """
    # Draw text with outline effect (centered both horizontally and vertically)
    txt = ax.text(
        x, y, text,
        fontsize=fontsize,
        color=color,
        ha='center',
        va='center',  # Changed from 'bottom' to 'center' for better centering
        weight='bold',
        zorder=11
    )
    txt.set_path_effects([
        path_effects.Stroke(linewidth=outline_width, foreground=outline_color),
        path_effects.Normal()
    ])


def plot_tracked_objects_ego_centric(
    ax: plt.Axes,
    tracked_objects: TrackedObjects,
    ego_state: EgoState,
    agent_id_map: dict,
    show_ids: bool = True,
    map_radius: Optional[float] = None
) -> None:
    """
    Render all tracked objects in ego-centric frame with type-based colors and agent IDs.

    Args:
        ax: Matplotlib axes object
        tracked_objects: Tracked objects collection
        ego_state: Ego vehicle state
        agent_id_map: Mapping from track_token to sequential ID (for vehicles)
        show_ids: Whether to display agent IDs (for vehicles)
        map_radius: Optional radius in meters to filter agents (only render agents within this distance)
    """
    if not hasattr(tracked_objects, 'tracked_objects'):
        logger.warning("TrackedObjects missing tracked_objects attribute")
        return

    # 用于记录每种类型是否已添加到legend
    legend_types = set()
    object_counts = {}

    for obj in tracked_objects.tracked_objects:
        obj_type = obj.tracked_object_type

        # 跳过EGO类型（自车单独绘制）
        if obj_type == TrackedObjectType.EGO:
            continue

        # Filter by distance if map_radius is specified
        if map_radius is not None:
            dx = obj.center.x - ego_state.rear_axle.x
            dy = obj.center.y - ego_state.rear_axle.y
            distance = np.sqrt(dx * dx + dy * dy)
            if distance > map_radius:
                continue

        try:
            # Extract object info
            world_x = obj.center.x
            world_y = obj.center.y
            world_heading = obj.center.heading
            length = obj.box.length
            width = obj.box.width

            # 获取该类型的颜色
            obj_color = OBJECT_TYPE_COLORS.get(obj_type, '#95A5A6')  # 默认灰色

            # 为该类型添加legend标签（每种类型只添加一次）
            if obj_type not in legend_types:
                label = OBJECT_TYPE_LABELS.get(obj_type, str(obj_type))
                legend_types.add(obj_type)
            else:
                label = None

            # Draw object box
            plot_vehicle_box_ego_centric(
                ax, world_x, world_y, world_heading, length, width,
                ego_state,
                color=obj_color,
                label=label,
                alpha=0.85,
                linewidth=2.0
            )

            # 对于车辆类型，添加agent ID标签（显示在框内）
            if show_ids and obj_type == TrackedObjectType.VEHICLE:
                seq_id = None
                if hasattr(obj.metadata, 'track_token'):
                    track_token = obj.metadata.track_token
                    if isinstance(track_token, str):
                        token_str = track_token
                    elif hasattr(track_token, 'hex'):
                        token_str = track_token.hex
                    else:
                        token_str = str(track_token)

                    # Get sequential ID
                    seq_id = agent_id_map.get(token_str)

                if seq_id is not None:
                    ego_x, ego_y = transform_to_ego_frame(world_x, world_y, ego_state)

                    # Position label in the center of vehicle box
                    add_text_with_outline(
                        ax, ego_x, ego_y,
                        f"{seq_id}",
                        fontsize=12,
                        color='white',
                        outline_color='black',
                        outline_width=3.0
                    )

            # 统计各类型数量
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        except Exception as e:
            logger.debug(f"Failed to render object: {e}")
            continue

    # 输出统计信息
    for obj_type, count in object_counts.items():
        type_name = OBJECT_TYPE_LABELS.get(obj_type, str(obj_type))
        logger.debug(f"Rendered {count} {type_name}")


def plot_planned_trajectory_ego_centric(
    ax: plt.Axes,
    trajectory,
    ego_state: EgoState,
    color: str = '#FF00FF',
    linewidth: float = 2.5,
    alpha: float = 0.8,
    label: str = 'Planned Trajectory'
) -> None:
    """
    Render planned trajectory in ego-centric frame.

    Args:
        ax: Matplotlib axes object
        trajectory: InterpolatedTrajectory object from simulation sample
        ego_state: Current ego vehicle state (for coordinate transformation)
        color: Trajectory line color
        linewidth: Line width
        alpha: Transparency
        label: Legend label
    """
    try:
        # Get sampled trajectory points
        sampled_traj = trajectory.get_sampled_trajectory()

        if len(sampled_traj) == 0:
            logger.debug("Empty trajectory, skipping")
            return

        # Transform trajectory points to ego frame
        traj_points_ego = []
        for state in sampled_traj:
            ego_x, ego_y = transform_to_ego_frame(
                state.rear_axle.x,
                state.rear_axle.y,
                ego_state
            )
            traj_points_ego.append((ego_x, ego_y))

        # Plot trajectory as a line
        if len(traj_points_ego) > 1:
            xs, ys = zip(*traj_points_ego)
            ax.plot(
                xs, ys,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                zorder=7,
                linestyle='-'
            )

            # Add markers at trajectory points for better visibility
            ax.scatter(
                xs, ys,
                color=color,
                s=20,
                alpha=alpha * 0.6,
                zorder=7,
                edgecolors='white',
                linewidths=0.5
            )

            # Mark the end point with a larger marker
            ax.scatter(
                xs[-1], ys[-1],
                color=color,
                s=80,
                alpha=alpha,
                zorder=7,
                marker='*',
                edgecolors='white',
                linewidths=1.5
            )

    except Exception as e:
        logger.debug(f"Failed to render trajectory: {e}")
