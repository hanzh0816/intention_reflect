# 轨迹与地图可视化说明

## 概述

这个脚本用于可视化自车轨迹并同时显示场景地图。与原始的 `visualize_ego_trajectories.py` 不同，本脚本具有以下特点：

1. **独立保存**：每个轨迹单独保存为一个图像文件
2. **地图上下文**：在轨迹上叠加显示场景的地图元素（车道、路段等）
3. **原始坐标系**：使用原始坐标系（不归一化），确保轨迹与地图正确对应
4. **时间戳命名**：输出目录自动添加时间戳后缀，格式如 `./work_dirs/trajectory_visualizations_20250131_143025/`

## 文件说明

- `visualize_ego_trajectories_with_map.py`: 主可视化脚本
- `config/training/visualize_trajectories_with_map.yaml`: 配置文件

## 使用方法

### 基本使用

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_trajectories_with_map \
    scenario_builder=nuplan_mini \
    scenario_filter.limit_total_scenarios=10
```

### 自定义参数

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_trajectories_with_map \
    scenario_builder=nuplan_mini \
    scenario_filter.limit_total_scenarios=20 \
    num_scenarios_to_visualize=20 \
    map_radius=100.0 \
    trajectory_output_dir=work_dirs/my_visualizations
```

### 按场景类型过滤

如果你只想可视化特定类型的场景（如超车、变道等），可以使用 `scenario_type` 参数：

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_trajectories_with_map \
    scenario_builder=nuplan_mini \
    scenario_filter.limit_total_scenarios=1000 \
    scenario_type=near_multiple_vehicles \
    num_scenarios_to_visualize=20
```

此命令会从整个数据集中筛选出所有 `near_multiple_vehicles` 类型的场景，然后从中随机采样20个进行可视化。

**注意**：使用 `scenario_type` 时，建议将 `scenario_filter.limit_total_scenarios` 设置较大，以确保加载足够多的场景进行筛选。

## 配置参数说明

### 可视化参数

- `num_scenarios_to_visualize`: 要可视化的场景数量（默认：10）
- `scenario_type`: 可选，按场景类型过滤（默认：None，即不过滤）
  - 可用的场景类型包括：`near_multiple_vehicles`、`following_lane`、`changing_lane` 等
  - 如果指定，只会从匹配该类型的场景中采样
  - 如果指定的类型不存在，脚本会显示数据集中所有可用的场景类型
- `trajectory_output_dir`: 输出目录基础路径（默认：`work_dirs/trajectory_visualizations`）
  - 注意：实际输出目录会自动添加时间戳后缀，如 `work_dirs/trajectory_visualizations_20250131_143025`
  - 如果指定了 `scenario_type`，目录名中会包含场景类型，如 `work_dirs/trajectory_visualizations_near_multiple_vehicles_20250131_143025`
- `map_radius`: 地图显示半径，单位米（默认：80.0）

### 轨迹提取参数

- `history_horizon`: 历史轨迹时间范围，单位秒（默认：2.0）
- `future_horizon`: 未来轨迹时间范围，单位秒（默认：8.0）
- `sample_interval`: 采样间隔，单位秒（默认：0.1）

## 输出说明

### 输出目录结构

输出目录自动添加运行时间戳，例如：
```
work_dirs/
└── trajectory_visualizations_20250131_143025/
    ├── trajectory_000_following_lane_af3c2b1e.png
    ├── trajectory_001_changing_lane_9d4e5f2a.png
    └── ...
```

如果指定了 `scenario_type`，目录名会包含场景类型：
```
work_dirs/
└── trajectory_visualizations_near_multiple_vehicles_20250131_143025/
    ├── trajectory_000_near_multiple_vehicles_af3c2b1e.png
    ├── trajectory_001_near_multiple_vehicles_9d4e5f2a.png
    └── ...
```

时间戳格式为 `YYYYMMDD_HHMMSS`（年月日_时分秒），这样可以精确区分每次运行的结果，避免覆盖。

### 文件命名

每个轨迹图像文件命名格式：
```
trajectory_{序号}_{场景类型}_{场景token}.png
```

例如：
```
trajectory_000_following_lane_af3c2b1e.png
trajectory_001_changing_lane_9d4e5f2a.png
```

### 可视化元素

每张图包含以下元素：

1. **地图元素**
   - 浅蓝色填充区域（交替深浅）：不同车道，使用填充多边形和交替颜色清晰区分
   - 深灰色实线：车道边界（加粗显示，更加清晰）
   - 灰色虚线：车道中心线
   - 蓝色线条：路线路段边界（route roadblocks）

2. **轨迹元素**
   - 亮蓝色粗线 + 圆点：历史轨迹（过去2秒）
   - 橙红色粗线 + 圆点：未来轨迹（未来8秒）
   - 蓝色方块（白色边框）：轨迹起点
   - 红色三角（白色边框）：轨迹终点

## 视觉设计特点

- **车道区分明显**：使用交替的浅蓝色填充不同的车道，配合加粗的深灰色边界线，使各车道清晰可辨
- **分层渲染**：使用zorder控制绘制层次
  - 底层（zorder=1）：车道填充
  - 中层（zorder=2-3）：车道边界、中心线、路段边界
  - 顶层（zorder=10-11）：轨迹和标记点
- **高对比度配色**：轨迹使用亮蓝色和橙红色，在地图背景上非常醒目
- **无自车绘制**：不绘制自车初始位置，避免视觉干扰，聚焦于轨迹本身

## 与原始脚本的区别

| 特性 | 原始脚本 | 新脚本 |
|------|---------|--------|
| 输出方式 | 所有轨迹在一张图上 | 每个轨迹单独一张图 |
| 坐标系 | 归一化到(0,0) | 原始坐标系 |
| 地图显示 | 无 | 有（车道、路段等） |
| 车道区分 | N/A | 填充+交替颜色+加粗边界 |
| 输出位置 | 单个文件 | 文件夹（work_dirs下） |
| 用途 | 轨迹形状对比 | 轨迹与场景关系分析 |

## 注意事项

1. **时间戳命名**：输出目录会自动添加当前时间戳（格式：YYYYMMDD_HHMMSS），每次运行会创建新的时间戳目录
2. **内存使用**：由于每个场景都需要加载地图数据，处理大量场景时可能需要较多内存
3. **处理时间**：绘制地图元素需要额外时间，处理速度比原始脚本慢
4. **坐标系统**：轨迹使用原始坐标（世界坐标系），与地图直接对应
5. **归一化问题已解决**：轨迹不再归一化，直接使用原始坐标，因此可以正确叠加在地图上

## 故障排除

### 地图元素未显示

如果某些地图元素未显示，可能是因为：
- 地图数据缺失
- `map_radius` 设置过小，增大该值可以显示更多周围环境

### 输出目录权限问题

确保 `work_dirs` 目录有写权限：
```bash
mkdir -p work_dirs
chmod 755 work_dirs
```

注意：无需手动创建时间戳目录，脚本会自动创建。

## 示例

查看示例输出效果，请运行基本命令后检查 `work_dirs/trajectory_visualizations_YYYYMMDD_HHMMSS/` 目录（时间戳为运行时刻）。

例如，2025年1月31日14:30:25运行后，输出目录为：
```
work_dirs/trajectory_visualizations_20250131_143025/
```
