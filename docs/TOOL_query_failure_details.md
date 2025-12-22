# Failure Details Query Tool

## 功能概述

`query_failure_details.py` 是一个查询工具，通过 scenario token 查询数据库，返回详细的失败信息：

- ✅ 失败类型（碰撞、超速、死锁、可行驶区域违规）
- ✅ 发生在哪一帧（可选）
- ✅ 精确时间戳（微秒）
- ✅ 责任判定（针对碰撞）
- ✅ 详细事件信息（速度、能量、进度等）

## 快速开始

### 基本用法（快速查询）

```bash
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"
```

输出示例：
```
================================================================================
Failure Case Details: 00cca24d240f5980
================================================================================

Basic Information:
  Scenario Name (Token): 00cca24d240f5980
  Scenario Type:         highway
  Log Name:              2021.07.16.20.45.29_veh-35_01095_01486
  Planner:               planTF
  Duration:              15.20 seconds
  Severity:              CRITICAL

Failure Types Summary:
  Collision:             Yes
  Speed Violation:       No
  Deadlock:              No
  Drivable Area:         No

--------------------------------------------------------------------------------
Collision Events (1 total)
--------------------------------------------------------------------------------

Collision #1:
  Timestamp:       1626462329123456 us
  Frame Number:    ?
  Collision Type:  ACTIVE_FRONT_COLLISION
  Object Type:     VEHICLE
  At Fault:        YES ⚠️
  Collision Energy: 45.23
  Ego Speed:       12.50 m/s
  Object Speed:    8.30 m/s

================================================================================

Tip: Use --show-frames to see frame numbers (requires loading simulation history)
```

### 完整查询（包含帧号）

```bash
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames
```

输出示例：
```
================================================================================
Failure Case Details: 00cca24d240f5980
================================================================================

Basic Information:
  Scenario Name (Token): 00cca24d240f5980
  Scenario Type:         highway
  Log Name:              2021.07.16.20.45.29_veh-35_01095_01486
  Planner:               planTF
  Duration:              15.20 seconds
  Severity:              CRITICAL

Failure Types Summary:
  Collision:             Yes
  Speed Violation:       Yes
  Deadlock:              No
  Drivable Area:         No

--------------------------------------------------------------------------------
Collision Events (1 total)
--------------------------------------------------------------------------------

Collision #1:
  Timestamp:       1626462329123456 us
  Frame Number:    304
  Collision Type:  ACTIVE_FRONT_COLLISION
  Object Type:     VEHICLE
  At Fault:        YES ⚠️
  Collision Energy: 45.23
  Ego Speed:       12.50 m/s
  Object Speed:    8.30 m/s
  Traffic Lights:  3 light(s) detected
    Light 1: 🔴 RED (Lane: 12345)
    Light 2: 🟢 GREEN (Lane: 12346)
    Light 3: 🟡 YELLOW (Lane: 12347)

--------------------------------------------------------------------------------
Speed Violations (1 total)
--------------------------------------------------------------------------------

Violation #1:
  Start Time:      1626462325000000 us
  End Time:        1626462330000000 us
  Frame 220 to 320
  Duration:        5.00 seconds
  Max Overspeed:   8.50 m/s
  Mean Overspeed:  6.20 m/s
  Speed Limit:     20.00 m/s

================================================================================
```

## 命令参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--scenario-name` | 要查询的 scenario token（必需） | - | `"00cca24d240f5980"` |
| `--database-path` | 数据库路径 | `work_dirs/exp/failure_cases.db` | `path/to/database.db` |
| `--show-frames` | 显示帧号（需要加载 SimulationHistory，较慢） | False | - |

## 输出信息详解

### 1. 基本信息 (Basic Information)

- **Scenario Name**: Scenario token（16位十六进制）
- **Scenario Type**: 场景类型（highway, urban 等）
- **Log Name**: 日志名称
- **Planner**: 规划器名称
- **Duration**: 仿真总时长（秒）
- **Severity**: 失败严重程度
  - `CRITICAL`: 有由自车导致的碰撞
  - `HIGH`: 有死锁或严重超速（>10 m/s）
  - `MEDIUM`: 其他违规

### 2. 碰撞事件 (Collision Events)

每个碰撞事件包含：
- **Timestamp**: 碰撞发生的精确时间戳（微秒）
- **Frame Number**: 碰撞发生的帧号（需要 `--show-frames`）
- **Collision Type**: 碰撞类型
  - `ACTIVE_FRONT_COLLISION`: 前向碰撞
  - `ACTIVE_LATERAL_COLLISION`: 侧向碰撞
  - `ACTIVE_REAR_COLLISION`: 后向碰撞
  - `STOPPED_TRACK_COLLISION`: 与停止车辆碰撞
- **Object Type**: 碰撞对象类型（VEHICLE, PEDESTRIAN 等）
- **At Fault**: 是否由自车导致 ⚠️
- **Collision Energy**: 碰撞能量（Delta-V）
- **Ego Speed**: 自车速度（m/s）
- **Object Speed**: 碰撞对象速度（m/s）
- **Traffic Lights**: 碰撞时刻的红绿灯状态（需要 `--show-frames`）✨
  - 显示附近所有红绿灯的状态
  - 🔴 RED（红灯）
  - 🟢 GREEN（绿灯）
  - 🟡 YELLOW（黄灯）
  - ⚪ UNKNOWN（未知状态）
  - 每个红绿灯显示对应的 Lane Connector ID

### 3. 超速违规 (Speed Violations)

每个超速事件包含：
- **Start Time / End Time**: 超速开始和结束时间戳（微秒）
- **Frame Range**: 超速帧号范围（需要 `--show-frames`）
- **Duration**: 超速持续时长（秒）
- **Max Overspeed**: 最大超速值（m/s）
- **Mean Overspeed**: 平均超速值（m/s）
- **Speed Limit**: 速度限制（m/s）

### 4. 死锁事件 (Deadlock Event)

死锁事件包含：
- **Start Time / End Time**: 死锁开始和结束时间戳（微秒）
- **Frame Range**: 死锁帧号范围（需要 `--show-frames`）
- **Duration**: 死锁持续时长（秒）
- **Total Progress**: 实际前进距离（米）
- **Expected Progress**: 预期前进距离（米）
- **Progress Ratio**: 进度比例（实际/预期）
- **Final Speed**: 最终速度（m/s）

### 5. 可行驶区域违规 (Drivable Area Violations)

每个违规事件包含：
- **Timestamp**: 违规发生的时间戳（微秒）
- **Frame Number**: 违规发生的帧号（需要 `--show-frames`）
- **Max Distance**: 距离可行驶区域的最大距离（米）
- **Duration**: 违规持续时长（秒）

## 使用场景

### 场景 1: 快速查看失败摘要

```bash
# 不加载 SimulationHistory，快速查看失败类型和时间戳
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"
```

**适用于**：
- 快速了解失败类型
- 查看时间戳信息
- 不需要知道具体帧号

### 场景 2: 定位具体失败帧

```bash
# 加载 SimulationHistory，获取精确帧号
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames
```

**适用于**：
- 需要在代码中定位具体帧
- 在 nuBoard 中查看特定帧
- 分析失败发生的前后帧

### 场景 3: 批量查询多个 scenarios

```bash
# 使用 shell 脚本批量查询
for scenario in "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1"; do
  echo "========================================"
  python scripts/query_failure_details.py --scenario-name "$scenario"
done
```

### 场景 4: 导出查询结果

```bash
# 将查询结果保存到文件
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames \
  > failure_analysis.txt
```

## 与其他工具配合使用

### 1. 列出所有 failure cases

```bash
# 先列出所有失败案例
python scripts/export_failure_cases.py --list

# 选择感兴趣的 scenario 查询详情
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"
```

### 2. 查询后导出可视化

```bash
# 查询失败详情
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames

# 导出到 nuBoard
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980"

# 启动 nuBoard 可视化
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 3. 筛选特定类型的失败

```bash
# 使用 SQL 查询所有碰撞案例
sqlite3 work_dirs/exp/failure_cases.db \
  "SELECT scenario_name FROM failure_cases WHERE has_collision=1" | \
while read scenario; do
  echo "Analyzing collision in: $scenario"
  python scripts/query_failure_details.py --scenario-name "$scenario" --show-frames
done
```

## 性能说明

### 不使用 `--show-frames`
- **速度**: 非常快（毫秒级）
- **功能**: 只查询数据库，返回时间戳信息
- **内存**: 低

### 使用 `--show-frames`
- **速度**: 较慢（秒级，取决于 SimulationHistory 大小）
- **功能**: 额外加载和反序列化 SimulationHistory，计算帧号
- **内存**: 需要加载完整历史记录（可能数百 MB）

**建议**：
- 快速查看时不使用 `--show-frames`
- 需要精确帧号时才使用 `--show-frames`
- 批量查询时避免使用 `--show-frames`

## 常见问题

### Q1: 时间戳和帧号的关系？

时间戳是绝对时间（微秒），帧号是相对索引（从0开始）。

```python
# SimulationHistory 中的关系
for idx, sample in enumerate(history.data):
    frame_number = idx  # 0, 1, 2, ...
    timestamp_us = sample.ego_state.time_point.time_us  # 绝对时间
```

### Q2: 如何找到碰撞前后的帧？

```bash
# 1. 查询碰撞帧号（假设是 304 帧）
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames

# 2. 在代码中加载历史记录
# collision_frame = 304
# before_frames = history.data[294:304]  # 碰撞前10帧
# collision_frame = history.data[304]    # 碰撞帧
# after_frames = history.data[305:315]   # 碰撞后10帧
```

### Q3: 为什么有些失败没有 "At Fault" 信息？

只有碰撞事件有责任判定。超速、死锁、可行驶区域违规都直接认为是自车的问题。

### Q4: 如何将查询结果用于程序分析？

可以将脚本导入为模块：

```python
from scripts.query_failure_details import FailureDetailsQuery

query = FailureDetailsQuery("work_dirs/exp/failure_cases.db")
details = query.query_failure_details("00cca24d240f5980", load_history=True)

# 访问碰撞信息
for collision in details['collisions']:
    print(f"Collision at frame {details['frame_mapping'][collision['timestamp_us']]}")
    print(f"At fault: {collision['is_at_fault']}")
```

## 技术实现

### 数据库查询

脚本直接查询 SQLite 数据库的 5 个表：
- `failure_cases`: 主表
- `collision_details`: 碰撞详情
- `speed_violation_details`: 超速详情
- `deadlock_details`: 死锁详情
- `drivable_area_violation_details`: 可行驶区域详情
- `simulation_histories`: 完整历史记录（仅在 `--show-frames` 时加载）

### 帧号计算

```python
# 从 SimulationHistory 构建时间戳到帧号的映射
frame_mapping = {}
for idx, sample in enumerate(history.data):
    timestamp_us = sample.ego_state.time_point.time_us
    frame_mapping[timestamp_us] = idx
```

## 相关文档

- **完整可视化指南**: `docs/failure_case_visualization.md`
- **数据库架构**: `src/evaluation/database_manager.py`
- **检测引擎**: `src/evaluation/failure_detectors/detection_engine.py`
- **碰撞检测**: `src/evaluation/failure_detectors/collision_detector.py`

---

现在你可以通过 scenario token 快速查询所有失败详情了！ ✨
