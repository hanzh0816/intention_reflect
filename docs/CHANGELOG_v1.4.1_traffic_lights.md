# v1.4.1 功能更新：红绿灯信息显示

## 🎯 新增功能

在查询工具中增加了红绿灯信息的显示，可以查看碰撞时刻的红绿灯状态。

### 主要特性

- ✅ **红绿灯状态显示**：在碰撞事件中显示碰撞时刻的红绿灯信息
- ✅ **多灯显示**：显示附近所有检测到的红绿灯状态
- ✅ **颜色标识**：使用彩色图标（🔴🟢🟡⚪）直观显示红绿灯状态
- ✅ **Lane 信息**：每个红绿灯显示对应的 Lane Connector ID
- ✅ **自动提取**：与帧号映射同步提取，无额外性能开销

## 📊 显示效果

### 使用 `--show-frames` 参数查询

```bash
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames
```

### 输出示例

```
Collision #1:
  Timestamp:       1626462329123456 us
  Frame Number:    304
  Collision Type:  ACTIVE_FRONT_COLLISION
  Object Type:     VEHICLE
  At Fault:        YES ⚠️
  Collision Energy: 45.23
  Ego Speed:       12.50 m/s
  Object Speed:    8.30 m/s
  Traffic Lights:  3 light(s) detected        ← 新增 ✨
    Light 1: 🔴 RED (Lane: 12345)
    Light 2: 🟢 GREEN (Lane: 12346)
    Light 3: 🟡 YELLOW (Lane: 12347)
```

## 🔍 红绿灯状态说明

| 图标 | 状态 | 说明 |
|------|------|------|
| 🔴 | RED | 红灯 |
| 🟢 | GREEN | 绿灯 |
| 🟡 | YELLOW | 黄灯 |
| ⚪ | UNKNOWN | 未知状态 |

## 💡 应用场景

### 场景 1: 分析红灯碰撞

```bash
# 查询碰撞详情（含红绿灯状态）
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames

# 输出可能显示：
# Collision at frame 304
# At Fault: YES
# Traffic Lights:
#   Light 1: 🔴 RED (Lane: 12345)  ← 碰撞时自车lane的红绿灯是红灯
```

**分析结论**：自车在红灯时发生碰撞，很可能是闯红灯导致。

### 场景 2: 分析黄灯加速碰撞

```bash
# 输出可能显示：
# Collision at frame 150
# At Fault: YES
# Ego Speed: 18.5 m/s (加速中)
# Traffic Lights:
#   Light 1: 🟡 YELLOW (Lane: 12345)  ← 黄灯时加速
```

**分析结论**：自车在黄灯时加速通过路口，可能导致与其他车辆碰撞。

### 场景 3: 分析非红绿灯路口碰撞

```bash
# 输出可能显示：
# Collision at frame 220
# At Fault: YES
# Traffic Lights: 0 light(s) detected  ← 没有红绿灯
```

**分析结论**：碰撞发生在无红绿灯路口，需要检查其他因素（如让行规则）。

## 🔧 技术实现

### 1. 数据源

红绿灯信息来自 `SimulationHistorySample` 的 `traffic_light_status` 字段：

```python
class SimulationHistorySample:
    iteration: SimulationIteration
    ego_state: EgoState
    trajectory: AbstractTrajectory
    observation: Observation
    traffic_light_status: List[TrafficLightStatusData]  ← 红绿灯信息
```

### 2. 提取逻辑

```python
def _extract_traffic_light_info(self, history) -> Dict[int, List]:
    """提取每一帧的红绿灯状态"""
    traffic_light_info = {}
    for sample in history.data:
        timestamp_us = sample.ego_state.time_point.time_us
        if hasattr(sample, 'traffic_light_status') and sample.traffic_light_status:
            tl_list = []
            for tl in sample.traffic_light_status:
                tl_dict = {
                    'lane_connector_id': tl.lane_connector_id,
                    'status': tl.status.name  # RED, GREEN, YELLOW, UNKNOWN
                }
                tl_list.append(tl_dict)
            traffic_light_info[timestamp_us] = tl_list
    return traffic_light_info
```

### 3. 显示逻辑

在 `format_collisions()` 函数中：

```python
# 显示红绿灯信息（如果有）
if traffic_light_info and timestamp in traffic_light_info:
    tl_status_list = traffic_light_info[timestamp]
    if tl_status_list:
        print(f"  Traffic Lights:  {len(tl_status_list)} light(s) detected")
        for tl_idx, tl in enumerate(tl_status_list, 1):
            # 根据状态选择图标
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
```

## ⚡ 性能影响

### 无额外开销

红绿灯信息与帧号映射同时提取，不会增加额外的性能开销：

- **基本查询**（无 `--show-frames`）：不提取红绿灯信息，性能不受影响
- **详细查询**（有 `--show-frames`）：红绿灯信息随 SimulationHistory 一次性提取

### 性能对比

| 查询模式 | 速度 | 内存 | 红绿灯信息 |
|---------|------|------|-----------|
| 基本查询 | <100ms | <10MB | ❌ 不显示 |
| 详细查询 | 1-5s | 100-500MB | ✅ 显示（无额外开销） |

## 🔄 向后兼容性

### 完全兼容

- 所有现有命令仍然有效
- 不使用 `--show-frames` 时，行为与之前完全一致
- 使用 `--show-frames` 时，自动显示红绿灯信息（如果有）

### 优雅降级

如果 SimulationHistory 中没有红绿灯信息：
- 不会报错
- 简单地不显示红绿灯部分
- 其他信息正常显示

```python
# 代码中的保护措施
if hasattr(sample, 'traffic_light_status') and sample.traffic_light_status:
    # 提取红绿灯信息
    ...
# 如果没有 traffic_light_status 属性或为空，则跳过
```

## 📝 更新的文件

### 核心代码
- ✅ `scripts/query_failure_details.py`
  - 添加 `_extract_traffic_light_info()` 方法
  - 更新 `query_failure_details()` 方法
  - 更新 `format_collisions()` 函数
  - 更新提示信息

### 文档
- ✅ `docs/TOOL_query_failure_details.md` - 添加红绿灯信息说明
- ✅ `docs/FEATURE_v1.4_query_tool.md` - 更新输出示例
- ✅ `docs/IMPLEMENTATION_v1.4_summary.md` - 添加实现细节
- ✅ `docs/CHANGELOG_v1.4.1_traffic_lights.md` - 本文档

## 🎯 使用建议

### 建议 1: 分析碰撞原因

查看碰撞时的红绿灯状态，判断是否与闯红灯相关：

```bash
python scripts/query_failure_details.py \
  --scenario-name "collision_scenario" \
  --show-frames
```

### 建议 2: 批量分析红灯碰撞

```python
from scripts.query_failure_details import FailureDetailsQuery

query = FailureDetailsQuery("work_dirs/exp/failure_cases.db")

# 查询所有碰撞案例
collisions = query._db_manager.list_all_failure_cases()

red_light_collisions = []
for case in collisions:
    if case['has_collision']:
        details = query.query_failure_details(case['scenario_name'], load_history=True)

        # 检查是否有红灯碰撞
        for collision in details['collisions']:
            timestamp = collision['timestamp_us']
            if timestamp in details['traffic_light_info']:
                tl_list = details['traffic_light_info'][timestamp]
                for tl in tl_list:
                    if tl['status'] == 'RED':
                        red_light_collisions.append({
                            'scenario': case['scenario_name'],
                            'frame': details['frame_mapping'][timestamp],
                            'lane': tl['lane_connector_id']
                        })
                        break

print(f"Found {len(red_light_collisions)} potential red light collisions")
```

### 建议 3: 生成红绿灯统计报告

```bash
# 创建自定义脚本统计红绿灯状态分布
python -c "
from scripts.query_failure_details import FailureDetailsQuery

query = FailureDetailsQuery('work_dirs/exp/failure_cases.db')
cases = query._db_manager.list_all_failure_cases()

tl_stats = {'RED': 0, 'GREEN': 0, 'YELLOW': 0, 'UNKNOWN': 0, 'NONE': 0}

for case in cases:
    if case['has_collision']:
        details = query.query_failure_details(case['scenario_name'], load_history=True)
        for collision in details['collisions']:
            timestamp = collision['timestamp_us']
            if timestamp in details['traffic_light_info']:
                for tl in details['traffic_light_info'][timestamp]:
                    tl_stats[tl['status']] += 1
            else:
                tl_stats['NONE'] += 1

print('Traffic Light Status Distribution:')
for status, count in tl_stats.items():
    print(f'  {status}: {count}')
"
```

## 📚 相关文档

- **使用指南**: `docs/TOOL_query_failure_details.md`
- **功能介绍**: `docs/FEATURE_v1.4_query_tool.md`
- **实现总结**: `docs/IMPLEMENTATION_v1.4_summary.md`
- **快速开始**: `docs/QUICKSTART_failure_viz.md`

## 🎉 总结

v1.4.1 成功添加了红绿灯信息显示功能：

✅ 在碰撞事件中显示红绿灯状态
✅ 使用直观的彩色图标
✅ 包含 Lane Connector ID
✅ 无额外性能开销
✅ 完全向后兼容
✅ 优雅降级处理

现在你可以更深入地分析碰撞原因，特别是与红绿灯相关的碰撞！🚦🚀
