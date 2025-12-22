# v1.4 功能更新：Failure Details 查询工具

## 🎯 新增功能

### 快速查询 Failure 详情

新增 `query_failure_details.py` 工具，可以通过 scenario token 查询数据库，返回详细的失败信息：

```bash
# 基本查询
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"

# 包含帧号的详细查询
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980" --show-frames
```

## 🔍 查询信息详解

### 1. 基本信息

- Scenario token（16位十六进制）
- 场景类型（highway, urban 等）
- 日志名称
- 规划器名称
- 仿真总时长
- 失败严重程度（CRITICAL / HIGH / MEDIUM）

### 2. 碰撞信息

每个碰撞事件包含：
- ✅ **时间戳**（微秒级精度）
- ✅ **帧号**（需要 `--show-frames`）
- ✅ **碰撞类型**（前向/侧向/后向/停止车辆）
- ✅ **责任判定**（是否由自车导致）⚠️
- ✅ **红绿灯状态**（碰撞时刻的红绿灯信息，需要 `--show-frames`）✨
- 碰撞对象类型（车辆/行人等）
- 碰撞能量（Delta-V）
- 自车和对象速度

### 3. 超速信息

每个超速事件包含：
- ✅ **起止时间戳**（微秒级精度）
- ✅ **起止帧号范围**（需要 `--show-frames`）
- 持续时长
- 最大/平均超速值
- 速度限制

### 4. 死锁信息

死锁事件包含：
- ✅ **起止时间戳**（微秒级精度）
- ✅ **起止帧号范围**（需要 `--show-frames`）
- 持续时长
- 实际/预期前进距离
- 进度比例
- 最终速度

### 5. 可行驶区域违规

每个违规事件包含：
- ✅ **时间戳**（微秒级精度）
- ✅ **帧号**（需要 `--show-frames`）
- 距离可行驶区域的最大距离
- 违规持续时长

## 📊 输出示例

### 基本查询（快速模式）

```bash
$ python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"

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
  Frame Number:    ?                          ← 需要 --show-frames 显示
  Collision Type:  ACTIVE_FRONT_COLLISION
  Object Type:     VEHICLE
  At Fault:        YES ⚠️
  Collision Energy: 45.23
  Ego Speed:       12.50 m/s
  Object Speed:    8.30 m/s

================================================================================

Tip: Use --show-frames to see frame numbers (requires loading simulation history)
```

### 详细查询（包含帧号）

```bash
$ python scripts/query_failure_details.py --scenario-name "00cca24d240f5980" --show-frames

Querying failure details for: 00cca24d240f5980
(Loading simulation history to compute frame numbers...)

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
  Frame Number:    304                        ← 碰撞发生在第304帧 ✅
  Collision Type:  ACTIVE_FRONT_COLLISION
  Object Type:     VEHICLE
  At Fault:        YES ⚠️
  Collision Energy: 45.23
  Ego Speed:       12.50 m/s
  Object Speed:    8.30 m/s
  Traffic Lights:  2 light(s) detected        ← 碰撞时刻的红绿灯状态 ✨
    Light 1: 🔴 RED (Lane: 12345)
    Light 2: 🟢 GREEN (Lane: 12346)

--------------------------------------------------------------------------------
Speed Violations (1 total)
--------------------------------------------------------------------------------

Violation #1:
  Start Time:      1626462325000000 us
  End Time:        1626462330000000 us
  Frame 220 to 320                           ← 超速帧号范围 ✅
  Duration:        5.00 seconds
  Max Overspeed:   8.50 m/s
  Mean Overspeed:  6.20 m/s
  Speed Limit:     20.00 m/s

================================================================================
```

## 🔧 使用场景

### 场景 1: 快速检查失败类型

```bash
# 不加载 SimulationHistory，快速查看失败摘要
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"
```

**优点**：
- 速度极快（毫秒级）
- 低内存占用
- 获取时间戳和失败类型

**适用于**：
- 快速浏览失败案例
- 批量查询多个 scenarios
- 只需要时间戳信息

### 场景 2: 定位具体失败帧

```bash
# 加载 SimulationHistory，获取精确帧号
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames
```

**优点**：
- 获取精确帧号
- 可以在代码中定位失败帧
- 方便在 nuBoard 中跳转到失败时刻

**适用于**：
- 需要在代码中分析失败前后帧
- 在 nuBoard 中精确定位失败时刻
- 深入分析失败原因

### 场景 3: 批量分析失败案例

```bash
# 查询所有碰撞案例
sqlite3 work_dirs/exp/failure_cases.db \
  "SELECT scenario_name FROM failure_cases WHERE has_collision=1" | \
while read scenario; do
  echo "============================================"
  echo "Analyzing collision in: $scenario"
  python scripts/query_failure_details.py --scenario-name "$scenario"
  echo ""
done
```

### 场景 4: 导出查询结果

```bash
# 将查询结果保存到文件
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames \
  > failure_analysis_00cca24d240f5980.txt
```

### 场景 5: 编程使用

```python
from scripts.query_failure_details import FailureDetailsQuery

# 创建查询对象
query = FailureDetailsQuery("work_dirs/exp/failure_cases.db")

# 查询失败详情
details = query.query_failure_details(
    scenario_name="00cca24d240f5980",
    load_history=True
)

# 访问碰撞信息
for collision in details['collisions']:
    timestamp = collision['timestamp_us']
    frame_num = details['frame_mapping'][timestamp]

    print(f"Collision at frame {frame_num}")
    print(f"Type: {collision['collision_type']}")
    print(f"At fault: {collision['is_at_fault']}")
    print(f"Energy: {collision['collision_energy']}")
```

## 🚀 与其他工具配合

### 工作流 1: 查询 → 导出 → 可视化

```bash
# 1. 查询失败详情
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames

# 2. 导出到 nuBoard 格式
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980"

# 3. 在 nuBoard 中可视化
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006

# 4. 在浏览器中打开并跳转到失败帧
# 根据查询结果中的帧号（如 304）在 nuBoard 时间轴中定位
```

### 工作流 2: 列表 → 查询 → 分析

```bash
# 1. 列出所有失败案例
python scripts/export_failure_cases.py --list

# 2. 选择感兴趣的案例查询详情
python scripts/query_failure_details.py \
  --scenario-name "00cca24d240f5980" \
  --show-frames

# 3. 在代码中加载并分析失败帧
python -c "
from nuplan.planning.simulation.simulation_log import SimulationLog

sim_log = SimulationLog.load_data('work_dirs/failure_viz/simulation/planTF/.../00cca24d240f5980.pkl.xz')

# 根据查询结果定位到碰撞帧（如 304）
collision_frame = sim_log.simulation_history.data[304]
print(f'Ego position: {collision_frame.ego_state.center}')
print(f'Ego speed: {collision_frame.ego_state.dynamic_car_state.speed}')
"
```

### 工作流 3: 筛选 → 批量查询

```bash
# 1. 筛选所有碰撞案例
sqlite3 work_dirs/exp/failure_cases.db \
  "SELECT scenario_name FROM failure_cases WHERE has_collision=1" \
  > collision_scenarios.txt

# 2. 批量查询详情
cat collision_scenarios.txt | while read scenario; do
  python scripts/query_failure_details.py --scenario-name "$scenario"
done > all_collision_details.txt

# 3. 分析汇总
grep "At Fault:" all_collision_details.txt | \
  grep -c "YES"  # 统计由自车导致的碰撞数量
```

## ⚡ 性能对比

| 模式 | 速度 | 内存 | 功能 |
|------|------|------|------|
| 基本查询 | 极快（<100ms） | 低（<10MB） | 时间戳、失败类型 |
| 详细查询（`--show-frames`） | 较慢（1-5s） | 高（100-500MB） | + 帧号映射 |

**建议**：
- 快速浏览时：使用基本查询
- 需要帧号时：使用详细查询
- 批量查询时：避免使用 `--show-frames`

## 📝 命令参数

```bash
python scripts/query_failure_details.py --help
```

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--scenario-name` | ✅ | - | Scenario token |
| `--database-path` | ❌ | `work_dirs/exp/failure_cases.db` | 数据库路径 |
| `--show-frames` | ❌ | `False` | 显示帧号（需加载历史） |

## 🔄 向后兼容性

- 完全兼容现有的 failure case 数据库
- 不影响导出和可视化工具
- 可以独立使用或与其他工具配合

## 🆚 与 --list 和 --info 的区别

| 功能 | `--list` | `--info` | `query_failure_details.py` |
|------|----------|----------|----------------------------|
| 场景 | 列出所有 | 显示基本信息 | 查询完整详情 |
| 碰撞时间戳 | ❌ | ❌ | ✅ |
| 碰撞帧号 | ❌ | ❌ | ✅（`--show-frames`） |
| 责任判定 | ❌ | ❌ | ✅ |
| 超速详情 | ❌ | ❌ | ✅（起止时间、帧号范围） |
| 死锁详情 | ❌ | ❌ | ✅（起止时间、进度比） |
| 速度 | 极快 | 快 | 快（基本）/ 较慢（`--show-frames`） |

## 💡 技巧

### 技巧 1: 使用快捷别名

```bash
# 添加到 ~/.bashrc
alias fc-query='python scripts/query_failure_details.py --scenario-name'

# 使用
fc-query "00cca24d240f5980"
fc-query "00cca24d240f5980" --show-frames
```

### 技巧 2: 查询并导出

```bash
# 先查询了解失败类型
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"

# 如果需要可视化，再导出
if [ 感兴趣 ]; then
  python scripts/export_failure_cases.py --scenario-name "00cca24d240f5980"
fi
```

### 技巧 3: 自定义查询

```bash
# 查询所有 CRITICAL 级别的碰撞
sqlite3 work_dirs/exp/failure_cases.db << EOF
SELECT fc.scenario_name, cd.timestamp_us, cd.collision_type
FROM failure_cases fc
JOIN collision_details cd ON cd.failure_case_id = fc.id
WHERE fc.failure_severity = 'CRITICAL' AND cd.is_at_fault = 1
ORDER BY cd.collision_energy DESC
LIMIT 10;
EOF
```

## 📚 相关文档

- **完整使用指南**: `docs/TOOL_query_failure_details.md`
- **快速开始**: `docs/QUICKSTART_failure_viz.md`
- **数据库架构**: `src/evaluation/database_manager.py`
- **检测引擎**: `src/evaluation/failure_detectors/detection_engine.py`

## 🎉 总结

v1.4 新增的查询工具提供了快速、灵活的失败信息查询能力：

✅ 通过 scenario token 快速查询失败详情
✅ 精确到帧号级别的失败定位
✅ 完整的责任判定和事件详情
✅ 灵活的查询模式（快速/详细）
✅ 可编程使用（Python 模块）
✅ 与现有工具完美配合

现在你可以更高效地分析和调试 failure cases 了！🚀
