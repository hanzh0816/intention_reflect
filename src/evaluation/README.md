# 失败案例收集系统使用说明

## 概述

失败案例收集系统（Failure Case Collector）是一个基于Callback机制的自动化工具，用于在开环仿真测试时检测和保存所有失败案例到SQLite数据库。

## 支持的失败类型

1. **碰撞（Collision）**：与其他车辆、行人或静态物体发生碰撞
   - 区分at-fault（自车责任）和not-at-fault碰撞
   - 记录碰撞能量、类型、对象类型等详细信息

2. **超速（Speed Violation）**：超出道路限速
   - 阈值：默认超速 >5 m/s
   - 记录违规时长、最大超速值、平均超速值

3. **锁死/停滞（Deadlock）**：长时间无法前进
   - 阈值：持续时间 >10s 且进度 <10%期望进度
   - 记录实际进度、期望进度、最终速度

4. **可行驶区域违规（Drivable Area Violation）**：驶出可行驶区域
   - 阈值：距离可行驶区域 >0.5m
   - 当前版本为简化实现

## 安装和配置

### 1. 启用失败案例收集

在你的评估配置文件中添加callback（例如 `config/local/my_eval.yaml`）：

```yaml
defaults:
  - callback:
      - simulation_log_callback
      - failure_case_collector  # 添加这一行
```

### 2. 自定义配置（可选）

如果需要自定义阈值，可以在配置文件中覆盖参数：

```yaml
callback:
  failure_case_collector:
    enable_collision_detection: true
    enable_speed_detection: true
    enable_deadlock_detection: true
    enable_drivable_area_detection: false  # 禁用可行驶区域检测

    # 调整阈值
    max_overspeed_threshold_mps: 8.0       # 改为8 m/s
    deadlock_duration_threshold_s: 15.0    # 改为15秒
    deadlock_min_progress_ratio: 0.05      # 改为5%
```

## 使用方法

### 运行评估

使用现有的评估脚本：

```bash
# 运行开环评估
./eval.sh my_config open_loop_boxes

# 或使用其他challenge
./eval.sh my_config closed_loop_nonreactive_agents
```

### 查看结果

失败案例数据库将保存在输出目录下：

```bash
# 数据库位置
<output_dir>/failure_cases.db

# 例如
exp/plantf/2025-01-01_12-00-00/failure_cases.db
```

## 数据库结构

### 主要表

1. **failure_cases**: 失败案例主表
   - scenario_type, scenario_name, log_name, planner_name
   - duration_seconds, failure_severity (CRITICAL/HIGH/MEDIUM)
   - has_collision, has_speed_violation, has_deadlock等标志

2. **collision_details**: 碰撞详情
   - timestamp_us, collision_type, tracked_object_type
   - collision_energy, is_at_fault, ego_speed, object_speed

3. **speed_violation_details**: 速度违规详情
   - start/end_timestamp_us, duration_us
   - max_overspeed_mps, mean_overspeed_mps, speed_limit_mps

4. **deadlock_details**: 锁死详情
   - duration_seconds, total_progress_meters, expected_progress_meters
   - progress_ratio, final_speed_mps

5. **simulation_histories**: 完整SimulationHistory（压缩保存）
   - history_blob (BLOB), serialization_format
   - blob_size_bytes, num_samples

## 查询失败案例

### 使用SQLite命令行

```bash
# 打开数据库
sqlite3 <output_dir>/failure_cases.db

# 查看所有失败案例
SELECT scenario_name, failure_severity,
       has_collision, has_speed_violation, has_deadlock
FROM failure_cases;

# 查看所有碰撞案例
SELECT fc.scenario_name, cd.collision_type, cd.is_at_fault, cd.collision_energy
FROM failure_cases fc
JOIN collision_details cd ON fc.id = cd.failure_case_id
WHERE fc.has_collision = 1;

# 按严重程度统计
SELECT failure_severity, COUNT(*) as count
FROM failure_cases
GROUP BY failure_severity;

# 按失败类型统计
SELECT
  SUM(has_collision) as collisions,
  SUM(has_speed_violation) as speed_violations,
  SUM(has_deadlock) as deadlocks
FROM failure_cases;
```

### 使用Python脚本

```python
import sqlite3
import pickle
import lzma

# 连接数据库
conn = sqlite3.connect('failure_cases.db')
conn.row_factory = sqlite3.Row

# 查询失败案例
cursor = conn.execute("""
    SELECT * FROM failure_cases
    WHERE has_collision = 1
    ORDER BY simulation_timestamp DESC
    LIMIT 10
""")

for row in cursor:
    print(f"Scenario: {row['scenario_name']}")
    print(f"  Severity: {row['failure_severity']}")
    print(f"  Duration: {row['duration_seconds']:.1f}s")

# 提取SimulationHistory
cursor = conn.execute("""
    SELECT history_blob FROM simulation_histories
    WHERE failure_case_id = ?
""", (1,))

row = cursor.fetchone()
if row:
    # 解压和反序列化
    compressed_data = row['history_blob']
    decompressed_data = lzma.decompress(compressed_data)
    history = pickle.loads(decompressed_data)
    print(f"History loaded: {len(history)} samples")

conn.close()
```

## 技术细节

### 失败判定逻辑

**碰撞（标准模式）**：
- 零容忍at-fault碰撞（前碰、与静止物体碰撞、不当变道时的侧碰）
- 记录所有碰撞，但只有at-fault碰撞标记为CRITICAL

**超速（标准模式）**：
- 超速 >5 m/s 才记录
- 连续超速时间合并为一次违规

**锁死（标准模式）**：
- 持续时间 >10秒
- 实际进度 <10% 期望进度（假设平均速度15 m/s）

### 数据压缩

- 使用pickle协议序列化SimulationHistory
- 使用lzma压缩（preset=3）
- 典型压缩率：10-30% of 原始大小

### 性能影响

- 失败检测在 `on_simulation_end` 进行
- 每个scenario增加约0.5-1秒开销
- 仅在检测到失败时保存数据

## 故障排除

### Callback未运行

检查配置文件是否正确添加了callback：
```bash
grep -r "failure_case_collector" config/local/
```

### 数据库未创建

检查输出目录权限和日志：
```bash
ls -la <output_dir>/
tail -f <output_dir>/log.txt | grep FailureCaseCollector
```

### 导入错误

确保PYTHONPATH包含项目根目录：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 扩展和定制

### 添加新的失败类型

1. 在 `detection_engine.py` 中定义新的数据类
2. 创建新的检测器（继承类似的模式）
3. 在 `database_manager.py` 中添加新表
4. 在 `FailureDetectionEngine` 中集成新检测器

### 调整检测逻辑

修改对应的检测器文件：
- `collision_detector.py` - 碰撞检测逻辑
- `speed_violation_detector.py` - 速度违规检测
- `deadlock_detector.py` - 锁死检测

## 版本信息

- **版本**: 1.0
- **兼容性**: PlanTF框架
- **Python**: 3.8+
- **依赖**: nuplan-devkit, SQLite3

## 支持

如有问题请查看：
- 计划文档: `/home/hzh/.claude/plans/frolicking-beaming-dragon.md`
- 代码位置: `src/evaluation/`
- 配置位置: `nuplan-devkit/nuplan/planning/script/config/simulation/callback/failure_case_collector.yaml`
