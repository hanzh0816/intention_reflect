# Failure Case 可视化使用指南

## 功能概述

Failure case 可视化工具可以将仿真阶段收集到的失败案例导出为 nuBoard 可加载的格式，支持交互式查看规划历史、轨迹、地图等信息。

### 核心特性

- ✅ 通过 scenario_name（token，如 `00cca24d240f5980`）精确查询失败案例
- ✅ 支持获取真实的 AbstractScenario 对象（需要提供 nuplan 数据库路径）
- ✅ 自动降级到 StubScenario（不需要额外数据时）
- ✅ 完整的 SimulationHistory 保存和反序列化
- ✅ 与 nuBoard 完全兼容的输出格式

## 使用流程

### 第一步：仿真阶段收集 failure cases

确保在评估配置中启用 `failure_case_collector` callback：

```yaml
# config/local/eval-xxx.yaml
callback:
  - failure_case_collector
```

运行评估：

```bash
bash eval.sh
```

这会生成失败案例数据库：`work_dirs/exp/failure_cases.db`

---

### 第二步：列出所有可用的 failure cases

```bash
python scripts/export_failure_cases.py \
  --list \
  --database-path work_dirs/exp/failure_cases.db
```

输出示例：

```
Found 3 failure case(s):

----------------------------------------------------------------------------------------------------------------------------
#    Scenario Name                            Planner         Severity   Type                      Duration
----------------------------------------------------------------------------------------------------------------------------
1    00cca24d240f5980                         planTF          CRITICAL   collision                 15.2s
2    01abd3e5120a4bc0                         planTF          HIGH       deadlock                  22.5s
3    02bcd4f6231b5cd1                         planTF          MEDIUM     speed_violation           10.8s
----------------------------------------------------------------------------------------------------------------------------

Total: 3 failure case(s)
```

---

### 第三步：导出特定 scenario

#### 方式 1：使用真实 scenario（推荐）

提供 nuplan 数据库路径，工具会自动通过 scenario_name（token）查询真实的 AbstractScenario 对象：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db
```

**参数说明：**

- `--scenario-name`: 要导出的 scenario token（16位十六进制字符串）
- `--database-path`: failure case 数据库路径
- `--output-dir`: 输出目录
- `--data-root`: nuplan 数据集根目录
- `--map-root`: nuplan 地图文件目录
- `--db-files`: nuplan 数据库文件路径（可以用逗号分隔多个文件）
- `--sensor-root`: (可选) 传感器数据路径
- `--map-version`: (可选) 地图版本，默认 `nuplan-maps-v1.0`

#### 方式 2：使用 StubScenario（简化方式）

如果不提供 nuplan 数据库路径，工具会自动使用轻量级的 StubScenario：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz
```

> **注意：** StubScenario 只包含基本元数据，真实 scenario 包含完整的场景信息（地图、轨迹、传感器数据等），可视化效果更好。

---

### 第四步：启动 nuBoard 可视化

#### 方式 1：使用便捷脚本

```bash
bash scripts/view_failure_case.sh work_dirs/failure_viz 5006
```

#### 方式 2：直接运行 nuBoard

```bash
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

然后在浏览器中访问：`http://localhost:5006`

---

## 技术细节

### scenario_name 与 token 的关系

在 nuplan 中：
- `scenario_name` 就是 `token`
- Token 是 16 位十六进制字符串（如 `00cca24d240f5980`）
- Token 对应初始 LIDAR PC (点云) 的唯一标识符

### 真实 Scenario 的获取流程

```
scenario_name (token)
    ↓
ScenarioFilter(scenario_tokens=[token])
    ↓
scenario_builder.get_scenarios(filter)
    ↓
SQL 查询 nuplan 数据库
    ↓
AbstractScenario 对象 (NuPlanScenario)
```

### 数据转换流程

```
SQLite DB (failure_cases.db)
    ↓
查询 scenario_name, log_name, planner_name, history_blob
    ↓
反序列化 SimulationHistory (lzma + pickle)
    ↓
获取 Scenario 对象:
  - 优先: scenario_builder.get_scenarios(token)
  - 降级: StubScenario(元数据)
    ↓
创建 SimulationLog(scenario, planner, history)
    ↓
保存为 .pkl.xz 文件
    ↓
NuBoard 加载可视化
```

### 输出目录结构

符合 nuBoard 标准格式：

```
work_dirs/failure_viz/
├── planTF/                          # planner 名称
│   ├── highway/                     # scenario_type
│   │   └── 2021.07.16.20.45.29_veh-35_01095_01486/  # log_name
│   │       └── 00cca24d240f5980/    # scenario_name
│   │           └── 00cca24d240f5980.pkl.xz  # SimulationLog 文件
│   └── urban/
│       └── ...
└── ...
```

---

## 常见问题

### Q1: 如何知道 scenario_name？

使用 `--list` 参数列出所有失败案例，第一列就是 scenario_name（token）。

### Q2: 必须提供 nuplan 数据库路径吗？

不必须。如果不提供，工具会使用 StubScenario，但可视化效果会受限。提供真实数据库可以获得完整的场景信息。

### Q3: 如何找到对应的 nuplan 数据库文件？

数据库文件路径通常在失败案例的 `log_name` 字段中可以找到线索。例如：
- `log_name`: `2021.07.16.20.45.29_veh-35_01095_01486`
- 对应的数据库文件可能在：`/data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db`

### Q4: 可以批量导出多个 scenarios 吗？

当前版本暂不支持批量导出。如需批量导出，可以编写简单的 shell 脚本循环调用导出工具。

### Q5: 导出失败怎么办？

1. 检查日志输出，了解具体错误
2. 确认数据库文件存在且完整
3. 如果是 scenario_builder 相关错误，尝试不提供数据库路径使用 StubScenario
4. 使用 `--verbose` 参数获取详细日志：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz \
  --verbose
```

---

## 完整示例

### 示例 1：完整工作流（使用真实 scenario）

```bash
# 1. 列出所有失败案例
python scripts/export_failure_cases.py \
  --list \
  --database-path work_dirs/exp/failure_cases.db

# 2. 导出指定 scenario（使用真实数据）
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db

# 3. 启动 nuBoard
bash scripts/view_failure_case.sh work_dirs/failure_viz 5006
```

### 示例 2：简化工作流（使用 StubScenario）

```bash
# 1. 列出所有失败案例
python scripts/export_failure_cases.py \
  --list \
  --database-path work_dirs/exp/failure_cases.db

# 2. 导出指定 scenario（不提供额外数据）
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz

# 3. 启动 nuBoard
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

---

## 代码架构

### 核心文件

| 文件 | 说明 |
|------|------|
| `src/evaluation/failure_case_exporter.py` | 核心导出器，负责数据转换 |
| `src/evaluation/database_manager.py` | 数据库查询管理 |
| `src/evaluation/history_serializer.py` | 历史数据序列化/反序列化 |
| `scripts/export_failure_cases.py` | 命令行工具入口 |
| `scripts/view_failure_case.sh` | nuBoard 启动便捷脚本 |

### 关键类

- `FailureCaseExporter`: 主导出器，支持真实 scenario 和 StubScenario
- `StubScenario`: 轻量级 scenario 对象（不需要额外数据时使用）
- `StubPlanner`: 轻量级 planner 对象
- `DatabaseManager`: SQLite 数据库操作
- `HistorySerializer`: SimulationHistory 序列化/反序列化

---

## 开发和扩展

如需扩展功能，可以修改 `FailureCaseExporter` 类：

- 支持批量导出：修改 `export_multiple()` 方法
- 支持过滤条件：在 `list_failure_cases()` 中添加过滤逻辑
- 自定义输出格式：修改 `export_failure_case()` 中的路径生成逻辑

---

## 更新日志

### v1.1 (当前版本)
- ✅ 支持通过 scenario_name (token) 获取真实的 AbstractScenario 对象
- ✅ 集成 NuPlanScenarioBuilder 动态查询 scenario
- ✅ 自动降级机制：真实 scenario → StubScenario
- ✅ 详细的错误处理和日志输出

### v1.0
- ✅ 基础导出功能
- ✅ StubScenario 支持
- ✅ nuBoard 格式兼容
