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

#### 导出模式

工具支持三种导出模式：

1. **单个 scenario**: 导出指定的一个 failure case
2. **多个 scenarios**: 导出指定的多个 failure cases
3. **所有 scenarios**: 导出数据库中的所有 failure cases

所有导出都会保存到同一个目录（默认 `work_dirs/failure_viz`），方便在 nuBoard 中一起可视化。

#### 方式 1：导出单个 scenario

提供 nuplan 数据库路径，工具会自动通过 scenario_name（token）查询真实的 AbstractScenario 对象：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db
```

#### 方式 2：导出多个 scenarios（新功能 ✨）

通过空格分隔多个 scenario tokens：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1" \
  --database-path work_dirs/exp/failure_cases.db
```

#### 方式 3：导出所有 failure cases（新功能 ✨）

使用 `--all` 参数导出数据库中的所有 failure cases：

```bash
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db
```

#### 使用真实 scenario（可选）

对于任何导出模式，都可以添加 nuplan 数据库参数来获取真实的 scenario 对象：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db

# 或者导出所有 failure cases
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db
```

**参数说明：**

- `--scenario-name`: 要导出的 scenario token（可指定多个，用空格分隔）
- `--all`: 导出所有 failure cases
- `--database-path`: failure case 数据库路径
- `--output-dir`: 输出目录（默认：`work_dirs/failure_viz`）
- `--data-root`: nuplan 数据集根目录（可选）
- `--map-root`: nuplan 地图文件目录（可选）
- `--db-files`: nuplan 数据库文件路径（可选，可以用逗号分隔多个文件）
- `--sensor-root`: (可选) 传感器数据路径
- `--map-version`: (可选) 地图版本，默认 `nuplan-maps-v1.0`

#### 使用 StubScenario（简化方式）

如果不提供 nuplan 数据库路径，工具会自动使用轻量级的 StubScenario：

```bash
# 单个 scenario
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db

# 多个 scenarios
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" \
  --database-path work_dirs/exp/failure_cases.db

# 所有 failure cases
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db
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
保存为 .pkl.xz 文件到 simulation/ 子目录
    ↓
创建 .nuboard 元数据文件
    ↓
NuBoard 加载可视化
```

### 输出目录结构

符合 nuBoard 标准格式：

```
work_dirs/failure_viz/
├── nuboard_1703123456.nuboard    # NuBoard 元数据文件（必需）
├── simulation/                    # 仿真日志目录
│   └── planTF/                    # planner 名称
│       ├── highway/               # scenario_type
│       │   └── 2021.07.16.20.45.29_veh-35_01095_01486/  # log_name
│       │       └── 00cca24d240f5980/  # scenario_name
│       │           └── 00cca24d240f5980.pkl.xz  # SimulationLog 文件
│       └── urban/
│           └── ...
└── metrics/                       # 空目录（无指标数据）
    └── aggregator_metric/         # 空目录
```

**重要说明：**
- `.nuboard` 文件是 nuBoard 的元数据文件，包含指向模拟日志的路径信息
- `simulation/` 子目录用于隔离仿真日志和元数据文件，避免目录遍历错误
- 导出工具会自动创建这个标准结构
- nuBoard 通过读取 `.nuboard` 文件来定位 `simulation/` 目录下的 SimulationLog 文件

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

✅ **v1.3+ 支持批量导出！** 有三种方式：

**方式 1: 指定多个 scenario tokens**
```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1" \
  --database-path work_dirs/exp/failure_cases.db
```

**方式 2: 导出所有 failure cases**
```bash
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db
```

**方式 3: 使用 shell 脚本（适用于旧版本）**
```bash
for scenario in "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1"; do
  python scripts/export_failure_cases.py \
    --scenario-name "$scenario" \
    --database-path work_dirs/exp/failure_cases.db
done
```

### Q5: nuBoard 报错 "No available nuBoard files are found" 怎么办？

这是因为导出目录中缺少 `.nuboard` 元数据文件。从 v1.2 版本开始，导出工具会自动创建这个文件。

**解决方案：**
1. 确保使用最新版本的导出工具
2. 重新导出 failure case，会自动生成 `.nuboard` 文件
3. 检查导出目录中是否存在 `nuboard_*.nuboard` 文件

**手动创建 .nuboard 文件（不推荐）：**
```python
from pathlib import Path
from nuplan.planning.nuboard.base.data_class import NuBoardFile

output_dir = Path("work_dirs/failure_viz")
nuboard_file = NuBoardFile(
    simulation_main_path=str(output_dir),
    simulation_folder="simulation",  # 注意：使用 "simulation" 子目录
    metric_main_path=str(output_dir),
    metric_folder="metrics",
    aggregator_metric_folder="aggregator_metric",
)
nuboard_file.save_nuboard_file(output_dir / "nuboard.nuboard")
```

### Q6: 导出失败怎么办？

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

### 示例 1：导出所有 failure cases（推荐）

```bash
# 1. 列出所有失败案例
python scripts/export_failure_cases.py \
  --list \
  --database-path work_dirs/exp/failure_cases.db

# 2. 导出所有 failure cases（使用默认输出目录）
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db

# 3. 启动 nuBoard
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 示例 2：导出指定的多个 scenarios

```bash
# 1. 列出所有失败案例，选择感兴趣的几个
python scripts/export_failure_cases.py \
  --list \
  --database-path work_dirs/exp/failure_cases.db

# 2. 导出指定的 scenarios
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1" \
  --database-path work_dirs/exp/failure_cases.db

# 3. 启动 nuBoard
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 示例 3：使用真实 scenario 导出

```bash
# 导出所有 failure cases，使用真实的 scenario 对象
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db

# 启动 nuBoard
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 示例 4：增量导出（追加更多 scenarios）

```bash
# 第一次导出部分 scenarios
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" \
  --database-path work_dirs/exp/failure_cases.db

# 后续追加更多 scenarios（会重用现有的 .nuboard 文件）
python scripts/export_failure_cases.py \
  --scenario-name "02bcd4f6231b5cd1" "03def789abcd1234" \
  --database-path work_dirs/exp/failure_cases.db

# 所有导出都在同一个目录下，可以一起查看
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

### v1.3 (当前版本)
- ✅ **批量导出功能**：支持导出多个或所有 failure cases
- ✅ 支持三种导出模式：单个、多个、全部
- ✅ 统一输出目录：所有导出保存到同一个 `work_dirs/failure_viz`，方便一起可视化
- ✅ 智能 `.nuboard` 文件管理：重用现有文件，避免重复创建
- ✅ 默认输出目录：`--output-dir` 现在默认为 `work_dirs/failure_viz`

### v1.2.1
- ✅ **修复目录遍历错误**：使用标准的 `simulation/` 子目录结构
- ✅ 避免 `.nuboard` 文件和 planner 目录混在一起导致的 `NotADirectoryError`
- ✅ 完全兼容 nuBoard 标准目录结构

### v1.2
- ✅ **修复 nuBoard 无法加载问题**：自动生成 `.nuboard` 元数据文件
- ✅ 导出工具现在会自动创建 nuBoard 所需的元数据文件
- ✅ 无需手动创建 `.nuboard` 文件

### v1.1
- ✅ 支持通过 scenario_name (token) 获取真实的 AbstractScenario 对象
- ✅ 集成 NuPlanScenarioBuilder 动态查询 scenario
- ✅ 自动降级机制：真实 scenario → StubScenario
- ✅ 详细的错误处理和日志输出

### v1.0
- ✅ 基础导出功能
- ✅ StubScenario 支持
- ✅ nuBoard 格式兼容
