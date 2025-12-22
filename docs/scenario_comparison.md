# 真实 Scenario vs StubScenario 详细对比

## 概念区别

### 真实 Scenario（NuPlanScenario）
**定义**：从 nuplan 原始数据库中完整加载的场景对象，包含该场景的所有历史数据。

**数据来源**：nuplan 数据库文件（.db）中的完整记录
```
/data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db
```

### StubScenario
**定义**：轻量级的 scenario 占位对象，只包含最基本的元数据。

**数据来源**：failure case 数据库中的元数据字段（scenario_type, scenario_name, log_name）+ SimulationHistory 中的 map_api

---

## 详细功能对比表

| 功能分类 | 真实 Scenario | StubScenario | 说明 |
|---------|-------------|-------------|------|
| **基本属性** |
| scenario_name | ✅ 从数据库读取 | ✅ 从元数据读取 | 都支持 |
| scenario_type | ✅ 从数据库读取 | ✅ 从元数据读取 | 都支持 |
| log_name | ✅ 从数据库读取 | ✅ 从元数据读取 | 都支持 |
| token | ✅ 真实 token | ✅ 使用 scenario_name | 都支持 |
| map_api | ✅ 从 map 数据库加载 | ✅ 从 SimulationHistory 获取 | 都支持 |
| database_interval | ✅ 0.05s (20Hz) | ✅ 固定 0.05s | 都支持 |
| **场景数据查询** |
| get_number_of_iterations() | ✅ 从数据库查询帧数 | ❌ 返回 0 | 真实 scenario 知道场景有多少帧 |
| get_time_point(iteration) | ✅ 查询指定帧时间戳 | ❌ NotImplementedError | 无法查询 |
| get_mission_goal() | ✅ 从数据库查询目标点 | ❌ 返回 None | 无任务目标 |
| get_expert_goal_state() | ✅ 专家轨迹终点 | ❌ NotImplementedError | 无法查询 |
| get_route_roadblock_ids() | ✅ 完整路由信息 | ❌ 返回空列表 [] | 无路由信息 |
| **自车状态** |
| get_ego_state_at_iteration(i) | ✅ 查询任意帧自车状态 | ❌ NotImplementedError | 无法查询历史状态 |
| get_ego_past_trajectory() | ✅ 查询过去轨迹 | ❌ NotImplementedError | 无法查询 |
| get_ego_future_trajectory() | ✅ 查询未来轨迹 | ❌ NotImplementedError | 无法查询 |
| **障碍物** |
| get_tracked_objects_at_iteration(i) | ✅ 查询障碍物 | ❌ NotImplementedError | 无法查询障碍物 |
| get_tracked_objects_within_time_window() | ✅ 时间窗口内障碍物 | ❌ NotImplementedError | 无法查询 |
| get_past_tracked_objects() | ✅ 过去障碍物轨迹 | ❌ NotImplementedError | 无法查询 |
| get_future_tracked_objects() | ✅ 未来障碍物轨迹 | ❌ NotImplementedError | 无法查询 |
| **传感器数据** |
| get_sensors_at_iteration(i) | ✅ LIDAR 点云 + 相机图像 | ❌ NotImplementedError | 无法获取传感器数据 |
| get_lidar_to_ego_transform() | ✅ LIDAR 到自车的变换 | ❌ NotImplementedError | 无法查询 |
| **时间戳** |
| get_past_timestamps() | ✅ 过去时间戳序列 | ❌ NotImplementedError | 无法查询 |
| get_future_timestamps() | ✅ 未来时间戳序列 | ❌ NotImplementedError | 无法查询 |
| **交通灯** |
| get_traffic_light_status_at_iteration(i) | ✅ 交通灯状态 | ❌ NotImplementedError | 无法查询 |

---

## 内部实现对比

### 真实 Scenario（NuPlanScenario）

```python
class NuPlanScenario(AbstractScenario):
    def __init__(
        self,
        data_root: str,                     # nuplan 数据根目录
        log_file_load_path: str,            # 数据库文件路径
        initial_lidar_token: str,           # 初始 token
        initial_lidar_timestamp: int,       # 初始时间戳
        scenario_type: str,
        map_root: str,                      # 地图根目录
        map_version: str,
        map_name: str,
        scenario_extraction_info: Optional[ScenarioExtractionInfo],
        ego_vehicle_parameters: VehicleParameters,
        sensor_root: Optional[str] = None,
    ):
        # 保存数据库文件路径
        self._log_file = download_file_if_necessary(data_root, log_file_load_path)
        # ... 其他初始化

    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        """从数据库查询自车状态"""
        return get_ego_state_for_lidarpc_token_from_db(
            self._log_file,                  # 查询数据库文件
            self._lidarpc_tokens[iteration]  # 指定帧的 token
        )

    def get_tracked_objects_at_iteration(self, iteration: int) -> DetectionsTracks:
        """从数据库查询障碍物"""
        return DetectionsTracks(
            extract_tracked_objects(
                self._lidarpc_tokens[iteration],
                self._log_file,              # 查询数据库文件
                future_trajectory_sampling
            )
        )

    def get_sensors_at_iteration(self, iteration: int) -> Sensors:
        """从数据库查询传感器数据（LIDAR 点云、相机图像）"""
        lidar_pc = next(
            get_sensor_data_from_sensor_data_tokens_from_db(
                self._log_file,              # 查询数据库文件
                get_lidarpc_sensor_data(),
                LidarPc,
                [self._lidarpc_tokens[iteration]]
            )
        )
        return self._get_sensor_data_from_lidar_pc(lidar_pc, channels)
```

### StubScenario（我们实现的）

```python
class StubScenario(AbstractScenario):
    def __init__(
        self,
        scenario_type: str,      # 仅基本元数据
        scenario_name: str,
        log_name: str,
        map_api: AbstractMap,    # 从 SimulationHistory 获取
        database_interval: float = 0.05,
    ):
        # 仅保存元数据，没有数据库文件
        self._scenario_type = scenario_type
        self._scenario_name = scenario_name
        self._log_name = log_name
        self._map_api = map_api
        self._database_interval = database_interval

    # 基本属性可以返回
    @property
    def scenario_name(self) -> str:
        return self._scenario_name

    @property
    def map_api(self) -> AbstractMap:
        return self._map_api

    # 大部分查询方法都不支持
    def get_ego_state_at_iteration(self, iteration: int) -> EgoState:
        raise NotImplementedError("StubScenario does not support get_ego_state_at_iteration")

    def get_tracked_objects_at_iteration(self, iteration: int) -> Any:
        raise NotImplementedError("StubScenario does not support get_tracked_objects_at_iteration")

    def get_sensors_at_iteration(self, iteration: int) -> Any:
        raise NotImplementedError("StubScenario does not support get_sensors_at_iteration")
```

---

## 为什么 StubScenario 能工作？

### 关键：SimulationHistory 已经包含了所有运行时数据

当我们保存 failure case 时，`SimulationHistory` 已经包含了仿真过程中的所有数据：

```python
class SimulationHistory:
    data: List[SimulationHistorySample]  # 每一帧的完整数据

class SimulationHistorySample:
    # 这一帧的所有数据
    observation: Observation              # 包含 ego_state, tracked_objects, sensors, map_api
    ego_state: EgoState                   # 自车状态
    trajectory: AbstractTrajectory        # 规划轨迹
    # ... 等等
```

**nuBoard 主要使用 SimulationHistory 中的数据进行可视化：**
- 自车轨迹：从 `SimulationHistory.data[i].ego_state`
- 障碍物：从 `SimulationHistory.data[i].observation.tracked_objects`
- 地图：从 `SimulationHistory.data[0].observation.map_api`
- 规划轨迹：从 `SimulationHistory.data[i].trajectory`

**Scenario 对象主要提供元数据：**
- scenario_name, scenario_type, log_name
- map_api（但 SimulationHistory 中也有）

所以即使使用 StubScenario，只要 SimulationHistory 完整，nuBoard 也能正常可视化！

---

## 在 nuBoard 中的实际使用差异

### 使用真实 Scenario

```python
simulation_log = SimulationLog(
    scenario=real_scenario,          # NuPlanScenario
    planner=planner,
    simulation_history=history,
)

# nuBoard 可以：
# 1. 显示完整的场景信息（scenario_type, 路由等）
# 2. 可以查询原始场景数据（如果需要）
# 3. 可以对比专家轨迹 vs 规划轨迹
# 4. 显示 mission goal
# 5. 可视化效果更丰富
```

### 使用 StubScenario

```python
simulation_log = SimulationLog(
    scenario=stub_scenario,          # StubScenario
    planner=planner,
    simulation_history=history,      # 包含所有运行时数据
)

# nuBoard 可以：
# 1. 显示基本场景信息（scenario_name, scenario_type, log_name）
# 2. 可视化自车轨迹（从 SimulationHistory）
# 3. 可视化障碍物（从 SimulationHistory）
# 4. 可视化地图（从 SimulationHistory.map_api）
# 5. 可视化规划轨迹（从 SimulationHistory）

# nuBoard 不能：
# 1. 查询原始场景数据（因为 StubScenario 不支持）
# 2. 对比专家轨迹（需要真实 scenario 查询）
# 3. 显示详细的 mission goal
```

---

## 数据流对比图

### 真实 Scenario 数据流

```
原始 nuplan 数据库 (.db)
    ↓
NuPlanScenario.get_ego_state_at_iteration(i)
    ↓
SQL 查询数据库
    ↓
返回 EgoState, TrackedObjects, Sensors 等
    ↓
nuBoard 可视化（可查询任意帧的原始数据）
```

### StubScenario 数据流

```
SimulationHistory（已保存的运行时数据）
    ↓
SimulationHistory.data[i].observation
    ↓
直接读取 EgoState, TrackedObjects, Sensors
    ↓
nuBoard 可视化（只能查看已保存的数据）
```

---

## 何时使用哪种？

### 使用真实 Scenario（推荐）

**优点**：
- ✅ 可以查询原始场景的完整数据
- ✅ 可以对比专家轨迹
- ✅ 可视化效果更丰富
- ✅ 支持所有 nuBoard 功能

**缺点**：
- ❌ 需要提供 nuplan 数据库文件路径
- ❌ 需要更多存储空间（数据库文件）
- ❌ 需要正确配置 data_root, map_root, db_files

**适用场景**：
- 详细分析失败案例
- 对比规划器 vs 专家驾驶
- 需要查询原始场景数据
- 需要完整的可视化效果

### 使用 StubScenario

**优点**：
- ✅ 不需要额外的数据库文件
- ✅ 配置简单，一条命令即可
- ✅ 占用存储少
- ✅ 基本可视化功能完全够用

**缺点**：
- ❌ 无法查询原始场景数据
- ❌ 无法对比专家轨迹
- ❌ 缺少部分高级可视化功能

**适用场景**：
- 快速查看失败案例
- 不需要原始场景数据
- 只关注规划器的表现
- 简化部署（无需完整 nuplan 数据集）

---

## 实际示例

### 真实 Scenario 示例

```bash
# 提供完整参数
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz \
  --data-root /data/sets/nuplan \
  --map-root /data/sets/nuplan/maps \
  --db-files /data/sets/nuplan/nuplan-v1.1/mini/2021.07.16.20.45.29_veh-35_01095_01486.db

# 输出日志：
# INFO - Building scenario_builder with nuplan database...
# INFO - ✓ Scenario builder created successfully
# INFO - Retrieved real scenario: 00cca24d240f5980 (type: lane_follow, log: ...)
# INFO - Using real scenario from scenario_builder
```

### StubScenario 示例

```bash
# 不提供 nuplan 数据库参数
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz

# 输出日志：
# INFO - Scenario builder parameters not provided, will use StubScenario
# INFO - scenario_builder not available or scenario not found, using StubScenario
```

---

## 技术总结

| 维度 | 真实 Scenario | StubScenario |
|-----|-------------|-------------|
| **数据完整性** | 完整的原始场景数据 | 仅元数据 + SimulationHistory |
| **功能支持** | 全部 AbstractScenario 方法 | 仅基本属性，其他 NotImplementedError |
| **可视化质量** | 完整可视化 + 专家对比 | 基础可视化（足够用） |
| **配置复杂度** | 需要多个路径参数 | 一个数据库路径即可 |
| **存储需求** | 需要原始 .db 文件（GB 级） | 仅 failure_cases.db（MB 级） |
| **适用场景** | 深度分析、论文研究 | 日常调试、快速查看 |

---

## 结论

- **真实 Scenario** = 完整功能，需要原始数据
- **StubScenario** = 轻量级替代，基于已保存的 SimulationHistory

两者都能满足 nuBoard 可视化的需求，选择取决于你的具体需求：
- 需要完整功能 → 真实 Scenario
- 快速简单查看 → StubScenario

**我们的实现支持两种方式自动切换**，优先使用真实 Scenario，自动降级到 StubScenario！
