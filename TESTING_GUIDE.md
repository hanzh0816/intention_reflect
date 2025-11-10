# PlanTF模型测试指南

本文档说明如何对训练好的PlanTF模型进行开环和闭环测试。

## 前提条件

1. 已完成模型训练，并有checkpoint文件（.ckpt）
2. Checkpoint位置通常在：`/data2/hzh/nuplan/exp/planTF/<timestamp>/checkpoints/`
3. 确保环境变量已正确配置

## 一、开环测试（Open-loop Testing）

### 什么是开环测试？

开环测试直接在数据集上评估模型的轨迹预测精度，不执行仿真：
- 使用recorded数据中的历史信息
- 模型预测未来轨迹
- 与expert轨迹对比，计算指标
- **优点**：快速、可重复
- **缺点**：无法评估真实驾驶场景中的交互表现

### 评估指标

- **minADE1**: 最佳mode的平均位移误差
- **minADE6**: Top-6 modes中最佳的平均位移误差
- **minFDE1**: 最佳mode的终点位移误差
- **minFDE6**: Top-6 modes中最佳的终点位移误差
- **MR**: Miss Rate（轨迹偏离率）

### 运行开环测试

```bash
# 使用test_open_loop.sh脚本
sh ./test_open_loop.sh /path/to/checkpoint.ckpt

# 例如：
sh ./test_open_loop.sh /data2/hzh/nuplan/exp/planTF/2025.11.09.23-24-55/checkpoints/epoch=10.ckpt
```

### 使用验证集进行评估

```bash
python run_training.py \
    py_func=validate \
    +training=train_planTF \
    cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M \
    cache.use_cache_without_dataset=true \
    model.intent_enabled=true \
    checkpoint=/path/to/checkpoint.ckpt \
    data_loader.params.batch_size=16
```

## 二、闭环测试（Closed-loop Testing）

### 什么是闭环测试？

闭环测试在仿真环境中执行模型预测的轨迹：
- 模型的预测会影响ego车辆的行为
- Ego车辆行为会影响后续观测
- 形成闭环反馈
- **优点**：评估真实驾驶性能
- **缺点**：计算成本高、需要仿真环境

### 仿真模式

**Non-reactive Agents** (推荐用于初步测试):
- Agent按照logged轨迹运动
- 不响应ego车辆的动作
- 计算快速，适合大规模评估

**Reactive Agents** (更真实):
- Agent使用IDM等模型响应ego车辆
- 更接近真实场景
- 计算成本高

### 评估指标

仿真会计算多个metrics，包括：
- **no_at_fault_collisions**: 无责碰撞分数
- **drivable_area_compliance**: 可行驶区域遵守度
- **driving_direction_compliance**: 行驶方向遵守度
- **speed_limit_compliance**: 速限遵守度
- **progress**: 任务完成度
- **time_to_collision_within_bound**: TTC安全指标
- **comfort**: 舒适度（加速度、急转等）

### 运行闭环测试

#### 1. 使用test脚本（推荐）

```bash
# 使用test14-random场景（280个场景，14种类型各20个）
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt test14-random

# 使用test14-hard场景（更困难的场景）
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt test14-hard

# 使用val14场景（验证集中的1400个场景）
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt val14
```

#### 2. 手动运行

```bash
python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    planner.imitation_planner.planner_ckpt=/path/to/checkpoint.ckpt \
    scenario_filter=test14-random \
    scenario_builder=nuplan \
    observation=box_observation \
    ego_controller=perfect_tracking_controller \
    number_of_gpus_allocated_per_simulation=0.25 \
    worker=sequential \
    experiment_name=my_simulation_test
```

### 查看结果

仿真完成后，结果保存在：
```
${NUPLAN_EXP_ROOT}/simulation_results/<experiment_name>/<timestamp>/
├── aggregator_metric/           # 汇总指标
│   └── *.parquet               # 包含所有场景的评分
├── simulation_log/              # 每个场景的详细log
│   └── <scenario_type>/<scenario_token>/
│       └── *.msgpack.xz        # 仿真轨迹数据
└── code/                        # 配置信息
```

查看汇总结果：
```python
import pandas as pd

df = pd.read_parquet('path/to/aggregator_metric/*.parquet')
final_score = df[df['scenario'] == 'final_score']
print(final_score)
```

或使用脚本中的自动打印功能。

## 三、场景选择

### 可用的scenario_filter

1. **test14-random**:
   - 14种场景类型，每种20个
   - 总共280个场景
   - 用于快速评估

2. **test14-hard**:
   - 14种场景类型的困难版本
   - 筛选出更具挑战性的场景

3. **val14**:
   - 验证集中的1400个场景（每种类型100个）
   - 更全面的评估

4. **自定义**:
   创建自己的scenario_filter配置文件在`config/scenario_filter/`

### 场景类型

包含的14种场景类型：
- `starting_left_turn` - 左转起步
- `starting_right_turn` - 右转起步
- `starting_straight_traffic_light_intersection_traversal` - 直行通过交叉口
- `stopping_with_lead` - 跟车停止
- `high_lateral_acceleration` - 高横向加速度
- `high_magnitude_speed` - 高速
- `low_magnitude_speed` - 低速
- `traversing_pickup_dropoff` - 通过上下车点
- `waiting_for_pedestrian_to_cross` - 等待行人过马路
- `behind_long_vehicle` - 跟随长车
- `stationary_in_traffic` - 交通中静止
- `near_multiple_vehicles` - 多车辆附近
- `changing_lane` - 变道
- `following_lane_with_lead` - 跟车

## 四、完整测试流程示例

### 1. 找到最佳checkpoint

```bash
# 查看训练结果，找到验证集表现最好的checkpoint
ls -lh /data2/hzh/nuplan/exp/planTF/*/checkpoints/
```

通常文件名包含epoch和val_minFDE，例如：
```
epoch=15-val_minFDE=1.234.ckpt
```

### 2. 快速开环测试

```bash
# 首先在验证集上做快速评估
CKPT=/data2/hzh/nuplan/exp/planTF/2025.11.09.23-24-55/checkpoints/epoch=15-val_minFDE=1.234.ckpt
sh ./test_open_loop.sh $CKPT
```

### 3. 闭环仿真（快速版）

```bash
# 使用test14-random进行快速闭环测试（~280个场景）
sh ./test_closed_loop.sh $CKPT test14-random
```

### 4. 完整闭环仿真（可选）

```bash
# 使用val14进行完整评估（~1400个场景）
# 这可能需要几个小时
sh ./test_closed_loop.sh $CKPT val14
```

## 五、性能优化

### 并行仿真

如果有多个GPU，可以并行运行仿真：

```bash
python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    planner.imitation_planner.planner_ckpt=/path/to/checkpoint.ckpt \
    scenario_filter=val14 \
    worker=ray_distributed \
    worker.threads_per_node=16 \
    number_of_gpus_allocated_per_simulation=0.25
```

### 调试模式

测试单个场景：
```bash
python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    planner.imitation_planner.planner_ckpt=/path/to/checkpoint.ckpt \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=1 \
    scenario_filter.scenario_types=['changing_lane']
```

## 六、常见问题

### Q1: checkpoint路径错误
确保checkpoint文件存在：
```bash
ls -lh /path/to/checkpoint.ckpt
```

### Q2: CUDA out of memory
减少batch size或GPU分配：
```bash
data_loader.params.batch_size=8
number_of_gpus_allocated_per_simulation=0.125
```

### Q3: 想要可视化结果
需要使用nuBoard工具：
```bash
python run_nuboard.py \
    simulation_path=/path/to/simulation_results
```

## 七、参考资料

- NuPlan官方文档: https://nuplan-devkit.readthedocs.io/
- Simulation配置: `nuplan-devkit/nuplan/planning/script/config/simulation/`
- Metric定义: `nuplan-devkit/nuplan/planning/metrics/`
