# 测试故障排除指南

## 常见错误及解决方案

### 1. 模型加载错误：Unexpected key(s) in state_dict

#### 错误信息
```
RuntimeError: Error(s) in loading state_dict for PlanningModel:
    Unexpected key(s) in state_dict: "lateral_intent_head.weight",
    "lateral_intent_head.bias", "longitudinal_intent_head.weight", ...
```

#### 原因
训练时使用了 `intent_enabled=true`，但测试时模型配置没有启用intent功能，导致模型结构不匹配。

#### 解决方案

**已修复**: `config/planner/planTF.yaml` 已经更新，包含intent参数。

如果你修改过配置或使用了不同的训练配置，请确保：

1. **检查训练时的配置**:
   ```bash
   # 查看训练时使用的配置
   cat /data2/hzh/nuplan/exp/planTF/<timestamp>/code/hydra/config.yaml | grep intent
   ```

2. **确保planner配置匹配**:
   编辑 `config/planner/planTF.yaml`:
   ```yaml
   planner:
     intent_enabled: true  # 必须与训练时一致
     intent_time_horizon: 2.0
     intent_embed_dim: 64
     lateral_classes: 5
     longitudinal_classes: 4
   ```

3. **或者在命令行覆盖** (不推荐，容易遗漏):
   ```bash
   python run_simulation.py \
       ... \
       planner.imitation_planner.planner.intent_enabled=true \
       planner.imitation_planner.planner.intent_time_horizon=2.0
   ```

### 2. 开环测试：Train fraction错误

#### 错误信息
```
AssertionError: Train fraction has to be larger than 0!
```

#### 原因
`CustomDataModule` 要求 `train_fraction > 0`，但脚本试图只使用测试集（`train_fraction=0.0`）。

#### 解决方案

**已修复**: `test_open_loop.sh` 已更新为使用 `py_func=validate` 在验证集上评估。

验证集评估的优点：
- ✅ Cache包含完整的validation split
- ✅ 与训练时的验证一致，便于对比
- ✅ 避免datamodule的限制

如果确实需要使用测试集，修改 `src/custom_training/custom_datamodule.py:133`:
```python
# 将 assert train_fraction > 0.0 改为:
assert train_fraction >= 0.0, "Train fraction has to be larger/equal than 0!"
```

### 3. 找不到checkpoint文件

#### 错误信息
```
Error: Checkpoint not found at /path/to/checkpoint.ckpt
```

#### 解决方案
```bash
# 使用find_checkpoints.sh查找可用的checkpoints
sh ./find_checkpoints.sh

# 或手动查找
find /data2/hzh/nuplan/exp/planTF -name "*.ckpt"
```

### 4. CUDA out of memory (OOM)

#### 错误信息
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

#### 解决方案

**闭环测试中**:
```bash
# 减少GPU分配
number_of_gpus_allocated_per_simulation=0.125  # 从0.25降到0.125

# 或使用CPU
number_of_gpus_allocated_per_simulation=0
```

**开环测试中**:
```bash
# 减小batch size
data_loader.params.batch_size=4  # 从16降到4或8
```

### 5. 场景数据库找不到

#### 错误信息
```
FileNotFoundError: No such file or directory: '/data2/hzh/nuplan/dataset/...'
```

#### 解决方案
确保环境变量正确设置：
```bash
export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

# 验证路径存在
ls $NUPLAN_DATA_ROOT
ls $NUPLAN_MAPS_ROOT
ls $NUPLAN_DB_FILES
```

### 6. ImportError: Cannot import module

#### 错误信息
```
ImportError: cannot import name 'XXX' from 'src.xxx'
```

#### 解决方案
确保PYTHONPATH包含项目根目录：
```bash
export PYTHONPATH=/home/hzh/code/planning/planTF:$PYTHONPATH

# 或在Python脚本开头添加
import sys
sys.path.insert(0, '/home/hzh/code/planning/planTF')
```

### 7. 仿真结果没有生成

#### 问题
仿真运行完成但找不到结果文件

#### 解决方案
```bash
# 检查输出目录
echo $NUPLAN_EXP_ROOT

# 查找最近的仿真结果
find $NUPLAN_EXP_ROOT -name "aggregator_metric" -type d -mtime -1

# 使用view_simulation_results.py自动查找
python view_simulation_results.py --root $NUPLAN_EXP_ROOT
```

### 8. 仿真速度太慢

#### 解决方案

1. **使用非响应式agents** (已在test_closed_loop.sh中使用):
   ```bash
   +simulation=closed_loop_nonreactive_agents
   ```

2. **减少场景数量**:
   ```bash
   # 使用test14-random（280个）而不是val14（1400个）
   scenario_filter=test14-random

   # 或限制总场景数
   scenario_filter.limit_total_scenarios=50
   ```

3. **并行执行** (需要多GPU):
   ```bash
   worker=ray_distributed
   worker.threads_per_node=8
   ```

### 9. 模型参数数量不匹配

#### 错误信息
```
RuntimeError: Error(s) in loading state_dict for PlanningModel:
    Missing key(s) in state_dict: "xxx"
    Unexpected key(s) in state_dict: "yyy"
```

#### 原因
模型架构配置（dim, num_modes等）与checkpoint不匹配

#### 解决方案

1. **检查训练配置**:
   ```bash
   # 查看训练时的模型配置
   cat /data2/hzh/nuplan/exp/planTF/<timestamp>/code/hydra/config.yaml | grep -A 20 "model:"
   ```

2. **确保所有参数匹配**:
   - `dim`: 特征维度（默认128）
   - `num_modes`: 轨迹模态数（默认6）
   - `history_steps`: 历史步数（默认21）
   - `future_steps`: 预测步数（默认80）
   - `encoder_depth`: 编码器深度（默认4）
   - `num_heads`: 注意力头数（默认8）
   - 所有intent相关参数

3. **对比配置文件**:
   ```bash
   # 对比训练和测试配置
   diff <(cat /data2/hzh/nuplan/exp/planTF/<timestamp>/code/hydra/config.yaml | grep -A 30 "model:") \
        <(cat config/planner/planTF.yaml | grep -A 30 "planner:")
   ```

### 10. 验证集/测试集加载失败

#### 错误信息
```
AssertionError: Splitter returned no validation/test samples
```

#### 解决方案

1. **检查数据划分配置**:
   ```bash
   # 确认使用正确的splitter
   splitter=nuplan
   ```

2. **确认scenario_filter设置**:
   ```bash
   # 对于闭环测试，使用专门的test场景
   scenario_filter=test14-random

   # 不要使用training_scenarios_1M（只包含train split）
   ```

### 11. Intent标签缺失

#### 错误信息
```
KeyError: 'intent' not in targets
```

#### 原因
使用的cache在生成时没有启用intent

#### 解决方案

1. **检查cache是否包含intent**:
   ```bash
   python check_intent_labels.py
   ```

2. **如果没有，重新生成cache**:
   ```bash
   sh ./cache.sh
   ```

3. **临时方案：使用不需要intent的checkpoint**:
   - 训练一个 `intent_enabled=false` 的模型
   - 或修改代码使intent变为可选

## 快速诊断流程

遇到问题时，按以下顺序检查：

1. ✅ **环境变量**: `echo $NUPLAN_DATA_ROOT`
2. ✅ **Checkpoint存在**: `ls /path/to/checkpoint.ckpt`
3. ✅ **配置匹配**: 对比训练和测试配置
4. ✅ **PYTHONPATH**: `echo $PYTHONPATH`
5. ✅ **依赖安装**: `pip list | grep -E "torch|nuplan"`
6. ✅ **显存充足**: `nvidia-smi`

## 获取帮助

如果以上都无法解决问题：

1. 查看完整错误堆栈
2. 检查日志文件: `<output_dir>/log.txt`
3. 尝试在单个场景上调试
4. 对比官方示例配置

## 调试技巧

### 测试单个场景
```bash
python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    planner.imitation_planner.planner_ckpt=/path/to/checkpoint.ckpt \
    scenario_filter.limit_total_scenarios=1 \
    scenario_filter.scenario_types=['changing_lane'] \
    scenario_builder=nuplan
```

### 查看详细日志
```bash
# 运行时添加verbose标志
python run_simulation.py ... --verbose

# 或设置日志级别
logger_level=debug
```

### 检查模型结构
```python
import torch
from src.models.planTF.planning_model import PlanningModel

# 加载checkpoint查看keys
ckpt = torch.load('/path/to/checkpoint.ckpt')
print("Checkpoint keys:", ckpt.keys())
if 'state_dict' in ckpt:
    print("Model keys:", ckpt['state_dict'].keys())

# 创建模型查看结构
model = PlanningModel(
    intent_enabled=True,
    intent_time_horizon=2.0,
    # ... 其他参数
)
print("Model structure:", model)
```
