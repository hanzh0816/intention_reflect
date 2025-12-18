# 配置系统使用指南 v2.0

## 概述

新的配置系统已重构为**直接对齐 Hydra 配置结构**，配置文件的结构即 Hydra 参数结构，无需额外的映射表。

## 三种配置方式

### 方式1：纯配置组选择

使用顶层字符串值选择配置组：

```yaml
training: "train_planTF"        # → +training=train_planTF
scenario_builder: "nuplan"      # → scenario_builder=nuplan
```

### 方式2：配置组选择 + 参数覆盖（推荐）

使用 `_select` 字段指定配置组，同时覆盖参数：

```yaml
worker:
  _select: "single_machine_thread_pool"  # 选择配置组
  max_workers: 64                         # 覆盖参数
  threads_per_node: 40                    # 额外参数
```

生成的 Hydra 参数：
```bash
worker=single_machine_thread_pool +worker.max_workers=64 +worker.threads_per_node=40
```

### 方式3：纯参数覆盖

直接覆盖配置参数，不选择配置组：

```yaml
cache:
  cache_path: "/data2/hzh/nuplan/train_cache"
  use_cache_without_dataset: true
  cleanup_cache: false
```

生成的 Hydra 参数：
```bash
cache.cache_path=/data2/hzh/nuplan/train_cache cache.use_cache_without_dataset=true ...
```

## 配置文件结构

### 元数据（不会生成 Hydra 参数）

```yaml
name: "config_name"      # 配置名称（用于记录）
version: "2.0"           # 版本号
gpu:
  devices: "0,1,2,3"     # GPU 设备列表
paths:                   # 环境变量路径
  nuplan_data_root: "/path"
  nuplan_maps_root: "/path"
  nuplan_exp_root: "/path"
```

### Hydra 配置（直接对应）

配置结构应与 Hydra 配置结构完全对齐：

```yaml
# 训练参数（顶层）
lr: 1e-3                 # → lr=1e-3
epochs: 25               # → epochs=25
warmup_epochs: 3         # → warmup_epochs=3

# 嵌套参数
data_loader:
  params:
    batch_size: 64       # → data_loader.params.batch_size=64
    num_workers: 64      # → data_loader.params.num_workers=64

# Lightning 参数
lightning:
  trainer:
    params:
      val_check_interval: 1.0  # → lightning.trainer.params.val_check_interval=1.0

# Wandb 参数
wandb:
  mode: "online"         # → wandb.mode=online
  project: "nuplan"      # → wandb.project=nuplan
```

## 智能 + 前缀

系统会自动检测配置键是否存在于 Hydra 配置中：
- 如果**已存在** → `key=value`（无 + 前缀）
- 如果**不存在** → `+key=value`（有 + 前缀）

示例：
```yaml
scenario_builder: "nuplan"  # 已存在 → scenario_builder=nuplan
worker:
  max_workers: 64            # 不存在 → +worker.max_workers=64
```

## 配置模板

### 训练模式（train）

参考：`config/local.example/default_v2.yaml`

必需配置：
- `training`: 训练配置组（如 "train_planTF"）
- `scenario_builder`: 场景构建器
- `scenario_filter`: 场景过滤器

### 缓存模式（cache）

参考：`config/local.example/cache_v2.yaml`

必需配置：
- `py_func: "cache"`: 指定为缓存模式
- `training`: 训练配置组
- `cache.cache_path`: 缓存路径

### 评估模式（eval）

参考：`config/local.example/eval_v2.yaml`

必需配置：
- `scenario_builder`: 通常使用 "nuplan_challenge"
- `scenario_filter`: 评估场景
- `planner.imitation_planner.planner_ckpt`: 检查点路径

## 使用示例

### 1. 创建配置文件

```bash
cp config/local.example/default_v2.yaml config/local/my_config.yaml
# 编辑 my_config.yaml
```

### 2. 运行训练

```bash
./train.sh my_config
```

### 3. 生成缓存

```bash
./cache.sh my_cache_config
```

### 4. 运行评估

```bash
./eval.sh my_eval_config closed_loop_nonreactive_agents
```

## 常见问题

### Q: 如何同时选择配置组和覆盖参数？

A: 使用 `_select` 字段：

```yaml
worker:
  _select: "single_machine_thread_pool"
  max_workers: 64
```

### Q: 配置冲突如何处理？

A: 不能同时使用字符串值和嵌套结构（除非使用 `_select`）：

```yaml
# ❌ 错误（冲突）
worker: "single_machine_thread_pool"
worker:
  max_workers: 64

# ✅ 正确（使用 _select）
worker:
  _select: "single_machine_thread_pool"
  max_workers: 64

# ✅ 正确（仅参数覆盖）
worker:
  max_workers: 64
```

### Q: 如何知道配置结构是否正确？

A: 测试配置加载：

```bash
python3 scripts/load_config.py config/local/my_config.yaml --type train
```

查看生成的 `HYDRA_PARAMS` 是否符合预期。

## 迁移指南

从旧配置（v1.0）迁移到新配置（v2.0）：

### 旧配置
```yaml
data:
  scenario_builder: "nuplan"
  scenario_filter: "training_scenarios_1M"
  cache:
    cache_path: "/path"

worker:
  type: "single_machine_thread_pool"
  max_workers: 64

training:
  batch_size: 64
  lr: 1e-3
```

### 新配置
```yaml
scenario_builder: "nuplan"
scenario_filter: "training_scenarios_1M"

cache:
  cache_path: "/path"

worker:
  _select: "single_machine_thread_pool"
  max_workers: 64

data_loader:
  params:
    batch_size: 64

lr: 1e-3
```

## 参考

- Hydra 文档: https://hydra.cc/
- 配置示例: `config/local.example/`
- 脚本实现: `scripts/load_config.py`
