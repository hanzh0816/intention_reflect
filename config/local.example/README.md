# 本地配置使用指南

本目录提供了配置模板示例。请按照以下步骤使用：

## 快速开始

### 1. 复制配置模板到本地目录

```bash
# 复制默认配置作为起点
cp config/local.example/default.yaml config/local/default.yaml

# （可选）复制其他预设配置
cp config/local.example/cache_default.yaml config/local/cache_default.yaml
cp config/local.example/eval_default.yaml config/local/eval_default.yaml
```

### 2. 编辑本地配置

打开 `config/local/default.yaml` 编辑你的个人配置：
- 修改 `gpu.devices` 为你使用的GPU
- 修改 `training.batch_size` 和 `num_workers` 根据你的硬件
- 修改 `wandb.mode` 和 `wandb.name` 等wandb设置
- 其他训练参数根据需要调整

### 3. 启动训练

```bash
# 使用默认配置启动训练
./train.sh

# 或指定特定配置
./train.sh debug
./train.sh full_train
./train.sh quick_test

# 或指定其他自定义配置
./train.sh my_custom_config
```

### 4. 启动评估

```bash
# 使用默认配置进行评估（一次评估一个challenge）
./eval.sh eval_default closed_loop_nonreactive_agents
./eval.sh eval_default closed_loop_reactive_agents
./eval.sh eval_default open_loop_boxes

# 或指定特定配置
./eval.sh my_eval_config closed_loop_nonreactive_agents
```

## 配置说明

### default.yaml
通用的默认配置，包含常用的参数设置。

### debug.yaml
快速调试配置，用于验证代码逻辑：
- 单GPU运行（GPU 0）
- 小batch_size（8）
- 少量epoch（2）
- Wandb离线模式

### full_train.yaml
完整训练配置，用于生产级训练：
- 多GPU运行（GPU 1,4,5,6）
- 较大batch_size（128）
- 更多epoch（50）
- Wandb在线模式

### quick_test.yaml
快速测试配置，用于验证和测试：
- 单GPU运行（GPU 0）
- 中等batch_size（32）
- 仅验证（epochs=1）
- Wandb离线模式

### eval_default.yaml
评估配置，用于在不同challenge上评估模型：
- 单GPU运行（GPU 0）
- scenario_builder: nuplan_challenge
- threads_per_node: 20
- 需要指定checkpoint路径
- 支持三个challenge：
  - closed_loop_nonreactive_agents
  - closed_loop_reactive_agents
  - open_loop_boxes

## 配置优先级

1. **命令行参数** > 2. **本地YAML配置** > 3. **项目默认配置**

这意味着如果你在运行脚本时附加命令行参数，它会覆盖YAML配置中的设置。

### 示例：临时覆盖配置

```bash
# 使用debug配置，但临时改为online wandb
./train.sh debug wandb.mode=online

# 使用default配置，临时改为单GPU
./train.sh default CUDA_VISIBLE_DEVICES=0
```

## 创建自定义配置

你可以创建任何自定义配置。例如创建一个用于特定实验的配置：

```bash
# 复制模板
cp config/local.example/default.yaml config/local/my_experiment.yaml

# 编辑配置
vim config/local/my_experiment.yaml

# 使用配置
./train.sh my_experiment
```

## 重要说明

- `config/local/` 目录被添加到 `.gitignore`，你的个人配置不会被提交到git
- 这样做的好处：
  - 不会因为配置修改而产生许多git commit
  - 不同的分支可以有不同的本地配置
  - 可以安全地修改配置而无需进行git操作

- 提交代码时无需担心配置：只需提交 `config/local.example/` 中的模板

## 配置快照和复现

每次训练时，脚本会自动：
1. **保存配置快照**：将使用的完整配置保存到实验目录（如 `work_dirs/exp_xxx/config.yaml`）
2. **记录到Wandb**：配置参数会记录到Wandb实验记录
3. **添加版本标签**：配置中的 `name` 和 `version` 会作为标签记录

这样便于后续的实验复现。

## 故障排除

### 配置文件不存在

如果出现 `Config file not found` 错误：
```bash
# 检查可用的配置
ls config/local/*.yaml

# 确保配置文件存在
ls config/local/default.yaml
```

### 配置不生效

确保：
1. 配置文件是有效的YAML格式
2. 配置键与脚本中定义的键名完全匹配
3. 值的数据类型正确（例如数字不要加引号）

### 临时测试配置

可以在命令行直接传参而无需修改配置文件：
```bash
# 临时使用不同的GPU
CUDA_VISIBLE_DEVICES=2,3 ./train.sh

# 临时改变batch size
./train.sh default data_loader.params.batch_size=32
```
