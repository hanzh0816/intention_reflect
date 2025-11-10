# Intent-Conditioned Trajectory Generation

## Overview

PlanTF模型集成了意图条件轨迹生成功能。系统从专家轨迹自动提取意图标签，并使用预测的意图来条件生成轨迹。

## Intent Categories

**Lateral Intent** (5类):
- `turn_left` (0): 急转左
- `turn_right` (1): 急转右
- `shift_left` (2): 缓变左
- `shift_right` (3): 缓变右
- `stay_in_lane` (4): 保持车道

**Longitudinal Intent** (4类):
- `accelerate` (0): 加速
- `maintain_speed` (1): 匀速
- `decelerate` (2): 减速
- `stop` (3): 停止

## Usage

### 1. 启用Intent功能

编辑 `config/model/planTF.yaml`:
```yaml
intent_enabled: true
intent_time_horizon: 2.0  # 可选: 2.0, 4.0, 8.0
```

### 2. 生成缓存（包含intent标签）

```bash
./cache.sh
```

确保 `cache.sh` 中设置了 `model.intent_enabled=true`。

### 3. 训练

```bash
python run_training.py \
  py_func=train \
  +training=train_planTF \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  model.intent_enabled=true
```

### 4. 验证缓存中的Intent标签

```bash
python check_intent_labels.py --cache_path /path/to/cache --max_samples 10
```

## Configuration

### Model Config (`config/model/planTF.yaml`)

```yaml
intent_enabled: true          # 启用/禁用intent功能
intent_time_horizon: 2.0      # 意图分类时间窗口(秒)
intent_embed_dim: 64          # 意图嵌入维度
lateral_classes: 5            # 横向意图类别数
longitudinal_classes: 4       # 纵向意图类别数
```

### Trainer Config (`config/custom_trainer/planTF.yaml`)

```yaml
intent_loss_weight: 1.0       # 意图分类loss权重
```

### Lightning Config (`config/lightning/custom_lightning.yaml`)

```yaml
strategy: ddp_find_unused_parameters_true  # Intent模式需要设置为true
```

## Hyperparameters

**时间窗口**:
- 2.0s (推荐): 短期意图，最稳定
- 4.0s: 中期意图
- 8.0s: 长期意图

**Intent Loss权重**:
- 0.5: 优先轨迹精度
- 1.0 (默认): 平衡
- 2.0: 强调意图学习

## Architecture

```
Input → Encoder → Intent Prediction (M modes)
                → Intent Embeddings
                → Conditioned Features
                → Trajectory Decoder
                → M intent-conditioned trajectories
```

**训练**: 所有M个modes学习相同的expert intent（交叉熵loss）

## Files

```
src/
├── utils/intent_classification.py         # 意图分类核心逻辑
├── features/intent_labels.py              # IntentLabels特征类
├── target_builders/intent_target_builder.py  # 意图目标构建器
└── models/planTF/
    ├── planning_model.py                  # 模型架构（intent heads + embeddings）
    └── lightning_trainer.py               # 训练逻辑（intent loss）

config/
├── model/planTF.yaml                      # intent参数配置
├── custom_trainer/planTF.yaml             # loss权重配置
└── lightning/custom_lightning.yaml        # DDP配置

check_intent_labels.py                     # 验证缓存工具
```

## Notes

- Intent模式下 `trajectory_decoder.multimodal_proj` 不被使用（正常现象）
- 需要设置 `strategy: ddp_find_unused_parameters_true`
- 缓存数据必须包含intent标签（使用`intent_enabled=true`生成）
