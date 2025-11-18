# SNN意图模块 - 使用文档

## 概述

本文档介绍了基于脉冲神经网络（SNN）的意图理解模块，包含Intention Decoder（两种实现）和Intent Head模块的完整实现。

## 核心功能

✅ **完整的SNN意图理解pipeline**
✅ **两种Intention Decoder实现：MLP和Transformer**
✅ **统一的SNN组件架构**
✅ **与现有PlanTF框架兼容**
✅ **灵活的配置和扩展性**
✅ **GPU加速支持（通过SpikingJelly）**

## 模块结构

```
src/models/planTF/modules/
├── snn_utils.py                    # 基础SNN工具和组件
├── snn_attention.py                # SNN注意力机制
├── snn_intention_mlp_decoder.py    # MLP版本的Intention Decoder
├── snn_intention_transformer_decoder.py  # Transformer版本的Intention Decoder
├── snn_intent_heads.py            # 横向和纵向意图分类头
├── snn_modules.py                 # 模块统一接口
└── planning_model_snn_intent.py   # 完整的SNN意图规划模型
```

## 快速开始

### 1. 安装依赖

```bash
# 安装SpikingJelly（如果未安装）
pip install spikingjelly
```

### 2. 基础使用

#### 创建SNN MLP意图解码器
```python
from src.models.planTF.modules.snn_modules import create_snn_intention_mlp_decoder

# 创建标准MLP意图解码器
intention_decoder = create_snn_intention_mlp_decoder(
    dim=128,                    # 特征维度
    decoder_type="standard",     # "standard" 或 "shallow"
    time_steps=4,               # SNN时间步数
    depth=2,                    # MLP层数
    dropout=0.1                # dropout比例
)

# 使用解码器
ego_feature = torch.randn(2, 128)  # [B=2, C=128]
intention_feature = intention_decoder(ego_feature)  # [B, 128]
```

#### 创建SNN Transformer意图解码器
```python
from src.models.planTF.modules.snn_modules import create_snn_intention_transformer_decoder

# 创建标准Transformer意图解码器
intention_decoder = create_snn_intention_transformer_decoder(
    decoder_type="standard",
    dim=128,
    size="standard",           # "tiny", "small", "standard", "large", "huge"
    time_steps=4,
    num_heads=8,               # 注意力头数
    depth=2                    # Transformer层数
)

# 使用解码器
intention_feature = intention_decoder(ego_feature)  # [B, 128]
```

#### 创建SNN意图分类头
```python
from src.models.planTF.modules.snn_modules import create_snn_intent_heads

# 创建标准意图分类头
intent_heads = create_snn_intent_heads(
    in_features=128,           # 输入特征维度
    lateral_classes=5,         # 横向意图类别数
    longitudinal_classes=4,    # 纵向意图类别数
    size="standard",           # "small", "standard", "large", "xlarge"
    time_steps=4,
    dropout=0.1
)

# 使用分类头
lateral_logits, longitudinal_logits = intent_heads(intention_feature)
# lateral_logits: [B, 5]
# longitudinal_logits: [B, 4]

# 获取预测结果
(lateral_pred, longitudinal_pred,
 lateral_conf, longitudinal_conf) = intent_heads.get_predictions(intention_feature)
```

### 3. 高级用法

#### 使用预设配置快速创建完整模型
```python
from src.models.planTF.modules.snn_modules import create_snn_intent_preset_model

# 创建轻量级模型（适用于资源受限场景）
lightweight_model = create_snn_intent_preset_model(
    dim=128,
    preset_name="lightweight"
)

# 创建平衡模型（推荐）
balanced_model = create_snn_intent_preset_model(
    dim=128,
    preset_name="balanced"
)

# 创建高性能模型
performance_model = create_snn_intent_preset_model(
    dim=128,
    preset_name="performance"
)
```

#### 自定义创建组件
```python
from src.models.planTF.modules.snn_modules import create_snn_intent_components

# 创建自定义SNN意图组件
intention_decoder, intent_heads = create_snn_intent_components(
    dim=128,
    lateral_classes=5,
    longitudinal_classes=4,
    intention_decoder_type="transformer",  # "mlp" 或 "transformer"
    intention_decoder_size="large",
    intent_head_size="large",
    time_steps=8,
    neuron_cfg={
        'spike_mode': 'plif',      # LIF, PLIF, IF, ILIF
        'tau': 2.0,
        'v_threshold': 1.0,
        'backend': 'torch'
    }
)
```

#### 集成到完整规划模型
```python
from src.models.planTF.modules.snn_modules import PlanningModelSNNIntent

# 创建支持SNN意图的完整规划模型
model = PlanningModelSNNIntent(
    dim=128,
    # ... 其他标准配置
    # SNN意图配置
    use_snn_intention=True,
    snn_intention_type="transformer",    # "mlp" 或 "transformer"
    snn_intention_size="standard",       # Model size
    snn_intent_head_size="standard",     # Head size
    snn_time_steps=4,                    # SNN time steps
    lateral_classes=5,
    longitudinal_classes=4,
    intent_time_horizon=2.0,            # Intent prediction horizon
    intention_decoder_depth=2
)

# 使用模型（与传统PlanTF API相同）
output = model(data)
# output包含：
# - trajectory: [B, num_modes, future_steps, 4]
# - probability: [B, num_modes]
# - prediction: [B, num_agents, future_steps, 2]
# - intent: {"lateral": [B, 5], "longitudinal": [B, 4]}
```

## 配置选项

### SNN神经元配置
```python
neuron_cfg = {
    'spike_mode': 'lif',      # 神经元类型：lif, plif, if, ilif
    'tau': 2.0,              # 膜电位时间常数
    'v_threshold': 1.0,      # 脉冲阈值电压
    'v_reset': 0.0,          # 重置电压
    'detach_reset': True,    # 是否分离梯度重置
    'backend': 'torch',      # 后端：torch, cupy
    'time_steps': 4          # SNN时间步数
}
```

### Intention Decoder尺寸配置

**MLP版本**:
- `"small"`: 1层MLP, hidden_dim=64
- `"standard"`: 2层MLP, hidden_dim=128
- `"large"`: 3层MLP, hidden_dim=256

**Transformer版本**:
- `"tiny"`: 1层, 2头, mlp_ratio=2.0
- `"small"`: 1层, 4头, mlp_ratio=2.0
- `"standard"`: 2层, 8头, mlp_ratio=4.0
- `"large"`: 3层, 8头, mlp_ratio=4.0
- `"huge"`: 4层, 16头, mlp_ratio=4.0

### Intent Head尺寸配置
- `"small"`: [32]隐藏层
- `"standard"`: [64]隐藏层
- `"large"`: [128, 64]隐藏层
- `"xlarge"`: [256, 128, 64]隐藏层

## 性能监控

### 监控SNN脉冲率
```python
# 获取SNN模块的脉冲发放率
spike_rates = model.get_snn_spike_rates()
print(f"SNN脉冲率: {spike_rates}")
# 输出: {'mlp_layer_0': 0.234, 'mlp_layer_1': 0.189, ...}
```

### 重置SNN神经元状态
```python
# 在新场景或批次开始时重置神经元状态
model.reset_snn_neurons()
```

## 与传统PlanningModel对比

| 特性 | 传统PlanningModel | SNN意图PlanningModel |
|------|-------------------|----------------------|
| 意图理解 | ❌ 无 | ✅ SNN-based |
| IntentionDecoder | Transformer | MLP/Transformer + LIF |
| IntentHead | 线性层 | SNN分类器 |
| 时间处理 | 单步 | 多时间步 |
| 神经机制 | ANN | ANN + SNN |
| GPU支持 | ✅ | ✅ (通过SpikingJelly) |

## 训练和推理

### 训练配置（Lightning Trainer）
```python
from src.models.planTF.modules.snn_modules import PlanningModelSNNIntentLightningTrainer

trainer = PlanningModelSNNIntentLightningTrainer(
    model=your_snn_model,
    lr=1e-4,
    weight_decay=1e-2,
    epochs=50,
    warmup_epochs=5,
    intent_loss_weight=1.0,      # 意图损失权重
    monitor_snn_spike_rates=True  # 监控SNN脉冲率
)

# 训练过程与普通PlanTF模型相同
```

### 推理
```python
model.eval()

# 可选：重置神经元状态
model.reset_snn_neurons()

# 推理
with torch.no_grad():
    output = model(test_data)
    trajectory = output["trajectory"]
    lateral_intent = output["intent"]["lateral"]
    longitudinal_intent = output["intent"]["longitudinal"]
```

## 常见问题

### 1. SpikingJelly安装问题
```bash
# 确保安装正确版本
pip install spikingjelly
check_spiking_jelly_available()  # True
```

### 2. GPU内存不足
```python
# 使用轻量级配置
model = create_snn_intent_preset_model(dim=128, preset_name="lightweight")

# 或减少时间步数
model.snn_time_steps = 2
```

### 3. 训练速度慢
```python
# 使用更小的模型尺寸
model = create_snn_intent_preset_model(dim=64, preset_name="small")

# 或使用MLP版本（比Transformer更快）
model.snn_intention_type = "mlp"
```

## 示例代码

见 `test_snn_intent_modules.py` 获取完整的使用示例和测试代码。

## 技术细节

### 架构概览
1. **Ego特征提取** ➜ **SNN Intention Decoder** ➜ **意图特征**
2. **意图特征** ➜ **SNN Intent Heads** ➜ **横向/纵向意图预测**
3. **[Ego特征 + 意图特征]** ➜ **轨迹解码** ➜ **最终轨迹预测**

### SNN时间处理
- 输入扩展: `[B, C]` → `[T, B, C]` (T = time_steps)
- LIF神经元处理: 每个时间步产生脉冲
- 时间平均: `[T, B, C]` → `[B, C]` (平均池化)

### 脉冲发放率监控
- 实时跟踪每个LIF层的脉冲率
- 用于分析和调优网络性能
- 有助于理解SNN的动态行为

## 版本信息

- 版本: 1.0.0
- 基于PlanTF框架扩展
- 支持PyTorch ≥ 1.10
- 支持SpikingJelly ≥ 0.0.13

---

如需更多帮助，请参考源码中的详细注释或联系开发团队。