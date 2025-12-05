# MNIST Classification 快速入门指南

## 环境要求

确保已安装以下依赖：
- PyTorch
- torchvision
- SpikingJelly
- scikit-learn
- matplotlib
- seaborn
- tqdm

## 训练模型

### 1. 训练ANN模型

```bash
cd /home/hzh/code/planning/planTF/mnist_cls
python scripts/train_ann.py
```

预期结果：
- 训练时间：约10分钟（50 epochs）
- 测试准确率：97-98%
- 模型保存：`checkpoints/ann/best_model.pth`

### 2. 训练SNN (BP模式)

```bash
python scripts/train_snn_bp.py
```

预期结果：
- 训练时间：约15分钟（50 epochs）
- 测试准确率：95-97%
- 模型保存：`checkpoints/snn_bp/best_model.pth`

关键特性：
- 使用LIF神经元
- 反向传播训练
- 时间步数：8

### 3. 训练SNN (STDP模式)

#### 方式1：从头训练
```bash
python scripts/train_snn_stdp.py
```

#### 方式2：使用SNN BP预训练模型初始化（推荐）
```bash
# 先训练SNN BP模型
python scripts/train_snn_bp.py

# 使用BP预训练权重初始化STDP训练
python scripts/train_snn_stdp.py --pretrain-checkpoint checkpoints/snn_bp/best_model.pth
```

预期结果：
- **从头训练**：测试准确率 88-93%
- **使用预训练**：测试准确率 92-95%（显著提升！）
- 训练时间：约30分钟（100 epochs）
- 模型保存：`checkpoints/snn_stdp/best_model.pth`

关键特性：
- 使用Reward-modulated STDP
- 无反向传播（直接权重更新）
- 仅更新输出层权重
- 支持加载预训练权重初始化

## 评估模型

### 评估ANN模型

```bash
python scripts/evaluate.py \
    --model-type ann \
    --checkpoint checkpoints/ann/best_model.pth \
    --save-cm
```

### 评估SNN BP模型

```bash
python scripts/evaluate.py \
    --model-type snn_bp \
    --checkpoint checkpoints/snn_bp/best_model.pth \
    --save-cm
```

### 评估SNN STDP模型

```bash
python scripts/evaluate.py \
    --model-type snn_stdp \
    --checkpoint checkpoints/snn_stdp/best_model.pth \
    --save-cm
```

`--save-cm` 选项会保存混淆矩阵图像。

## 配置调整

### 修改训练参数

编辑对应的配置文件：
- ANN: `configs/ann_config.py`
- SNN BP: `configs/snn_bp_config.py`
- SNN STDP: `configs/snn_stdp_config.py`

常用参数：
```python
# 训练配置
'batch_size': 128,      # 批次大小
'epochs': 50,           # 训练轮数
'lr': 1e-3,             # 学习率

# SNN特定配置
'time_steps': 8,        # 时间步数（建议8-16）
'population_size': 1,   # 群体编码大小（1=禁用）

# STDP特定配置
'stdp_lr': 0.001,       # STDP学习率
'stdp_a_pre': 0.01,     # LTP幅度
'stdp_a_post': -0.01,   # LTD幅度
```

## 模型架构

### ANN MLP
```
输入层: 784 → 512
隐藏层: 512 → 256
输出层: 256 → 10
激活: ReLU + BatchNorm + Dropout
```

### SNN MLP
```
时间扩展: [B, 784] → [T, B, 784]
隐藏层1: 784 → 512 (LIF)
隐藏层2: 512 → 256 (LIF)
输出层: 256 → 10 (可选LIF)
```

## 性能优化建议

### SNN BP优化
1. 增加时间步数到16
2. 使用群体编码（population_size=10）
3. 调整LIF参数（tau, v_threshold）

### SNN STDP优化
1. 先用SNN BP预训练隐藏层
2. 增加训练epoch到200
3. 调整STDP参数：
   - `stdp_lr`: 0.0001 ~ 0.01
   - `A_pre / A_post`: 调整比例
4. 使用群体编码提升性能

## 故障排除

### CUDA Out of Memory
- 减小batch_size（如64或32）
- 减小隐藏层维度

### SNN训练不稳定
- 增加time_steps
- 调整LIF参数tau
- 使用更小的学习率

### STDP性能低
- 增加训练epoch
- 调整STDP学习率
- 先用BP预训练
