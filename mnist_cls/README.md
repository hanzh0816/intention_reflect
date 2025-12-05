# MNIST Classification Framework

MNIST数字识别训练框架，包含ANN和SNN两个版本。

## 特性

- **ANN版本**: 标准3层MLP (784→512→256→10)
- **SNN版本**: 基于LIF神经元的脉冲神经网络
  - 支持BP训练（反向传播）
  - 支持STDP训练（脉冲时序依赖可塑性）
  - 复用PlanTF代码库的SNN组件

## 快速开始

### 训练ANN模型
```bash
python scripts/train_ann.py
```

### 训练SNN (BP模式)
```bash
python scripts/train_snn_bp.py
```

### 训练SNN (STDP模式)
```bash
# 从头训练
python scripts/train_snn_stdp.py

# 使用预训练模型初始化（推荐）
python scripts/train_snn_stdp.py --pretrain-checkpoint checkpoints/snn_bp/best_model.pth
```

### 评估模型
```bash
python scripts/evaluate.py --model-type ann --checkpoint checkpoints/ann/best_model.pth
python scripts/evaluate.py --model-type snn_bp --checkpoint checkpoints/snn_bp/best_model.pth
python scripts/evaluate.py --model-type snn_stdp --checkpoint checkpoints/snn_stdp/best_model.pth
```

## 项目结构

```
mnist_cls/
├── models/          # 模型定义
├── trainers/        # 训练器
├── data/            # 数据加载
├── utils/           # 工具函数
├── configs/         # 配置文件
├── scripts/         # 训练/评估脚本
└── checkpoints/     # 模型保存
```

## 预期性能

| 模型 | 测试准确率 | 训练时间 |
|------|-----------|---------|
| ANN MLP | 97-98% | ~10分钟 |
| SNN BP | 95-97% | ~15分钟 |
| SNN STDP | 88-93% | ~30分钟 |
