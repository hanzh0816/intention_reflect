#!/bin/bash
# 完整训练流程：ANN → SNN BP → SNN STDP (with pretraining)

set -e  # 出错时退出

echo "=========================================="
echo "MNIST Classification - Full Training Pipeline"
echo "=========================================="
echo ""

# 进入项目目录
cd /home/hzh/code/planning/planTF/mnist_cls

# 1. 训练ANN模型
echo "Step 1/5: Training ANN model..."
echo "------------------------------------------"
python scripts/train_ann.py
echo ""

# 2. 训练SNN BP模型
echo "Step 2/5: Training SNN BP model..."
echo "------------------------------------------"
python scripts/train_snn_bp.py
echo ""

# 3. 使用预训练训练SNN STDP模型
echo "Step 3/5: Training SNN STDP model (with BP pretraining)..."
echo "------------------------------------------"
python scripts/train_snn_stdp.py --pretrain-checkpoint checkpoints/snn_bp/best_model.pth
echo ""

# 4. 评估所有模型
echo "Step 4/5: Evaluating all models..."
echo "------------------------------------------"

echo "Evaluating ANN..."
python scripts/evaluate.py --model-type ann --checkpoint checkpoints/ann/best_model.pth --save-cm
echo ""

echo "Evaluating SNN BP..."
python scripts/evaluate.py --model-type snn_bp --checkpoint checkpoints/snn_bp/best_model.pth --save-cm
echo ""

echo "Evaluating SNN STDP..."
python scripts/evaluate.py --model-type snn_stdp --checkpoint checkpoints/snn_stdp/best_model.pth --save-cm
echo ""

# 5. 总结
echo "=========================================="
echo "Step 5/5: Training Pipeline Completed!"
echo "=========================================="
echo ""
echo "Model checkpoints saved in:"
echo "  - checkpoints/ann/best_model.pth"
echo "  - checkpoints/snn_bp/best_model.pth"
echo "  - checkpoints/snn_stdp/best_model.pth"
echo ""
echo "Confusion matrices saved with each checkpoint."
echo "=========================================="
