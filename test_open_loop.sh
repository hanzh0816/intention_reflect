#!/bin/bash
# Open-loop testing script for PlanTF model
# Open-loop: 在验证集上直接评估轨迹预测精度（ADE, FDE等指标）

export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data2/hzh/nuplan/exp"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

# 设置checkpoint路径
# 例如: /data2/hzh/nuplan/exp/planTF/2025.11.09.23-24-55/checkpoints/epoch=10-val_minFDE=2.345.ckpt
CHECKPOINT_PATH="${1:-/path/to/your/checkpoint.ckpt}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Usage: $0 <checkpoint_path>"
    exit 1
fi

echo "Running open-loop test with checkpoint: $CHECKPOINT_PATH"
echo "==============================================="

python run_training.py \
    py_func=validate \
    +training=train_planTF \
    cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M \
    cache.use_cache_without_dataset=true \
    checkpoint=$CHECKPOINT_PATH \
    data_loader.params.batch_size=32

echo ""
echo "Open-loop test completed!"
echo "Metrics: minADE1, minADE6, minFDE1, minFDE6, MR"
