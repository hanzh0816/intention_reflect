export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data2/hzh/nuplan/exp"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

export PYTHONPATH=$PYTHONPATH:$(pwd)

#!/bin/bash

# Training script for SNN Planner
# Usage: bash train_snn.sh

CUDA_VISIBLE_DEVICES=2 python run_training.py \
    py_func=train +training=train_snn_planner \
    worker=single_machine_thread_pool worker.max_workers=32 \
    scenario_builder=nuplan cache.cache_path=/data2/hzh/nuplan/exp/cache_snn_planner \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
    lr=5e-4 epochs=25 warmup_epochs=3 weight_decay=0.0001
