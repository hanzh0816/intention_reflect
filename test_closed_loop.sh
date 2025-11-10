#!/bin/bash
# Closed-loop testing script for PlanTF model
# Closed-loop: 在仿真环境中执行轨迹，评估实际驾驶性能

export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data2/hzh/nuplan/exp"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

# 设置checkpoint路径
CHECKPOINT_PATH="${1:-/path/to/your/checkpoint.ckpt}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Usage: $0 <checkpoint_path> [scenario_filter]"
    exit 1
fi

# scenario_filter可选: test14-random, test14-hard, val14
SCENARIO_FILTER="${2:-test14-random}"

echo "Running closed-loop simulation with checkpoint: $CHECKPOINT_PATH"
echo "Scenario filter: $SCENARIO_FILTER"
echo "==============================================="

python run_simulation.py \
    +simulation=closed_loop_nonreactive_agents \
    planner=planTF \
    planner.imitation_planner.planner_ckpt=$CHECKPOINT_PATH \
    scenario_filter=$SCENARIO_FILTER \
    scenario_builder=nuplan \
    observation=box_observation \
    ego_controller=perfect_tracking_controller \
    number_of_gpus_allocated_per_simulation=0.25 \
    number_of_cpus_allocated_per_simulation=1 \
    worker=sequential \
    experiment_name=closed_loop_test_$(basename $CHECKPOINT_PATH .ckpt) \
    group=${NUPLAN_EXP_ROOT}/simulation_results

echo ""
echo "Closed-loop simulation completed!"
echo "Results saved to: ${NUPLAN_EXP_ROOT}/simulation_results"
echo "Check aggregator_metric/*.parquet for detailed metrics"
