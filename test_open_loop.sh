export PYTHONPATH=$PYTHONPATH:$(pwd)
export NUPLAN_EXP_ROOT="/home/hzh/code/planning/planTF/work_dirs"
export NUPLAN_DATA_ROOT="/media/admin-vcciv-delta/Elements SE/nuplan/data/cache"
export NUPLAN_MAPS_ROOT="/media/admin-vcciv-delta/Elements SE/nuplan/maps"

CHECKPOINT_PATH=$1
NAME="plantf_open_loop_test"
EXP_NAME="simulation/$NAME"
PLANNER=planTF
BUILDER=nuplan
FILTER=mini
CHALLENGE="open_loop_boxes"

CUDA_VISIBLE_DEVICES=1 python run_simulation.py \
    +simulation=$CHALLENGE \
    experiment_name="$EXP_NAME" \
    planner=$PLANNER \
    scenario_builder=$BUILDER \
    scenario_filter=$FILTER \
    worker.threads_per_node=40 \
    verbose=true \
    experiment_uid="$PLANNER/$FILTER" \
    planner.imitation_planner.planner_ckpt=$CHECKPOINT_PATH