export NUPLAN_DATA_ROOT="/media/admin-vcciv-delta/Elements SE/nuplan/data/cache"
export NUPLAN_MAPS_ROOT="/media/admin-vcciv-delta/Elements SE/nuplan/maps"
export NUPLAN_EXP_ROOT="/media/admin-vcciv-delta/Elements SE/hzh/nuplan/exp"

export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_training.py \
    py_func=cache +training=train_planTF \
    scenario_builder=nuplan \
    cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40