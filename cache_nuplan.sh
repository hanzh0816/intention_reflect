export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data2/hzh/nuplan/exp"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

export PYTHONPATH=$PYTHONPATH:$(pwd)

# python run_training.py \
# py_func=cache +training=train_planTF \
# scenario_builder=nuplan \
# cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M \
# cache.cleanup_cache=true \
# scenario_filter=training_scenarios_1M \
# worker.threads_per_node=40

  python run_training.py \
  py_func=cache \
  +training=train_snn_planner \
  worker=single_machine_thread_pool \
  worker.max_workers=32 \
  scenario_builder=nuplan \
  cache.cache_path=${NUPLAN_EXP_ROOT}/cache_snn_planner \
  cache.use_cache_without_dataset=true \
  scenario_filter=training_scenarios_1M \

