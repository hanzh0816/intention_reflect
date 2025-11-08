export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/data2/hzh/nuplan/exp"
export NUPLAN_DB_FILES="/data2/hzh/nuplan/dataset/nuplan-v1.1/splits/mini"

python run_training.py \
    py_func=train \
    +training=train_planTF \
    cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M \
    cache.use_cache_without_dataset=true \
    model.intent_enabled=true