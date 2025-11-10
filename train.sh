export NUPLAN_DATA_ROOT="/data2/hzh/nuplan/dataset"
export NUPLAN_MAPS_ROOT="/data2/hzh/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="/home/hzh/code/planning/planTF/work_dirs"

CUDA_VISIBLE_DEVICES=0,1 python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/data2/hzh/nuplan/exp/cache_plantf_1M cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=128 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=1.0 \
  wandb.mode=online wandb.project=nuplan wandb.name=plantf