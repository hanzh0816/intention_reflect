import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from typing import Tuple
from run_training import main as main_train

if __name__ == "__main__":
    # Initialize configuration management system
    CONFIG_PATH = "./config"
    CONFIG_NAME = "default_training"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    os.environ["http_proxy"] = "http://127.0.0.1:11234"
    os.environ["https_proxy"] = "http://127.0.0.1:11234"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    os.environ["NUPLAN_MAPS_ROOT"] = "/data2/hzh/nuplan/dataset/maps"
    os.environ["NUPLAN_DATA_ROOT"] = "/data2/hzh/nuplan/dataset"
    os.environ["NUPLAN_EXP_ROOT"] = "/data2/hzh/nuplan/exp"
    cfg = hydra.compose(
        config_name=CONFIG_NAME,
        overrides=[
            "py_func=train",
            "+training=train_snn_planner",
            "worker=sequential",  # Single-threaded for debugging
            "scenario_builder=nuplan",
            "cache.cache_path=/data2/hzh/nuplan/exp/cache_snn_planner",
            "cache.use_cache_without_dataset=true",
            "data_loader.params.batch_size=32",  # Small batch for debugging
            "data_loader.params.num_workers=0",  # Single-threaded dataloader
            "lr=1e-3",
            "epochs=25",
            "warmup_epochs=3",
            "weight_decay=0.0001",
            # "lightning.trainer.params.val_check_interval=0.5",
        ],
    )
    main_train(cfg)
