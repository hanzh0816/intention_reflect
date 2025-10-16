import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from typing import Tuple
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    TargetsType,
    ScenarioListType,
)
from src.custom_training.custom_training_builder import (
    build_training_engine,
    update_config_for_training,
)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Debug script for data processing in planTF.
    Demonstrates the simplified pipeline: dataset -> dataloader -> model
    """
    pl.seed_everything(cfg.seed, workers=True)

    # Build training engine
    update_config_for_training(cfg)
    worker = build_worker(cfg)
    engine = build_training_engine(cfg, worker)

    # Setup datamodule and get dataloader
    datamodule = engine.datamodule
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()

    # Get one batch
    batch: Tuple[FeaturesType, TargetsType, ScenarioListType] = next(iter(train_dataloader))
    features, targets, scenario_info = batch

    # Get model and move to device (Lightning handles device placement)
    model = engine.model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.eval()

    # Forward pass - using the same data structure as lightning_trainer._step()
    with torch.no_grad():
        # Transfer batch to device using Lightning's method
        # batch_on_device = model.transfer_batch_to_device(batch, device, 0)
        # features_on_device, _, _ = batch_on_device

        # Forward pass following lightning_trainer forward logic
        output = model(features["feature"].data)


if __name__ == "__main__":
    # Initialize configuration management system
    CONFIG_PATH = "./config"
    CONFIG_NAME = "default_training"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=CONFIG_PATH)

    os.environ["http_proxy"] = "http://127.0.0.1:11234"
    os.environ["https_proxy"] = "http://127.0.0.1:11234"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    os.environ["NUPLAN_MAPS_ROOT"] = "/data2/hzh/nuplan/dataset/maps"
    os.environ["NUPLAN_DATA_ROOT"] = "/data2/hzh/nuplan/dataset"
    os.environ["NUPLAN_EXP_ROOT"] = "/data2/hzh/nuplan/exp"
    cfg = hydra.compose(
        config_name=CONFIG_NAME,
        overrides=[
            "py_func=train",
            "+training=train_planTF",
            "worker=sequential",  # Single-threaded for debugging
            "scenario_builder=nuplan_mini",
            "scenario_filter.limit_total_scenarios=10",  # Reduced for debugging
            "cache.cache_path=null",
            "data_loader.params.batch_size=2",  # Small batch for debugging
            "data_loader.params.num_workers=0",  # Single-threaded dataloader
            "lr=1e-3",
            "epochs=25",
            "warmup_epochs=3",
            "weight_decay=0.0001",
            "lightning.trainer.params.val_check_interval=0.5",
        ],
    )

    main(cfg)
