import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path
from src.custom_training.custom_training_builder import (
    build_training_engine,
    update_config_for_training,
)
from src.feature_builders.common.route_utils import route_roadblock_correction
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


def visualize_roadblocks(dataset, sample_idx=0, output_path="roadblock_visualization.png"):
    """
    Visualize roadblocks for a specific sample in the dataset.

    Args:
        dataset: NuPlanDataset instance
        sample_idx: Index of the sample to visualize
        output_path: Path to save the visualization
    """
    # Get scenario from dataset
    scenario = dataset._scenarios[sample_idx]
    map_api = scenario.map_api
    ego_state = scenario.initial_ego_state

    # Get original route roadblocks
    original_route_roadblock_ids = scenario.get_route_roadblock_ids()

    # Get corrected route roadblocks
    corrected_route_roadblock_ids = route_roadblock_correction(
        ego_state, map_api, original_route_roadblock_ids
    )

    # Get all roadblocks in a radius for context
    proximal_roadblocks = map_api.get_proximal_map_objects(
        ego_state.center, 100, [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    )

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot original roadblocks (in blue)
    for roadblock_id in original_route_roadblock_ids:
        try:
            roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            if not roadblock:
                roadblock = map_api.get_map_object(
                    roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
                )
            if roadblock:
                ax.plot(
                    *roadblock.polygon.exterior.xy,
                    color="blue",
                    linewidth=2,
                    label=(
                        "Original Roadblock"
                        if "Original Roadblock" not in [l.get_label() for l in ax.get_lines()]
                        else ""
                    ),
                )
                center = roadblock.polygon.centroid
                ax.text(center.x, center.y, str(roadblock_id), fontsize=8, color="blue")
        except Exception as e:
            print(f"Could not plot original roadblock {roadblock_id}: {e}")

    # Plot corrected roadblocks (in red)
    for roadblock_id in corrected_route_roadblock_ids:
        try:
            roadblock = map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
            if not roadblock:
                roadblock = map_api.get_map_object(
                    roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
                )
            if roadblock:
                ax.plot(
                    *roadblock.polygon.exterior.xy,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=(
                        "Corrected Roadblock"
                        if "Corrected Roadblock" not in [l.get_label() for l in ax.get_lines()]
                        else ""
                    ),
                )
                center = roadblock.polygon.centroid
                ax.text(center.x, center.y, str(roadblock_id), fontsize=8, color="red")
        except Exception as e:
            print(f"Could not plot corrected roadblock {roadblock_id}: {e}")

    # Plot ego vehicle
    ego_footprint = ego_state.car_footprint.geometry
    ax.plot(*ego_footprint.exterior.xy, color="green", label="Ego Vehicle")
    ax.plot(ego_state.center.x, ego_state.center.y, "go")  # Ego center

    # Plot nearby lanes for context
    proximal_lanes = map_api.get_proximal_map_objects(
        ego_state.center, 50, [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    )
    lanes = proximal_lanes[SemanticMapLayer.LANE] + proximal_lanes[SemanticMapLayer.LANE_CONNECTOR]
    for lane in lanes:
        # Plot left boundary
        left_boundary_points = lane.left_boundary.discrete_path
        lx = [p.x for p in left_boundary_points]
        ly = [p.y for p in left_boundary_points]
        ax.plot(lx, ly, color="lightgrey")

        # Plot right boundary
        right_boundary_points = lane.right_boundary.discrete_path
        rx = [p.x for p in right_boundary_points]
        ry = [p.y for p in right_boundary_points]
        ax.plot(rx, ry, color="lightgrey")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Roadblock Visualization - Scenario {sample_idx}")
    ax.legend()
    ax.grid(True)

    # Zoom in around ego
    ax.set_xlim(ego_state.center.x - 50, ego_state.center.x + 50)
    ax.set_ylim(ego_state.center.y - 50, ego_state.center.y + 50)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved roadblock visualization to {output_path}")


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Debug script for visualizing roadblocks in planTF dataset.
    """
    pl.seed_everything(cfg.seed, workers=True)

    # Build training engine
    update_config_for_training(cfg)
    worker = build_worker(cfg)
    engine = build_training_engine(cfg, worker)

    # Setup datamodule and get dataset
    datamodule = engine.datamodule
    datamodule.setup("fit")
    dataset = datamodule.train_dataloader().dataset

    print(f"Total number of samples in dataset: {len(dataset)}")

    # Visualize first sample
    visualize_roadblocks(dataset, sample_idx=0, output_path="roadblock_vis_0.png")


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