# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlanTF is a learning-based autonomous driving planner built on the nuPlan dataset. It implements a pure learning-based baseline model that achieves performance without rule-based strategies or post-optimization. The repository includes a complete pipeline for data preprocessing, training, and evaluation of motion planning models.

## Environment Setup

Before development, set up the environment:

1. Create conda environment: `conda create -n plantf python=3.9 && conda activate plantf`
2. Install nuplan-devkit first: `cd nuplan-devkit && pip install -e . && pip install -r ./requirements.txt`
3. Install project dependencies: `sh ./script/setup_env.sh`
4. Set environment variables (modify paths as needed):
   ```bash
   export NUPLAN_DATA_ROOT="/path/to/nuplan/dataset"
   export NUPLAN_MAPS_ROOT="/path/to/nuplan/dataset/maps"
   export NUPLAN_EXP_ROOT="/path/to/nuplan/exp"
   export NUPLAN_DB_FILES="/path/to/nuplan/dataset/nuplan-v1.1/splits/mini"
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

## Common Development Commands

### Data Preprocessing
Cache features for training (required before training):
```bash
python run_training.py \
   py_func=cache +training=train_planTF \
   scenario_builder=nuplan \
   cache.cache_path=/path/to/cache \
   cache.cleanup_cache=true \
   scenario_filter=training_scenarios_1M \
   worker.threads_per_node=40
```

### Training
Train the PlanTF model:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py \
  py_func=train +training=train_planTF \
  worker=single_machine_thread_pool worker.max_workers=32 \
  scenario_builder=nuplan cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  data_loader.params.batch_size=32 data_loader.params.num_workers=32 \
  lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
  lightning.trainer.params.val_check_interval=0.5 \
  wandb.mode=online wandb.project=nuplan wandb.name=plantf
```

### Evaluation
- Quick sanity check: `sh ./script/plantf_single_scenarios.sh`
- Run benchmarks: `sh ./script/plantf_benchmarks.sh [test14-random|test14-hard|val14]`

## Architecture Overview

### Core Model Structure (`src/models/planTF/`)
- **planning_model.py**: Main PlanTF model inheriting from TorchModuleWrapper
- **modules/**: Core model components
  - `agent_encoder.py`: Encodes surrounding agent information
  - `map_encoder.py`: Processes map/road network data
  - `trajectory_decoder.py`: Generates future trajectory predictions
- **layers/**: Building blocks including transformer encoder layers and common MLPs
- **lightning_trainer.py**: PyTorch Lightning training wrapper

### Configuration System (`config/`)
- **model/**: Model architecture configurations (planTF.yaml)
- **scenario_filter/**: Dataset filtering (training_scenarios_1M.yaml, test14-random.yaml, etc.)
- **custom_trainer/**: Training-specific configurations
- **planner/**: Simulation planner configurations

### Training Pipeline (`src/custom_training/`)
- **custom_datamodule.py**: Data loading and preprocessing
- **custom_training_builder.py**: Training engine construction

### Main Entry Points
- **run_training.py**: Training and caching pipeline
- **run_simulation.py**: Model evaluation and benchmarking
- **run_nuboard.py**: Visualization tool

### Feature Processing
- **src/feature_builders/**: Custom feature builders for nuPlan dataset
- Handles agent states, map polygons, and trajectory history

## Key Dependencies

- PyTorch Lightning 2.0.1 for training framework
- nuplan-devkit for dataset interface and evaluation
- timm for vision model components
- wandb for experiment tracking
- natten for neighborhood attention operations

## Model Checkpoints

Place trained models in `checkpoints/` directory. The main PlanTF checkpoint should be named `planTF.ckpt`.