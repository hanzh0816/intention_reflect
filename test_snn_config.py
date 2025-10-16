"""
Quick test to verify SNN planner configuration is correct

This script checks if:
1. Config files can be loaded
2. Model can be instantiated
3. Trainer can be instantiated
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(config_path="config", config_name="default_training")
def test_config(cfg: DictConfig):
    print("=" * 60)
    print("Testing SNN Planner Configuration")
    print("=" * 60)

    # Print config
    print("\n1. Configuration loaded successfully!")
    print(f"   Job name: {cfg.job_name}")
    print(f"   Model: {cfg.model._target_}")
    print(f"   Trainer: {cfg.custom_trainer._target_}")

    # Test model instantiation
    print("\n2. Testing model instantiation...")
    try:
        model = hydra.utils.instantiate(cfg.model)
        print(f"   ✓ Model created: {type(model).__name__}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✓ Total parameters: {total_params:,}")
        print(f"   ✓ Trainable parameters: {trainable_params:,}")

        # Check SNN-specific attributes
        if hasattr(model, 'time_steps'):
            print(f"   ✓ SNN time steps: {model.time_steps}")
        if hasattr(model, 'tau'):
            print(f"   ✓ LIF tau: {model.tau}")

    except Exception as e:
        print(f"   ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test trainer instantiation
    print("\n3. Testing trainer instantiation...")
    try:
        trainer_cfg = cfg.custom_trainer
        trainer_cfg.model = model
        trainer_cfg.lr = cfg.get('lr', 1e-3)
        trainer_cfg.weight_decay = cfg.get('weight_decay', 0.0001)
        trainer_cfg.epochs = cfg.get('epochs', 25)
        trainer_cfg.warmup_epochs = cfg.get('warmup_epochs', 3)

        trainer = hydra.utils.instantiate(trainer_cfg)
        print(f"   ✓ Trainer created: {type(trainer).__name__}")

    except Exception as e:
        print(f"   ✗ Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("✓ All configuration tests passed!")
    print("=" * 60)
    print("\nYou can now run training with:")
    print("  bash train_snn.sh")
    print("\nOr directly:")
    print("  python run_training.py +training=train_snn_planner")


if __name__ == "__main__":
    test_config()
