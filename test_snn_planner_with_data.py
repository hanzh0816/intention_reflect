"""
Test script for SNN Planner with Real Dataset

This script validates spike_planner by:
1. Loading a real sample from the dataset
2. Performing forward pass
3. Computing loss (using SNNLightningTrainer logic)
4. Performing backward pass
5. Checking gradients and model state

Based on train_snn_planner configuration.
"""

import os
import sys
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from spikingjelly.clock_driven import functional

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


def print_section(title: str, width: int = 80):
    """Print a section header"""
    print(f"\n{'='*width}")
    print(f"{title}")
    print(f"{'='*width}")


def print_subsection(title: str, width: int = 80):
    """Print a subsection header"""
    print(f"\n{'-'*width}")
    print(f"{title}")
    print(f"{'-'*width}")


def analyze_data_batch(batch: Tuple[FeaturesType, TargetsType, ScenarioListType]):
    """Analyze and display information about a data batch"""
    print_subsection("Data Batch Analysis")

    features, targets, scenarios = batch
    data = features["feature"].data

    print(f"Batch size: {len(scenarios)}")
    print(f"\nAgent data:")
    print(f"  position shape: {data['agent']['position'].shape}")
    print(f"  heading shape: {data['agent']['heading'].shape}")
    print(f"  velocity shape: {data['agent']['velocity'].shape}")
    print(f"  valid_mask shape: {data['agent']['valid_mask'].shape}")
    print(f"  target shape: {data['agent']['target'].shape}")

    print(f"\nMap data:")
    print(f"  polygon_center shape: {data['map']['polygon_center'].shape}")
    print(f"  point_position shape: {data['map']['point_position'].shape}")
    print(f"  valid_mask shape: {data['map']['valid_mask'].shape}")

    # Check validity
    agent_valid_ratio = data['agent']['valid_mask'].float().mean().item()
    map_valid_ratio = data['map']['valid_mask'].float().mean().item()
    print(f"\nValidity ratios:")
    print(f"  Agents: {agent_valid_ratio*100:.1f}%")
    print(f"  Map: {map_valid_ratio*100:.1f}%")

    return data


def test_forward_pass(model, data: Dict, device: torch.device):
    """Test forward pass of the model"""
    print_subsection("Forward Pass Test")

    try:
        # Move data to device
        def move_to_device(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            elif isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [move_to_device(v) for v in obj]
            else:
                return obj

        data = move_to_device(data)
        model = model.to(device)
        model.train()  # Use train mode for loss computation

        # Forward pass
        print("Running forward pass...")
        output = model(data)

        print(f"✓ Forward pass successful!")
        print(f"\nOutput keys: {list(output.keys())}")

        # Check outputs
        if "trajectory" in output:
            traj = output["trajectory"]
            print(f"\nTrajectory:")
            print(f"  Shape: {traj.shape}")
            print(f"  Range: [{traj.min():.3f}, {traj.max():.3f}]")
            print(f"  Mean: {traj.mean():.3f}, Std: {traj.std():.3f}")

        if "probability" in output:
            prob = output["probability"]
            print(f"\nProbability:")
            print(f"  Shape: {prob.shape}")
            print(f"  Range: [{prob.min():.3f}, {prob.max():.3f}]")
            print(f"  Sum per sample: {prob.sum(dim=-1)}")

        if "prediction" in output:
            pred = output["prediction"]
            print(f"\nAgent Prediction:")
            print(f"  Shape: {pred.shape}")
            print(f"  Range: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"  Mean: {pred.mean():.3f}, Std: {pred.std():.3f}")

        # Check for NaN or Inf
        has_nan = False
        has_inf = False
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any():
                    print(f"  ⚠ Warning: {key} contains NaN values!")
                    has_nan = True
                if torch.isinf(value).any():
                    print(f"  ⚠ Warning: {key} contains Inf values!")
                    has_inf = True

        if not has_nan and not has_inf:
            print(f"\n✓ No NaN or Inf values in outputs")

        return output, True

    except Exception as e:
        print(f"✗ Forward pass failed!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False


def compute_loss(output: Dict, data: Dict):
    """
    Compute loss following SNNLightningTrainer._compute_objectives logic
    """
    print_subsection("Loss Computation")

    try:
        trajectory = output["trajectory"]
        probability = output["probability"]
        prediction = output["prediction"]

        # Get device from output
        device = trajectory.device

        targets = data["agent"]["target"].to(device)
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2]:].to(device)

        # Extract ego target
        ego_target_pos = targets[:, 0, :, :2]
        ego_target_heading = targets[:, 0, :, 2]
        ego_target = torch.cat([
            ego_target_pos,
            torch.stack([
                ego_target_heading.cos(),
                ego_target_heading.sin()
            ], dim=-1),
        ], dim=-1)

        agent_target = targets[:, 1:, :, :2]
        agent_mask = valid_mask[:, 1:]

        print(f"Target shapes:")
        print(f"  ego_target: {ego_target.shape}")
        print(f"  agent_target: {agent_target.shape}")
        print(f"  agent_mask: {agent_mask.shape}")

        # Ego trajectory regression loss (best mode)
        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)

        # Ego mode classification loss
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        # Agent prediction loss
        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask],
            agent_target[agent_mask]
        )

        # Total loss
        total_loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

        print(f"\nLoss components:")
        print(f"  Ego regression loss: {ego_reg_loss.item():.6f}")
        print(f"  Ego classification loss: {ego_cls_loss.item():.6f}")
        print(f"  Agent prediction loss: {agent_reg_loss.item():.6f}")
        print(f"  Total loss: {total_loss.item():.6f}")

        print(f"\n✓ Loss computation successful!")

        losses = {
            "total_loss": total_loss,
            "ego_reg_loss": ego_reg_loss,
            "ego_cls_loss": ego_cls_loss,
            "agent_reg_loss": agent_reg_loss,
        }

        return losses, True

    except Exception as e:
        print(f"✗ Loss computation failed!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, False


def test_backward_pass(loss: torch.Tensor, model):
    """Test backward pass and check gradients"""
    print_subsection("Backward Pass Test")

    try:
        # Backward pass
        print("Running backward pass...")
        loss.backward()

        print(f"✓ Backward pass successful!")

        # Check gradients
        grad_stats = []
        no_grad_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_stats.append((name, grad_norm, param.grad.numel()))
                else:
                    no_grad_params.append(name)

        if no_grad_params:
            print(f"\n⚠ Warning: {len(no_grad_params)} parameters have no gradients")
            print(f"  Examples: {no_grad_params[:3]}")

        # Sort by gradient norm
        grad_stats.sort(key=lambda x: x[1], reverse=True)

        print(f"\nGradient statistics:")
        print(f"  Total parameters with gradients: {len(grad_stats)}")

        if grad_stats:
            print(f"\nTop 10 parameters by gradient norm:")
            for name, grad_norm, numel in grad_stats[:10]:
                print(f"  {name:60s}: {grad_norm:.6f} (numel: {numel})")

            print(f"\nBottom 10 parameters by gradient norm:")
            for name, grad_norm, numel in grad_stats[-10:]:
                print(f"  {name:60s}: {grad_norm:.6f} (numel: {numel})")

            # Check for zero gradients
            zero_grads = [name for name, grad_norm, _ in grad_stats if grad_norm < 1e-10]
            if zero_grads:
                print(f"\n⚠ Warning: {len(zero_grads)} parameters have near-zero gradients")
                print(f"  Examples: {zero_grads[:5]}")
            else:
                print(f"\n✓ All parameters have non-zero gradients")

            # Check for NaN or Inf gradients
            has_nan_grad = False
            has_inf_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"  ⚠ Warning: {name} has NaN gradients!")
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        print(f"  ⚠ Warning: {name} has Inf gradients!")
                        has_inf_grad = True

            if not has_nan_grad and not has_inf_grad:
                print(f"✓ No NaN or Inf gradients detected")

        return True

    except Exception as e:
        print(f"✗ Backward pass failed!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """
    Main test function
    """
    print_section("SNN Planner Test with Real Dataset")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Build training engine
    print_section("Building Training Engine")
    try:
        update_config_for_training(cfg)
        worker = build_worker(cfg)
        engine = build_training_engine(cfg, worker)
        print("✓ Training engine built successfully")
    except Exception as e:
        print(f"✗ Failed to build training engine!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Setup datamodule
    print_section("Loading Dataset")
    try:
        datamodule = engine.datamodule
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        print(f"✓ Dataloader created")
        print(f"  Total batches: {len(train_dataloader)}")
    except Exception as e:
        print(f"✗ Failed to setup datamodule!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Get one batch
    print_subsection("Loading One Batch")
    try:
        batch = next(iter(train_dataloader))
        print("✓ Successfully loaded one batch")
        data = analyze_data_batch(batch)
    except Exception as e:
        print(f"✗ Failed to load batch!")
        print(f"  Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Get model
    print_section("Model Information")
    model = engine.model.model  # Get the actual model from Lightning wrapper

    print(f"Model class: {type(model).__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print_section("Forward Pass Test")
    output, forward_ok = test_forward_pass(model, data, device)

    if not forward_ok:
        print("\n✗ Forward pass failed - cannot proceed with backward pass")
        return

    # Reset SNN states after forward
    functional.reset_net(model)

    # Compute loss
    print_section("Loss Computation Test")
    losses, loss_ok = compute_loss(output, data)

    if not loss_ok:
        print("\n✗ Loss computation failed - cannot proceed with backward pass")
        return

    # Test backward pass
    print_section("Backward Pass Test")
    backward_ok = test_backward_pass(losses["total_loss"], model)

    # Reset SNN states after backward
    functional.reset_net(model)

    # Final summary
    print_section("Test Summary")

    results = [
        ("Data loading", True),
        ("Model creation", True),
        ("Forward pass", forward_ok),
        ("Loss computation", loss_ok),
        ("Backward pass", backward_ok),
    ]

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:20s}: {status}")

    all_passed = all(result for _, result in results)

    print(f"\n{'='*80}")
    if all_passed:
        print("✓ All infrastructure tests PASSED!")
        print("  The spike_planner can perform forward and backward propagation.")
    else:
        print("✗ Some tests FAILED - please check the errors above")
    print(f"{'='*80}")

    # Additional diagnostic information
    print_section("Diagnostic Information")

    if output is not None:
        traj_zeros = (output["trajectory"] == 0).all().item()
        prob_zeros = (output["probability"] == 0).all().item()
        pred_has_values = (output["prediction"].abs() > 0).any().item()

        print(f"\nModel output diagnosis:")
        print(f"  Trajectory all zeros: {traj_zeros}")
        print(f"  Probability all zeros: {prob_zeros}")
        print(f"  Prediction has values: {pred_has_values}")

        if traj_zeros or prob_zeros:
            print(f"\n⚠ WARNING: trajectory_decoder outputs are all zeros!")
            print(f"  This is likely due to SNN neurons not firing (insufficient input).")
            print(f"  Possible causes:")
            print(f"    1. LIF neurons need stronger input to fire")
            print(f"    2. Time constant (tau) may need tuning")
            print(f"    3. Voltage threshold may be too high")
            print(f"    4. Input scaling may need adjustment")
            print(f"\n  Suggestions:")
            print(f"    - Increase tau (e.g., tau=4.0 or 8.0)")
            print(f"    - Lower v_threshold (e.g., 0.1 or 0.3)")
            print(f"    - Add input scaling/normalization")
            print(f"    - Check if model needs warm-up iterations")

    if backward_ok and losses is not None:
        has_nan_grads = any("NaN" in str(p.grad) if p.grad is not None else False
                           for p in model.parameters())
        if has_nan_grads:
            print(f"\n⚠ WARNING: NaN gradients detected!")
            print(f"  This will prevent training from converging.")
            print(f"  Possible causes:")
            print(f"    1. Gradient explosion due to zero outputs")
            print(f"    2. Division by zero in loss computation")
            print(f"    3. Numerical instability in SNN dynamics")
            print(f"\n  Suggestions:")
            print(f"    - Use gradient clipping during training")
            print(f"    - Review loss computation for numerical stability")
            print(f"    - Check SNN parameter settings")

    print(f"\n{'='*80}")
    print("Test complete! See diagnostics above for model issues.")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Set environment variables if not already set
    # You can modify these paths according to your setup
    if "NUPLAN_DATA_ROOT" not in os.environ:
        os.environ["NUPLAN_DATA_ROOT"] = "/data2/hzh/nuplan/dataset"
    if "NUPLAN_MAPS_ROOT" not in os.environ:
        os.environ["NUPLAN_MAPS_ROOT"] = "/data2/hzh/nuplan/dataset/maps"
    if "NUPLAN_EXP_ROOT" not in os.environ:
        os.environ["NUPLAN_EXP_ROOT"] = "/data2/hzh/nuplan/exp"

    # Optional: Set proxy if needed
    # os.environ["http_proxy"] = "http://127.0.0.1:11234"
    # os.environ["https_proxy"] = "http://127.0.0.1:11234"

    # Optional: Set specific GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Run with Hydra
    sys.argv = [
        "test_snn_planner_with_data.py",
        "py_func=train",
        "+training=train_snn_planner",
        "worker=sequential",
        "scenario_builder=nuplan_mini",
        "scenario_filter.limit_total_scenarios=10",
        "cache.cache_path=${oc.env:NUPLAN_EXP_ROOT}/cache_snn_planner",
        "cache.use_cache_without_dataset=true",
        "data_loader.params.batch_size=2",
        "data_loader.params.num_workers=0",
        "lr=1e-3",
        "epochs=25",
        "warmup_epochs=3",
        "weight_decay=0.0001",
    ]

    main()
