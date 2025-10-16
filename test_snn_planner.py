"""
Test script for SNN Planner

This script tests:
1. Forward pass
2. Backward pass
3. Output format compatibility
4. Comparison with original PlanTF
"""

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import functional

# Import both models
from src.models.spike_planner import SNNPlanningModel
from src.models.planTF.planning_model import PlanningModel


def create_dummy_data(batch_size=2, num_agents=10, num_polygons=20):
    """
    Create dummy input data matching nuPlan format

    Args:
        batch_size: Batch size
        num_agents: Number of agents
        num_polygons: Number of map polygons

    Returns:
        dict: Dummy data dictionary
    """
    history_steps = 21
    future_steps = 80

    data = {
        "agent": {
            "position": torch.randn(batch_size, num_agents, history_steps + future_steps, 2),
            "heading": torch.randn(batch_size, num_agents, history_steps + future_steps),
            "velocity": torch.randn(batch_size, num_agents, history_steps + future_steps, 2),
            "shape": torch.randn(batch_size, num_agents, history_steps + future_steps, 2),  # (width, length)
            "category": torch.randint(0, 4, (batch_size, num_agents)),
            "valid_mask": torch.randint(0, 2, (batch_size, num_agents, history_steps + future_steps)).bool(),
            "target": torch.randn(batch_size, num_agents, future_steps, 3),  # (x, y, heading)
        },
        "map": {
            "polygon_center": torch.randn(batch_size, num_polygons, 3),  # (x, y, heading)
            "polygon_type": torch.randint(0, 3, (batch_size, num_polygons)),
            "polygon_on_route": torch.randint(0, 2, (batch_size, num_polygons)),
            "polygon_tl_status": torch.randint(0, 4, (batch_size, num_polygons)),
            "polygon_has_speed_limit": torch.randint(0, 2, (batch_size, num_polygons)).bool(),
            "polygon_speed_limit": torch.randn(batch_size, num_polygons).abs() * 30 + 10,
            "point_position": torch.randn(batch_size, num_polygons, 3, 20, 2),  # 3 lanes
            "point_vector": torch.randn(batch_size, num_polygons, 3, 20, 2),  # 3 lanes
            "point_orientation": torch.randn(batch_size, num_polygons, 3, 20),  # 3 lanes
            "valid_mask": torch.randint(0, 2, (batch_size, num_polygons, 3, 20)).bool(),  # 3 lanes
        },
        "current_state": torch.randn(batch_size, 10),  # Ego state features
    }

    # Ensure at least first agent is valid
    data["agent"]["valid_mask"][:, 0, :] = True

    # Ensure at least some map polygons are valid
    data["map"]["valid_mask"][:, :5, :10] = True

    return data


def test_forward_pass(model, model_name="Model"):
    """Test forward pass of the model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} - Forward Pass")
    print(f"{'='*60}")

    try:
        # Create dummy data
        data = create_dummy_data(batch_size=2)

        # Set model to eval mode
        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(data)

        # Check output format
        print(f"✓ Forward pass successful!")
        print(f"\nOutput keys: {output.keys()}")

        if "trajectory" in output:
            print(f"  trajectory shape: {output['trajectory'].shape}")
            print(f"  trajectory range: [{output['trajectory'].min():.3f}, {output['trajectory'].max():.3f}]")

        if "probability" in output:
            print(f"  probability shape: {output['probability'].shape}")
            print(f"  probability range: [{output['probability'].min():.3f}, {output['probability'].max():.3f}]")

        if "prediction" in output:
            print(f"  prediction shape: {output['prediction'].shape}")
            print(f"  prediction range: [{output['prediction'].min():.3f}, {output['prediction'].max():.3f}]")

        if "output_trajectory" in output:
            print(f"  output_trajectory shape: {output['output_trajectory'].shape}")
            print(f"  output_trajectory range: [{output['output_trajectory'].min():.3f}, {output['output_trajectory'].max():.3f}]")

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
            print(f"✓ No NaN or Inf values detected")

        return output

    except Exception as e:
        print(f"✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_backward_pass(model, model_name="Model"):
    """Test backward pass of the model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} - Backward Pass")
    print(f"{'='*60}")

    try:
        # Create dummy data
        data = create_dummy_data(batch_size=2)

        # Set model to train mode
        model.train()

        # Forward pass
        output = model(data)

        # Compute dummy loss
        trajectory = output["trajectory"]
        probability = output["probability"]
        prediction = output["prediction"]

        # Simple MSE loss on trajectory
        target_traj = torch.randn_like(trajectory)
        loss_traj = torch.nn.functional.mse_loss(trajectory, target_traj)

        # Cross entropy on probability
        target_mode = torch.randint(0, probability.shape[1], (probability.shape[0],))
        loss_prob = torch.nn.functional.cross_entropy(probability, target_mode)

        # MSE loss on prediction
        target_pred = torch.randn_like(prediction)
        loss_pred = torch.nn.functional.mse_loss(prediction, target_pred)

        total_loss = loss_traj + loss_prob + loss_pred

        print(f"Loss components:")
        print(f"  Trajectory loss: {loss_traj.item():.4f}")
        print(f"  Probability loss: {loss_prob.item():.4f}")
        print(f"  Prediction loss: {loss_pred.item():.4f}")
        print(f"  Total loss: {total_loss.item():.4f}")

        # Backward pass
        total_loss.backward()

        print(f"✓ Backward pass successful!")

        # Check gradients
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats.append((name, grad_norm))

        # Sort by gradient norm
        grad_stats.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 5 gradients by norm:")
        for name, grad_norm in grad_stats[:5]:
            print(f"  {name}: {grad_norm:.6f}")

        print(f"\nBottom 5 gradients by norm:")
        for name, grad_norm in grad_stats[-5:]:
            print(f"  {name}: {grad_norm:.6f}")

        # Check for zero gradients
        zero_grads = [name for name, grad_norm in grad_stats if grad_norm < 1e-10]
        if zero_grads:
            print(f"\n⚠ Warning: {len(zero_grads)} parameters have near-zero gradients")
            print(f"  Examples: {zero_grads[:3]}")
        else:
            print(f"✓ All parameters have non-zero gradients")

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
        print(f"✗ Backward pass failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Reset SNN states if applicable
        if "SNN" in model_name:
            functional.reset_net(model)


def compare_output_formats(snn_output, ann_output):
    """Compare output formats between SNN and ANN models"""
    print(f"\n{'='*60}")
    print(f"Comparing Output Formats")
    print(f"{'='*60}")

    if snn_output is None or ann_output is None:
        print("✗ Cannot compare - one or both models failed")
        return

    # Check keys
    snn_keys = set(snn_output.keys())
    ann_keys = set(ann_output.keys())

    print(f"SNN output keys: {snn_keys}")
    print(f"ANN output keys: {ann_keys}")

    if snn_keys == ann_keys:
        print(f"✓ Output keys match!")
    else:
        missing_in_snn = ann_keys - snn_keys
        missing_in_ann = snn_keys - ann_keys
        if missing_in_snn:
            print(f"⚠ Keys missing in SNN: {missing_in_snn}")
        if missing_in_ann:
            print(f"⚠ Extra keys in SNN: {missing_in_ann}")

    # Compare shapes
    print(f"\nShape comparison:")
    for key in snn_keys & ann_keys:
        snn_shape = snn_output[key].shape
        ann_shape = ann_output[key].shape
        match = "✓" if snn_shape == ann_shape else "✗"
        print(f"  {match} {key:20s}: SNN {str(snn_shape):30s} | ANN {str(ann_shape):30s}")

    # Compare value ranges
    print(f"\nValue range comparison:")
    for key in snn_keys & ann_keys:
        snn_min = snn_output[key].min().item()
        snn_max = snn_output[key].max().item()
        ann_min = ann_output[key].min().item()
        ann_max = ann_output[key].max().item()
        print(f"  {key:20s}:")
        print(f"    SNN: [{snn_min:8.3f}, {snn_max:8.3f}]")
        print(f"    ANN: [{ann_min:8.3f}, {ann_max:8.3f}]")


def test_trajectory_format(output, model_name="Model"):
    """Test if trajectory output format is correct for nuPlan"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name} - Trajectory Format")
    print(f"{'='*60}")

    if output is None:
        print("✗ Cannot test - model failed")
        return False

    # Check required keys
    required_keys = ["trajectory", "probability", "prediction"]
    for key in required_keys:
        if key not in output:
            print(f"✗ Missing required key: {key}")
            return False

    print(f"✓ All required keys present")

    # Check trajectory shape [B, num_modes, future_steps, out_channels]
    trajectory = output["trajectory"]
    if len(trajectory.shape) != 4:
        print(f"✗ Trajectory should be 4D, got {len(trajectory.shape)}D")
        return False

    bs, num_modes, future_steps, out_channels = trajectory.shape
    print(f"✓ Trajectory shape: [batch={bs}, modes={num_modes}, steps={future_steps}, channels={out_channels}]")

    if out_channels != 4:
        print(f"⚠ Warning: Expected 4 output channels (x, y, cos_heading, sin_heading), got {out_channels}")

    # Check probability shape [B, num_modes]
    probability = output["probability"]
    if len(probability.shape) != 2:
        print(f"✗ Probability should be 2D, got {len(probability.shape)}D")
        return False

    if probability.shape != (bs, num_modes):
        print(f"✗ Probability shape mismatch: expected [{bs}, {num_modes}], got {list(probability.shape)}")
        return False

    print(f"✓ Probability shape: {list(probability.shape)}")

    # Check prediction shape [B, num_agents-1, future_steps, 2]
    prediction = output["prediction"]
    if len(prediction.shape) != 4:
        print(f"✗ Prediction should be 4D, got {len(prediction.shape)}D")
        return False

    print(f"✓ Prediction shape: {list(prediction.shape)}")

    # Check output_trajectory in eval mode
    if "output_trajectory" in output:
        output_traj = output["output_trajectory"]
        if output_traj.shape != (bs, future_steps, 3):
            print(f"⚠ output_trajectory shape: expected [{bs}, {future_steps}, 3], got {list(output_traj.shape)}")
        else:
            print(f"✓ Output trajectory shape: {list(output_traj.shape)} (x, y, heading)")

    print(f"✓ All trajectory formats correct!")
    return True


def main():
    print("="*60)
    print("SNN Planner Test Suite")
    print("="*60)

    # Model parameters
    model_params = {
        "dim": 64,  # Smaller for faster testing
        "encoder_depth": 2,  # Smaller for faster testing
        "num_heads": 4,
        "num_modes": 6,
        "history_steps": 21,
        "future_steps": 80,
    }

    print(f"\nModel parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")

    # Create SNN model
    print(f"\n{'='*60}")
    print("Creating SNN Model")
    print(f"{'='*60}")
    try:
        snn_model = SNNPlanningModel(
            **model_params,
            time_steps=4,
            tau=2.0,
            v_threshold=0.5,
            scale=0.25,
            backend='torch',
        )
        print(f"✓ SNN model created successfully")

        # Count parameters
        snn_params = sum(p.numel() for p in snn_model.parameters())
        snn_trainable = sum(p.numel() for p in snn_model.parameters() if p.requires_grad)
        print(f"  Total parameters: {snn_params:,}")
        print(f"  Trainable parameters: {snn_trainable:,}")

    except Exception as e:
        print(f"✗ Failed to create SNN model:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Create ANN model for comparison
    print(f"\n{'='*60}")
    print("Creating ANN Model (for comparison)")
    print(f"{'='*60}")
    try:
        ann_model = PlanningModel(**model_params)
        print(f"✓ ANN model created successfully")

        # Count parameters
        ann_params = sum(p.numel() for p in ann_model.parameters())
        ann_trainable = sum(p.numel() for p in ann_model.parameters() if p.requires_grad)
        print(f"  Total parameters: {ann_params:,}")
        print(f"  Trainable parameters: {ann_trainable:,}")

        print(f"\nParameter comparison:")
        print(f"  SNN/ANN ratio: {snn_params/ann_params:.2f}x")

    except Exception as e:
        print(f"✗ Failed to create ANN model:")
        print(f"  {type(e).__name__}: {str(e)}")
        ann_model = None

    # Test SNN model
    snn_output = test_forward_pass(snn_model, "SNN Model")
    test_trajectory_format(snn_output, "SNN Model")

    # Reset SNN states
    functional.reset_net(snn_model)

    snn_backward_ok = test_backward_pass(snn_model, "SNN Model")

    # Reset SNN states again
    functional.reset_net(snn_model)

    # Test ANN model
    if ann_model is not None:
        ann_output = test_forward_pass(ann_model, "ANN Model")
        test_trajectory_format(ann_output, "ANN Model")
        ann_backward_ok = test_backward_pass(ann_model, "ANN Model")

        # Compare outputs
        compare_output_formats(snn_output, ann_output)

    # Final summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    if snn_output is not None:
        print(f"✓ SNN forward pass: PASSED")
    else:
        print(f"✗ SNN forward pass: FAILED")

    if snn_backward_ok:
        print(f"✓ SNN backward pass: PASSED")
    else:
        print(f"✗ SNN backward pass: FAILED")

    if ann_model is not None:
        if ann_output is not None:
            print(f"✓ ANN forward pass: PASSED")
        else:
            print(f"✗ ANN forward pass: FAILED")

        if ann_backward_ok:
            print(f"✓ ANN backward pass: PASSED")
        else:
            print(f"✗ ANN backward pass: FAILED")

    print(f"\n{'='*60}")
    if snn_output is not None and snn_backward_ok:
        print("✓ All SNN tests PASSED!")
    else:
        print("✗ Some tests FAILED - please check the errors above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
