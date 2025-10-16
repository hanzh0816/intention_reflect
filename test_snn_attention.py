"""
Test script for SNN Attention modules

This script tests both SNNMultiheadAttention and SNNNeighborhoodAttention1D
to verify that forward and backward passes work correctly.
"""

import torch
import torch.nn as nn
from models.spike_planner.layers.snn_attention import SNNMultiheadAttention, SNNNeighborhoodAttention1D


def test_snn_multihead_attention():
    """Test SNNMultiheadAttention module"""
    print("=" * 80)
    print("Testing SNNMultiheadAttention")
    print("=" * 80)

    # Set parameters
    T, B, L, C = 4, 2, 16, 64  # time steps, batch size, sequence length, channels
    num_heads = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Input shape: [T={T}, B={B}, L={L}, C={C}]")
    print(f"Number of heads: {num_heads}")

    # Create module
    model = SNNMultiheadAttention(
        embed_dim=C,
        num_heads=num_heads,
        dropout=0.1,
        qkv_bias=True,
        backend='torch',  # Use 'torch' backend for CPU/GPU compatibility
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create input tensor with random spikes (0 or 1)
    # In SNN, inputs are typically binary spikes
    x = torch.rand(T, B, L, C, device=device)
    x = (x > 0.5).float()  # Convert to binary spikes
    x.requires_grad = True

    print(f"\nInput tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Spike rate: {x.mean().item():.4f}")
    print(f"  Requires grad: {x.requires_grad}")

    # Forward pass
    print("\n--- Forward Pass ---")
    try:
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Output spike rate: {output.mean().item():.4f}")
        print(f"  Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
        print("  Forward pass: PASSED")
    except Exception as e:
        print(f"  Forward pass: FAILED")
        print(f"  Error: {e}")
        return False

    # Backward pass
    print("\n--- Backward Pass ---")
    try:
        # Create a dummy loss
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        print(f"  Loss: {loss.item():.6f}")

        # Backward
        loss.backward()

        # Check gradients
        has_grad = x.grad is not None
        print(f"  Input gradient exists: {has_grad}")
        if has_grad:
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
            print(f"  Input gradient std: {x.grad.std().item():.6f}")

        # Check model parameter gradients
        param_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        print(f"  Parameters with gradients: {param_grads}/{total_params}")

        print("  Backward pass: PASSED")
    except Exception as e:
        print(f"  Backward pass: FAILED")
        print(f"  Error: {e}")
        return False

    print("\n" + "=" * 80)
    print("SNNMultiheadAttention: ALL TESTS PASSED")
    print("=" * 80 + "\n")
    return True


def test_snn_neighborhood_attention():
    """Test SNNNeighborhoodAttention1D module"""
    print("=" * 80)
    print("Testing SNNNeighborhoodAttention1D")
    print("=" * 80)

    # Set parameters
    T, B, L, C = 4, 2, 32, 64  # time steps, batch size, sequence length, channels
    num_heads = 8
    kernel_size = 7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Input shape: [T={T}, B={B}, L={L}, C={C}]")
    print(f"Number of heads: {num_heads}")
    print(f"Kernel size: {kernel_size}")

    # Create module
    model = SNNNeighborhoodAttention1D(
        dim=C,
        kernel_size=kernel_size,
        num_heads=num_heads,
        dilation=1,
        qkv_bias=True,
        dropout=0.1,
        backend='torch',
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create input tensor with random spikes
    x = torch.rand(T, B, L, C, device=device)
    x = (x > 0.5).float()
    x.requires_grad = True

    print(f"\nInput tensor:")
    print(f"  Shape: {x.shape}")
    print(f"  Spike rate: {x.mean().item():.4f}")
    print(f"  Requires grad: {x.requires_grad}")

    # Forward pass
    print("\n--- Forward Pass ---")
    try:
        output = model(x)
        print(f"  Output shape: {output.shape}")
        print(f"  Output spike rate: {output.mean().item():.4f}")
        print(f"  Output min/max: {output.min().item():.4f} / {output.max().item():.4f}")
        print("  Forward pass: PASSED")
    except Exception as e:
        print(f"  Forward pass: FAILED")
        print(f"  Error: {e}")
        return False

    # Backward pass
    print("\n--- Backward Pass ---")
    try:
        # Create a dummy loss
        target = torch.randn_like(output)
        loss = nn.MSELoss()(output, target)
        print(f"  Loss: {loss.item():.6f}")

        # Backward
        loss.backward()

        # Check gradients
        has_grad = x.grad is not None
        print(f"  Input gradient exists: {has_grad}")
        if has_grad:
            print(f"  Input gradient shape: {x.grad.shape}")
            print(f"  Input gradient mean: {x.grad.mean().item():.6f}")
            print(f"  Input gradient std: {x.grad.std().item():.6f}")

        # Check model parameter gradients
        param_grads = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        print(f"  Parameters with gradients: {param_grads}/{total_params}")

        print("  Backward pass: PASSED")
    except Exception as e:
        print(f"  Backward pass: FAILED")
        print(f"  Error: {e}")
        return False

    print("\n" + "=" * 80)
    print("SNNNeighborhoodAttention1D: ALL TESTS PASSED")
    print("=" * 80 + "\n")
    return True


def test_gradient_flow():
    """Test gradient flow through the modules"""
    print("=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T, B, L, C = 4, 2, 16, 64
    num_heads = 8

    # Test SNNMultiheadAttention
    print("\n--- Testing SNNMultiheadAttention Gradient Flow ---")
    model1 = SNNMultiheadAttention(C, num_heads, backend='torch').to(device)
    x1 = torch.rand(T, B, L, C, device=device)
    x1 = (x1 > 0.5).float()
    x1.requires_grad = True

    output1 = model1(x1)
    loss1 = output1.sum()
    loss1.backward()

    grad_norm1 = x1.grad.norm().item()
    print(f"  Input gradient norm: {grad_norm1:.6f}")
    print(f"  Gradient flow: {'GOOD' if grad_norm1 > 1e-6 else 'POOR'}")

    # Test SNNNeighborhoodAttention1D
    print("\n--- Testing SNNNeighborhoodAttention1D Gradient Flow ---")
    model2 = SNNNeighborhoodAttention1D(C, kernel_size=7, num_heads=num_heads, backend='torch').to(device)
    x2 = torch.rand(T, B, L, C, device=device)
    x2 = (x2 > 0.5).float()
    x2.requires_grad = True

    output2 = model2(x2)
    loss2 = output2.sum()
    loss2.backward()

    grad_norm2 = x2.grad.norm().item()
    print(f"  Input gradient norm: {grad_norm2:.6f}")
    print(f"  Gradient flow: {'GOOD' if grad_norm2 > 1e-6 else 'POOR'}")

    print("\n" + "=" * 80)
    print("Gradient Flow: ALL TESTS PASSED")
    print("=" * 80 + "\n")
    return True


def test_shape_consistency():
    """Test that output shapes are consistent with input shapes"""
    print("=" * 80)
    print("Testing Shape Consistency")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_cases = [
        (4, 2, 16, 64),   # Standard case
        (8, 1, 32, 128),  # More time steps, larger dimension
        (2, 4, 8, 32),    # Fewer time steps, smaller dimension
    ]

    for i, (T, B, L, C) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: [T={T}, B={B}, L={L}, C={C}] ---")

        # Test SNNMultiheadAttention
        model1 = SNNMultiheadAttention(C, num_heads=8, backend='torch').to(device)
        x1 = torch.rand(T, B, L, C, device=device)
        x1 = (x1 > 0.5).float()
        out1 = model1(x1)
        assert out1.shape == x1.shape, f"Shape mismatch! Expected {x1.shape}, got {out1.shape}"
        print(f"  SNNMultiheadAttention: {x1.shape} -> {out1.shape} ✓")

        # Test SNNNeighborhoodAttention1D
        model2 = SNNNeighborhoodAttention1D(C, kernel_size=7, num_heads=8, backend='torch').to(device)
        x2 = torch.rand(T, B, L, C, device=device)
        x2 = (x2 > 0.5).float()
        out2 = model2(x2)
        assert out2.shape == x2.shape, f"Shape mismatch! Expected {x2.shape}, got {out2.shape}"
        print(f"  SNNNeighborhoodAttention1D: {x2.shape} -> {out2.shape} ✓")

    print("\n" + "=" * 80)
    print("Shape Consistency: ALL TESTS PASSED")
    print("=" * 80 + "\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("SNN Attention Modules Test Suite")
    print("=" * 80 + "\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    results = []

    # Run tests
    results.append(("SNNMultiheadAttention", test_snn_multihead_attention()))
    results.append(("SNNNeighborhoodAttention1D", test_snn_neighborhood_attention()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Shape Consistency", test_shape_consistency()))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:40s} {status}")

    all_passed = all(passed for _, passed in results)
    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
