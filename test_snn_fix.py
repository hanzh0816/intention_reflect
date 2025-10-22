"""Quick test for SNNStateAttentionEncoder fix"""

import torch
from src.models.spike_planner.modules.snn_agent_encoder import SNNStateAttentionEncoder

print("Testing SNNStateAttentionEncoder...")

# Create encoder
encoder = SNNStateAttentionEncoder(
    state_channel=6,
    dim=128,
    state_dropout=0.75,
    tau=2.0,
    backend='torch'
)

# Create dummy input
batch_size = 4
time_steps = 4
x = torch.randn(batch_size, 6)  # [B, state_channel]

print(f"Input shape: {x.shape}")

try:
    # Forward pass
    output = encoder(x, time_steps=time_steps)
    print(f"✓ Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{time_steps}, {batch_size}, 128]")

    if output.shape == (time_steps, batch_size, 128):
        print("✓ Output shape is correct!")
    else:
        print(f"✗ Output shape mismatch!")

    # Check for NaN/Inf
    if torch.isnan(output).any():
        print("⚠ Warning: Output contains NaN")
    elif torch.isinf(output).any():
        print("⚠ Warning: Output contains Inf")
    else:
        print("✓ No NaN or Inf in output")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")
