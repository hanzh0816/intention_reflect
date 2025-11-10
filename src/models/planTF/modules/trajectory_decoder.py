import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryDecoder(nn.Module):
    """
    Single-mode trajectory decoder for intent-conditioned planning.

    Takes concatenated [ego_feature; intention_feature] and decodes to a single trajectory.
    """

    def __init__(self, embed_dim, future_steps, out_channels) -> None:
        """
        Initialize Trajectory Decoder.

        Args:
            embed_dim: Dimension of input features (should be 2*ego_dim for concatenated features)
            future_steps: Number of future timesteps to predict
            out_channels: Number of output channels per timestep (typically 4 for x,y,cos,sin)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.out_channels = out_channels

        # Project concatenated features to intermediate dimension
        self.input_proj = nn.Linear(embed_dim, embed_dim // 2)

        # Trajectory prediction head
        hidden = embed_dim  # Use full embed_dim for hidden layer
        self.loc = nn.Sequential(
            nn.Linear(embed_dim // 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Concatenated feature tensor [B, embed_dim]

        Returns:
            trajectory: Single-mode trajectory [B, future_steps, out_channels]
        """
        # Project input
        x = self.input_proj(x)

        # Predict trajectory
        trajectory = self.loc(x).view(-1, self.future_steps, self.out_channels)

        return trajectory
