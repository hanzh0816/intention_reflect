import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryDecoder(nn.Module):
    """
    Multi-modal trajectory decoder for intent-enhanced planning.

    Takes concatenated [ego_feature; intention_feature] and decodes to multiple trajectory modes.
    Each mode is enhanced by the intent information.
    """

    def __init__(self, embed_dim, future_steps, out_channels, num_modes=6) -> None:
        """
        Initialize Multi-modal Trajectory Decoder.

        Args:
            embed_dim: Dimension of input features (should be 2*ego_dim for concatenated features)
            future_steps: Number of future timesteps to predict
            out_channels: Number of output channels per timestep (typically 4 for x,y,cos,sin)
            num_modes: Number of trajectory modes to predict (default: 6)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.future_steps = future_steps
        self.out_channels = out_channels
        self.num_modes = num_modes

        # Project concatenated features to intermediate dimension
        self.input_proj = nn.Linear(embed_dim, embed_dim // 2)

        # Multi-modal projection: project to multiple mode-specific features
        self.multimodal_proj = nn.Linear(embed_dim // 2, num_modes * (embed_dim // 2))

        # Trajectory prediction head (shared across modes)
        hidden = embed_dim  # Use full embed_dim for hidden layer
        self.loc = nn.Sequential(
            nn.Linear(embed_dim // 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, future_steps * out_channels),
        )

        # Mode probability head
        self.pi = nn.Sequential(
            nn.Linear(embed_dim // 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Concatenated feature tensor [B, embed_dim]

        Returns:
            loc: Multi-modal trajectories [B, num_modes, future_steps, out_channels]
            pi: Mode probabilities (logits) [B, num_modes]
        """
        batch_size = x.shape[0]

        # Project input: [B, embed_dim] -> [B, embed_dim//2]
        x = self.input_proj(x)

        # Multi-modal projection: [B, embed_dim//2] -> [B, num_modes * (embed_dim//2)]
        x_multimodal = self.multimodal_proj(x)

        # Reshape to separate modes: [B, num_modes, embed_dim//2]
        x_multimodal = x_multimodal.view(batch_size, self.num_modes, -1)

        # Predict trajectories for each mode: [B, num_modes, future_steps * out_channels]
        loc = self.loc(x_multimodal)

        # Reshape trajectories: [B, num_modes, future_steps, out_channels]
        loc = loc.view(batch_size, self.num_modes, self.future_steps, self.out_channels)

        # Predict mode probabilities: [B, num_modes, 1] -> [B, num_modes]
        pi = self.pi(x_multimodal).squeeze(-1)

        return loc, pi
