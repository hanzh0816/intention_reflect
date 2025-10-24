"""
SNN version of PlanTF Planning Model

This is the main SNN-based planning model that integrates all SNN components
for autonomous driving trajectory prediction.
"""

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder

from .layers.snn_transformer_encoder_layer import SNNTransformerEncoderLayer
from .layers.snn_neuron import LIFNeuron
from .modules.snn_agent_encoder import SNNAgentEncoder
from .modules.snn_map_encoder import SNNMapEncoder
from .modules.snn_trajectory_decoder import SNNTrajectoryDecoder

# Required by nuplan (no actual meaning)
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class SNNPlanningModel(TorchModuleWrapper):
    """
    SNN version of PlanTF Planning Model

    This model uses Spiking Neural Networks for processing agent and map features
    to generate trajectory predictions.
    """

    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        drop_path=0.2,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        time_steps=4,
        scale=0.25,
        neuron_cfg: dict = None,
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
    ):
        """
        Args:
            dim: Model dimension
            state_channel: Number of ego state channels
            polygon_channel: Number of map polygon feature channels
            history_channel: Number of history feature channels
            history_steps: Number of historical time steps
            future_steps: Number of future steps to predict
            encoder_depth: Depth of transformer encoder
            drop_path: Drop path rate
            num_heads: Number of attention heads
            num_modes: Number of trajectory modes
            use_ego_history: Whether to use ego history
            state_attn_encoder: Whether to use attention for state encoding
            state_dropout: Dropout rate for state features
            time_steps: Number of SNN time steps (T)
            scale: Scaling factor for attention
            neuron_cfg: Dictionary containing neuron configuration (spike_mode, tau,
                       v_threshold, v_reset, detach_reset, backend, etc.)
            feature_builder: Feature builder for nuplan
        """
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.time_steps = time_steps

        # Set default neuron configuration
        if neuron_cfg is None:
            neuron_cfg = {
                'spike_mode': 'lif',
                'tau': 2.0,
                'v_threshold': 1.0,
                'v_reset': 0.0,
                'detach_reset': True,
                'backend': 'torch',
            }


        # Position embedding MLP
        self.pos_emb_fc1 = nn.Linear(4, dim, bias=False)
        self.pos_emb_bn1 = nn.BatchNorm1d(dim)
        self.pos_emb_lif1 = LIFNeuron(**neuron_cfg)

        self.pos_emb_fc2 = nn.Linear(dim, dim, bias=False)
        self.pos_emb_bn2 = nn.BatchNorm1d(dim)
        self.pos_emb_lif2 = LIFNeuron(**neuron_cfg)

        # Agent encoder
        self.agent_encoder = SNNAgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
            neuron_cfg=neuron_cfg,
        )

        # Map encoder
        self.map_encoder = SNNMapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            neuron_cfg=neuron_cfg,
        )

        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            SNNTransformerEncoderLayer(
                dim=dim,
                num_heads=num_heads,
                drop_path=dp,
                scale=scale,
                neuron_cfg=neuron_cfg,
            )
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        ])

        # Layer normalization (applied per time step)
        self.norm = nn.LayerNorm(dim)

        # Trajectory decoder
        self.trajectory_decoder = SNNTrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
            neuron_cfg=neuron_cfg,
        )

        # Agent prediction head
        self.agent_pred_fc1 = nn.Linear(dim, dim * 2, bias=False)
        self.agent_pred_bn1 = nn.BatchNorm1d(dim * 2)
        self.agent_pred_lif1 = LIFNeuron(**neuron_cfg)

        self.agent_pred_fc2 = nn.Linear(dim * 2, future_steps * 2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, data):
        """
        Forward pass

        Args:
            data: Input data dict

        Returns:
            dict: Output predictions containing trajectory, probability, and prediction
        """
        T = self.time_steps

        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, :self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]
        M = polygon_center.shape[1]

        # Prepare position features
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat([
            position,
            torch.stack([angle.cos(), angle.sin()], dim=-1)
        ], dim=-1)  # [B, A+M, 4]

        # Position embedding with SNN
        pos_flat = pos.reshape(bs * (A + M), 4)
        pos_emb = self.pos_emb_fc1(pos_flat)
        pos_emb = self.pos_emb_bn1(pos_emb)
        pos_emb = pos_emb.unsqueeze(0).repeat(T, 1, 1)  # [T, B*(A+M), dim]
        pos_emb = self.pos_emb_lif1(pos_emb)

        pos_emb_flat = pos_emb.reshape(T * bs * (A + M), self.dim)
        pos_emb = self.pos_emb_fc2(pos_emb_flat)
        pos_emb = self.pos_emb_bn2(pos_emb)
        pos_emb = pos_emb.reshape(T, bs, A + M, self.dim)
        pos_emb = self.pos_emb_lif2(pos_emb)

        # Prepare padding masks
        agent_key_padding = ~(agent_mask.any(-1))
        # polygon_mask is [B, M, 3, P], we need to reduce to [B, M]
        polygon_key_padding = ~(polygon_mask.reshape(bs, M, -1).any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        # Encode agents and map
        x_agent = self.agent_encoder(data, time_steps=T)  # [T, B, A, dim]
        x_polygon = self.map_encoder(data, time_steps=T)  # [T, B, M, dim]

        # Concatenate and add position embedding
        x = torch.cat([x_agent, x_polygon], dim=2) + pos_emb  # [T, B, A+M, dim]

        # Pass through transformer encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        # Apply layer normalization
        x = torch.stack([self.norm(x[t]) for t in range(T)], dim=0)

        # Decode trajectory from ego features
        trajectory, probability = self.trajectory_decoder(x[:, :, 0])  # x[:, :, 0] is [T, B, dim]

        # Predict other agents' trajectories
        x_agents = x[:, :, 1:A]  # [T, B, A-1, dim]
        x_agents_flat = x_agents.reshape(T * bs * (A - 1), self.dim)

        pred = self.agent_pred_fc1(x_agents_flat)
        pred = self.agent_pred_bn1(pred)
        pred = pred.reshape(T, bs, A - 1, -1)
        pred = self.agent_pred_lif1(pred)

        # Average over time
        pred = pred.mean(dim=0)  # [B, A-1, dim*2]
        pred = pred.reshape(bs * (A - 1), -1)
        pred = self.agent_pred_fc2(pred)
        prediction = pred.view(bs, A - 1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
        }

        # During inference, select best mode
        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat([
                output_trajectory[..., :2],
                angle.unsqueeze(-1)
            ], dim=-1)

        return out
