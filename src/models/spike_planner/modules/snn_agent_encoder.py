"""
SNN version of AgentEncoder module

This module encodes agent information including historical trajectories
and current states using SNN components.
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

from ..layers.snn_common_layers import build_snn_mlp
from ..layers.snn_embedding import SNNNATSequenceEncoder
from ..layers.snn_attention import SNNMultiheadAttention


class SNNAgentEncoder(nn.Module):
    """
    SNN version of AgentEncoder

    Encodes agent historical trajectories and current states with SNN components.
    """

    def __init__(
        self,
        state_channel=6,
        history_channel=9,
        dim=128,
        hist_steps=21,
        use_ego_history=False,
        drop_path=0.2,
        state_attn_encoder=True,
        state_dropout=0.75,
        tau=2.0,
        backend='torch',
    ):
        super().__init__()
        self.dim = dim
        self.state_channel = state_channel
        self.use_ego_history = use_ego_history
        self.hist_steps = hist_steps
        self.state_attn_encoder = state_attn_encoder

        # History encoder using SNN NAT
        self.history_encoder = SNNNATSequenceEncoder(
            in_chans=history_channel,
            embed_dim=dim // 4,
            drop_path_rate=drop_path,
            tau=tau,
            backend=backend,
        )

        if not use_ego_history:
            if not self.state_attn_encoder:
                # Simple MLP encoder for ego state
                self.ego_state_fc1 = nn.Linear(state_channel, dim, bias=False)
                self.ego_state_bn1 = nn.BatchNorm1d(dim)
                self.ego_state_lif1 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)

                self.ego_state_fc2 = nn.Linear(dim, dim, bias=False)
                self.ego_state_bn2 = nn.BatchNorm1d(dim)
                self.ego_state_lif2 = MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend)
            else:
                # Attention-based encoder for ego state
                self.ego_state_emb = SNNStateAttentionEncoder(
                    state_channel, dim, state_dropout, tau=tau, backend=backend
                )

        self.type_emb = nn.Embedding(4, dim)

    @staticmethod
    def to_vector(feat, valid_mask):
        """Convert features to velocity vectors"""
        vec_mask = valid_mask[..., :-1] & valid_mask[..., 1:]

        while len(vec_mask.shape) < len(feat.shape):
            vec_mask = vec_mask.unsqueeze(-1)

        return torch.where(
            vec_mask,
            feat[:, :, 1:, ...] - feat[:, :, :-1, ...],
            torch.zeros_like(feat[:, :, 1:, ...]),
        )

    def forward(self, data, time_steps=4):
        """
        Args:
            data: Input feature dict
            time_steps: Number of SNN time steps (T)

        Returns:
            [T, B, A, C] - encoded agent features
        """
        T_snn = time_steps
        T = self.hist_steps

        position = data["agent"]["position"][:, :, :T]
        heading = data["agent"]["heading"][:, :, :T]
        velocity = data["agent"]["velocity"][:, :, :T]
        shape = data["agent"]["shape"][:, :, :T]
        category = data["agent"]["category"].long()
        valid_mask = data["agent"]["valid_mask"][:, :, :T]

        heading_vec = self.to_vector(heading, valid_mask)
        valid_mask_vec = valid_mask[..., 1:] & valid_mask[..., :-1]

        agent_feature = torch.cat([
            self.to_vector(position, valid_mask),
            self.to_vector(velocity, valid_mask),
            torch.stack([heading_vec.cos(), heading_vec.sin()], dim=-1),
            shape[:, :, 1:],
            valid_mask_vec.float().unsqueeze(-1),
        ], dim=-1)

        bs, A, T_hist, C_feat = agent_feature.shape
        agent_feature = agent_feature.view(bs * A, T_hist, C_feat)
        valid_agent_mask = valid_mask.any(-1).flatten()

        # Prepare input for SNN history encoder: [T_snn, B, C, L]
        # agent_feature is [bs*A, T_hist, C_feat], need to transpose to [bs*A, C_feat, T_hist]
        agent_feature_valid = agent_feature[valid_agent_mask].permute(0, 2, 1)

        # Expand to time dimension for SNN
        agent_feature_snn = agent_feature_valid.unsqueeze(0).repeat(T_snn, 1, 1, 1)

        # Encode with SNN NAT: [T_snn, B_valid, C_feat, T_hist] -> [T_snn, B_valid, dim]
        x_agent_tmp = self.history_encoder(agent_feature_snn)

        # Scatter back to full batch
        x_agent = torch.zeros(T_snn, bs * A, self.dim, device=position.device)
        x_agent[:, valid_agent_mask] = x_agent_tmp
        x_agent = x_agent.view(T_snn, bs, A, self.dim)

        # Handle ego vehicle state
        if not self.use_ego_history:
            ego_feature = data["current_state"][:, :self.state_channel]

            if not self.state_attn_encoder:
                # Simple MLP encoding
                x_ego = self.ego_state_fc1(ego_feature)
                x_ego = self.ego_state_bn1(x_ego)
                x_ego = x_ego.unsqueeze(0).repeat(T_snn, 1, 1)
                x_ego = self.ego_state_lif1(x_ego)

                x_ego_flat = x_ego.reshape(T_snn * bs, self.dim)
                x_ego = self.ego_state_fc2(x_ego_flat)
                x_ego = self.ego_state_bn2(x_ego)
                x_ego = x_ego.reshape(T_snn, bs, self.dim)
                x_ego = self.ego_state_lif2(x_ego)
            else:
                # Attention-based encoding
                x_ego = self.ego_state_emb(ego_feature, time_steps=T_snn)

            x_agent[:, :, 0] = x_ego

        # Add type embedding (not time-dependent)
        x_type = self.type_emb(category)  # [B, A, dim]
        x_type = x_type.unsqueeze(0).repeat(T_snn, 1, 1, 1)  # [T, B, A, dim]

        return x_agent + x_type


class SNNStateAttentionEncoder(nn.Module):
    """
    SNN version of StateAttentionEncoder for encoding ego vehicle state

    Uses attention to aggregate different state features.
    """

    def __init__(self, state_channel, dim, state_dropout=0.5, tau=2.0, backend='torch'):
        super().__init__()

        self.state_channel = state_channel
        self.state_dropout = state_dropout
        self.dim = dim

        # Linear projection for each state channel
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, dim, bias=False),
                nn.BatchNorm1d(dim),
                MultiStepLIFNode(tau=tau, detach_reset=True, backend=backend),
            )
            for _ in range(state_channel)
        ])

        # Attention mechanism
        self.attn = SNNMultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            qkv_bias=True,
            tau=tau,
            backend=backend,
        )

        self.pos_embed = nn.Parameter(torch.Tensor(1, state_channel, dim))
        self.query = nn.Parameter(torch.Tensor(1, 1, dim))

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x, time_steps=4):
        """
        Args:
            x: [B, state_channel] - ego state features
            time_steps: Number of SNN time steps

        Returns:
            [T, B, dim] - encoded ego state
        """
        T = time_steps
        B = x.shape[0]

        # Project each channel independently
        x_embed_list = []
        for i, linear_module in enumerate(self.linears):
            x_i = x[:, i:i+1]  # [B, 1]

            # Process through Linear + BatchNorm (without time dimension)
            x_i = linear_module[0](x_i)  # Linear: [B, 1] -> [B, dim]
            x_i = linear_module[1](x_i)  # BatchNorm: [B, dim] -> [B, dim]

            # Expand to time dimension for LIF
            x_i = x_i.unsqueeze(0).repeat(T, 1, 1)  # [T, B, dim]

            # Pass through LIF
            x_i = linear_module[2](x_i)  # LIF: [T, B, dim] -> [T, B, dim]

            x_embed_list.append(x_i)

        # Stack embeddings: [T, B, state_channel, dim]
        x_embed = torch.stack(x_embed_list, dim=2)

        # Add positional embedding
        # pos_embed: [1, state_channel, dim] -> [T, B, state_channel, dim]
        pos_embed = self.pos_embed.unsqueeze(0).repeat(T, B, 1, 1)
        x_embed = x_embed + pos_embed

        # Apply dropout mask during training
        if self.training and self.state_dropout > 0:
            # Keep first 3 tokens visible, randomly dropout others
            visible_tokens = torch.zeros((B, 3), device=x.device, dtype=torch.bool)
            dropout_tokens = (
                torch.rand((B, self.state_channel - 3), device=x.device) < self.state_dropout
            )
            key_padding_mask = torch.cat([visible_tokens, dropout_tokens], dim=1)
        else:
            key_padding_mask = None

        # Prepare query: [T, B, 1, dim]
        query = self.query.unsqueeze(0).repeat(T, B, 1, 1)

        # Attention: both query and x_embed are [T, B, L, C]
        # For attention, we treat x_embed as keys and values
        x_state = self.attn(
            x_embed,
            key_padding_mask=key_padding_mask,
        )  # [T, B, state_channel, dim]

        # Use query to extract single output
        # Simple approach: average over state channels
        x_state = x_state.mean(dim=2)  # [T, B, dim]

        return x_state
