import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder
from src.target_builders.intent_target_builder import IntentTargetBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.trajectory_decoder import TrajectoryDecoder

# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModel(TorchModuleWrapper):
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
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        # Intent-related parameters
        intent_enabled=False,
        intent_time_horizon=2.0,
        intent_embed_dim=64,
        lateral_classes=5,
        longitudinal_classes=4,
    ) -> None:
        # Build target builders list
        target_builders_list = [EgoTrajectoryTargetBuilder(trajectory_sampling)]
        if intent_enabled:
            target_builders_list.append(
                IntentTargetBuilder(
                    time_horizon=intent_time_horizon,
                    sample_interval=0.1
                )
            )

        super().__init__(
            feature_builders=[feature_builder],
            target_builders=target_builders_list,
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.num_modes = num_modes

        # Intent-related attributes
        self.intent_enabled = intent_enabled
        self.intent_embed_dim = intent_embed_dim
        self.lateral_classes = lateral_classes
        self.longitudinal_classes = longitudinal_classes

        self.pos_emb = build_mlp(4, [dim] * 2)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
        )

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

        # Intent prediction heads and embeddings
        if self.intent_enabled:
            # Intent prediction: predict M intent hypotheses (one per trajectory mode)
            self.lateral_intent_head = nn.Linear(dim, num_modes * lateral_classes)
            self.longitudinal_intent_head = nn.Linear(dim, num_modes * longitudinal_classes)

            # Intent embeddings: convert predicted intent to embeddings
            self.lateral_intent_embed = nn.Embedding(lateral_classes, intent_embed_dim)
            self.longitudinal_intent_embed = nn.Embedding(longitudinal_classes, intent_embed_dim)

            # Fusion layer: combine lateral + longitudinal embeddings -> dim
            self.intent_fusion = nn.Linear(intent_embed_dim * 2, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat(
            [position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1
        )
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Extract ego feature (first token)
        ego_feature = x[:, 0]  # [B, dim]

        # Intent-conditioned trajectory generation
        if self.intent_enabled:
            # Predict M intent hypotheses (one per mode)
            lateral_logits = self.lateral_intent_head(ego_feature)  # [B, M*C_lat]
            longitudinal_logits = self.longitudinal_intent_head(ego_feature)  # [B, M*C_long]

            # Reshape to [B, M, C]
            lateral_logits = lateral_logits.view(bs, self.num_modes, self.lateral_classes)
            longitudinal_logits = longitudinal_logits.view(bs, self.num_modes, self.longitudinal_classes)

            # Get predicted intent indices (for embedding lookup)
            if self.training:
                # During training, use gumbel softmax for differentiable sampling
                lateral_probs = F.gumbel_softmax(lateral_logits, tau=1.0, hard=True)  # [B, M, C_lat]
                longitudinal_probs = F.gumbel_softmax(longitudinal_logits, tau=1.0, hard=True)  # [B, M, C_long]

                lateral_indices = lateral_probs.argmax(dim=-1)  # [B, M]
                longitudinal_indices = longitudinal_probs.argmax(dim=-1)  # [B, M]
            else:
                # During inference, use argmax
                lateral_indices = lateral_logits.argmax(dim=-1)  # [B, M]
                longitudinal_indices = longitudinal_logits.argmax(dim=-1)  # [B, M]

            # Compute intent embeddings
            lateral_embeds = self.lateral_intent_embed(lateral_indices)  # [B, M, intent_embed_dim]
            longitudinal_embeds = self.longitudinal_intent_embed(longitudinal_indices)  # [B, M, intent_embed_dim]

            # Fuse lateral and longitudinal embeddings
            intent_embeds = torch.cat([lateral_embeds, longitudinal_embeds], dim=-1)  # [B, M, 2*intent_embed_dim]
            intent_features = self.intent_fusion(intent_embeds)  # [B, M, dim]

            # Add intent features to ego feature for each mode
            ego_feature_expanded = ego_feature.unsqueeze(1).expand(-1, self.num_modes, -1)  # [B, M, dim]
            conditioned_features = ego_feature_expanded + intent_features  # [B, M, dim]

            # Decode trajectories from conditioned features
            # Reshape to [B*M, dim] for decoder
            conditioned_features_flat = conditioned_features.view(bs * self.num_modes, self.dim)

            # Use modified decoder forward (treating each mode independently)
            trajectory_flat = []
            probability_flat = []
            for i in range(self.num_modes):
                feat = conditioned_features[:, i, :]  # [B, dim]
                # Directly predict trajectory for this mode
                loc = self.trajectory_decoder.loc(feat).view(bs, 1, self.future_steps, 4)  # [B, 1, T, 4]
                pi = self.trajectory_decoder.pi(feat).squeeze(-1)  # [B]
                trajectory_flat.append(loc)
                probability_flat.append(pi)

            trajectory = torch.cat(trajectory_flat, dim=1)  # [B, M, T, 4]
            probability = torch.stack(probability_flat, dim=1)  # [B, M]

            # Add intent predictions to output
            out = {
                "trajectory": trajectory,
                "probability": probability,
                "prediction": self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2),
                "intent": {
                    "lateral": lateral_logits,  # [B, M, C_lat]
                    "longitudinal": longitudinal_logits,  # [B, M, C_long]
                }
            }
        else:
            # Original trajectory decoding (no intent conditioning)
            trajectory, probability = self.trajectory_decoder(ego_feature)
            prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

            out = {
                "trajectory": trajectory,
                "probability": probability,
                "prediction": prediction,
            }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out
