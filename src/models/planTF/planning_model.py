import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.nuplan_feature_builder import NuplanFeatureBuilder

from .layers.common_layers import build_mlp
from .layers.transformer_encoder_layer import TransformerEncoderLayer
from .modules.agent_encoder import AgentEncoder
from .modules.map_encoder import MapEncoder
from .modules.intention_decoder import IntentionDecoder
from .modules.trajectory_decoder import TrajectoryDecoder

from .modules.snn_layers import SNNIntentHeads, SNNIntentionMLPDecoder
from .modules.snn_utlis import get_default_snn_config

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
        intent_time_horizon=2.0,
        intention_decoder_depth=2,
        lateral_classes=5,
        longitudinal_classes=4,
        snn_cfg=None,  # 统一的SNN配置
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        # 处理SNN配置：如果为None，使用默认配置
        if snn_cfg is None:
            snn_cfg = get_default_snn_config()

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps

        # Intent-related attributes
        self.intention_decoder_depth = intention_decoder_depth
        self.lateral_classes = lateral_classes
        self.longitudinal_classes = longitudinal_classes
        self.intent_time_horizon = intent_time_horizon

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
        self.intention_decoder = SNNIntentionMLPDecoder(
            dim=dim,
            depth=intention_decoder_depth,
            hidden_dim=dim * 4,
            snn_cfg=snn_cfg,
        )
        self.intent_head = SNNIntentHeads(
            in_features=dim,
            lateral_classes=lateral_classes,
            longitudinal_classes=longitudinal_classes,
            snn_cfg=snn_cfg,
        )

        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=2 * dim,
            num_modes=num_modes,
            future_steps=future_steps,
            out_channels=4,
        )
        self.agent_predictor = build_mlp(dim, [dim * 2, future_steps * 2], norm="ln")

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
        """
        Forward pass with new intent-conditioned single-mode architecture.

        Pipeline:
        1. Encoding (unchanged) -> ego_feature
        2. Intention Decoder -> intention_feature
        3. Intent Classification -> A_pred (lateral + longitudinal)
        4. Trajectory Decoder([ego_feature; intention_feature]) -> T_pred
        """
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        pos = torch.cat([position, torch.stack([angle.cos(), angle.sin()], dim=-1)], dim=-1)
        pos_embed = self.pos_emb(pos)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        # === Encoding (unchanged) ===
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Extract ego feature (first token)
        ego_feature = x[:, 0]  # [B, dim]

        # === Ego Trajectory Decoding ===

        # Step 1: Intention Decoder
        intention_feature = self.intention_decoder(ego_feature)  # [B, dim]

        # Step 2: Intent Classification from intention_feature
        lateral_result, longitudinal_result = self.intent_head(
            intention_feature
        )

        # 兼容BP和STDP两种模式
        if isinstance(lateral_result, dict):
            # STDP模式：提取logits并保存完整信息
            lateral_logits = lateral_result['logits']
            longitudinal_logits = longitudinal_result['logits']
            intent_dict = {
                "lateral": lateral_logits,
                "longitudinal": longitudinal_logits,
                # 保存完整的STDP信息供trainer使用
                "lateral_full": lateral_result,
                "longitudinal_full": longitudinal_result,
            }
        else:
            # BP模式：直接使用张量
            lateral_logits = lateral_result
            longitudinal_logits = longitudinal_result
            intent_dict = {
                "lateral": lateral_logits,
                "longitudinal": longitudinal_logits,
            }

        # Step 3: Trajectory Decoding from concatenated features
        combined_feature = torch.cat([ego_feature, intention_feature], dim=-1)  # [B, 2*dim]
        trajectory, probability = self.trajectory_decoder(combined_feature)
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "intent": intent_dict,
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out
