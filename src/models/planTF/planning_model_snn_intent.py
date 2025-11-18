"""
支持SNN意图解码的PlanningModel
集成基于SNN的IntentionDecoder和IntentHeads
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
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
from .modules.trajectory_decoder import TrajectoryDecoder

# SNN意图模块
from .modules.snn_intention_mlp_decoder import (
    SNNIntentionMLPDecoder, create_snn_intention_mlp_decoder
)
from .modules.snn_intention_transformer_decoder import (
    SNNIntentionTransformerDecoder, create_snn_intention_transformer_decoder
)
from .modules.snn_intent_heads import SNNIntentHeads, create_snn_intent_heads


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class PlanningModelSNNIntent(TorchModuleWrapper):
    """
    支持SNN意图解码的PlanningModel

    提供两种SNN IntentionDecoder选择：
    1. MLP版本：多层感知机 + LIF神经元
    2. Transformer版本：SNN Transformer编码器 + 注意力机制
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
        feature_builder: NuplanFeatureBuilder = NuplanFeatureBuilder(),
        # 意图相关参数
        intent_time_horizon=2.0,
        intention_decoder_depth=2,
        lateral_classes=5,
        longitudinal_classes=4,
        # SNN相关参数
        use_snn_intention=True,  # 是否使用SNN意图解码器
        snn_intention_type="mlp",  # "mlp" 或 "transformer"
        snn_intention_size="standard",  # "tiny", "small", "standard", "large"
        snn_intent_head_size="standard",  # "small", "standard", "large", "xlarge"
        snn_time_steps=4,  # SNN时间步数
        snn_neuron_cfg=None,  # SNN神经元配置
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps

        # 意图相关属性
        self.intention_decoder_depth = intention_decoder_depth
        self.lateral_classes = lateral_classes
        self.longitudinal_classes = longitudinal_classes
        self.intent_time_horizon = intent_time_horizon

        # SNN相关属性
        self.use_snn_intention = use_snn_intention
        self.snn_intention_type = snn_intention_type
        self.snn_time_steps = snn_time_steps
        self.snn_neuron_cfg = snn_neuron_cfg

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

        # SNN意图解码器
        if self.use_snn_intention:
            if snn_intention_type == "mlp":
                self.intention_decoder = create_snn_intention_mlp_decoder(
                    decoder_type="standard",
                    dim=dim,
                    depth=intention_decoder_depth,
                    neuron_cfg=snn_neuron_cfg,
                    time_steps=snn_time_steps,
                    size=snn_intention_size,
                )
            elif snn_intention_type == "transformer":
                self.intention_decoder = create_snn_intention_transformer_decoder(
                    decoder_type="standard",
                    dim=dim,
                    depth=intention_decoder_depth,
                    num_heads=num_heads,
                    neuron_cfg=snn_neuron_cfg,
                    time_steps=snn_time_steps,
                    size=snn_intention_size,
                )
            else:
                raise ValueError(f"Unsupported SNN intention type: {snn_intention_type}")

            # SNN意图分类头
            self.intent_heads = create_snn_intent_heads(
                in_features=dim,
                lateral_classes=lateral_classes,
                longitudinal_classes=longitudinal_classes,
                size=snn_intent_head_size,
                neuron_cfg=snn_neuron_cfg,
                time_steps=snn_time_steps,
            )
        else:
            # 回退到传统实现（可选）
            from .modules.intention_decoder import IntentionDecoder
            self.intention_decoder = IntentionDecoder(
                dim=dim,
                depth=intention_decoder_depth,
                num_heads=num_heads,
                drop_path=drop_path,
            )
            # 传统的线性分类头
            self.lateral_intent_head = nn.Linear(dim, lateral_classes)
            self.longitudinal_intent_head = nn.Linear(dim, longitudinal_classes)
            self.intent_heads = None

        # 轨迹解码器接受拼接的特征
        self.trajectory_decoder = TrajectoryDecoder(
            embed_dim=2 * dim,  # ego特征 + 意图特征
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

    def enable_snn_time_step_processing(self):
        """
        启用SNN时间步处理模式
        这会配置所有SNN模块以支持时间序列输入
        """
        if self.use_snn_intention:
            # SNN模块会自动处理时间步
            pass

    def reset_snn_neurons(self):
        """
        重置所有SNN神经元的状态
        在新的场景或批次开始时调用
        """
        if self.use_snn_intention and hasattr(self.intention_decoder, 'reset_neurons'):
            self.intention_decoder.reset_neurons()

        if self.use_snn_intention and self.intent_heads and hasattr(self.intent_heads, 'reset_neurons'):
            self.intent_heads.reset_neurons()

    def get_snn_spike_rates(self):
        """
        获取SNN模块的脉冲发放率
        用于分析和可视化
        """
        spike_rates = {}

        if self.use_snn_intention and hasattr(self.intention_decoder, 'get_spike_rates'):
            decoder_rates = self.intention_decoder.get_spike_rates()
            for key, value in decoder_rates.items():
                spike_rates[f"intention_decoder_{key}"] = value

        return spike_rates

    def forward(self, data):
        """
        支持SNN的前向传播

        Pipeline:
        1. Encoding -> ego_feature, x (所有agents和map的特征)
        2. SNN Intention Decoder -> intention_feature
        3. SNN Intent Classification -> lateral_logits, longitudinal_logits
        4. SNN Trajectory Decoder([ego_feature; intention_feature]) -> trajectory, probability
        """
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, :self.history_steps]
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

        # === 编码阶段 ===
        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)

        x = torch.cat([x_agent, x_polygon], dim=1) + pos_embed

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # 提取ego特征（第一个token）
        ego_feature = x[:, 0]  # [B, dim]

        # === SNN意图解码 ===
        intention_feature = self.intention_decoder(ego_feature)  # [B, dim]

        # === SNN意图分类 ===
        if self.use_snn_intention and self.intent_heads:
            lateral_logits, longitudinal_logits = self.intent_heads(intention_feature)
        else:
            # 传统的线性分类头
            lateral_logits = self.lateral_intent_head(intention_feature)
            longitudinal_logits = self.longitudinal_intent_head(intention_feature)

        # === 轨迹解码（使用拼接特征） ===
        combined_feature = torch.cat([ego_feature, intention_feature], dim=-1)  # [B, 2*dim]
        trajectory, probability = self.trajectory_decoder(combined_feature)
        prediction = self.agent_predictor(x[:, 1:A]).view(bs, -1, self.future_steps, 2)

        out = {
            "trajectory": trajectory,
            "probability": probability,
            "prediction": prediction,
            "intent": {
                "lateral": lateral_logits,
                "longitudinal": longitudinal_logits,
            },
        }

        if not self.training:
            best_mode = probability.argmax(dim=-1)
            output_trajectory = trajectory[torch.arange(bs), best_mode]
            angle = torch.atan2(output_trajectory[..., 3], output_trajectory[..., 2])
            out["output_trajectory"] = torch.cat(
                [output_trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
            )

        return out


class PlanningModelSNNIntentLightningTrainer(pl.LightningModule):
    """
    支持SNN意图解码的Lightning Trainer

    添加了SNN特定的监控和调试功能
    """

    def __init__(
        self,
        model: PlanningModelSNNIntent,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        intent_loss_weight: float = 1.0,
        monitor_snn_spike_rates: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.intent_loss_weight = intent_loss_weight
        self.monitor_snn_spike_rates = monitor_snn_spike_rates

    def on_train_epoch_start(self) -> None:
        """每个训练周期开始时重置SNN神经元状态"""
        if hasattr(self.model, 'reset_snn_neurons'):
            self.model.reset_snn_neurons()

    def on_validation_epoch_start(self) -> None:
        """每个验证周期开始时重置SNN神经元状态"""
        if hasattr(self.model, 'reset_snn_neurons'):
            self.model.reset_snn_neurons()

    def training_step(self, batch, batch_idx):
        """训练步骤，支持SNN监控"""
        loss = super().training_step(batch, batch_idx)

        # 监控SNN脉冲率
        if self.monitor_snn_spike_rates and hasattr(self.model, 'get_snn_spike_rates'):
            spike_rates = self.model.get_snn_spike_rates()
            for key, rate in spike_rates.items():
                self.log(f"snn_spike_rates/train_{key}", rate, on_step=True, on_epoch=False)

        return loss