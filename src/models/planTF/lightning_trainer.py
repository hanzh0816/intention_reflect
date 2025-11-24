import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR
from src.utils.intent_cls import classify_intent_from_cached_trajectory

import spikingjelly.clock_driven.functional as functional

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        intent_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.intent_loss_weight = intent_loss_weight

    def on_fit_start(self) -> None:
        metrics_collection = MetricCollection(
            {
                "minADE1": minADE(k=1).to(self.device),
                "minADE6": minADE(k=6).to(self.device),
                "minFDE1": minFDE(k=1).to(self.device),
                "minFDE6": minFDE(k=6).to(self.device),
                "MR": MR().to(self.device),
            }
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        features, _, _ = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        # 处理STDP更新（如果启用）
        if hasattr(self.model, 'intent_head') and self.model.intent_head.use_stdp:
            stdp_metrics = self._stdp_update_step(res, features["feature"].data)
            if stdp_metrics:
                self._log_stdp_metrics(stdp_metrics, prefix)

        functional.reset_net(self.model)

        return losses["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        trajectory, probability, prediction = (
            res["trajectory"],
            res["probability"],
            res["prediction"],
        )
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        ego_target_pos, ego_target_heading = targets[:, 0, :, :2], targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack([ego_target_heading.cos(), ego_target_heading.sin()], dim=-1),
            ],
            dim=-1,
        )
        agent_target, agent_mask = targets[:, 1:], valid_mask[:, 1:]

        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        agent_reg_loss = F.smooth_l1_loss(prediction[agent_mask], agent_target[agent_mask][:, :2])

        lateral_logits = res["intent"]["lateral"]  # [B, C_lat]
        longitudinal_logits = res["intent"]["longitudinal"]  # [B, C_long]

        # Intent classification loss: CE(A_pred, A_gt)
        # Build ground-truth intent labels from expert (ego) target trajectory
        # targets: [B, A, T, 3] with (x, y, heading)
        bs = targets.shape[0]
        ego_expert_traj = targets[:, 0]  # [B, T, 3]

        # Try to read time horizon and dt from model config; fall back to sensible defaults
        time_horizon = getattr(getattr(self.model, "model", self.model), "intent_time_horizon", 2.0)
        try:
            sample_interval = self.model.feature_builders[0].sample_interval  # type: ignore[attr-defined]
        except Exception:
            sample_interval = 0.1

        lateral_list = []
        longitudinal_list = []
        for i in range(bs):
            lat_idx, lon_idx = classify_intent_from_cached_trajectory(
                trajectory_data=ego_expert_traj[i],
                time_horizon=time_horizon,
                sample_interval=sample_interval,
            )
            lateral_list.append(lat_idx)
            longitudinal_list.append(lon_idx)

        lateral_target = torch.as_tensor(
            lateral_list, dtype=torch.long, device=lateral_logits.device
        )
        longitudinal_target = torch.as_tensor(
            longitudinal_list, dtype=torch.long, device=longitudinal_logits.device
        )

        lateral_intent_loss = F.cross_entropy(lateral_logits, lateral_target)
        longitudinal_intent_loss = F.cross_entropy(longitudinal_logits, longitudinal_target)
        intent_cls_loss = lateral_intent_loss + longitudinal_intent_loss

        loss = (
            ego_reg_loss + ego_cls_loss + agent_reg_loss + self.intent_loss_weight * intent_cls_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss,
            "cls_loss": ego_cls_loss,
            "prediction_loss": agent_reg_loss,
            "intent_cls_loss": intent_cls_loss,
        }

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _stdp_update_step(self, res, data) -> Dict:
        """
        执行STDP权重更新（仅当模型使用STDP模式时调用）

        Args:
            res: 模型输出
            data: 批次数据

        Returns:
            stdp_metrics: STDP相关指标字典
        """
        intent_head = self.model.intent_head

        # 从res中获取完整的STDP信息
        # STDP模式：lateral_full/longitudinal_full包含spikes和hidden outputs
        # BP模式：这些键不存在，返回的是logits张量
        lateral_result = res["intent"].get("lateral_full", res["intent"]["lateral"])
        longitudinal_result = res["intent"].get("longitudinal_full", res["intent"]["longitudinal"])

        # 获取targets中的intent标签
        targets = data["agent"]["target"]
        bs = targets.shape[0]
        ego_expert_traj = targets[:, 0]  # [B, T, 3]

        time_horizon = getattr(getattr(self.model, "model", self.model), "intent_time_horizon", 2.0)
        try:
            sample_interval = self.model.feature_builders[0].sample_interval
        except Exception:
            sample_interval = 0.1

        lateral_list = []
        longitudinal_list = []
        for i in range(bs):
            lat_idx, lon_idx = classify_intent_from_cached_trajectory(
                trajectory_data=ego_expert_traj[i],
                time_horizon=time_horizon,
                sample_interval=sample_interval,
            )
            lateral_list.append(lat_idx)
            longitudinal_list.append(lon_idx)

        lateral_labels = torch.as_tensor(
            lateral_list, dtype=torch.long, device=res["intent"]["lateral"].device
        )
        longitudinal_labels = torch.as_tensor(
            longitudinal_list, dtype=torch.long, device=res["intent"]["longitudinal"].device
        )

        try:
            # 检查是否为STDP模式（lateral_result应该是字典）
            if isinstance(lateral_result, dict) and isinstance(longitudinal_result, dict):
                stdp_metrics = intent_head.stdp_update(
                    lateral_result=lateral_result,
                    longitudinal_result=longitudinal_result,
                    lateral_labels=lateral_labels,
                    longitudinal_labels=longitudinal_labels,
                )
                return stdp_metrics
            else:
                # BP模式，不执行STDP更新
                return {}
        except Exception as e:
            logger.warning(f"STDP更新失败: {e}")
            return {}

    def _log_stdp_metrics(self, stdp_metrics: Dict, prefix: str) -> None:
        """
        记录STDP相关指标

        Args:
            stdp_metrics: STDP指标字典
            prefix: 前缀（train/val）
        """
        if not stdp_metrics:
            return

        for head_name, metrics in stdp_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    self.log(
                        f"stdp/{prefix}_{head_name}_{metric_name}",
                        value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                    )

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            # Log original metrics dict (may contain keys like "val/minFDE1")
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
            # Additionally log flat aliases for checkpoint filename formatting
            # e.g., "val_minFDE1", "val_minADE1"
            if prefix == "val":
                alias_keys = ["val/minFDE1", "val/minADE1"]
                for k in alias_keys:
                    if k in metrics:
                        self.log(
                            k.replace("/", "_"),
                            metrics[k],
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                        )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
