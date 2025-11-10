import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
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
from src.utils.intent_classification import classify_intent_from_trajectory

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        intent_loss_weight=1.0,
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
        # Multi-modal metrics with k=6 modes
        metrics_collection = MetricCollection(
            {
                "ADE": minADE(k=6).to(self.device),
                "FDE": minFDE(k=6).to(self.device),
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
        features, targets, _ = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data, targets)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]

    def _compute_objectives(self, res, data, targets=None) -> Dict[str, torch.Tensor]:
        """
        Compute loss with multi-modal intent-enhanced architecture.

        Loss components:
        1. L_ego_reg: Smooth L1 loss for best trajectory mode (selected by ADE)
        2. L_ego_cls: Cross-entropy for mode classification
        3. L_intent_cls: Cross-entropy for intent classification (lateral + longitudinal)
        4. L_agent: Agent prediction loss (unchanged)
        """
        trajectory = res["trajectory"]  # [B, num_modes, T, 4] - multi-modal
        probability = res["probability"]  # [B, num_modes]
        prediction = res["prediction"]  # [B, A-1, T, 2]

        traj_targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2] :]

        # Ground truth ego trajectory
        ego_target_pos, ego_target_heading = traj_targets[:, 0, :, :2], traj_targets[:, 0, :, 2]
        ego_target = torch.cat(
            [
                ego_target_pos,
                torch.stack(
                    [ego_target_heading.cos(), ego_target_heading.sin()], dim=-1
                ),
            ],
            dim=-1,
        )  # [B, T, 4]

        agent_target, agent_mask = traj_targets[:, 1:], valid_mask[:, 1:]

        # === 1. Ego Trajectory Regression Loss (Multi-modal) ===
        # Select best mode based on ADE
        batch_size, num_modes = trajectory.shape[0], trajectory.shape[1]

        # Compute ADE for each mode: [B, num_modes]
        ade_per_mode = torch.norm(
            trajectory[:, :, :, :2] - ego_target[:, None, :, :2],
            p=2,
            dim=-1
        ).mean(dim=-1)

        # Select best mode (minimum ADE) for each sample
        best_mode = ade_per_mode.argmin(dim=-1)  # [B]

        # Extract best trajectory for each sample
        best_traj = trajectory[torch.arange(batch_size), best_mode]  # [B, T, 4]

        # Compute regression loss on best trajectory
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)

        # === 2. Ego Mode Classification Loss ===
        # Classify which mode is best (based on ADE selection)
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        # === 3. Agent Prediction Loss (unchanged) ===
        agent_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        # Initialize objectives dict
        objectives_dict = {
            "ego_reg_loss": ego_reg_loss,
            "ego_cls_loss": ego_cls_loss,
            "agent_loss": agent_loss,
        }

        total_loss = ego_reg_loss + ego_cls_loss + agent_loss

        # === 4. Intent Classification Loss ===
        if "intent" in res and targets is not None and "intent" in targets:
            intent_labels = targets["intent"]
            lateral_target = intent_labels.lateral_intent  # [B]
            longitudinal_target = intent_labels.longitudinal_intent  # [B]

            lateral_logits = res["intent"]["lateral"]  # [B, C_lat]
            longitudinal_logits = res["intent"]["longitudinal"]  # [B, C_long]

            # Intent classification loss: CE(A_pred, A_gt)
            lateral_intent_loss = F.cross_entropy(lateral_logits, lateral_target)
            longitudinal_intent_loss = F.cross_entropy(longitudinal_logits, longitudinal_target)
            intent_cls_loss = lateral_intent_loss + longitudinal_intent_loss

            total_loss = total_loss + self.intent_loss_weight * intent_cls_loss

            objectives_dict["intent_cls_loss"] = intent_cls_loss
            objectives_dict["lateral_intent_loss"] = lateral_intent_loss
            objectives_dict["longitudinal_intent_loss"] = longitudinal_intent_loss

        objectives_dict["loss"] = total_loss

        return objectives_dict

    def _compute_metrics(self, output, data, prefix) -> Dict[str, torch.Tensor]:
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

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
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
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
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
