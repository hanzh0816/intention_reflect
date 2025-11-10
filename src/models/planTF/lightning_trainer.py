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
        consistency_loss_weight=1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.intent_loss_weight = intent_loss_weight
        self.consistency_loss_weight = consistency_loss_weight

    def on_fit_start(self) -> None:
        # Single-mode metrics (no multi-modal k=6 anymore)
        metrics_collection = MetricCollection(
            {
                "ADE": minADE(k=1).to(self.device),
                "FDE": minFDE(k=1).to(self.device),
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
        Compute loss with new single-mode intent-conditioned architecture.

        Loss components:
        1. L_trajectory: L1 loss between predicted and ground truth trajectory
        2. L_intent_cls: Cross-entropy for intent classification (lateral + longitudinal)
        3. L_consistency: Consistency between reclassified trajectory and predicted intent
        4. L_agent: Agent prediction loss (unchanged)
        """
        trajectory = res["trajectory"]  # [B, T, 4] - single mode
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

        # === 1. Trajectory Loss ===
        trajectory_loss = F.l1_loss(trajectory, ego_target)

        # === 2. Agent Prediction Loss (unchanged) ===
        agent_loss = F.smooth_l1_loss(
            prediction[agent_mask], agent_target[agent_mask][:, :2]
        )

        # Initialize objectives dict
        objectives_dict = {
            "trajectory_loss": trajectory_loss,
            "agent_loss": agent_loss,
        }

        total_loss = trajectory_loss + agent_loss

        # === 3. Intent Classification Loss ===
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

            # === 4. Consistency Loss: Reclassify predicted trajectory ===
            # Detach predicted intent to stop gradient flow
            lateral_pred_detached = lateral_logits.detach()
            longitudinal_pred_detached = longitudinal_logits.detach()

            # Reclassify predicted trajectories
            batch_size = trajectory.shape[0]
            lateral_reclassified = []
            longitudinal_reclassified = []

            for i in range(batch_size):
                traj_i = trajectory[i].detach().cpu().numpy()  # [T, 4]
                lat_idx, long_idx = classify_intent_from_trajectory(
                    traj_i, dt=0.1, time_horizon=2.0
                )
                lateral_reclassified.append(lat_idx)
                longitudinal_reclassified.append(long_idx)

            lateral_reclassified = torch.tensor(
                lateral_reclassified, dtype=torch.long, device=trajectory.device
            )  # [B]
            longitudinal_reclassified = torch.tensor(
                longitudinal_reclassified, dtype=torch.long, device=trajectory.device
            )  # [B]

            # Consistency loss: CE(A_reclassified, A_pred.detach())
            lateral_consistency_loss = F.cross_entropy(
                lateral_pred_detached, lateral_reclassified
            )
            longitudinal_consistency_loss = F.cross_entropy(
                longitudinal_pred_detached, longitudinal_reclassified
            )
            consistency_loss = lateral_consistency_loss + longitudinal_consistency_loss

            total_loss = total_loss + self.consistency_loss_weight * consistency_loss

            objectives_dict["consistency_loss"] = consistency_loss
            objectives_dict["lateral_consistency_loss"] = lateral_consistency_loss
            objectives_dict["longitudinal_consistency_loss"] = longitudinal_consistency_loss

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
