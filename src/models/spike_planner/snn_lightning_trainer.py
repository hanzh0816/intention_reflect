"""
SNN version of Lightning Trainer for PlanTF

This module provides the PyTorch Lightning training wrapper for the SNN-based
planning model, including training/validation loops, metrics, and optimization.
"""

import logging
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
from spikingjelly.clock_driven import functional

from src.metrics import MR, minADE, minFDE
from src.optim.warmup_cos_lr import WarmupCosLR

logger = logging.getLogger(__name__)


class SNNLightningTrainer(pl.LightningModule):
    """
    PyTorch Lightning training module for SNN-based planning model

    This trainer handles the training loop, validation, metrics computation,
    and optimization for the SNN planning model.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
    ):
        """
        Args:
            model: The SNN planning model
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            epochs: Total number of training epochs
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def on_fit_start(self):
        """Initialize metrics at the start of training"""
        metrics_collection = MetricCollection({
            "minADE1": minADE(k=1).to(self.device),
            "minADE6": minADE(k=6).to(self.device),
            "minFDE1": minFDE(k=1).to(self.device),
            "minFDE6": minFDE(k=6).to(self.device),
            "MR": MR().to(self.device),
        })
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
        }

    def _step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        prefix: str,
    ) -> torch.Tensor:
        """
        Execute a single training/validation step

        Args:
            batch: Input batch
            prefix: 'train' or 'val'

        Returns:
            Loss tensor
        """
        features, _, _ = batch

        # Forward pass
        res = self.forward(features["feature"].data)

        # Compute losses and metrics
        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"]

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        """
        Compute training objectives (losses)

        Args:
            res: Model output dict
            data: Ground truth data

        Returns:
            Dict of loss values
        """
        trajectory, probability, prediction = (
            res["trajectory"],
            res["probability"],
            res["prediction"],
        )
        targets = data["agent"]["target"]
        valid_mask = data["agent"]["valid_mask"][:, :, -trajectory.shape[-2]:]

        # Extract ego target
        ego_target_pos = targets[:, 0, :, :2]
        ego_target_heading = targets[:, 0, :, 2]
        ego_target = torch.cat([
            ego_target_pos,
            torch.stack([
                ego_target_heading.cos(),
                ego_target_heading.sin()
            ], dim=-1),
        ], dim=-1)

        agent_target = targets[:, 1:]
        agent_mask = valid_mask[:, 1:]

        # Ego trajectory regression loss (best mode)
        ade = torch.norm(trajectory[..., :2] - ego_target[:, None, :, :2], dim=-1)
        best_mode = torch.argmin(ade.sum(-1), dim=-1)
        best_traj = trajectory[torch.arange(trajectory.shape[0]), best_mode]
        ego_reg_loss = F.smooth_l1_loss(best_traj, ego_target)

        # Ego mode classification loss
        ego_cls_loss = F.cross_entropy(probability, best_mode.detach())

        # Agent prediction loss
        agent_reg_loss = F.smooth_l1_loss(
            prediction[agent_mask],
            agent_target[agent_mask][:, :2]
        )

        # Total loss
        loss = ego_reg_loss + ego_cls_loss + agent_reg_loss

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss,
            "cls_loss": ego_cls_loss,
            "prediction_loss": agent_reg_loss,
        }

    def _compute_metrics(
        self,
        output,
        data,
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics

        Args:
            output: Model output
            data: Ground truth data
            prefix: 'train' or 'val'

        Returns:
            Dict of metric values
        """
        metrics = self.metrics[prefix](output, data["agent"]["target"][:, 0])
        return metrics

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ):
        """
        Log losses and metrics

        Args:
            loss: Main loss value
            objectives: Dict of loss components
            metrics: Dict of metric values
            prefix: 'train' or 'val'
            loss_name: Name for the main loss
        """
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
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Validation step

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self,
        batch: Tuple[FeaturesType, TargetsType, ScenarioListType],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Test step

        Args:
            batch: Test batch
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Forward pass through the model

        Args:
            features: Input features

        Returns:
            Model predictions
        """
        return self.model(features)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Reset SNN states at the end of each training batch

        This is crucial for SNN models to prevent state accumulation across batches.
        """
        functional.reset_net(self.model)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        """Reset SNN states at the end of each validation batch"""
        functional.reset_net(self.model)

    def on_test_batch_end(self, outputs, batch, batch_idx):
        """Reset SNN states at the end of each test batch"""
        functional.reset_net(self.model)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configure optimizers and learning rate schedulers

        Returns:
            Optimizer and scheduler configuration
        """
        # Separate parameters into decay and no-decay groups
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
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
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)

        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both decay and no_decay: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in any group: {param_dict.keys() - union_params}"

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

        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Cosine learning rate scheduler with warmup
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]
