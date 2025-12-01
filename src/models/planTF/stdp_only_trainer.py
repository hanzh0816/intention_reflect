import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

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
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from omegaconf import OmegaConf

from src.utils.intent_cls import classify_intent_from_cached_trajectory
import spikingjelly.clock_driven.functional as functional

logger = logging.getLogger(__name__)


class StdpOnlyTrainer(pl.LightningModule):
    """
    STDP-Only 训练器：仅使用STDP规则直接更新intent_head权重，其他参数冻结

    核心特性：
    - 直接修改权重（weight.data），不使用梯度优化
    - 冻结所有参数（requires_grad=False）
    - 不使用优化器（manual_optimization 模式）
    - 仅在训练阶段更新权重，验证/测试阶段只评估

    工作流：
    1. 从checkpoint加载模型权重（仅权重，不加载optimizer state）
    2. 冻结所有参数（因为STDP直接修改weight.data）
    3. 仅对intent_head的输出层进行STDP权重更新
    4. 从头开始训练（新的epoch计数）

    Args:
        model: TorchModuleWrapper模型
        epochs: 训练总epoch数
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        epochs: int = 50,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # 关键：启用手动优化模式（STDP直接修改权重，不使用优化器）
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.epochs = epochs
        self.intent_loss_weight = 1.0

        # 检查模型是否有intent_head且启用了STDP
        if not hasattr(self.model, "intent_head"):
            raise RuntimeError("模型未包含intent_head模块")

        if not self.model.intent_head.use_stdp:
            raise RuntimeError("intent_head未启用STDP模式。请在配置中设置use_stdp=True")

    def on_fit_start(self) -> None:
        # Initialize metrics
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

        # 保存配置快照（仅在主进程执行）
        if self.trainer.is_global_zero:
            self._save_config_snapshot()
        self._freeze_all_parameters()

        # 打印冻结信息
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"✓ STDP-Only训练已启用\n"
            f"  - 可训练参数: {trainable_params} / 总参数: {total_params}\n"
            f"  - 自动优化: {not self.automatic_optimization}\n"
            f"  - 权重更新方式: 直接修改weight.data"
        )

        # 保存配置快照（仅在主进程执行）
        if self.trainer.is_global_zero:
            self._save_config_snapshot()

    def _freeze_all_parameters(self) -> None:
        """冻结所有参数

        STDP训练直接修改weight.data，不需要requires_grad=True
        所有参数统一冻结，包括intent_head
        """
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("已冻结所有参数（STDP通过direct weight update训练）")

    def _step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str) -> None:
        """
        训练步骤

        Args:
            batch: 批次数据
            prefix: 步骤前缀 ('train'/'val'/'test')
        """
        features, _, _ = batch
        res = self.forward(features["feature"].data)

        losses = self._compute_objectives(res, features["feature"].data)
        metrics = self._compute_metrics(res, features["feature"].data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        if prefix == "train":
            # 设置模型为训练模式（为了获取SNN脉冲信息）
            self.model.train()
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
        执行STDP权重更新

        直接修改weight.data，不使用梯度：
        weight.data = weight.data + learning_rate * weight_delta

        Args:
            res: 模型输出
            data: 批次数据

        Returns:
            stdp_metrics: STDP相关指标字典
        """
        intent_head = self.model.intent_head

        # 从res中获取完整的STDP信息
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

        # 分类intent标签
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
            # 检查是否为STDP模式
            if isinstance(lateral_result, dict) and isinstance(longitudinal_result, dict):
                stdp_metrics = intent_head.stdp_update(
                    lateral_result=lateral_result,
                    longitudinal_result=longitudinal_result,
                    lateral_labels=lateral_labels,
                    longitudinal_labels=longitudinal_labels,
                )
                return stdp_metrics
            else:
                logger.warning("模型未返回STDP格式的输出。请确保intent_head的use_stdp=True")
                return {}
        except Exception as e:
            logger.warning(f"STDP更新失败: {e}")
            return {}

    def _log_stdp_metrics(self, stdp_metrics: Dict, prefix: str) -> None:
        """
        记录STDP相关指标

        Args:
            stdp_metrics: STDP指标字典
            prefix: 前缀（train/val/test）
        """
        if not stdp_metrics:
            return

        for head_name, metrics in stdp_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    try:
                        float_value = (
                            float(value) if not isinstance(value, torch.Tensor) else value.item()
                        )
                    except (ValueError, TypeError):
                        logger.debug(f"跳过STDP指标 {head_name}/{metric_name}: 无法转换为float")
                        continue

                    self.log(
                        f"stdp/{prefix}_{head_name}_{metric_name}",
                        float_value,
                        on_step=True,
                        on_epoch=True,
                        sync_dist=True,
                    )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> None:
        """
        训练步骤 - 手动优化模式

        不返回loss（因为不使用自动优化）
        直接执行STDP权重更新
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

    def forward(self, features: FeaturesType) -> TargetsType:
        """前向传播"""
        return self.model(features)

    def configure_optimizers(self):
        """
        配置优化器

        STDP训练使用manual_optimization模式，直接修改weight.data，不需要优化器
        """
        logger.info("STDPOnlyTrainer: Configuring optimizers (none needed)")
        return None

    def _save_config_snapshot(self) -> None:
        """
        从Hydra生成的配置中创建可访问的副本和易读摘要
        """
        try:
            work_dir = os.getcwd()
            hydra_config_path = os.path.join(work_dir, "code", "hydra", "config.yaml")

            if not os.path.exists(hydra_config_path):
                logger.debug(f"Hydra config not found at {hydra_config_path}")
                return

            cfg = OmegaConf.load(hydra_config_path)

            # 复制config.yaml到顶级目录
            config_copy_path = os.path.join(work_dir, "config.yaml")
            with open(config_copy_path, "w") as f:
                OmegaConf.save(cfg, f)

            # 生成易读摘要
            summary_path = os.path.join(work_dir, "config_summary.txt")
            with open(summary_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("STDP-ONLY TRAINING CONFIGURATION SUMMARY\n")
                f.write("=" * 80 + "\n\n")

                try:
                    commit_hash = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=os.path.dirname(
                                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                            ),
                            stderr=subprocess.DEVNULL,
                        )
                        .decode("utf-8")
                        .strip()
                    )
                    f.write("COMMIT INFORMATION:\n")
                    f.write(f"  Commit Hash: {commit_hash}\n\n")
                except Exception:
                    logger.debug("Failed to retrieve commit hash")

                f.write("KEY STDP TRAINING PARAMETERS:\n")
                f.write(f"  Epochs: {self.epochs}\n")
                f.write(f"  Optimization Mode: Manual (STDP direct weight update)\n")
                f.write(f"  Optimizer: None (not needed)\n\n")

                f.write("=" * 80 + "\n")
                f.write("FULL CONFIGURATION (YAML):\n")
                f.write("=" * 80 + "\n")
                OmegaConf.save(cfg, f)

            logger.info(f"配置快照已保存到 {work_dir}")

        except Exception as e:
            logger.warning(f"保存配置快照失败: {e}")


def load_checkpoint_for_stdp(checkpoint_path: str, model: TorchModuleWrapper) -> None:
    """
    从checkpoint加载模型权重（仅权重，不加载optimizer state）

    Args:
        checkpoint_path: checkpoint文件路径
        model: 目标模型

    Raises:
        FileNotFoundError: 如果checkpoint文件不存在

    Example:
        >>> model = PlanningModel(...)
        >>> load_checkpoint_for_stdp("/path/to/last.ckpt", model)
        >>> trainer = StdpOnlyTrainer(model, stdp_a_pre=0.01, epochs=50)
    """
    from src.planners.planner_utils import load_checkpoint
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    state_dict = load_checkpoint(checkpoint_path)
    model.load_state_dict(state_dict)
    logger.info(f"✓ 已从 {checkpoint_path} 加载权重（仅权重，不加载optimizer state）")
