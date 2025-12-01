import logging
import os
import subprocess
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
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

        # 检查模型是否有intent_head且启用了STDP
        if not hasattr(self.model, "intent_head"):
            raise RuntimeError("模型未包含intent_head模块")

        if not self.model.intent_head.use_stdp:
            raise RuntimeError("intent_head未启用STDP模式。请在配置中设置use_stdp=True")

    def on_fit_start(self) -> None:
        """训练开始时冻结所有参数"""
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

        # 设置模型为训练模式（为了获取SNN脉冲信息）
        self.model.train()

        # 前向传播 - 获取脉冲和隐藏输出用于STDP更新
        res = self.forward(features["feature"].data)

        # 执行STDP权重更新（直接修改weight.data）
        stdp_metrics = self._stdp_update_step(res, features["feature"].data)

        # 记录指标
        if stdp_metrics:
            self._log_stdp_metrics(stdp_metrics, prefix)

        # 重置SNN网络状态（重置膜电位）
        functional.reset_net(self.model)

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
    ) -> None:
        """
        验证步骤 - 仅评估，不更新权重

        使用torch.no_grad()确保不修改权重
        """
        features, _, _ = batch

        # 进入eval模式，确保不修改权重
        self.model.eval()

        with torch.no_grad():
            res = self.forward(features["feature"].data)

            # 获取logits用于计算准确率
            lateral_logits = res["intent"]["lateral"]
            longitudinal_logits = res["intent"]["longitudinal"]

            # 获取targets中的intent标签（仅用于计算准确率）
            targets = features["feature"].data["agent"]["target"]
            bs = targets.shape[0]
            ego_expert_traj = targets[:, 0]

            time_horizon = getattr(
                getattr(self.model, "model", self.model), "intent_time_horizon", 2.0
            )
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
                lateral_list, dtype=torch.long, device=lateral_logits.device
            )
            longitudinal_labels = torch.as_tensor(
                longitudinal_list, dtype=torch.long, device=longitudinal_logits.device
            )

            # 计算准确率
            lateral_pred = torch.argmax(lateral_logits, dim=-1)
            longitudinal_pred = torch.argmax(longitudinal_logits, dim=-1)

            lateral_acc = (lateral_pred == lateral_labels).float().mean()
            longitudinal_acc = (longitudinal_pred == longitudinal_labels).float().mean()

            self.log("val/lateral_acc", lateral_acc, on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                "val/longitudinal_acc",
                longitudinal_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # 重置SNN网络状态
        functional.reset_net(self.model)

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> None:
        """
        测试步骤 - 仅评估，不更新权重

        使用torch.no_grad()确保不修改权重
        """
        features, _, _ = batch

        # 进入eval模式，确保不修改权重
        self.model.eval()

        with torch.no_grad():
            res = self.forward(features["feature"].data)

            # 获取logits用于计算准确率
            lateral_logits = res["intent"]["lateral"]
            longitudinal_logits = res["intent"]["longitudinal"]

            # 获取targets中的intent标签（仅用于计算准确率）
            targets = features["feature"].data["agent"]["target"]
            bs = targets.shape[0]
            ego_expert_traj = targets[:, 0]

            time_horizon = getattr(
                getattr(self.model, "model", self.model), "intent_time_horizon", 2.0
            )
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
                lateral_list, dtype=torch.long, device=lateral_logits.device
            )
            longitudinal_labels = torch.as_tensor(
                longitudinal_list, dtype=torch.long, device=longitudinal_logits.device
            )

            # 计算准确率
            lateral_pred = torch.argmax(lateral_logits, dim=-1)
            longitudinal_pred = torch.argmax(longitudinal_logits, dim=-1)

            lateral_acc = (lateral_pred == lateral_labels).float().mean()
            longitudinal_acc = (longitudinal_pred == longitudinal_labels).float().mean()

            self.log("test/lateral_acc", lateral_acc, on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                "test/longitudinal_acc",
                longitudinal_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # 重置SNN网络状态
        functional.reset_net(self.model)

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
