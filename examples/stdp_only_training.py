"""
STDP-Only 训练示例

这个脚本演示如何使用 StdpOnlyTrainer 进行纯STDP训练：
1. 从checkpoint加载模型权重
2. 冻结除intent_head外的所有参数
3. 仅对intent_head进行STDP权重更新
4. 从头开始训练（新的lr、epoch等）

使用方法：
    python examples/stdp_only_training.py --ckpt-path /path/to/last.ckpt --epochs 50 --stdp-lr 0.001
"""

import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

# 导入必要的模块
from src.models.planTF.planning_model import PlanningModel
from src.models.planTF.stdp_only_trainer import StdpOnlyTrainer, load_checkpoint_for_stdp
from src.models.planTF.modules.snn_utlis import get_default_snn_config


def main(args):
    """主训练函数"""

    # ========== 模型初始化 ==========
    # 初始化模型，确保启用STDP
    snn_cfg = get_default_snn_config()
    snn_cfg["use_stdp"] = True  # 确保启用STDP模式

    model = PlanningModel(
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        num_heads=8,
        num_modes=6,
        intention_decoder_depth=2,
        lateral_classes=5,
        longitudinal_classes=4,
        snn_cfg=snn_cfg,
    )

    # ========== 加载checkpoint ==========
    print(f"从 {args.ckpt_path} 加载checkpoint...")
    load_checkpoint_for_stdp(args.ckpt_path, model)

    # ========== 创建STDP训练器 ==========
    trainer_module = StdpOnlyTrainer(
        model=model,
        stdp_a_pre=args.stdp_a_pre,
        stdp_a_post=args.stdp_a_post,
        epochs=args.epochs,
        log_metrics=True,
    )

    # ========== 配置PyTorch Lightning Trainer ==========
    callbacks = [
        ModelCheckpoint(
            monitor="loss/train_loss",
            save_top_k=3,
            mode="min",
            dirpath="checkpoints/stdp",
            filename="stdp-{epoch:02d}-{loss/train_loss:.4f}",
        )
    ]

    # 如果指定了wandb项目，使用WandbLogger
    logger = None
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name or "stdp-only-training",
            save_dir="./logs",
        )

    pl_trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,  # STDP训练中跳过检查
    )

    # ========== 开始训练 ==========
    print("\n" + "=" * 60)
    print("开始STDP-Only训练")
    print("=" * 60)
    print(f"STDP A_pre (LTP): {args.stdp_a_pre}")
    print(f"STDP A_post (LTD): {args.stdp_a_post}")
    print(f"Epochs: {args.epochs}")
    print(f"优化模式: Manual (直接weight.data更新)")
    print(f"所有参数: 冻结（requires_grad=False）")
    print("=" * 60 + "\n")

    # 获取数据加载器（这里假设你已经有数据加载器的配置）
    # train_loader, val_loader = setup_dataloaders(args)
    # pl_trainer.fit(trainer_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("注意：需要提供train_loader和val_loader才能启动训练")
    print("请根据项目的数据加载器配置进行调整")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STDP-Only 训练脚本")

    # 模型和checkpoint相关
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="要加载的checkpoint文件路径",
    )

    # STDP参数（直接weight update，不需要学习率）
    parser.add_argument(
        "--stdp-a-pre",
        type=float,
        default=0.01,
        help="STDP LTP幅度（正值）",
    )
    parser.add_argument(
        "--stdp-a-post",
        type=float,
        default=-0.01,
        help="STDP LTD幅度（负值）",
    )

    # 训练配置
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练总epoch数",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="使用的GPU设备数量",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="每N步记录一次指标",
    )


    # Wandb配置
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb项目名称（可选）",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run名称（可选）",
    )

    args = parser.parse_args()
    main(args)
