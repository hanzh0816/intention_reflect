import sys

sys.path.append("/home/hzh/code/planning/planTF")
sys.path.append("/home/hzh/code/planning/planTF/mnist_cls")

import argparse
import torch
import os
from models.snn_mlp import SNNMLP
from trainers.snn_stdp_trainer import SNNSTDPTrainer
from data.mnist_dataset import MNISTDataModule
from configs.snn_stdp_config import SNN_STDP_CONFIG
from spikingjelly.activation_based import functional  # 更新为activation_based


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    args = parser.parse_args()

    torch.manual_seed(42)
    data_module = MNISTDataModule(
        batch_size=SNN_STDP_CONFIG["training"]["batch_size"],
        num_workers=SNN_STDP_CONFIG["training"]["num_workers"],
        **SNN_STDP_CONFIG["data"],
    )
    train_loader, val_loader, test_loader = data_module.get_loaders()

    model = SNNMLP(**SNN_STDP_CONFIG["model"])

    if args.pretrain_checkpoint and os.path.exists(args.pretrain_checkpoint):
        checkpoint = torch.load(args.pretrain_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded pretrained: {args.pretrain_checkpoint}")

    # Override epochs if specified
    training_cfg = SNN_STDP_CONFIG["training"].copy()
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
        print(f"Running with {args.epochs} epochs (overriding config)")

    trainer = SNNSTDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **training_cfg,
        **SNN_STDP_CONFIG["logging"],
        stdp_cfg=SNN_STDP_CONFIG["model"]["snn_cfg"]["stdp_cfg"],
    )

    history = trainer.train()

    best_ckpt = os.path.join(SNN_STDP_CONFIG["logging"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(SNN_STDP_CONFIG["training"]["device"]), target.to(
                    SNN_STDP_CONFIG["training"]["device"]
                )
                output = model(data)
                logits = output["logits"] if isinstance(output, dict) else output
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                functional.reset_net(model)
        print(f"Test Accuracy: {100. * correct / total:.2f}%")


if __name__ == "__main__":
    main()
