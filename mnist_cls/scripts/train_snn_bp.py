import sys

sys.path.append("/home/hzh/code/planning/planTF")
sys.path.append("/home/hzh/code/planning/planTF/mnist_cls")

import torch
from models.snn_mlp import SNNMLP
from trainers.snn_bp_trainer import SNNBPTrainer
from data.mnist_dataset import MNISTDataModule
from configs.snn_bp_config import SNN_BP_CONFIG
from spikingjelly.clock_driven import functional


def main():
    torch.manual_seed(42)
    data_module = MNISTDataModule(
        batch_size=SNN_BP_CONFIG["training"]["batch_size"],
        num_workers=SNN_BP_CONFIG["training"]["num_workers"],
        **SNN_BP_CONFIG["data"],
    )
    train_loader, val_loader, test_loader = data_module.get_loaders()

    model = SNNMLP(**SNN_BP_CONFIG["model"])
    trainer = SNNBPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **SNN_BP_CONFIG["training"],
        **SNN_BP_CONFIG["logging"],
    )

    history = trainer.train()

    import os

    best_ckpt = os.path.join(SNN_BP_CONFIG["logging"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(SNN_BP_CONFIG["training"]["device"]), target.to(
                    SNN_BP_CONFIG["training"]["device"]
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
