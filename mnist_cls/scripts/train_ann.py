import sys

sys.path.append("/home/hzh/code/planning/planTF/mnist_cls")

import torch
from models.ann_mlp import ANNMLP
from trainers.ann_trainer import ANNTrainer
from data.mnist_dataset import MNISTDataModule
from configs.ann_config import ANN_CONFIG


def main():
    torch.manual_seed(42)
    data_module = MNISTDataModule(
        batch_size=ANN_CONFIG["training"]["batch_size"],
        num_workers=ANN_CONFIG["training"]["num_workers"],
        **ANN_CONFIG["data"],
    )
    train_loader, val_loader, test_loader = data_module.get_loaders()

    model = ANNMLP(**ANN_CONFIG["model"])
    trainer = ANNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=ANN_CONFIG["training"]["device"],
        lr=ANN_CONFIG["training"]["lr"],
        weight_decay=ANN_CONFIG["training"]["weight_decay"],
        epochs=ANN_CONFIG["training"]["epochs"],
        checkpoint_dir=ANN_CONFIG["logging"]["checkpoint_dir"],
        log_interval=ANN_CONFIG["logging"]["log_interval"],
    )

    history = trainer.train()

    import os

    best_ckpt = os.path.join(ANN_CONFIG["logging"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(ANN_CONFIG["training"]["device"]), target.to(
                    ANN_CONFIG["training"]["device"]
                )
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        print(f"Test Accuracy: {100. * correct / total:.2f}%")


if __name__ == "__main__":
    main()
