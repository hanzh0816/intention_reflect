import sys

sys.path.append("/home/hzh/code/planning/planTF")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import os
from spikingjelly.activation_based import functional
from src.models.planTF.modules.snn_stdp import SpikingJellySTDPWrapper


class SNNSTDPTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        stdp_cfg: dict = None,
        epochs: int = 100,
        checkpoint_dir: str = None,
        log_interval: int = 100,
        *args,
        **kwargs,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval

        # 冻结所有参数（STDP使用局部学习规则）
        for param in self.model.parameters():
            param.requires_grad = False

        # 创建SpikingJelly STDP wrapper
        self.stdp_wrapper = SpikingJellySTDPWrapper(
            layer=self.model.output_linear,
            neuron=self.model.output_lif,
            learning_rate=stdp_cfg.get('learning_rate', 0.001),
            tau_pre=stdp_cfg.get('tau_pre', 10.0),
            tau_post=stdp_cfg.get('tau_post', 10.0),
        )

        # 启用monitors (hidden layer的LIF神经元 -> output layer)
        self.stdp_wrapper.enable_monitors(self.model.hidden.lif)

        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "stdp_weight_change": [],
        }
        self.best_val_acc = 0.0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_weight_change = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [STDP]")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            with torch.no_grad():
                output = self.model(data)

            if not isinstance(output, dict):
                raise RuntimeError("Model must return dict in STDP mode")

            logits = output["logits"]
            loss = self.criterion(logits, target)

            # STDP weight update (使用SpikingJelly的trace更新公式)
            weight_delta = self.stdp_wrapper.update_weight(
                logits=logits,
                labels=target,
                pre_spikes_sequence=output["hidden_output"],
                post_spikes_sequence=output["spike_trains"],
            )
            self.stdp_wrapper.apply_update(weight_delta)

            # Reset network and STDP
            functional.reset_net(self.model)
            self.stdp_wrapper.reset()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            total_weight_change += weight_delta.abs().mean().item()

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
                )

        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": 100.0 * correct / total,
            "weight_change": total_weight_change / len(self.train_loader),
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                logits = output["logits"] if isinstance(output, dict) else output
                loss = self.criterion(logits, target)
                functional.reset_net(self.model)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.2f}%"}
                )

        return {"loss": total_loss / len(self.val_loader), "accuracy": 100.0 * correct / total}

    def train(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["stdp_weight_change"].append(train_metrics["weight_change"])

            print(
                f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}% | "
                f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                    epoch,
                    val_metrics["accuracy"],
                )

        print(f"STDP Training completed. Best Val Acc: {self.best_val_acc:.2f}%")
        return self.history

    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_acc": val_acc,
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.history = checkpoint["history"]
