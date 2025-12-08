"""评估指标工具"""

import sys

sys.path.append("/home/hzh/code/planning/planTF")

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from spikingjelly.clock_driven import functional


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """计算准确率

    Args:
        predictions: 预测标签 [B]
        targets: 真实标签 [B]

    Returns:
        准确率（百分比）
    """
    correct = predictions.eq(targets).sum().item()
    total = targets.size(0)
    return 100.0 * correct / total


def compute_confusion_matrix(predictions: np.ndarray, targets: np.ndarray):
    """计算混淆矩阵

    Args:
        predictions: 预测标签
        targets: 真实标签

    Returns:
        混淆矩阵
    """
    return confusion_matrix(targets, predictions)


def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    """绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def evaluate_model(model, test_loader, device="cuda", is_snn=False):
    """完整模型评估

    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        is_snn: 是否为SNN模型

    Returns:
        评估结果字典
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # 处理不同输出格式
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output

            pred = logits.argmax(dim=1)

            all_predictions.append(pred.cpu())
            all_targets.append(target.cpu())

            # 重置SNN状态
            if is_snn:
                functional.reset_net(model)

    all_predictions = torch.cat(all_predictions).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # 计算指标
    accuracy = 100.0 * (all_predictions == all_targets).sum() / len(all_targets)
    cm = compute_confusion_matrix(all_predictions, all_targets)

    # 分类报告
    report = classification_report(
        all_targets, all_predictions, target_names=[str(i) for i in range(10)], digits=4
    )

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_predictions,
        "targets": all_targets,
    }
