"""统一评估脚本"""

import sys
sys.path.append('/home/hzh/code/planning/planTF')
sys.path.append('/home/hzh/code/planning/planTF/mnist_cls')

import argparse
import torch
import os
from models.ann_mlp import ANNMLP
from models.snn_mlp import SNNMLP
from data.mnist_dataset import MNISTDataModule
from utils.metrics import evaluate_model, plot_confusion_matrix
from configs.ann_config import ANN_CONFIG
from configs.snn_bp_config import SNN_BP_CONFIG
from configs.snn_stdp_config import SNN_STDP_CONFIG


def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST models')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['ann', 'snn_bp', 'snn_stdp'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save-cm', action='store_true',
                       help='Save confusion matrix')

    args = parser.parse_args()

    print("="*60)
    print(f"Evaluating {args.model_type.upper()} Model")
    print("="*60)

    # 加载数据
    print("\nLoading MNIST test dataset...")
    data_module = MNISTDataModule(batch_size=args.batch_size)
    _, _, test_loader = data_module.get_loaders()
    print(f"  Test samples: {len(test_loader.dataset)}")

    # 加载模型
    print(f"\nLoading {args.model_type} model...")
    if args.model_type == 'ann':
        model = ANNMLP(**ANN_CONFIG['model'])
        is_snn = False
    elif args.model_type == 'snn_bp':
        model = SNNMLP(**SNN_BP_CONFIG['model'])
        is_snn = True
    else:  # snn_stdp
        model = SNNMLP(**SNN_STDP_CONFIG['model'])
        is_snn = True

    # 加载权重
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)

    print(f"  Checkpoint loaded: {args.checkpoint}")
    print(f"  Trained epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # 评估
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)

    results = evaluate_model(model, test_loader, args.device, is_snn=is_snn)

    print(f"\n{'='*60}")
    print(f"Test Accuracy: {results['accuracy']:.2f}%")
    print(f"{'='*60}")
    print("\nClassification Report:")
    print(results['classification_report'])

    # 保存混淆矩阵
    if args.save_cm:
        cm_path = args.checkpoint.replace('.pth', '_confusion_matrix.png')
        plot_confusion_matrix(results['confusion_matrix'], cm_path)

    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
