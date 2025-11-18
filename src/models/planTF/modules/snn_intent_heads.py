"""
基于SNN的意图分类头
实现lateral_intent_head和longitudinal_intent_head的SNN版本
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, List

from .snn_utils import SNNClassifier, TimeDimExpander, TimeDimAverage, get_default_neuron_config


class SNNLateralIntentHead(nn.Module):
    """
    基于SNN的横向意图分类头

    分类类别：
    - turn_left: 左转
    - turn_right: 右转
    - shift_left: 左偏移
    - shift_right: 右偏移
    - stay_in_lane: 保持车道

    Args:
        in_features: 输入特征维度
        num_classes: 分类数（默认5）
        hidden_dims: 隐藏层维度列表（默认[64]）
        neuron_cfg: LIF神经元配置
        dropout: dropout比例
        time_steps: 时间步数
    """

    def __init__(self, in_features: int, num_classes: int = 5,
                 hidden_dims: Optional[List[int]] = None,
                 neuron_cfg: Optional[Dict] = None, dropout: float = 0.0,
                 time_steps: int = 4):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.time_steps = time_steps

        if hidden_dims is None:
            hidden_dims = [64]  # 默认单层隐藏层

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        # 更新神经元配置中的时间步
        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # SNN分类器
        self.classifier = SNNClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            neuron_cfg=self.neuron_cfg,
            dropout=dropout
        )

        # 时间维度扩展器（用于输入特征）
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # 特征归一化
        self.feature_norm = nn.LayerNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, intention_feature: torch.Tensor):
        """
        前向传播

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            lateral_logits: [B, num_classes] 横向意图logits
        """
        # 特征归一化
        x = self.feature_norm(intention_feature)  # [B, in_features]

        # 扩展时间维度
        x = self.time_expander(x)  # [T, B, in_features]

        # SNN分类
        lateral_logits = self.classifier(x)  # [B, num_classes]

        return lateral_logits

    def get_prediction_confidence(self, intention_feature: torch.Tensor):
        """
        获取预测置信度

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            confidence: [B] 置信度分数
            predictions: [B] 预测类别
        """
        logits = self.forward(intention_feature)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predictions = torch.max(probabilities, dim=-1)
        return confidence, predictions

    def reset_neurons(self):
        """重置神经元状态"""
        # 重置分类器中的神经元
        for module in self.classifier.modules():
            if hasattr(module, 'lif') and hasattr(module.lif, 'reset'):
                module.lif.reset()


class SNNLongitudinalIntentHead(nn.Module):
    """
    基于SNN的纵向意图分类头

    分类类别：
    - accelerate: 加速
    - maintain_speed: 保持速度
    - decelerate: 减速
    - stop: 停止

    Args:
        in_features: 输入特征维度
        num_classes: 分类数（默认4）
        hidden_dims: 隐藏层维度列表（默认[64]）
        neuron_cfg: LIF神经元配置
        dropout: dropout比例
        time_steps: 时间步数
    """

    def __init__(self, in_features: int, num_classes: int = 4,
                 hidden_dims: Optional[List[int]] = None,
                 neuron_cfg: Optional[Dict] = None, dropout: float = 0.0,
                 time_steps: int = 4):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.time_steps = time_steps

        if hidden_dims is None:
            hidden_dims = [64]  # 默认单层隐藏层

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        # 更新神经元配置中的时间步
        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg['time_steps'] = time_steps

        # SNN分类器
        self.classifier = SNNClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            neuron_cfg=self.neuron_cfg,
            dropout=dropout
        )

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # 特征归一化
        self.feature_norm = nn.LayerNorm(in_features)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, intention_feature: torch.Tensor):
        """
        前向传播

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            longitudinal_logits: [B, num_classes] 纵向意图logits
        """
        # 特征归一化
        x = self.feature_norm(intention_feature)  # [B, in_features]

        # 扩展时间维度
        x = self.time_expander(x)  # [T, B, in_features]

        # SNN分类
        longitudinal_logits = self.classifier(x)  # [B, num_classes]

        return longitudinal_logits

    def get_prediction_confidence(self, intention_feature: torch.Tensor):
        """
        获取预测置信度

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            confidence: [B] 置信度分数
            predictions: [B] 预测类别
        """
        logits = self.forward(intention_feature)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predictions = torch.max(probabilities, dim=-1)
        return confidence, predictions

    def reset_neurons(self):
        """重置神经元状态"""
        # 重置分类器中的神经元
        for module in self.classifier.modules():
            if hasattr(module, 'lif') and hasattr(module.lif, 'reset'):
                module.lif.reset()


class SNNIntentHeads(nn.Module):
    """
    统一的SNN意图分类头，包含横向和纵向意图分类

    Args:
        in_features: 输入特征维度
        lateral_classes: 横向意图类别数（默认5）
        longitudinal_classes: 纵向意图类别数（默认4）
        lateral_hidden_dims: 横向分类隐藏层维度
        longitudinal_hidden_dims: 纵向分类隐藏层维度
        neuron_cfg: LIF神经元配置
        dropout: dropout比例
        time_steps: 时间步数
    """

    def __init__(self, in_features: int, lateral_classes: int = 5,
                 longitudinal_classes: int = 4,
                 lateral_hidden_dims: Optional[List[int]] = None,
                 longitudinal_hidden_dims: Optional[List[int]] = None,
                 neuron_cfg: Optional[Dict] = None, dropout: float = 0.0,
                 time_steps: int = 4):
        super().__init__()
        self.in_features = in_features
        self.lateral_classes = lateral_classes
        self.longitudinal_classes = longitudinal_classes

        if lateral_hidden_dims is None:
            lateral_hidden_dims = [64]
        if longitudinal_hidden_dims is None:
            longitudinal_hidden_dims = [64]

        # 横向意图头
        self.lateral_head = SNNLateralIntentHead(
            in_features=in_features,
            num_classes=lateral_classes,
            hidden_dims=lateral_hidden_dims,
            neuron_cfg=neuron_cfg,
            dropout=dropout,
            time_steps=time_steps
        )

        # 纵向意图头
        self.longitudinal_head = SNNLongitudinalIntentHead(
            in_features=in_features,
            num_classes=longitudinal_classes,
            hidden_dims=longitudinal_hidden_dims,
            neuron_cfg=neuron_cfg,
            dropout=dropout,
            time_steps=time_steps
        )

    def forward(self, intention_feature: torch.Tensor):
        """
        前向传播

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            lateral_logits: [B, lateral_classes] 横向意图logits
            longitudinal_logits: [B, longitudinal_classes] 纵向意图logits
        """
        lateral_logits = self.lateral_head(intention_feature)
        longitudinal_logits = self.longitudinal_head(intention_feature)

        return lateral_logits, longitudinal_logits

    def get_predictions(self, intention_feature: torch.Tensor):
        """
        获取预测结果和置信度

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            lateral_predictions: [B] 横向意图预测
            longitudinal_predictions: [B] 纵向意图预测
            lateral_confidence: [B] 横向置信度
            longitudinal_confidence: [B] 纵向置信度
        """
        lateral_logits, longitudinal_logits = self.forward(intention_feature)

        # 计算概率和预测
        lateral_probs = torch.softmax(lateral_logits, dim=-1)
        longitudinal_probs = torch.softmax(longitudinal_logits, dim=-1)

        lateral_confidence, lateral_predictions = torch.max(lateral_probs, dim=-1)
        longitudinal_confidence, longitudinal_predictions = torch.max(longitudinal_probs, dim=-1)

        return (lateral_predictions, longitudinal_predictions,
                lateral_confidence, longitudinal_confidence)

    def get_intent_vectors(self, intention_feature: torch.Tensor):
        """
        获取意图向量表示（用于可视化或下游任务）

        Args:
            intention_feature: [B, in_features] 意图特征

        Returns:
            lateral_vectors: [B, lateral_classes] 横向意图向量
            longitudinal_vectors: [B, longitudinal_classes] 纵向意图向量
        """
        lateral_logits, longitudinal_logits = self.forward(intention_feature)
        return lateral_logits, longitudinal_logits

    def reset_neurons(self):
        """重置所有神经元状态"""
        self.lateral_head.reset_neurons()
        self.longitudinal_head.reset_neurons()


# 预定义配置
SNN_INTENT_HEAD_CONFIGS = {
    "small": {
        "lateral_hidden_dims": [32],
        "longitudinal_hidden_dims": [32],
        "dropout": 0.05
    },
    "standard": {
        "lateral_hidden_dims": [64],
        "longitudinal_hidden_dims": [64],
        "dropout": 0.1
    },
    "large": {
        "lateral_hidden_dims": [128, 64],
        "longitudinal_hidden_dims": [128, 64],
        "dropout": 0.15
    },
    "xlarge": {
        "lateral_hidden_dims": [256, 128, 64],
        "longitudinal_hidden_dims": [256, 128, 64],
        "dropout": 0.2
    }
}


def create_snn_intent_heads(in_features: int, lateral_classes: int = 5,
                          longitudinal_classes: int = 4, size: str = "standard",
                          neuron_cfg: Optional[Dict] = None, time_steps: int = 4,
                          **kwargs):
    """
    工厂函数：创建SNN意图分类头

    Args:
        in_features: 输入特征维度
        lateral_classes: 横向意图类别数
        longitudinal_classes: 纵向意图类别数
        size: "small", "standard", "large", "xlarge"
        neuron_cfg: 神经元配置
        time_steps: 时间步数
        **kwargs: 额外的参数

    Returns:
        SNNIntentHeads 实例
    """
    if size not in SNN_INTENT_HEAD_CONFIGS:
        raise ValueError(f"Unsupported size: {size}. Available: {list(SNN_INTENT_HEAD_CONFIGS.keys())}")

    config = SNN_INTENT_HEAD_CONFIGS[size].copy()
    config.update(kwargs)

    return SNNIntentHeads(
        in_features=in_features,
        lateral_classes=lateral_classes,
        longitudinal_classes=longitudinal_classes,
        neuron_cfg=neuron_cfg,
        time_steps=time_steps,
        **config
    )