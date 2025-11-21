import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union

from .snn_utlis import LIFNeuron, get_default_neuron_config


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

    def __init__(
        self,
        in_features: int,
        num_classes: int = 5,
        hidden_dims: Optional[List[int]] = None,
        neuron_cfg: Optional[Dict] = None,
        dropout: float = 0.0,
        time_steps: int = 4,
    ):
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
        self.neuron_cfg["time_steps"] = time_steps

        # SNN分类器
        self.classifier = SNNClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            neuron_cfg=self.neuron_cfg,
            dropout=dropout,
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
            if hasattr(module, "lif") and hasattr(module.lif, "reset"):
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

    def __init__(
        self,
        in_features: int,
        num_classes: int = 4,
        hidden_dims: Optional[List[int]] = None,
        neuron_cfg: Optional[Dict] = None,
        dropout: float = 0.0,
        time_steps: int = 4,
    ):
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
        self.neuron_cfg["time_steps"] = time_steps

        # SNN分类器
        self.classifier = SNNClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            neuron_cfg=self.neuron_cfg,
            dropout=dropout,
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
            if hasattr(module, "lif") and hasattr(module.lif, "reset"):
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

    def __init__(
        self,
        in_features: int,
        lateral_classes: int = 5,
        longitudinal_classes: int = 4,
        lateral_hidden_dims: Optional[List[int]] = None,
        longitudinal_hidden_dims: Optional[List[int]] = None,
        neuron_cfg: Optional[Dict] = None,
        dropout: float = 0.0,
        time_steps: int = 4,
    ):
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
            time_steps=time_steps,
        )

        # 纵向意图头
        self.longitudinal_head = SNNLongitudinalIntentHead(
            in_features=in_features,
            num_classes=longitudinal_classes,
            hidden_dims=longitudinal_hidden_dims,
            neuron_cfg=neuron_cfg,
            dropout=dropout,
            time_steps=time_steps,
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

        return (
            lateral_predictions,
            longitudinal_predictions,
            lateral_confidence,
            longitudinal_confidence,
        )

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


class SNNIntentionMLPDecoder(nn.Module):
    """
    基于SNN的意图解码器 - MLP版本

    使用多层感知机 + LIF神经元处理ego车辆特征，
    输出意图感知的特征表示

    Args:
        dim: 特征维度
        depth: MLP层数（默认2）
        hidden_dim: 隐藏层维度（默认与输入相同）
        neuron_cfg: LIF神经元配置
        dropout: dropout比例
        time_steps: 时间步数
    """

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        hidden_dim: Optional[int] = None,
        neuron_cfg: Optional[Dict] = None,
        dropout: float = 0.0,
        time_steps: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.time_steps = time_steps

        if hidden_dim is None:
            hidden_dim = dim

        if neuron_cfg is None:
            neuron_cfg = get_default_neuron_config()

        # 更新神经元配置中的时间步
        self.neuron_cfg = neuron_cfg.copy()
        self.neuron_cfg["time_steps"] = time_steps

        # 时间维度扩展器
        self.time_expander = TimeDimExpander(time_steps=time_steps)

        # MLP层
        self.mlp_layers = nn.ModuleList()

        # 输入层
        self.mlp_layers.append(SNNLinearBlock(dim, hidden_dim, self.neuron_cfg, dropout=dropout))

        # 隐藏层
        for i in range(depth - 1):
            self.mlp_layers.append(
                SNNLinearBlock(hidden_dim, hidden_dim, self.neuron_cfg, dropout=dropout)
            )

        # 输出层（无LIF激活，保持特征空间）
        self.output_linear = nn.Linear(hidden_dim, dim, bias=True)

        # 时间维度平均
        self.time_average = TimeDimAverage()

        # LayerNorm（在最后应用）
        self.norm = nn.LayerNorm(dim)

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
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, ego_feature: torch.Tensor):
        """
        前向传播

        Args:
            ego_feature: [B, dim] ego车辆特征

        Returns:
            intention_feature: [B, dim] 意图特征
        """
        B, C = ego_feature.shape

        # 扩展时间维度
        x = self.time_expander(ego_feature)  # [T, B, C]

        # 通过MLP层
        for layer in self.mlp_layers:
            x = layer(x)  # [T, B, hidden_dim]

        # 输出线性层（无LIF激活）
        T = x.shape[0]
        x_flat = x.reshape(T * B, -1)  # [T*B, hidden_dim]
        x = self.output_linear(x_flat)  # [T*B, dim]
        x = x.reshape(T, B, C)  # [T, B, dim]

        # 时间维度平均
        x = self.time_average(x)  # [B, dim]

        # LayerNorm
        intention_feature = self.norm(x)

        return intention_feature

    def get_spike_rates(self):
        """获取各层的脉冲发放率（用于分析）"""
        spike_rates = {}

        # 从LIF神经元获取脉冲率
        for i, layer in enumerate(self.mlp_layers):
            if hasattr(layer, "lif") and hasattr(layer.lif, "get_spike_rate"):
                spike_rates[f"mlp_layer_{i}"] = layer.lif.get_spike_rate()

        return spike_rates


class SNNLinearBlock(nn.Module):
    """SNN线性块：Linear -> BatchNorm -> LIF"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neuron_cfg: Dict,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron_cfg = neuron_cfg

        # 线性层（无偏置，因为后面有BatchNorm）
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # BatchNorm（在通道维度上）
        self.bn = nn.BatchNorm1d(out_features)

        # Dropout（可选）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # LIF神经元
        self.lif = LIFNeuron(**neuron_cfg)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, L, C_in] 或 [T, B, C_in]
        Returns:
            [T, B, L, C_out] 或 [T, B, C_out]
        """
        original_shape = x.shape
        T = x.shape[0]

        # 处理不同输入形状
        if len(original_shape) == 4:
            # [T, B, L, C_in]
            T, B, L, C_in = x.shape
            x_flat = x.reshape(T * B * L, C_in)
            x = self.linear(x_flat)  # [T*B*L, C_out]

            # BatchNorm：需要 [N, C] 格式
            x = self.bn(x)  # [T*B*L, C_out]

            # 恢复形状
            x = x.reshape(T, B, L, self.out_features)

        elif len(original_shape) == 3:
            # [T, B, C_in]
            T, B, C_in = x.shape
            x_flat = x.reshape(T * B, C_in)
            x = self.linear(x_flat)  # [T*B, C_out]
            x = self.bn(x)  # [T*B, C_out]
            x = x.reshape(T, B, self.out_features)

        else:
            raise ValueError(f"Unsupported input shape: {original_shape}")

        # Dropout（在LIF之前）
        if self.dropout is not None:
            x = self.dropout(x)

        # LIF神经元
        x = self.lif(x)

        return x


class SNNClassifier(nn.Module):
    """SNN分类器：用于意图分类任务"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: list,
        neuron_cfg: Dict,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # 构建隐藏层
        layers = []
        prev_dim = in_features

        for hidden_dim in hidden_dims:
            layers.append(SNNLinearBlock(prev_dim, hidden_dim, neuron_cfg, dropout=dropout))
            prev_dim = hidden_dim

        # 输出层（无LIF激活）
        self.output_linear = nn.Linear(prev_dim, num_classes, bias=True)

        self.hidden_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, C_in] 时间步×批次×特征
        Returns:
            [B, num_classes] 分类logits
        """
        # 通过隐藏层
        for layer in self.hidden_layers:
            x = layer(x)

        # 最终线性层（无LIF激活）
        T, B, C = x.shape
        x_flat = x.reshape(T * B, C)
        x = self.output_linear(x_flat)  # [T*B, num_classes]
        x = x.reshape(T, B, self.num_classes)

        # 时间维度平均得到最终logits
        x = x.mean(dim=0)  # [B, num_classes]

        return x


class TimeDimAverage(nn.Module):
    """时间维度平均池化"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, ...] 时间步×批次×其他维度
        Returns:
            [B, ...] 时间平均后的输出
        """
        return x.mean(dim=0)


class TimeDimExpander(nn.Module):
    """扩展时间维度"""

    def __init__(self, time_steps: int = 4):
        super().__init__()
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, ...] 输入张量
        Returns:
            [T, B, ...] 扩展时间维度后的张量
        """
        return x.unsqueeze(0).repeat(self.time_steps, 1, *([1] * (x.dim() - 1)))
