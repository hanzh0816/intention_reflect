"""
SNN模块归总 - 基于SNN的意图理解模块

该模块提供了基于脉冲神经网络的完整意图理解解决方案：

主要组件：
1. SNN Utils - 基础SNN工具和组件
2. SNN Intention MLP Decoder - 基于MLP的意图解码器
3. SNN Intention Transformer Decoder - 基于Transformer的意图解码器
4. SNN Intent Heads - 横向和纵向意图分类头
5. PlanningModel with SNN Intent - 完整的SNN意图规划模型

架构特点：
- 支持多种SNN神经元类型（LIF, PLIF, IF, ILIF）
- 统一的时间步处理接口
- 与传统ANN模块兼容
- 灵活的配置和扩展性
- GPU加速支持（通过SpikingJelly）

作者：基于PlanTF框架的SNN意图扩展
"""

# 基础SNN工具
from .snn_utils import (
    LIFNeuron, SNNLinearBlock, SNNClassifier,
    TimeDimAverage, TimeDimExpander,
    get_default_neuron_config, check_spiking_jelly_available
)

# SNN意图解码器 - MLP版本
from .snn_intention_mlp_decoder import (
    SNNIntentionMLPDecoder, SNNIntentionShallowDecoder,
    create_snn_intention_mlp_decoder,
    SNN_INTENTION_MLP_CONFIGS
)

# SNN意图解码器 - Transformer版本
from .snn_intention_transformer_decoder import (
    SNNIntentionTransformerDecoder, SNNIntentionLightTransformerDecoder,
    create_snn_intention_transformer_decoder,
    SNN_INTENTION_TRANSFORMER_CONFIGS
)

# SNN意图分类头
from .snn_intent_heads import (
    SNNLateralIntentHead, SNNLongitudinalIntentHead, SNNIntentHeads,
    create_snn_intent_heads,
    SNN_INTENT_HEAD_CONFIGS
)

# SNN注意力机制（内部使用）
from .snn_attention import SNNMultiheadAttention, SNNTransformerEncoderLayer, SNNMlp

# SNN意图规划模型 - 条件导入
try:
    from ..planning_model_snn_intent import (
        PlanningModelSNNIntent, PlanningModelSNNIntentLightningTrainer
    )
    PLANNING_MODEL_AVAILABLE = True
except ImportError:
    try:
        from src.models.planTF.planning_model_snn_intent import (
            PlanningModelSNNIntent, PlanningModelSNNIntentLightningTrainer
        )
        PLANNING_MODEL_AVAILABLE = True
    except ImportError:
        PLANNING_MODEL_AVAILABLE = False
        PlanningModelSNNIntent = None
        PlanningModelSNNIntentLightningTrainer = None

# 向后兼容性别名
if PLANNING_MODEL_AVAILABLE:
    PlanningModelWithSNNIntent = PlanningModelSNNIntent
    SNNIntentionPlanningModel = PlanningModelSNNIntent

__all__ = [
    # 基础工具
    'LIFNeuron', 'SNNLinearBlock', 'SNNClassifier',
    'TimeDimAverage', 'TimeDimExpander',
    'get_default_neuron_config', 'check_spiking_jelly_available',

    # SNN MLP意图解码器
    'SNNIntentionMLPDecoder', 'SNNIntentionShallowDecoder',
    'create_snn_intention_mlp_decoder',
    'SNN_INTENTION_MLP_CONFIGS',

    # SNN Transformer意图解码器
    'SNNIntentionTransformerDecoder', 'SNNIntentionLightTransformerDecoder',
    'create_snn_intention_transformer_decoder',
    'SNN_INTENTION_TRANSFORMER_CONFIGS',

    # SNN意图分类头
    'SNNLateralIntentHead', 'SNNLongitudinalIntentHead', 'SNNIntentHeads',
    'create_snn_intent_heads',
    'SNN_INTENT_HEAD_CONFIGS',

    # SNN注意力机制
    'SNNMultiheadAttention', 'SNNTransformerEncoderLayer', 'SNNMlp',
]

# 只有当规划模型可用时才添加到__all__
if PLANNING_MODEL_AVAILABLE:
    __all__ += [
        'PlanningModelSNNIntent', 'PlanningModelSNNIntentLightningTrainer',
        'PlanningModelWithSNNIntent', 'SNNIntentionPlanningModel'
    ]

# 版本信息
__version__ = "1.0.0"
__author__ = "PlanTF SNN Team"

# 预定义配置组合（便于快速使用）
SNN_INTENT_PRESETS = {
    "lightweight": {
        "intention_decoder": {
            "type": "mlp",
            "size": "small"
        },
        "intent_heads": {
            "size": "small"
        },
        "time_steps": 2,
        "neuron_type": "lif"
    },

    "balanced": {
        "intention_decoder": {
            "type": "transformer",
            "size": "standard"
        },
        "intent_heads": {
            "size": "standard"
        },
        "time_steps": 4,
        "neuron_type": "lif"
    },

    "performance": {
        "intention_decoder": {
            "type": "transformer",
            "size": "large"
        },
        "intent_heads": {
            "size": "large"
        },
        "time_steps": 8,
        "neuron_type": "plif"
    },

    "maximum": {
        "intention_decoder": {
            "type": "transformer",
            "size": "huge"
        },
        "intent_heads": {
            "size": "xlarge"
        },
        "time_steps": 16,
        "neuron_type": "plif"
    }
}


def create_snn_intent_preset_model(dim: int, preset_name: str = "balanced",
                                 **kwargs):
    """
    使用预设配置快速创建SNN意图模型

    Args:
        dim: 特征维度
        preset_name: 预设配置名称 ("lightweight", "balanced", "performance", "maximum")
        **kwargs: 额外的参数

    Returns:
        PlanningModelSNNIntent 实例
    """
    if PlanningModelSNNIntent is None:
        raise ImportError("PlanningModelSNNIntent module not available. Please check dependencies.")

    if preset_name not in SNN_INTENT_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(SNN_INTENT_PRESETS.keys())}")

    preset = SNN_INTENT_PRESETS[preset_name]

    # 构建神经元配置
    neuron_cfg = get_default_neuron_config()
    neuron_cfg['spike_mode'] = preset['neuron_type']
    neuron_cfg['time_steps'] = preset['time_steps']

    # 构建模型配置
    model_config = {
        'dim': dim,
        'use_snn_intention': True,
        'snn_intention_type': preset['intention_decoder']['type'],
        'snn_intention_size': preset['intention_decoder']['size'],
        'snn_intent_head_size': preset['intent_heads']['size'],
        'snn_time_steps': preset['time_steps'],
        'snn_neuron_cfg': neuron_cfg,
    }

    # 合并额外的参数
    model_config.update(kwargs)

    return PlanningModelSNNIntent(**model_config)


def create_snn_intent_components(dim: int, lateral_classes: int = 5,
                               longitudinal_classes: int = 4,
                               intention_decoder_type: str = "mlp",
                               intention_decoder_size: str = "standard",
                               intent_head_size: str = "standard",
                               time_steps: int = 4,
                               neuron_cfg: dict = None):
    """
    创建SNN意图组件（用于自定义集成）

    Args:
        dim: 特征维度
        lateral_classes: 横向意图类别数
        longitudinal_classes: 纵向意图类别数
        intention_decoder_type: "mlp" 或 "transformer"
        intention_decoder_size: "tiny", "small", "standard", "large", "huge"
        intent_head_size: "small", "standard", "large", "xlarge"
        time_steps: SNN时间步数
        neuron_cfg: 神经元配置

    Returns:
        (intention_decoder, intent_heads) 元组
    """
    # 意图解码器
    if intention_decoder_type == "mlp":
        intention_decoder = create_snn_intention_mlp_decoder(
            dim=dim,
            size=intention_decoder_size,
            time_steps=time_steps,
            neuron_cfg=neuron_cfg,
        )
    elif intention_decoder_type == "transformer":
        intention_decoder = create_snn_intention_transformer_decoder(
            dim=dim,
            size=intention_decoder_size,
            time_steps=time_steps,
            neuron_cfg=neuron_cfg,
        )
    else:
        raise ValueError(f"Unsupported intention decoder type: {intention_decoder_type}")

    # 意图分类头
    intent_heads = create_snn_intent_heads(
        in_features=dim,
        lateral_classes=lateral_classes,
        longitudinal_classes=longitudinal_classes,
        size=intent_head_size,
        time_steps=time_steps,
        neuron_cfg=neuron_cfg,
    )

    return intention_decoder, intent_heads


print(f"SNN Intent modules loaded. Version: {__version__}")
if PLANNING_MODEL_AVAILABLE:
    print(f"Available presets: {list(SNN_INTENT_PRESETS.keys())}")
else:
    print("PlanningModelSNNIntent not available - core SNN modules can still be used")
print(f"SpikingJelly available: {check_spiking_jelly_available()}")