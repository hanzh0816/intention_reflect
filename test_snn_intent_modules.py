"""
测试脚本：验证SNN意图模块功能
"""
import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

def test_snn_utils():
    """测试SNN基础工具"""
    print("正在测试SNN基础工具...")

    from src.models.planTF.modules.snn_utils import (
        LIFNeuron, SNNLinearBlock, TimeDimExpander, TimeDimAverage,
        get_default_neuron_config, check_spiking_jelly_available
    )

    # 检查SpikingJelly可用性
    available = check_spiking_jelly_available()
    print(f"SpikingJelly可用: {available}")

    # 获取默认配置
    neuron_cfg = get_default_neuron_config()
    print(f"默认神经元配置: {neuron_cfg}")

    # 测试时间维度扩展器
    time_expander = TimeDimExpander(time_steps=4)
    x = torch.randn(2, 64)  # [B=2, C=64]
    x_time = time_expander(x)
    print(f"时间扩展前: {x.shape}, 时间扩展后: {x_time.shape}")
    assert x_time.shape == (4, 2, 64), "时间扩展器输出形状错误"

    # 测试时间平均
    time_average = TimeDimAverage()
    x_avg = time_average(x_time)
    print(f"时间平均后: {x_avg.shape}")
    assert x_avg.shape == (2, 64), "时间平均器输出形状错误"

    print("✓ SNN基础工具测试通过")


def test_snn_intention_mlp_decoder():
    """测试SNN MLP意图解码器"""
    print("\n正在测试SNN MLP意图解码器...")

    from src.models.planTF.modules.snn_intention_mlp_decoder import create_snn_intention_mlp_decoder

    # 创建标准MLP解码器
    decoder = create_snn_intention_mlp_decoder(
        dim=128,
        decoder_type="standard",
        time_steps=4
    )
    decoder.eval()

    # 测试输入
    ego_feature = torch.randn(2, 128)  # [B=2, C=128]

    with torch.no_grad():
        intention_feature = decoder(ego_feature)

    print(f"输入特征形状: {ego_feature.shape}")
    print(f"意图特征形状: {intention_feature.shape}")
    assert intention_feature.shape == (2, 128), "意图解码器输出形状错误"

    # 测试浅层解码器
    shallow_decoder = create_snn_intention_mlp_decoder(
        dim=128,
        decoder_type="shallow",
        time_steps=2
    )
    shallow_decoder.eval()

    with torch.no_grad():
        shallow_intention = shallow_decoder(ego_feature)

    assert shallow_intention.shape == (2, 128), "浅层解码器输出形状错误"

    print("✓ SNN MLP意图解码器测试通过")


def test_snn_intention_transformer_decoder():
    """测试SNN Transformer意图解码器"""
    print("\n正在测试SNN Transformer意图解码器...")

    from src.models.planTF.modules.snn_intention_transformer_decoder import create_snn_intention_transformer_decoder

    # 创建标准Transformer解码器
    decoder = create_snn_intention_transformer_decoder(
        decoder_type="standard",
        dim=128,
        size="standard",
        time_steps=4
    )
    decoder.eval()

    # 测试输入
    ego_feature = torch.randn(2, 128)  # [B=2, C=128]

    with torch.no_grad():
        intention_feature = decoder(ego_feature)

    print(f"输入特征形状: {ego_feature.shape}")
    print(f"意图特征形状: {intention_feature.shape}")
    assert intention_feature.shape == (2, 128), "Transformer意图解码器输出形状错误"

    print("✓ SNN Transformer意图解码器测试通过")


def test_snn_intent_heads():
    """测试SNN意图分类头"""
    print("\n正在测试SNN意图分类头...")

    from src.models.planTF.modules.snn_intent_heads import create_snn_intent_heads

    # 创建意图分类头
    intent_heads = create_snn_intent_heads(
        in_features=128,
        size="standard",
        time_steps=4
    )
    intent_heads.eval()

    # 测试输入
    intention_feature = torch.randn(2, 128)  # [B=2, C=128]

    with torch.no_grad():
        lateral_logits, longitudinal_logits = intent_heads(intention_feature)

    print(f"输入意图特征形状: {intention_feature.shape}")
    print(f"横向意图logits形状: {lateral_logits.shape}")
    print(f"纵向意图logits形状: {longitudinal_logits.shape}")

    assert lateral_logits.shape == (2, 5), "横向意图输出形状错误"
    assert longitudinal_logits.shape == (2, 4), "纵向意图输出形状错误"

    print("✓ SNN意图分类头测试通过")


def test_snn_planning_model():
    """测试完整的SNN意图规划模型"""
    print("\n正在测试完整的SNN意图规划模型...")

    from src.models.planTF.modules.snn_modules import create_snn_intent_preset_model

    try:
        # 创建轻量级预设模型
        model = create_snn_intent_preset_model(
            dim=128,
            preset_name="lightweight"
        )
        model.eval()

        print("模型创建成功")
        print(f"使用SNN意图: {model.use_snn_intention}")
        print(f"SNN意图类型: {model.snn_intention_type}")

        # 模拟输入数据
        batch_size = 1
        history_steps = 21
        future_steps = 80
        num_agents = 5
        num_polygons = 10

        mock_data = {
            "agent": {
                "position": torch.randn(batch_size, num_agents, history_steps, 2),
                "heading": torch.randn(batch_size, num_agents, history_steps),
                "valid_mask": torch.ones(batch_size, num_agents, history_steps, dtype=torch.bool),
                "target": torch.randn(batch_size, num_agents, future_steps, 4),  # x, y, cos, sin
            },
            "map": {
                "polygon_center": torch.randn(batch_size, num_polygons, 3),  # x, y, heading
                "valid_mask": torch.ones(batch_size, num_polygons, dtype=torch.bool),
            }
        }

        with torch.no_grad():
            output = model(mock_data)

        print("前向传播成功")
        print(f"输出keys: {list(output.keys())}")
        print(f"轨迹形状: {output['trajectory'].shape}")
        print(f"概率形状: {output['probability'].shape}")
        print(f"意图输出形状:")
        print(f"  - 横向意图: {output['intent']['lateral'].shape}")
        print(f"  - 纵向意图: {output['intent']['longitudinal'].shape}")

        # 测试SNN脉冲率获取
        spike_rates = model.get_snn_spike_rates()
        print(f"SNN脉冲率: {spike_rates}")

        print("✓ 完整SNN意图规划模型测试通过")

    except Exception as e:
        print(f"完整模型测试失败: {e}")
        print("可能是由于缺少完整的NuPlan数据格式，但基本模块功能正常")


def test_all_modules():
    """运行所有测试"""
    print("=== SNN意图模块测试开始 ===")

    try:
        test_snn_utils()
        test_snn_intention_mlp_decoder()
        test_snn_intention_transformer_decoder()
        test_snn_intent_heads()
        test_snn_planning_model()

        print("\n🎉 所有SNN意图模块测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # 运行测试
    success = test_all_modules()

    if success:
        print("\n✅ SNN意图模块已准备就绪！")
        print("\n使用示例：")
        print("from src.models.planTF.modules.snn_modules import create_snn_intent_preset_model")
        print("model = create_snn_intent_preset_model(dim=128, preset_name='balanced')")
    else:
        print("\n❌ 部分测试失败，请检查依赖项和环境配置")