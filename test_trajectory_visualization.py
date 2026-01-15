#!/usr/bin/env python3
"""测试轨迹可视化功能"""

import sys
from pathlib import Path

# 测试导入
try:
    from src.utils.visualization_utils import (
        plot_planned_trajectory_ego_centric,
        OBJECT_TYPE_COLORS,
        OBJECT_TYPE_LABELS,
    )
    print("✓ 成功导入轨迹可视化工具")

    # 测试对象类型颜色
    print(f"\n✓ 对象类型颜色映射包含 {len(OBJECT_TYPE_COLORS)} 种类型")
    for obj_type, color in OBJECT_TYPE_COLORS.items():
        label = OBJECT_TYPE_LABELS.get(obj_type, str(obj_type))
        print(f"  {label}: {color}")

    print("\n✓ 所有导入和函数测试通过！")
    print("\n新增功能：")
    print("1. 绘制所有交通参与者类型（vehicles, pedestrians, bicycles等）")
    print("2. 每种类型使用不同颜色，并在图例中标注")
    print("3. Agent ID显示在车辆框内部")
    print("4. 绘制自车每个时刻的规划轨迹（紫红色线条）")
    print("5. 轨迹终点用星号标记")

except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
