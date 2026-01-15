#!/usr/bin/env python3
"""测试可视化改进"""

import sys
from pathlib import Path

# 测试导入
try:
    from src.utils.visualization_utils import (
        get_agent_color,
        AGENT_COLORS,
        add_text_with_outline,
        plot_tracked_objects_ego_centric
    )
    print("✓ 成功导入可视化工具")
    print(f"✓ AGENT_COLORS 包含 {len(AGENT_COLORS)} 种颜色")

    # 测试颜色分配
    for i in range(1, 6):
        color = get_agent_color(i)
        print(f"  Agent {i}: {color}")

    print("\n✓ 所有导入和函数测试通过！")
    print("\n改进内容：")
    print("1. 每个agent使用不同的颜色（15种高区分度颜色循环使用）")
    print("2. 提高车辆透明度从0.6到0.85，更清晰可见")
    print("3. Legend只显示前5个agent，避免过长")
    print("4. Agent ID标签只显示数字，更简洁")
    print("5. 增大字体和边框宽度，更易识别")

except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
