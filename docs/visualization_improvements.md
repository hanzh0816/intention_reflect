# 可视化改进说明

## 问题分析

### 原始问题
1. **所有agent颜色相同**：所有车辆都使用灰色 `#95A5A6`，无法区分不同的agent
2. **Legend重复无意义**：每个车辆都添加 "Other Vehicles" 标签，导致legend中出现大量重复项
3. **透明度过低**：alpha=0.6 使得车辆显示不够清晰
4. **标签冗余**：显示 "Agent X" 文字过长

### msgpack.xz 文件加载问题

#### 文件生成过程（三层包装）
在 nuplan 仿真过程中，`SimulationLog` 保存为 `.msgpack.xz` 文件经过了三层包装：

```python
# nuplan-devkit/nuplan/planning/simulation/simulation_log.py:33-40
def _dump_to_msgpack(self) -> None:
    # 1. pickle 序列化
    pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
    # 2. msgpack 打包
    msg_packed_bytes = msgpack.packb(pickle_object)
    # 3. lzma 压缩
    save_buffer(self.file_path, lzma.compress(msg_packed_bytes, preset=0))
```

#### 正确的加载方法
必须按照相反顺序解包：

```python
# 正确方法（使用 nuplan 官方 API）
simulation_log = SimulationLog.load_data(log_file)

# 内部实现
with lzma.open(str(file_path), "rb") as f:
    data = msgpack.unpackb(f.read())  # 1. msgpack 解包
    data = pickle.loads(data)          # 2. pickle 反序列化
```

#### 错误原因
原代码只做了 lzma 解压 + pickle 反序列化，**缺少了 msgpack 解包步骤**：

```python
# 错误方法（缺少 msgpack 解包）
decompressed_data = lzma.decompress(compressed_data)
simulation_log = pickle.loads(decompressed_data)  # ❌ 报错！
```

错误信息 `invalid load key, '\xc6'` 中的 `\xc6` 是 msgpack 的格式标记字节，pickle 无法识别。

## 改进方案

### 1. 文件加载修复

**修改文件**: `src/utils/simulation_video.py`

```python
# 添加导入
import msgpack
from nuplan.planning.simulation.simulation_log import SimulationLog

# 修改加载逻辑
if log_file.suffix == '.xz' or str(log_file).endswith(('.msgpack.xz', '.pkl.xz')):
    logger.info(f"Loading compressed simulation log: {log_file}")
    simulation_log = SimulationLog.load_data(log_file)  # 使用官方 API
```

### 2. 颜色区分系统

**修改文件**: `src/utils/visualization_utils.py`

定义了15种高区分度的颜色：

```python
AGENT_COLORS = [
    '#FF6B6B',  # 红色
    '#4ECDC4',  # 青色
    '#45B7D1',  # 蓝色
    '#FFA07A',  # 浅橙色
    '#98D8C8',  # 薄荷绿
    '#F7DC6F',  # 黄色
    '#BB8FCE',  # 紫色
    '#85C1E2',  # 天蓝色
    '#F8B88B',  # 桃色
    '#52B788',  # 绿色
    '#E76F51',  # 橙红色
    '#2A9D8F',  # 深青色
    '#E9C46A',  # 金黄色
    '#F4A261',  # 橙色
    '#264653',  # 深蓝灰
]

def get_agent_color(agent_id: int) -> str:
    """根据agent ID获取对应的颜色"""
    return AGENT_COLORS[(agent_id - 1) % len(AGENT_COLORS)]
```

### 3. Legend 优化

只为前5个agent添加legend标签，避免legend过长：

```python
legend_agents = set()

for obj in tracked_objects.tracked_objects:
    seq_id = agent_id_map.get(token_str)

    if seq_id is not None:
        agent_color = get_agent_color(seq_id)
        # 只为前5个agent添加legend
        if seq_id not in legend_agents and len(legend_agents) < 5:
            label = f'Agent {seq_id}'
            legend_agents.add(seq_id)
        else:
            label = None
```

### 4. 视觉效果增强

- **提高透明度**: alpha 从 0.6 提升到 0.85
- **增加边框宽度**: linewidth 从 1.5 提升到 2.0
- **简化标签**: 从 "Agent X" 改为只显示数字 "X"
- **增大字体**: fontsize 从 9 提升到 11
- **增强轮廓**: outline_width 从 2.0 提升到 3.0

### 5. 文本轮廓优化

使用 `matplotlib.patheffects` 实现更好的文本轮廓效果：

```python
import matplotlib.patheffects as path_effects

def add_text_with_outline(...):
    txt = ax.text(x, y, text, ...)
    txt.set_path_effects([
        path_effects.Stroke(linewidth=outline_width, foreground=outline_color),
        path_effects.Normal()
    ])
```

## 效果对比

### 改进前
- ❌ 所有车辆都是灰色，无法区分
- ❌ Legend 中大量重复的 "Other Vehicles"
- ❌ 车辆透明度低，不够清晰
- ❌ 标签文字冗长

### 改进后
- ✅ 每个agent使用不同颜色（15种颜色循环）
- ✅ Legend 只显示前5个agent，简洁明了
- ✅ 车辆透明度提高到0.85，更清晰可见
- ✅ 标签只显示数字，简洁易读
- ✅ 字体和边框更粗，更易识别

## 参考资料

改进方案参考了 `Intent_label` 分支的可视化实现：
- `src/target_builders/intent_utils/visualization.py`
- 使用了更好的颜色方案和视觉效果
- 优化了legend和标签显示逻辑

## 使用方法

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token 0cccf1639991539a \
  --simulation_mode closed_loop_reactive_agents
```

生成的视频将包含：
- 彩色区分的agent车辆
- 清晰的agent ID标签
- 简洁的legend
- 更好的视觉效果
