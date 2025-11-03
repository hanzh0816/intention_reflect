# 快速开始：超车场景可视化

## 一键运行

可视化数据集中的**所有超车场景**：

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_overtaking_scenarios \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=null
```

## 结果

运行完成后会自动：
1. ✅ 过滤出所有超车场景（关键词：overtaking, overtake, passing）
2. ✅ 为每个超车场景生成独立的轨迹+地图可视化图像
3. ✅ 保存到带时间戳的目录：`work_dirs/overtaking_visualizations_YYYYMMDD_HHMMSS/`

## 预期输出

```
INFO: Found 156 overtaking scenarios out of 10000 total
INFO: Processing all 156 overtaking scenarios
...
✓ Successfully created 156 overtaking trajectory visualizations
✓ Total overtaking scenarios found: 156
✓ Output directory: work_dirs/overtaking_visualizations_20250131_143025/
```

## 可视化内容

每张图包含：
- 🔵 **蓝色粗线**：历史轨迹（过去2秒）
- 🔴 **红色粗线**：未来轨迹（未来8秒）
- 🗺️ **浅蓝色填充**：车道（交替深浅颜色区分不同车道）
- ⬛ **深灰色线**：车道边界
- 📍 **蓝色方块**：轨迹起点
- 📍 **红色三角**：轨迹终点

## 仅可视化部分超车场景

如果数据集很大，只想看前10个超车场景：

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_overtaking_scenarios \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=1000 \
    num_scenarios_to_visualize=10
```

## 调整地图显示范围

增大地图显示半径到100米：

```bash
python visualize_ego_trajectories_with_map.py \
    +training=visualize_overtaking_scenarios \
    scenario_builder=nuplan \
    scenario_filter.limit_total_scenarios=null \
    map_radius=100.0
```

## 常见问题

### Q: 如何知道数据集中有多少超车场景？

运行脚本后查看日志，会显示：
```
INFO: Found XXX overtaking scenarios out of YYYY total
```

### Q: 可以过滤其他场景类型吗？

可以！修改代码中的 `overtaking_keywords` 列表即可，例如：
```python
overtaking_keywords = ['changing_lane', 'lane_change']  # 变道场景
overtaking_keywords = ['stopping', 'stop']  # 停车场景
```

### Q: 文件命名规则是什么？

格式：`trajectory_{序号}_{场景类型}_{token前8位}.png`

例如：`trajectory_000_on_lane_overtaking_a1b2c3d4.png`

## 更多信息

详细文档请查看：`README_TRAJECTORY_WITH_MAP_VISUALIZATION.md`
