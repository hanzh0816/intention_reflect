# 场景仿真与视频可视化工具

## 快速开始

使用与 `eval.sh` 相同的方式，通过 local config 运行单个场景的仿真并生成视频：

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token <你的场景token> \
  --simulation_mode closed_loop_nonreactive
```

## 主要特性

- ✅ **使用 local config**：像 `eval.sh` 一样，自动从 `config/local/` 加载所有配置
- ✅ **继承评估设置**：自动获取 planner、checkpoint、路径、GPU 等所有配置
- ✅ **单场景仿真**：通过 scenario_token 指定要仿真的场景
- ✅ **三种仿真模式**：open_loop、closed_loop_nonreactive、closed_loop_reactive
- ✅ **自车视角视频**：生成以自车为中心的视频（自车始终在中心，世界围绕它旋转）
- ✅ **Agent ID 标注**：所有 agent 显示序号（Agent 1, Agent 2, ...）
- ✅ **ID 映射文件**：生成 JSON 文件记录序号与 track_token 的对应关系

## 参数说明

### 必需参数

- `--config`: local config 名称（如 `251218_eval-plantf`）
- `--scenario_token`: 要仿真的场景 token
- `--simulation_mode`: 仿真模式
  - `open_loop`: agents 遵循记录的轨迹
  - `closed_loop_nonreactive`: agents 使用 IDM 模型
  - `closed_loop_reactive`: agents 使用 IDM 模型（reactive 指标）

### 可选参数

- `--output_dir`: 输出目录（默认：自动生成）
- `--video_fps`: 视频帧率（默认：10）
- `--video_resolution`: 视频分辨率（默认：1920x1080）
- `--map_radius`: 地图显示半径（米，默认：80）
- `--skip_video`: 跳过视频生成（仅运行仿真）

## 使用示例

### 示例 1：使用现有评估配置

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token abc123def456 \
  --simulation_mode closed_loop_nonreactive
```

### 示例 2：自定义视频设置

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token abc123def456 \
  --simulation_mode open_loop \
  --video_fps 20 \
  --video_resolution 2560x1440 \
  --map_radius 100
```

### 示例 3：仅运行仿真（不生成视频）

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token abc123def456 \
  --simulation_mode closed_loop_reactive \
  --skip_video
```

## 输出文件

运行后会在输出目录生成以下文件：

1. **仿真结果**
   - `aggregator_metric/`: 聚合指标
   - `simulation_logs/`: 详细仿真日志
   - 各种指标文件

2. **视频文件**
   - `simulation_<token>_<mode>.mp4`: 自车视角视频
   - `simulation_<token>_<mode>_agent_ids.json`: Agent ID 映射文件

### Agent ID 映射格式

```json
{
  "1": "a3f2b1c4e5d6f7a8b9c0d1e2f3a4b5c6",
  "2": "b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
  "3": "c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6"
}
```

- **键**：视频中显示的序号（"1", "2", "3"）
- **值**：完整的 track_token（十六进制字符串）

## 与 eval.sh 的对比

| 特性 | eval.sh | simulate_and_visualize.py |
|------|---------|---------------------------|
| 使用 local config | ✅ | ✅ |
| 继承所有配置 | ✅ | ✅ |
| 批量评估场景 | ✅ | ❌ |
| 单场景仿真 | ❌ | ✅ |
| 生成视频 | ❌ | ✅ |
| Agent ID 标注 | ❌ | ✅ |

## 常见问题

### Q: 如何找到可用的 local config？

A: 查看 `config/local/` 目录下的 `.yaml` 文件：

```bash
ls config/local/*.yaml
```

### Q: 如何获取场景 token？

A: 可以从之前的评估结果中获取，或使用 nuplan 的场景查询工具。

### Q: 视频生成失败怎么办？

A: 检查是否安装了 opencv-python：

```bash
pip install opencv-python
```

### Q: 输出目录在哪里？

A: 默认在 `<nuplan_exp_root>/scenario_sim/<token>_<mode>/`，其中 `nuplan_exp_root` 来自 local config。

## 详细文档

更多详细信息请参考：[docs/scenario_simulation_video.md](docs/scenario_simulation_video.md)
