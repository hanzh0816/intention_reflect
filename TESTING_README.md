# PlanTF 测试工具快速指南

## 快速开始

### 1. 查找checkpoint

```bash
# 查找最近训练的checkpoints
sh ./find_checkpoints.sh
```

这会显示：
- 最新实验的目录
- 所有可用的checkpoints
- 推荐使用的checkpoint（最低val_minFDE或last.ckpt）
- 快速测试命令

### 2. 开环测试（推荐先做）

开环测试快速评估模型的轨迹预测精度：

```bash
# 方法1: 使用脚本
sh ./test_open_loop.sh /path/to/checkpoint.ckpt

# 方法2: 使用find_checkpoints找到的checkpoint
CKPT=$(sh ./find_checkpoints.sh | grep "最佳checkpoint" | awk '{print $NF}')
sh ./test_open_loop.sh $CKPT
```

**输出指标**:
- minADE1/minADE6: 平均位移误差
- minFDE1/minFDE6: 终点位移误差
- MR: 轨迹偏离率

### 3. 闭环测试（完整评估）

闭环测试在仿真环境中评估真实驾驶性能：

```bash
# 快速测试（280个场景，~30分钟）
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt test14-random

# 完整测试（1400个场景，~2-3小时）
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt val14
```

**场景选项**:
- `test14-random`: 14种类型各20个场景（280个）
- `test14-hard`: 困难场景（280个）
- `val14`: 验证集场景（1400个）

### 4. 查看仿真结果

```bash
# 自动查找并显示最新结果
python view_simulation_results.py

# 查看特定结果
python view_simulation_results.py --result_dir /path/to/simulation/result

# 显示按场景类型的详细分解
python view_simulation_results.py --breakdown
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `find_checkpoints.sh` | 查找最近训练的checkpoints |
| `test_open_loop.sh` | 开环测试脚本 |
| `test_closed_loop.sh` | 闭环仿真脚本 |
| `view_simulation_results.py` | 查看仿真结果工具 |
| `TESTING_GUIDE.md` | 详细测试指南 |

## ⚠️ 重要提示：Intent-Conditioned模型

如果你的checkpoint是用 `intent_enabled=true` 训练的（默认配置），确保测试时也启用intent。

**已自动配置**: `config/planner/planTF.yaml` 已包含正确的intent参数。

如果遇到模型加载错误，查看 `TESTING_TROUBLESHOOTING.md` → 第1节。

## 完整测试流程示例

```bash
# 步骤1: 查找checkpoint
sh ./find_checkpoints.sh

# 假设输出显示最佳checkpoint为:
# /data2/hzh/nuplan/exp/planTF/2025.11.09.23-24-55/checkpoints/epoch=15-val_minFDE=1.234.ckpt

CKPT="/data2/hzh/nuplan/exp/planTF/2025.11.09.23-24-55/checkpoints/epoch=15-val_minFDE=1.234.ckpt"

# 步骤2: 开环测试
sh ./test_open_loop.sh $CKPT

# 步骤3: 闭环测试（快速）
sh ./test_closed_loop.sh $CKPT test14-random

# 步骤4: 查看结果
python view_simulation_results.py

# 如果性能满意，运行完整测试
# sh ./test_closed_loop.sh $CKPT val14
```

## 常见问题

### Q: 如何知道模型性能好不好？

**开环测试基准** (nuPlan val set):
- minADE1 < 2.0: 较好
- minADE1 < 1.5: 很好
- minFDE1 < 3.0: 较好
- minFDE1 < 2.0: 很好

**闭环测试基准**:
- no_at_fault_collisions > 0.95: 安全
- drivable_area_compliance > 0.95: 良好
- driving_direction_compliance > 0.95: 良好
- ego_progress > 0.80: 任务完成度高

### Q: 闭环测试太慢怎么办？

1. 先用`test14-random`（280个场景）快速验证
2. 只在满意后运行完整的`val14`测试
3. 或者创建自定义场景filter，只测试特定类型

### Q: 如何可视化仿真轨迹？

使用nuBoard工具：
```bash
python nuplan-devkit/nuplan/planning/script/run_nuboard.py \
    simulation_path=/path/to/simulation_results
```

然后在浏览器打开显示的URL。

### Q: checkpoint文件在哪里？

训练完成后，checkpoints保存在：
```
${NUPLAN_EXP_ROOT}/planTF/<timestamp>/checkpoints/
```

通常包括：
- `last.ckpt`: 最后一个epoch
- `epoch=N-val_minFDE=X.XXX.ckpt`: 验证集表现最好的checkpoints

## 更多信息

详细说明请参考 `TESTING_GUIDE.md`
