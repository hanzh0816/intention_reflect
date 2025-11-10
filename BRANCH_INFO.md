# 分支说明

## 分支结构

### `main` 分支
- 基础PlanTF实现
- 不包含intent功能
- 最新commit: `6888e93 update .gitignore`

### `intent-plantf` 分支（新建）✨
- 基于 `main` 分支创建
- 添加intent-conditioned trajectory generation功能
- 只包含核心代码、配置和测试工具
- **推荐用于后续开发和更新**

### `vis` 分支
- Intent功能的原始开发分支
- 包含所有功能（核心 + 可视化 + 额外工具）
- 用于可视化和实验

---

## `intent-plantf` vs `vis` 的区别

### intent-plantf 包含（从vis移植）

#### ✅ 核心代码
- `src/features/intent_labels.py` (166 lines)
- `src/target_builders/intent_target_builder.py` (134 lines)
- `src/utils/intent_classification.py` (432 lines)
- `src/models/planTF/planning_model.py` (修改)
- `src/models/planTF/lightning_trainer.py` (修改)

#### ✅ 配置文件
- `config/model/planTF.yaml`
- `config/planner/planTF.yaml`
- `config/custom_trainer/planTF.yaml`
- `config/lightning/custom_lightning.yaml`

#### ✅ 测试框架
- `test_open_loop.sh` - 开环测试
- `test_closed_loop.sh` - 闭环仿真
- `find_checkpoints.sh` - 查找checkpoint
- `view_simulation_results.py` - 结果查看
- `TESTING_README.md` - 测试快速指南
- `TESTING_GUIDE.md` - 详细测试文档
- `TESTING_TROUBLESHOOTING.md` - 故障排除

#### ✅ 工具脚本
- `cache.sh` - 生成cache
- `train.sh` - 训练脚本
- `check_intent_labels.py` - 验证intent标签
- `check_data_split.py` - 分析数据划分

#### ✅ 文档
- `INTENT_INTEGRATION_README.md` - Intent集成文档
- `.gitignore` - 更新的ignore规则

### vis 额外包含（未移植）

#### ❌ 可视化工具
- `visualize_ego_trajectories_with_map.py` (369 lines)
- `visualize_short_term_intent.py` (711 lines)
- `config/training/visualize_trajectories.yaml`
- `config/training/visualize_trajectories_with_map.yaml`
- `VISUALIZATION_GUIDE.md` (514 lines)
- `ego_trajectories.png`

#### ❌ 辅助脚本
- `print_scenario_types.py`
- `run_print_scenario_types.sh`
- `run_short_term_intent.sh`

#### ❌ README更新
- `README.md` 的部分更新（可视化相关）

---

## 为什么选择 `intent-plantf`

### ✅ 优点

1. **干净简洁**
   - 只包含核心功能和测试工具
   - 代码量：2491行新增（vs vis的4272行）
   - 避免可视化代码混入

2. **易于维护**
   - 基于main分支，便于合并上游更新
   - 清晰的功能边界
   - 减少merge冲突

3. **专注开发**
   - Intent功能的核心实现
   - 完整的测试框架
   - 生产就绪

4. **按需扩展**
   - 需要可视化时，可以从vis分支单独复制
   - 保持主开发分支简洁

### ❌ vis分支的问题

1. 包含大量可视化代码（1500+行）
2. 与main分支diff过大，难以同步
3. 混合了开发和可视化功能

---

## 推荐工作流程

### 日常开发
```bash
# 在intent-plantf分支进行开发
git checkout intent-plantf

# 开发intent相关功能
# ...

# 提交改动
git add .
git commit -m "Feature: ..."
```

### 同步main分支更新
```bash
# 定期合并main的更新
git checkout intent-plantf
git merge main

# 解决冲突（如果有）
# ...
```

### 需要可视化工具时
```bash
# 临时切换到vis分支
git checkout vis

# 或者单独复制可视化文件到intent-plantf
git checkout intent-plantf
git checkout vis -- visualize_ego_trajectories_with_map.py
```

### 准备合并到main（如果需要）
```bash
# intent-plantf分支稳定后，可以合并回main
git checkout main
git merge intent-plantf
```

---

## 快速开始

在 `intent-plantf` 分支上工作：

```bash
# 1. 切换到分支
git checkout intent-plantf

# 2. 查找checkpoint
sh ./find_checkpoints.sh

# 3. 运行测试
sh ./test_open_loop.sh /path/to/checkpoint.ckpt
sh ./test_closed_loop.sh /path/to/checkpoint.ckpt test14-random

# 4. 查看文档
cat TESTING_README.md
cat INTENT_INTEGRATION_README.md
```

---

## 文件统计

### intent-plantf vs main
- **23个文件修改**
- **+2491行，-18行**
- **16个新文件**
- **7个修改的文件**

### vis vs main
- **33个文件修改**
- **+4272行，-18行**
- **26个新文件**
- **7个修改的文件**

### 差异（vis - intent-plantf）
- **10个额外文件**（主要是可视化相关）
- **+1781行**（可视化代码）

---

## 总结

✅ **推荐使用 `intent-plantf` 进行后续开发**
- 干净的intent功能实现
- 完整的测试框架
- 便于维护和更新
- 生产就绪

📊 **保留 `vis` 作为参考和可视化工具库**
- 需要可视化时从这里取
- 实验性功能测试

🔄 **定期同步 `main` 分支的更新**
- 保持与上游一致
- 避免分支diverge
