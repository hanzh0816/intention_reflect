# Bug Fix Summary: v1.2.1 - NotADirectoryError 修复

## 问题描述

在 v1.2 版本中，用户运行 `run_nuboard.py` 时遇到以下错误：

```
NotADirectoryError: [Errno 20] Not a directory: '.../failure_viz/nuboard_xxx.nuboard'
```

## 根本原因

v1.2 的目录结构如下：
```
failure_viz/
├── nuboard_xxx.nuboard  ← 文件
└── planTF/              ← 目录
```

nuBoard 在遍历 `simulation_path` 时：
1. 调用 `simulation_path.iterdir()` 列出所有内容（包括文件和目录）
2. 对每一项调用 `iterdir()` 遍历 scenario_type
3. 当遇到 `.nuboard` 文件时，尝试对文件调用 `iterdir()` 导致 `NotADirectoryError`

**问题本质：** `.nuboard` 文件和 planner 目录混在同一层级，nuBoard 无法区分文件和目录。

## 解决方案

采用 nuBoard 标准的目录结构，使用 `simulation/` 子目录隔离仿真日志和元数据文件：

```
failure_viz/
├── nuboard_xxx.nuboard    ← 元数据文件
├── simulation/            ← 仿真日志目录（新增）
│   └── planTF/            ← planner 目录
└── metrics/               ← 指标目录（空）
```

这样，nuBoard 只会遍历 `simulation/` 子目录，避免接触到 `.nuboard` 文件。

## 代码修改

### 1. `src/evaluation/failure_case_exporter.py`

**修改 1: 导出路径添加 `simulation/` 子目录**
```python
# 修改前
output_dir = (
    self._output_base
    / failure_case['planner_name']
    / failure_case['scenario_type']
    / ...
)

# 修改后
output_dir = (
    self._output_base
    / "simulation"  # 添加 simulation 子目录
    / failure_case['planner_name']
    / failure_case['scenario_type']
    / ...
)
```

**修改 2: 更新 NuBoardFile 配置**
```python
# 修改前
nuboard_file = NuBoardFile(
    simulation_main_path=str(self._output_base),
    simulation_folder=".",  # 根目录
    ...
)

# 修改后
nuboard_file = NuBoardFile(
    simulation_main_path=str(self._output_base),
    simulation_folder="simulation",  # 使用 simulation 子目录
    ...
)
```

### 2. `scripts/create_nuboard_file.py`

同步更新 `simulation_folder="simulation"`

### 3. 文档更新

- `docs/failure_case_visualization.md`: 更新目录结构说明，添加 v1.2.1 更新日志
- `docs/nuboard_fix_guide.md`: 添加目录遍历错误说明，更新版本信息

## 测试验证

修复后，正确的使用流程：

```bash
# 1. 导出 failure case
python scripts/export_failure_cases.py \
  --scenario-name "scenario_xxx" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz

# 2. 验证目录结构
ls -la work_dirs/failure_viz/
# 应该看到：
# - nuboard_xxx.nuboard
# - simulation/
# - metrics/

# 3. 启动 nuBoard
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006

# 4. 在浏览器访问 http://localhost:5006
```

## 兼容性说明

### 新导出（v1.2.1+）
- ✅ 使用 `simulation/` 子目录
- ✅ 自动生成 `.nuboard` 文件
- ✅ 完全兼容 nuBoard

### 旧导出（v1.0-v1.2）
如果已经有旧版本导出的数据：

**选项 1: 重新导出（推荐）**
```bash
python scripts/export_failure_cases.py ...
```

**选项 2: 手动迁移**
```bash
cd work_dirs/failure_viz
mkdir simulation
mv planTF simulation/
python scripts/create_nuboard_file.py .
```

## 影响范围

- ✅ 不影响数据库结构
- ✅ 不影响 SimulationLog 文件格式
- ✅ 不影响已有的 failure case 数据
- ⚠️ 需要重新导出或手动迁移旧的导出目录

## 版本历史

| 版本 | 状态 | 说明 |
|------|------|------|
| v1.0-v1.1 | ❌ 不可用 | 无 `.nuboard` 文件 |
| v1.2 | ⚠️ 部分可用 | 有 `.nuboard` 文件，但可能遇到目录遍历错误 |
| v1.2.1 | ✅ 推荐 | 完整修复，使用标准目录结构 |

## 相关文件

修改的文件：
- `src/evaluation/failure_case_exporter.py`
- `scripts/create_nuboard_file.py`
- `docs/failure_case_visualization.md`
- `docs/nuboard_fix_guide.md`

新增文件：
- `docs/BUGFIX_v1.2.1_summary.md`（本文档）

## 总结

通过添加 `simulation/` 子目录，完全遵循 nuBoard 的标准目录结构，彻底解决了目录遍历错误问题。用户现在可以正常使用 nuBoard 可视化 failure cases。
