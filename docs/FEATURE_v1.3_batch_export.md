# v1.3 功能更新：批量导出与统一可视化

## 🎯 新增功能

### 1. 三种导出模式

**单个 scenario**（已有功能）
```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" \
  --database-path work_dirs/exp/failure_cases.db
```

**多个 scenarios**（新功能 ✨）
```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1" \
  --database-path work_dirs/exp/failure_cases.db
```

**所有 failure cases**（新功能 ✨）
```bash
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db
```

### 2. 统一输出目录

**核心特性：**
- 所有导出都保存到同一个目录（默认 `work_dirs/failure_viz`）
- 方便在 nuBoard 中一起可视化多个 failure cases
- 支持增量导出：可以多次运行导出命令，追加更多 scenarios

**目录结构：**
```
work_dirs/failure_viz/
├── nuboard_xxx.nuboard          # 元数据文件（自动创建/重用）
├── simulation/
│   └── planTF/
│       ├── highway/
│       │   ├── log1/
│       │   │   └── scenario1/
│       │   │       └── scenario1.pkl.xz
│       │   └── log2/
│       │       └── scenario2/
│       │           └── scenario2.pkl.xz
│       └── urban/
│           └── log3/
│               └── scenario3/
│                   └── scenario3.pkl.xz
└── metrics/
```

### 3. 智能 .nuboard 文件管理

**新功能：**
- 检测现有的 `.nuboard` 文件并重用
- 避免每次导出都创建新的 `.nuboard` 文件
- 只在第一次导出时创建，后续导出重用

**代码示例：**
```python
# FailureCaseExporter.create_nuboard_file() 方法
def create_nuboard_file(self, force: bool = False) -> Path:
    # 检查是否已存在 .nuboard 文件
    existing_nuboard_files = list(self._output_base.glob("*.nuboard"))

    if existing_nuboard_files and not force:
        # 重用现有文件
        return existing_nuboard_files[0]

    # 创建新文件
    ...
```

### 4. 默认输出目录

**变更：**
- `--output-dir` 参数现在有默认值：`work_dirs/failure_viz`
- 简化命令：不需要每次都指定输出目录

**之前（v1.2）：**
```bash
python scripts/export_failure_cases.py \
  --scenario-name "xxx" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz  # 必须指定
```

**现在（v1.3）：**
```bash
python scripts/export_failure_cases.py \
  --scenario-name "xxx" \
  --database-path work_dirs/exp/failure_cases.db
  # --output-dir 默认为 work_dirs/failure_viz
```

## 📊 批量导出性能

### 导出进度显示

导出多个 scenarios 时会显示详细的进度信息：

```
Exporting 3 failure case(s)...
================================================================================

[1/3] Exporting: 00cca24d240f5980
--------------------------------------------------------------------------------
✓ Successfully exported to: work_dirs/failure_viz/simulation/planTF/highway/...

[2/3] Exporting: 01abd3e5120a4bc0
--------------------------------------------------------------------------------
✓ Successfully exported to: work_dirs/failure_viz/simulation/planTF/urban/...

[3/3] Exporting: 02bcd4f6231b5cd1
--------------------------------------------------------------------------------
✓ Successfully exported to: work_dirs/failure_viz/simulation/planTF/highway/...

================================================================================
Export Summary
================================================================================
  Total scenarios: 3
  ✓ Succeeded: 3
  ✗ Failed: 0

Exported files:
  - work_dirs/failure_viz/simulation/planTF/highway/.../00cca24d240f5980.pkl.xz
  - work_dirs/failure_viz/simulation/planTF/urban/.../01abd3e5120a4bc0.pkl.xz
  - work_dirs/failure_viz/simulation/planTF/highway/.../02bcd4f6231b5cd1.pkl.xz

--------------------------------------------------------------------------------
✓ Created .nuboard file: work_dirs/failure_viz/nuboard_1703123456.nuboard

================================================================================
To visualize with nuBoard, run:
================================================================================

  python run_nuboard.py \
    simulation_path=work_dirs/failure_viz \
    port_number=5006

  Then open: http://localhost:5006

================================================================================
```

### 错误处理

即使部分导出失败，也会继续处理其余的 scenarios：

```
Export Summary
================================================================================
  Total scenarios: 5
  ✓ Succeeded: 4
  ✗ Failed: 1

Failed exports:
  - invalid_scenario_name: Scenario 'invalid_scenario_name' not found in database.
```

## 🔧 使用场景

### 场景 1：分析所有失败案例

```bash
# 一次性导出所有 failure cases
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db

# 在 nuBoard 中浏览和对比所有失败案例
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 场景 2：对比特定类型的失败

```bash
# 先列出所有失败案例
python scripts/export_failure_cases.py --list

# 选择几个相似的失败案例（比如都是碰撞类型）
python scripts/export_failure_cases.py \
  --scenario-name "scenario1" "scenario2" "scenario3"

# 在 nuBoard 中对比分析
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

### 场景 3：增量分析

```bash
# 第一天：导出初始的几个案例
python scripts/export_failure_cases.py \
  --scenario-name "case1" "case2"

# 第二天：新增更多案例（自动追加到同一目录）
python scripts/export_failure_cases.py \
  --scenario-name "case3" "case4" "case5"

# 第三天：再添加更多
python scripts/export_failure_cases.py \
  --scenario-name "case6" "case7"

# 所有案例都在同一个目录下，可以一起查看
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

## 🔄 向后兼容性

**完全兼容 v1.2.1：**
- 所有 v1.2.1 的命令在 v1.3 中仍然有效
- 目录结构保持一致
- 只是增加了新的功能选项

**迁移建议：**
- 旧命令无需修改即可继续使用
- 建议使用新的批量导出功能提高效率
- 可以省略 `--output-dir` 参数，使用默认值

## 📝 参数参考

```bash
python scripts/export_failure_cases.py --help
```

**主要参数：**
- `--list`: 列出所有 failure cases
- `--info SCENARIO`: 显示指定 scenario 的详细信息
- `--scenario-name SCENARIO [SCENARIO ...]`: 导出一个或多个 scenarios
- `--all`: 导出所有 failure cases
- `--database-path PATH`: failure case 数据库路径（默认：work_dirs/exp/failure_cases.db）
- `--output-dir PATH`: 输出目录（默认：work_dirs/failure_viz）
- `--data-root PATH`: nuplan 数据集根目录（可选，用于真实 scenario）
- `--map-root PATH`: nuplan 地图目录（可选）
- `--db-files PATH`: nuplan 数据库文件（可选）
- `--verbose`: 详细日志

## 🆚 版本对比

| 功能 | v1.2.1 | v1.3 |
|------|--------|------|
| 单个导出 | ✅ | ✅ |
| 多个导出 | ❌ | ✅ |
| 全部导出 | ❌ | ✅ |
| 默认输出目录 | ❌ | ✅ |
| 智能 .nuboard 管理 | ❌ | ✅ |
| 增量导出支持 | ⚠️ 手动 | ✅ 自动 |
| 批量进度显示 | - | ✅ |
| 错误容错处理 | - | ✅ |

## 🚀 后续改进方向

可能的未来增强功能：
1. 按失败类型过滤导出（如只导出碰撞类型）
2. 按严重程度过滤导出（如只导出 CRITICAL 级别）
3. 支持从文件读取 scenario 列表
4. 并行导出以提高速度
5. 导出统计报告（失败类型分布、严重程度统计等）

## 📚 相关文档

- 完整使用指南：`docs/failure_case_visualization.md`
- 问题修复指南：`docs/nuboard_fix_guide.md`
- v1.2.1 修复说明：`docs/BUGFIX_v1.2.1_summary.md`
