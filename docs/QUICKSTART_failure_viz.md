# Failure Case 可视化 - 快速开始

## 🚀 最简单的使用方式

### 1. 导出所有 failure cases（一行命令）

```bash
python scripts/export_failure_cases.py --all --database-path work_dirs/exp/failure_cases.db
```

### 2. 启动 nuBoard 可视化

```bash
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

### 3. 在浏览器中打开

```
http://localhost:5006
```

就这么简单！✨

---

## 📋 常用命令速查

### 查看可用的 failure cases

```bash
python scripts/export_failure_cases.py --list
```

### 查询 failure case 详细信息（新功能 ✨）

```bash
# 快速查询失败类型和时间戳
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980"

# 详细查询（包含帧号）
python scripts/query_failure_details.py --scenario-name "00cca24d240f5980" --show-frames
```

### 导出单个 scenario

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980"
```

### 导出多个 scenarios

```bash
python scripts/export_failure_cases.py \
  --scenario-name "00cca24d240f5980" "01abd3e5120a4bc0" "02bcd4f6231b5cd1"
```

### 导出所有 failure cases

```bash
python scripts/export_failure_cases.py --all
```

### 查看 scenario 详细信息

```bash
python scripts/export_failure_cases.py \
  --info "00cca24d240f5980"
```

---

## 💡 小技巧

### 技巧 1: 省略常用参数

所有命令都有默认值，可以简化：

```bash
# 完整命令
python scripts/export_failure_cases.py \
  --all \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz

# 简化版（使用默认值）
python scripts/export_failure_cases.py --all
```

### 技巧 2: 增量导出

可以多次运行导出命令，所有结果会追加到同一个目录：

```bash
# 第一次：导出部分
python scripts/export_failure_cases.py \
  --scenario-name "case1" "case2"

# 第二次：追加更多（自动重用现有的 .nuboard 文件）
python scripts/export_failure_cases.py \
  --scenario-name "case3" "case4"

# 所有案例都可以一起查看
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

### 技巧 3: 验证目录结构

导出后验证目录结构是否正确：

```bash
python scripts/verify_nuboard_structure.py work_dirs/failure_viz
```

---

## 🔍 故障排除

### 问题：No available nuBoard files are found

**解决方案：** 检查是否有 `.nuboard` 文件

```bash
ls work_dirs/failure_viz/*.nuboard
```

如果没有，创建一个：

```bash
python scripts/create_nuboard_file.py work_dirs/failure_viz
```

### 问题：NotADirectoryError

**解决方案：** 确保使用 v1.2.1+ 版本，目录结构应该是：

```
work_dirs/failure_viz/
├── nuboard_xxx.nuboard
├── simulation/          ← 必须有这个子目录
│   └── planTF/
└── metrics/
```

如果结构不对，重新导出即可。

### 问题：Scenario not found in database

**解决方案：** 先列出所有可用的 scenarios

```bash
python scripts/export_failure_cases.py --list
```

然后使用列表中显示的 scenario name。

---

## 📊 使用场景示例

### 场景 1: 快速浏览所有失败

```bash
# 一次性导出并查看所有失败案例
python scripts/export_failure_cases.py --all
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

### 场景 2: 分析特定类型的失败

```bash
# 1. 列出所有案例
python scripts/export_failure_cases.py --list

# 2. 选择几个碰撞类型的案例（从列表中挑选）
python scripts/export_failure_cases.py \
  --scenario-name "collision_case1" "collision_case2" "collision_case3"

# 3. 对比分析
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
```

### 场景 3: 逐步调试

```bash
# Day 1: 导出第一个问题案例
python scripts/export_failure_cases.py --scenario-name "problematic_case1"
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
# 分析，发现问题...

# Day 2: 添加相似的案例进行对比
python scripts/export_failure_cases.py --scenario-name "similar_case2" "similar_case3"
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
# 对比分析，找到规律...

# Day 3: 添加正常案例作为参照
python scripts/export_failure_cases.py --scenario-name "normal_case1"
python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006
# 对比正常和失败案例，确定根因...
```

---

## 🎯 最佳实践

### 1. 使用默认目录

始终使用默认的 `work_dirs/failure_viz` 目录，这样：
- 不需要记住路径
- 多次导出会自动追加
- nuBoard 命令更简单

### 2. 先列出，再选择

```bash
# 先看看有什么
python scripts/export_failure_cases.py --list

# 再决定导出什么
python scripts/export_failure_cases.py --scenario-name "..."
```

### 3. 验证结构

导出后验证一下：

```bash
python scripts/verify_nuboard_structure.py work_dirs/failure_viz
```

### 4. 清理旧数据

如果想重新开始：

```bash
rm -rf work_dirs/failure_viz
```

然后重新导出即可。

---

## 📚 需要更多帮助？

- **完整文档：** `docs/failure_case_visualization.md`
- **查询工具：** `docs/TOOL_query_failure_details.md` ✨
- **功能介绍：** `docs/FEATURE_v1.3_batch_export.md`
- **故障排除：** `docs/nuboard_fix_guide.md`
- **Bug 修复历史：** `docs/BUGFIX_v1.2.1_summary.md`

---

## ⚡ 快捷别名（可选）

添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# Failure case 导出别名
alias fc-list='python scripts/export_failure_cases.py --list'
alias fc-all='python scripts/export_failure_cases.py --all'
alias fc-export='python scripts/export_failure_cases.py --scenario-name'
alias fc-query='python scripts/query_failure_details.py --scenario-name'
alias fc-view='python run_nuboard.py simulation_path=work_dirs/failure_viz port_number=5006'
alias fc-verify='python scripts/verify_nuboard_structure.py work_dirs/failure_viz'
```

使用：

```bash
fc-list                    # 列出所有
fc-all                     # 导出所有
fc-export "case1" "case2"  # 导出指定的
fc-query "case1"           # 查询失败详情
fc-query "case1" --show-frames  # 查询详情（含帧号）
fc-view                    # 启动 nuBoard
fc-verify                  # 验证结构
```

---

现在你已经掌握了 failure case 可视化的所有基础知识！🎉
