# nuBoard 加载失败案例修复指南

## 问题症状

运行 `run_nuboard.py` 时出现错误：
```
INFO:nuplan.planning.nuboard.utils.utils:No available nuBoard files are found.
```

## 问题原因

nuBoard 需要一个 `.nuboard` 元数据文件来定位 SimulationLog 文件，但早期版本的 failure_case 导出工具没有自动创建这个文件。

## 解决方案

### 方案 1：重新导出（推荐）

使用最新版本的导出工具重新导出 failure case，会自动生成 `.nuboard` 文件：

```bash
python scripts/export_failure_cases.py \
  --scenario-name "your_scenario_name" \
  --database-path work_dirs/exp/failure_cases.db \
  --output-dir work_dirs/failure_viz
```

### 方案 2：为已导出的目录创建 .nuboard 文件

如果你已经导出了 failure case，可以使用工具脚本快速创建 `.nuboard` 文件：

```bash
python scripts/create_nuboard_file.py work_dirs/failure_viz
```

然后启动 nuBoard：

```bash
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

### 方案 3：手动创建（开发者）

如果需要自定义配置，可以使用 Python 手动创建：

```python
from pathlib import Path
from nuplan.planning.nuboard.base.data_class import NuBoardFile
import time

output_dir = Path("work_dirs/failure_viz")
nuboard_filename = output_dir / f"nuboard_{int(time.time())}.nuboard"

nuboard_file = NuBoardFile(
    simulation_main_path=str(output_dir),
    simulation_folder=".",
    metric_main_path=str(output_dir),
    metric_folder="metrics",
    aggregator_metric_folder="aggregator_metric",
)

nuboard_file.save_nuboard_file(nuboard_filename)
print(f"Created: {nuboard_filename}")
```

## 目录结构说明

正确的导出目录结构应该是：

```
work_dirs/failure_viz/
├── nuboard_1703123456.nuboard    # 必需的元数据文件
├── planTF/                        # planner 名称
│   └── highway/                   # scenario_type
│       └── 2021.07.16.xxx/        # log_name
│           └── 00cca24d240f5980/  # scenario_name
│               └── 00cca24d240f5980.pkl.xz  # SimulationLog
└── metrics/                       # 空目录（可选）
    └── aggregator_metric/         # 空目录（可选）
```

## 验证

1. 检查 `.nuboard` 文件是否存在：
```bash
ls -la work_dirs/failure_viz/*.nuboard
```

2. 启动 nuBoard：
```bash
python run_nuboard.py \
  simulation_path=work_dirs/failure_viz \
  port_number=5006
```

3. 在浏览器中访问：http://localhost:5006

4. 如果成功，你应该能在 nuBoard 界面中看到导出的 failure case

## 版本信息

- **v1.0-v1.1**: 不自动生成 `.nuboard` 文件（需要手动创建）
- **v1.2+**: 自动生成 `.nuboard` 文件（推荐）

## 更多信息

详细文档：`docs/failure_case_visualization.md`
