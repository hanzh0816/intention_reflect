"""
查看闭环仿真结果的工具脚本
"""
import argparse
import pandas as pd
from pathlib import Path
import pprint


def find_latest_result(root_dir):
    """找到最新的仿真结果目录"""
    root = Path(root_dir)
    if not root.exists():
        return None

    # 找到所有aggregator_metric目录
    metric_dirs = list(root.rglob('aggregator_metric'))
    if not metric_dirs:
        return None

    # 按修改时间排序，返回最新的
    metric_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return metric_dirs[0]


def load_metrics(metric_dir):
    """加载metrics parquet文件"""
    parquet_files = list(Path(metric_dir).glob('*.parquet'))
    if not parquet_files:
        print(f"未找到parquet文件在: {metric_dir}")
        return None

    # 通常只有一个parquet文件
    df = pd.read_parquet(parquet_files[0])
    return df


def print_summary(df):
    """打印汇总结果"""
    print("=" * 80)
    print("仿真结果汇总 (Final Score)")
    print("=" * 80)

    final_score = df[df['scenario'] == 'final_score']
    if final_score.empty:
        print("未找到final_score")
        return

    final_score_dict = final_score.to_dict(orient='records')[0]

    # 分类显示metrics
    safety_metrics = {}
    traffic_metrics = {}
    comfort_metrics = {}
    progress_metrics = {}
    other_metrics = {}

    for key, value in final_score_dict.items():
        if key in ['scenario', 'planner', 'scenario_type']:
            continue

        if 'collision' in key.lower() or 'ttc' in key.lower():
            safety_metrics[key] = value
        elif 'compliance' in key.lower() or 'limit' in key.lower():
            traffic_metrics[key] = value
        elif 'comfort' in key.lower() or 'jerk' in key.lower() or 'acceleration' in key.lower():
            comfort_metrics[key] = value
        elif 'progress' in key.lower():
            progress_metrics[key] = value
        else:
            other_metrics[key] = value

    if safety_metrics:
        print("\n【安全指标】")
        for k, v in safety_metrics.items():
            print(f"  {k}: {v:.4f}")

    if traffic_metrics:
        print("\n【交通规则遵守】")
        for k, v in traffic_metrics.items():
            print(f"  {k}: {v:.4f}")

    if comfort_metrics:
        print("\n【舒适度指标】")
        for k, v in comfort_metrics.items():
            print(f"  {k}: {v:.4f}")

    if progress_metrics:
        print("\n【任务完成度】")
        for k, v in progress_metrics.items():
            print(f"  {k}: {v:.4f}")

    if other_metrics:
        print("\n【其他指标】")
        for k, v in other_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


def print_scenario_breakdown(df):
    """按场景类型打印结果"""
    print("\n" + "=" * 80)
    print("按场景类型统计")
    print("=" * 80)

    # 排除final_score
    scenario_data = df[df['scenario'] != 'final_score']

    if scenario_data.empty:
        print("没有场景级别的数据")
        return

    # 按scenario_type分组
    if 'scenario_type' in scenario_data.columns:
        grouped = scenario_data.groupby('scenario_type')

        for scenario_type, group in grouped:
            print(f"\n{scenario_type} ({len(group)} 场景):")

            # 选择关键指标
            key_metrics = [
                'no_at_fault_collisions',
                'drivable_area_compliance',
                'driving_direction_compliance',
                'ego_progress_along_expert_route'
            ]

            for metric in key_metrics:
                if metric in group.columns:
                    mean_val = group[metric].mean()
                    print(f"  {metric}: {mean_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='查看PlanTF仿真结果')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='仿真结果目录（如果不指定，自动查找最新）')
    parser.add_argument('--root', type=str, default='/data2/hzh/nuplan/exp/simulation_results',
                        help='仿真结果根目录')
    parser.add_argument('--breakdown', action='store_true',
                        help='显示按场景类型的详细分解')

    args = parser.parse_args()

    # 查找结果目录
    if args.result_dir:
        metric_dir = Path(args.result_dir) / 'aggregator_metric'
        if not metric_dir.exists():
            metric_dir = Path(args.result_dir)
    else:
        print(f"在 {args.root} 中查找最新的仿真结果...")
        metric_dir = find_latest_result(args.root)
        if not metric_dir:
            print(f"未找到任何仿真结果在: {args.root}")
            return
        print(f"找到结果目录: {metric_dir.parent}")

    # 加载metrics
    print(f"\n加载metrics从: {metric_dir}")
    df = load_metrics(metric_dir)

    if df is None:
        return

    # 打印汇总
    print_summary(df)

    # 如果需要，打印场景分解
    if args.breakdown:
        print_scenario_breakdown(df)

    print("\n" + "=" * 80)
    print(f"总场景数: {len(df[df['scenario'] != 'final_score'])}")
    print("=" * 80)


if __name__ == '__main__':
    main()
