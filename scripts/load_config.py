#!/usr/bin/env python3
"""
配置文件加载工具
用于读取本地YAML配置并生成Hydra命令行参数和环境变量
"""

import yaml
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_file: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in config file: {e}", file=sys.stderr)
        sys.exit(1)


def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, str]:
    """
    将嵌套字典展平为点号分隔的键值对
    例如: {'training': {'batch_size': 64}} -> {'training.batch_size': '64'}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def get_gpu_devices(config: Dict[str, Any]) -> str:
    """从配置中获取GPU设备列表"""
    gpu_config = config.get('gpu', {})
    devices = gpu_config.get('devices', '0')
    return str(devices)


def get_env_vars(config: Dict[str, Any]) -> Dict[str, str]:
    """
    从配置中提取环境变量
    """
    env_vars = {}

    # 路径相关环境变量
    paths = config.get('paths', {})
    if 'nuplan_data_root' in paths:
        env_vars['NUPLAN_DATA_ROOT'] = str(paths['nuplan_data_root'])
    if 'nuplan_maps_root' in paths:
        env_vars['NUPLAN_MAPS_ROOT'] = str(paths['nuplan_maps_root'])
    if 'nuplan_exp_root' in paths:
        env_vars['NUPLAN_EXP_ROOT'] = str(paths['nuplan_exp_root'])
    if 'nuplan_db_files' in paths:
        env_vars['NUPLAN_DB_FILES'] = str(paths['nuplan_db_files'])

    return env_vars


def get_config_params(config: Dict[str, Any]) -> List[str]:
    """
    将配置转换为Hydra命令行参数列表（用于训练）
    """
    params = []

    # 添加Worker配置
    worker = config.get('worker', {})
    worker_type = worker.get('type', 'single_machine_thread_pool')
    worker_max = worker.get('max_workers', 64)
    params.append(f"worker={worker_type}")
    params.append(f"worker.max_workers={worker_max}")

    # 添加Scenario和Cache配置
    data = config.get('data', {})
    scenario_builder = data.get('scenario_builder', 'nuplan')
    scenario_filter = data.get('scenario_filter', 'training_scenarios_1M')
    params.append(f"scenario_builder={scenario_builder}")
    params.append(f"+training=train_planTF")
    params.append(f"scenario_filter={scenario_filter}")

    # Cache配置
    cache = data.get('cache', {})
    cache_path = cache.get('cache_path', '/data2/hzh/nuplan/train_cache')
    use_cache = cache.get('use_cache_without_dataset', True)
    cleanup = cache.get('cleanup_cache', False)

    params.append(f"cache.cache_path={cache_path}")
    params.append(f"cache.use_cache_without_dataset={str(use_cache).lower()}")
    params.append(f"cache.cleanup_cache={str(cleanup).lower()}")

    # 添加训练参数
    training = config.get('training', {})
    params.append(f"data_loader.params.batch_size={training.get('batch_size', 64)}")
    params.append(f"data_loader.params.num_workers={training.get('num_workers', 64)}")
    params.append(f"lr={training.get('lr', '1e-3')}")
    params.append(f"epochs={training.get('epochs', 25)}")
    params.append(f"warmup_epochs={training.get('warmup_epochs', 3)}")
    params.append(f"weight_decay={training.get('weight_decay', '0.0001')}")

    # 添加Lightning参数
    lightning = config.get('lightning', {})
    params.append(f"lightning.trainer.params.val_check_interval={lightning.get('val_check_interval', 1.0)}")

    # 添加Wandb参数
    wandb = config.get('wandb', {})
    params.append(f"wandb.mode={wandb.get('mode', 'disable')}")
    params.append(f"wandb.project={wandb.get('project', 'nuplan')}")
    params.append(f"wandb.name={wandb.get('name', 'experiment')}")

    # 添加配置版本信息
    config_name = config.get('name', 'default')
    config_version = config.get('version', '1.0')
    params.append(f"+local_config.name={config_name}")
    params.append(f"+local_config.version={config_version}")

    return params


def get_cache_params(config: Dict[str, Any]) -> List[str]:
    """
    将配置转换为Hydra命令行参数列表（用于缓存）
    """
    params = []

    # 添加基础配置
    data = config.get('data', {})
    scenario_builder = data.get('scenario_builder', 'nuplan')
    scenario_filter = data.get('scenario_filter', 'training_scenarios_1M')

    params.append(f"py_func=cache")
    params.append(f"+training=train_planTF")
    params.append(f"scenario_builder={scenario_builder}")
    params.append(f"scenario_filter={scenario_filter}")

    # Cache配置
    cache = data.get('cache', {})
    cache_path = cache.get('cache_path', '/data2/hzh/nuplan/exp/cache_plantf_1M')
    cleanup = cache.get('cleanup_cache', True)

    params.append(f"cache.cache_path={cache_path}")
    params.append(f"cache.cleanup_cache={str(cleanup).lower()}")

    # Worker配置（cache使用threads_per_node）
    worker = config.get('worker', {})
    threads_per_node = worker.get('threads_per_node', 40)
    params.append(f"worker.threads_per_node={threads_per_node}")

    return params


def get_config_metadata(config: Dict[str, Any]) -> Dict[str, str]:
    """获取配置元数据（用于日志记录）"""
    return {
        'name': config.get('name', 'default'),
        'version': config.get('version', '1.0'),
        'gpu_devices': get_gpu_devices(config),
        'batch_size': str(config.get('training', {}).get('batch_size', 64)),
        'wandb_mode': config.get('wandb', {}).get('mode', 'disable'),
    }


def main():
    parser = argparse.ArgumentParser(description='配置文件加载工具')
    parser.add_argument('config_file', help='YAML配置文件路径')
    parser.add_argument('--mode', choices=['gpu', 'params', 'metadata', 'env', 'all', 'bash', 'cache'],
                       default='all', help='输出模式')
    parser.add_argument('--type', choices=['train', 'cache'], default='train',
                       help='配置类型：train为训练，cache为缓存')

    args = parser.parse_args()

    # 检查配置文件是否存在
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    # 加载配置
    config = load_config(str(config_path))

    # 根据模式输出
    if args.mode in ['gpu', 'all', 'cache']:
        print(f"GPU_DEVICES={get_gpu_devices(config)}")

    if args.mode in ['env', 'all', 'cache']:
        env_vars = get_env_vars(config)
        for key, value in env_vars.items():
            print(f"{key}={value}")

    if args.mode in ['params', 'all', 'bash', 'cache']:
        # 将HYDRA_PARAMS作为一个数组，每个参数一行（用于bash读取）
        if args.type == 'cache' or args.mode == 'cache':
            params = get_cache_params(config)
        else:
            params = get_config_params(config)

        if args.mode == 'bash':
            # bash模式：每行一个参数
            for param in params:
                print(f"'{param}'")
        else:
            # 默认模式：单行输出
            print("HYDRA_PARAMS=" + " ".join(params))

    if args.mode in ['metadata', 'all']:
        metadata = get_config_metadata(config)
        for key, value in metadata.items():
            print(f"CONFIG_{key.upper()}={value}")


if __name__ == '__main__':
    main()

