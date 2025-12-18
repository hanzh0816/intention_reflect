"""配置工具函数模块"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf


def load_yaml(config_file: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_file}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML: {e}", file=sys.stderr)
        sys.exit(1)


def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, str]:
    """展平嵌套字典为点号分隔的键值对"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def format_value(value: Any) -> str:
    """格式化配置值"""
    return str(value).lower() if isinstance(value, bool) else str(value)


def load_base_hydra_config(mode: str) -> DictConfig:
    """加载 Hydra 基础配置"""
    script_dir = Path(__file__).parent.parent
    config_dir = str(script_dir / "config")
    config_name = "default_training" if mode in ["train", "cache"] else "default_simulation"

    try:
        with initialize_config_dir(config_dir=config_dir):
            base_overrides = (
                ["py_func=train", "splitter=nuplan"]
                if mode in ["train", "cache"]
                else []
            )
            cfg = compose(config_name=config_name, overrides=base_overrides)
        return cfg
    except Exception as e:
        print(f"Error: Failed to load base config: {e}", file=sys.stderr)
        raise


def config_key_exists(cfg: DictConfig, key_path: str) -> bool:
    """检查配置键是否存在"""
    try:
        keys = key_path.split(".")
        current = cfg
        for key in keys:
            if not OmegaConf.is_dict(current) or key not in current:
                return False
            current = current[key]
        return not OmegaConf.is_missing(cfg, key_path)
    except Exception:
        return False


def apply_override(cfg: DictConfig, key_path: str, value: Any) -> None:
    """应用覆盖值到配置"""
    keys = key_path.split(".")
    current = cfg
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_gpu_devices(config: Dict[str, Any]) -> str:
    """获取 GPU 设备"""
    return str(config.get("gpu", {}).get("devices", "0"))


def get_env_vars(config: Dict[str, Any]) -> Dict[str, str]:
    """提取环境变量"""
    env_vars = {}
    paths = config.get("paths", {})
    env_mapping = {
        "nuplan_data_root": "NUPLAN_DATA_ROOT",
        "nuplan_maps_root": "NUPLAN_MAPS_ROOT",
        "nuplan_exp_root": "NUPLAN_EXP_ROOT",
        "nuplan_db_files": "NUPLAN_DB_FILES",
    }
    for key, env_key in env_mapping.items():
        if key in paths:
            env_vars[env_key] = str(paths[key])
    return env_vars


def get_config_metadata(config: Dict[str, Any]) -> Dict[str, str]:
    """获取配置元数据"""
    return {
        "name": config.get("name", "default"),
        "version": config.get("version", "1.0"),
        "gpu_devices": get_gpu_devices(config),
    }


def validate_hydra_config(params: list, mode: str) -> bool:
    """验证 Hydra 配置是否有效"""
    script_dir = Path(__file__).parent.parent
    config_dir = str(script_dir / "config")
    config_name = (
        "default_training" if mode in ["train", "cache"] else "default_simulation"
    )

    try:
        with initialize_config_dir(config_dir=config_dir):
            compose(config_name=config_name, overrides=params)
        print("\n✓ Configuration validated successfully")
        return True
    except Exception as e:
        print(f"\n⚠️  Configuration validation failed:")
        print(f"   {e}")
        return False
