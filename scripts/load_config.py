#!/usr/bin/env python3
"""配置文件加载工具 v2.0 - 精简版"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

from config_utils import (
    load_yaml,
    flatten_dict,
    format_value,
    load_base_hydra_config,
    config_key_exists,
    apply_override,
    get_gpu_devices,
    get_env_vars,
    get_config_metadata,
    validate_hydra_config,
)


def build_hydra_params(config: Dict[str, Any], mode: str) -> List[str]:
    """构建 Hydra 命令行参数"""
    cfg = load_base_hydra_config(mode)
    params = []
    excluded_keys = {"name", "version", "gpu", "paths"}

    for key, value in config.items():
        if key in excluded_keys:
            continue

        if isinstance(value, str):
            params.append(_build_param(cfg, key, value))
            _try_update_cfg(cfg, key, value)

        elif isinstance(value, dict):
            if "_select" in value:
                group_name = value["_select"]
                params.append(_build_param(cfg, key, group_name))
                _try_update_cfg(cfg, key, group_name)

            filtered = {k: v for k, v in value.items() if k != "_select"}
            if filtered:
                for nested_key, nested_val in flatten_dict({key: filtered}).items():
                    params.append(_build_param(cfg, nested_key, nested_val))
                    _try_update_cfg(cfg, nested_key, nested_val)

        else:
            params.append(_build_param(cfg, key, format_value(value)))
            _try_update_cfg(cfg, key, value)

    if mode == "train":
        params.extend(
            [
                f"+local_config.name={config.get('name', 'default')}",
                f"+local_config.version={config.get('version', '1.0')}",
            ]
        )

    return params


def _build_param(cfg, key: str, value: str) -> str:
    """构建单个参数"""
    prefix = "" if config_key_exists(cfg, key) else "+"
    # Quote value if it contains special characters that Hydra might misinterpret
    if any(char in str(value) for char in ['=', ',', '[', ']', '{', '}', ' ']):
        # Escape single quotes in the value and wrap in single quotes
        value = str(value).replace("'", "\\'")
        return f"{prefix}{key}='{value}'"
    return f"{prefix}{key}={value}"


def _try_update_cfg(cfg, key: str, value: Any) -> None:
    """尝试更新配置"""
    try:
        apply_override(cfg, key, value)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="配置文件加载工具 v2.0")
    parser.add_argument("config_file", help="配置文件路径")
    parser.add_argument(
        "--type",
        choices=["train", "cache", "eval"],
        default="train",
        help="配置类型",
    )
    parser.add_argument("--validate", action="store_true", help="验证配置")
    args = parser.parse_args()

    if not Path(args.config_file).exists():
        print(f"Error: Config file not found: {args.config_file}", file=sys.stderr)
        sys.exit(1)

    config = load_yaml(args.config_file)

    print(f"GPU_DEVICES={get_gpu_devices(config)}")

    for key, value in get_env_vars(config).items():
        print(f"{key}={value}")

    params = build_hydra_params(config, args.type)
    print("HYDRA_PARAMS=" + " ".join(params))

    for key, value in get_config_metadata(config).items():
        print(f"CONFIG_{key.upper()}={value}")

    # 配置验证放在最后
    if args.validate and not validate_hydra_config(params, args.type):
        sys.exit(1)


if __name__ == "__main__":
    main()
