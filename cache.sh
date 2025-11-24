#!/bin/bash
# 缓存生成脚本 - 支持从本地YAML配置读取参数
# 使用方式:
#   ./cache.sh              # 使用默认配置 (config/local/cache_default.yaml)
#   ./cache.sh cache_1M     # 使用指定配置 (config/local/cache_1M.yaml)

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# 配置文件路径
CONFIG_NAME="${1:-cache_default}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"

# 移除第一个参数（配置文件名），剩余参数用于Hydra override
shift 2>/dev/null || true

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available cache configurations in config/local/:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/cache_*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo ""
        echo "Please create your local config by copying from config/local.example/:"
        echo "  cp config/local.example/cache_default.yaml config/local/cache_default.yaml"
        echo "  # Then edit config/local/cache_default.yaml with your settings"
    fi
    exit 1
fi

echo "Loading cache configuration from: $CONFIG_FILE"
echo ""

# 使用Python加载所有配置信息
# 存储为临时文件以安全传递复杂参数
TEMP_CONFIG_FILE=$(mktemp)
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --mode cache --type cache > "$TEMP_CONFIG_FILE"

# 从输出中提取各种配置信息（逐行读取，避免参数被误分割）
while IFS='=' read -r key value; do
    case "$key" in
        GPU_DEVICES)
            GPU_DEVICES="$value"
            ;;
        HYDRA_PARAMS)
            HYDRA_PARAMS="$value"
            ;;
        NUPLAN_DATA_ROOT)
            export NUPLAN_DATA_ROOT="$value"
            ;;
        NUPLAN_MAPS_ROOT)
            export NUPLAN_MAPS_ROOT="$value"
            ;;
        NUPLAN_EXP_ROOT)
            export NUPLAN_EXP_ROOT="$value"
            ;;
        NUPLAN_DB_FILES)
            export NUPLAN_DB_FILES="$value"
            ;;
    esac
done < "$TEMP_CONFIG_FILE"

# 打印配置信息
echo "Configuration loaded:"
echo "  Config Name: $CONFIG_NAME"
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo ""
echo "Starting cache generation..."
echo ""

# 执行缓存生成
# HYDRA_PARAMS 已包含从配置文件读取的所有参数（包括py_func=cache）
# 额外的命令行参数可以追加（用于临时覆盖配置）
python run_training.py $HYDRA_PARAMS "$@"

# 捕获返回码
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Cache generation completed successfully!"
else
    echo ""
    echo "Cache generation failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
