#!/bin/bash
# 评估脚本 - 支持从本地YAML配置读取参数
# 一次只评估一个challenge
# 使用方式:
#   ./eval.sh test closed_loop_nonreactive_agents            # 使用test配置评估closed_loop_nonreactive_agents challenge
#   ./eval.sh test closed_loop_reactive_agents               # 评估closed_loop_reactive_agents challenge
#   ./eval.sh test open_loop_boxes                           # 评估open_loop_boxes challenge

# 获取脚本所在目录
export PYTHONPATH=$PYTHONPATH:$(pwd)
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# 获取配置名和challenge
CONFIG_NAME="${1:-eval_default}"
CHALLENGE="${2:-}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"

# 检查是否提供了challenge
if [ -z "$CHALLENGE" ]; then
    echo "Error: Challenge name is required"
    echo ""
    echo "Usage: $0 <config_name> <challenge>"
    echo ""
    echo "Available challenges:"
    echo "  - closed_loop_nonreactive_agents"
    echo "  - closed_loop_reactive_agents"
    echo "  - open_loop_boxes"
    echo ""
    echo "Example:"
    echo "  $0 eval_default closed_loop_nonreactive_agents"
    exit 1
fi

# 移除前两个参数，剩余参数用于Hydra override
shift 2 2>/dev/null || true

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available evaluation configurations in config/local/:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/eval_*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo ""
        echo "Please create your local config by copying from config/local.example/:"
        echo "  cp config/local.example/eval_default.yaml config/local/eval_default.yaml"
        echo "  # Then edit config/local/eval_default.yaml with your settings"
    fi
    exit 1
fi

echo "Loading evaluation configuration from: $CONFIG_FILE"
echo "Challenge: $CHALLENGE"
echo ""

# 使用Python加载所有配置信息
# 存储为临时文件以安全传递复杂参数
TEMP_CONFIG_FILE=$(mktemp)
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --mode eval --type eval > "$TEMP_CONFIG_FILE"

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

# 打印评估配置
echo "Configuration loaded:"
echo "  Config Name: $CONFIG_NAME"
echo "  Challenge: $CHALLENGE"
[ -n "$GPU_DEVICES" ] && echo "  GPU Devices: $GPU_DEVICES"
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

echo ""
echo "Starting evaluation..."
echo "==============================================="
echo ""

# 执行评估
# HYDRA_PARAMS 已包含从配置文件读取的所有参数
# 额外的命令行参数可以追加（用于临时覆盖配置）
python run_simulation.py \
    +simulation=$CHALLENGE \
    $HYDRA_PARAMS \
    "$@"

# 捕获返回码
EXIT_CODE=$?

echo ""
echo "==============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Challenge: $CHALLENGE"
else
    echo "Evaluation failed with exit code: $EXIT_CODE"
fi
echo "==============================================="
echo ""

exit $EXIT_CODE
