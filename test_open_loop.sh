#!/bin/bash
# Open-loop testing script for PlanTF model
# Open-loop: 在验证集上直接评估轨迹预测精度（ADE, FDE等指标）
#
# 使用方式:
#   ./test_open_loop.sh /path/to/checkpoint.ckpt              # 使用默认配置

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 获取checkpoint路径和配置名
CHECKPOINT_PATH="${1:-/path/to/your/checkpoint.ckpt}"
CONFIG_NAME="${2:-default}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Usage: $0 <checkpoint_path> [config_name]"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configurations in config/local/:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo ""
        echo "Please create your local config by copying from config/local.example/:"
        echo "  cp config/local.example/default.yaml config/local/default.yaml"
    fi
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
echo ""

# 使用Python加载所有配置信息
# 存储为临时文件以安全传递复杂参数
TEMP_CONFIG_FILE=$(mktemp)
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --mode all > "$TEMP_CONFIG_FILE"

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

# 打印测试配置
echo "Test configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  GPU Devices: $GPU_DEVICES"
echo "  Config Name: $CONFIG_NAME"
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

echo ""
echo "Running open-loop test..."
echo "==============================================="
echo ""

# 执行测试
# HYDRA_PARAMS 已包含从配置文件读取的所有参数，但需要移除部分训练相关参数
# 对于测试，我们需要覆盖 py_func 和 checkpoint
python run_training.py \
    py_func=validate \
    checkpoint=$CHECKPOINT_PATH \
    $HYDRA_PARAMS \
    "$@"

# 捕获返回码
EXIT_CODE=$?

echo ""
echo "==============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Open-loop test completed successfully!"
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo "Metrics: minADE1, minADE6, minFDE1, minFDE6, MR"
else
    echo "Open-loop test failed with exit code: $EXIT_CODE"
fi
echo "==============================================="
echo ""

exit $EXIT_CODE

