#!/bin/bash
# 训练脚本 - 使用方式: ./train.sh [config_name]

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONFIG_NAME="${1:-default}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"
shift 2>/dev/null || true

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available configurations:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo "  Please copy from config/local.example/"
    fi
    exit 1
fi

echo "Loading configuration from: $CONFIG_FILE"
echo ""

mkdir -p "${SCRIPT_DIR}/tmp"
TEMP_CONFIG_FILE="${SCRIPT_DIR}/tmp/train_config_$$"
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --type train > "$TEMP_CONFIG_FILE"

while IFS='=' read -r key value; do
    case "$key" in
        GPU_DEVICES) GPU_DEVICES="$value" ;;
        HYDRA_PARAMS) HYDRA_PARAMS="$value" ;;
        NUPLAN_DATA_ROOT) export NUPLAN_DATA_ROOT="$value" ;;
        NUPLAN_MAPS_ROOT) export NUPLAN_MAPS_ROOT="$value" ;;
        NUPLAN_EXP_ROOT) export NUPLAN_EXP_ROOT="$value" ;;
        NUPLAN_DB_FILES) export NUPLAN_DB_FILES="$value" ;;
    esac
done < "$TEMP_CONFIG_FILE"

echo "Configuration loaded:"
echo "  GPU Devices: $GPU_DEVICES"
echo "  Config Name: $CONFIG_NAME"
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

echo ""
echo "Starting training..."
echo ""

python run_training.py py_func=train $HYDRA_PARAMS "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
else
    echo ""
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
