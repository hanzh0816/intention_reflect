#!/bin/bash
# 评估脚本 - 使用方式: ./eval.sh <config_name> <challenge>

export PYTHONPATH=$PYTHONPATH:$(pwd)
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

CONFIG_NAME="${1:-eval_default}"
CHALLENGE="${2:-}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"

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
    echo "Example: $0 eval_default closed_loop_nonreactive_agents"
    exit 1
fi

shift 2 2>/dev/null || true

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available evaluation configurations:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/eval_*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo "  Please copy from config/local.example/"
    fi
    exit 1
fi

echo "Loading evaluation configuration from: $CONFIG_FILE"
echo "Challenge: $CHALLENGE"
echo ""

mkdir -p "${SCRIPT_DIR}/tmp"
TEMP_CONFIG_FILE="${SCRIPT_DIR}/tmp/eval_config_$$"
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --type eval > "$TEMP_CONFIG_FILE"

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
echo "  Config Name: $CONFIG_NAME"
echo "  Challenge: $CHALLENGE"
[ -n "$GPU_DEVICES" ] && echo "  GPU Devices: $GPU_DEVICES"
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

echo ""
echo "Starting evaluation..."
echo "==============================================="
echo ""

python run_simulation.py \
    +simulation=$CHALLENGE \
    $HYDRA_PARAMS \
    "$@"

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
