#!/bin/bash
# 缓存生成脚本 - 使用方式: ./cache.sh [config_name]

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONFIG_NAME="${1:-cache_default}"
CONFIG_FILE="${SCRIPT_DIR}/config/local/${CONFIG_NAME}.yaml"
shift 2>/dev/null || true

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Available cache configurations:"
    if [ -d "${SCRIPT_DIR}/config/local" ]; then
        ls -1 "${SCRIPT_DIR}/config/local"/cache_*.yaml 2>/dev/null | sed 's/.*\//  - /' | sed 's/.yaml//'
    else
        echo "  (config/local directory not found)"
        echo "  Please copy from config/local.example/"
    fi
    exit 1
fi

echo "Loading cache configuration from: $CONFIG_FILE"
echo ""

mkdir -p "${SCRIPT_DIR}/tmp"
TEMP_CONFIG_FILE="${SCRIPT_DIR}/tmp/cache_config_$$"
trap "rm -f $TEMP_CONFIG_FILE" EXIT

python3 "${SCRIPT_DIR}/scripts/load_config.py" "$CONFIG_FILE" --type cache > "$TEMP_CONFIG_FILE"

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
[ -n "$NUPLAN_DATA_ROOT" ] && echo "  NUPLAN_DATA_ROOT: $NUPLAN_DATA_ROOT"
[ -n "$NUPLAN_MAPS_ROOT" ] && echo "  NUPLAN_MAPS_ROOT: $NUPLAN_MAPS_ROOT"
[ -n "$NUPLAN_EXP_ROOT" ] && echo "  NUPLAN_EXP_ROOT: $NUPLAN_EXP_ROOT"

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo ""
echo "Starting cache generation..."
echo ""

python run_training.py $HYDRA_PARAMS "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Cache generation completed successfully!"
else
    echo ""
    echo "Cache generation failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
