#!/bin/bash
#
# Convenience script to launch nuBoard for failure case visualization.
#
# Usage:
#   ./scripts/view_failure_case.sh <output_dir> [port]
#
# Examples:
#   ./scripts/view_failure_case.sh work_dirs/failure_viz
#   ./scripts/view_failure_case.sh work_dirs/failure_viz 5007
#

set -e

# Parse arguments
OUTPUT_DIR="${1:-.}"
PORT="${2:-5006}"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory not found: $OUTPUT_DIR"
    echo ""
    echo "Usage: $0 <output_dir> [port]"
    echo ""
    echo "Make sure you have exported failure cases first using:"
    echo "  python scripts/export_failure_cases.py --scenario-name <name> --output-dir <dir>"
    exit 1
fi

echo "Starting nuBoard for failure case visualization..."
echo "  Output directory: $OUTPUT_DIR"
echo "  Port: $PORT"
echo ""
echo "Opening nuBoard at: http://localhost:$PORT"
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# Launch nuBoard
python run_nuboard.py \
    simulation_path="$OUTPUT_DIR" \
    port_number="$PORT"
