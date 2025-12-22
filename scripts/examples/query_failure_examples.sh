#!/bin/bash
# Example usage of query_failure_details.py

echo "=================================================="
echo "Failure Details Query Tool - Usage Examples"
echo "=================================================="
echo ""

# Check if database exists
if [ ! -f "work_dirs/exp/failure_cases.db" ]; then
    echo "⚠️  Warning: Database not found at work_dirs/exp/failure_cases.db"
    echo ""
    echo "To create the database:"
    echo "  1. Run simulation with failure_case_collector callback enabled"
    echo "  2. Make sure your eval config includes:"
    echo "     callback:"
    echo "       - failure_case_collector"
    echo ""
    exit 1
fi

echo "✓ Database found"
echo ""

# Example 1: List all available scenarios
echo "Example 1: List all available failure cases"
echo "------------------------------------------------------------"
echo "Command: python scripts/export_failure_cases.py --list"
echo ""
python scripts/export_failure_cases.py --list --database-path work_dirs/exp/failure_cases.db 2>/dev/null | head -20
echo ""
echo "Press Enter to continue..."
read

# Example 2: Quick query (without frame numbers)
echo ""
echo "Example 2: Quick query for a specific scenario"
echo "------------------------------------------------------------"
echo "Command: python scripts/query_failure_details.py --scenario-name <TOKEN>"
echo ""

# Get first scenario from database
FIRST_SCENARIO=$(sqlite3 work_dirs/exp/failure_cases.db "SELECT scenario_name FROM failure_cases LIMIT 1" 2>/dev/null)

if [ -n "$FIRST_SCENARIO" ]; then
    echo "Querying scenario: $FIRST_SCENARIO"
    echo ""
    python scripts/query_failure_details.py --scenario-name "$FIRST_SCENARIO"
else
    echo "No scenarios found in database"
fi

echo ""
echo "Press Enter to continue..."
read

# Example 3: Detailed query with frame numbers
echo ""
echo "Example 3: Detailed query with frame numbers"
echo "------------------------------------------------------------"
echo "Command: python scripts/query_failure_details.py --scenario-name <TOKEN> --show-frames"
echo ""

if [ -n "$FIRST_SCENARIO" ]; then
    echo "Querying scenario: $FIRST_SCENARIO (with frame numbers)"
    echo ""
    python scripts/query_failure_details.py --scenario-name "$FIRST_SCENARIO" --show-frames
else
    echo "No scenarios found in database"
fi

echo ""
echo "Press Enter to continue..."
read

# Example 4: Query specific failure types
echo ""
echo "Example 4: Query only collision failures"
echo "------------------------------------------------------------"
echo "Command: Query all scenarios with collisions"
echo ""

COLLISION_SCENARIOS=$(sqlite3 work_dirs/exp/failure_cases.db \
    "SELECT scenario_name FROM failure_cases WHERE has_collision=1 LIMIT 3" 2>/dev/null)

if [ -n "$COLLISION_SCENARIOS" ]; then
    echo "$COLLISION_SCENARIOS" | while read scenario; do
        echo ""
        echo "Analyzing collision in: $scenario"
        echo "----------------------------------------"
        python scripts/query_failure_details.py --scenario-name "$scenario"
        echo ""
    done
else
    echo "No collision scenarios found"
fi

echo ""
echo "=================================================="
echo "Examples completed!"
echo "=================================================="
echo ""
echo "For more information, see: docs/TOOL_query_failure_details.md"
