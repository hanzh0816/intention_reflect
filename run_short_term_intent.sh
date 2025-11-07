#!/bin/bash

# Script to visualize ego trajectories with short-term intent classification (v3.0)
#
# Usage:
#   ./run_short_term_intent.sh [num_scenarios] [scenario_type]
#
# Examples:
#   ./run_short_term_intent.sh 10
#   ./run_short_term_intent.sh 20 traversing_crosswalk
#
# Intent Categories:
#   Lateral: turn_left, turn_right, shift_left, shift_right, stay_in_lane
#   Longitudinal: accelerate, maintain_speed, decelerate, stop

NUM_SCENARIOS=${1:-10}
SCENARIO_TYPE=${2:-""}

if [ -z "$SCENARIO_TYPE" ]; then
    echo "Running short-term intent visualization for $NUM_SCENARIOS random scenarios..."
    python visualize_short_term_intent.py \
        job_name=short_term_intent \
        py_func=train \
        scenario_builder=nuplan_mini \
        scenario_filter=training_scenarios_1M \
        splitter=nuplan \
        worker=sequential \
        +num_scenarios_to_visualize=$NUM_SCENARIOS
else
    echo "Running short-term intent visualization for $NUM_SCENARIOS scenarios of type: $SCENARIO_TYPE..."
    python visualize_short_term_intent.py \
        job_name=short_term_intent \
        py_func=train \
        scenario_builder=nuplan_mini \
        scenario_filter=training_scenarios_1M \
        splitter=nuplan \
        worker=sequential \
        +num_scenarios_to_visualize=$NUM_SCENARIOS \
        +scenario_type=$SCENARIO_TYPE
fi
