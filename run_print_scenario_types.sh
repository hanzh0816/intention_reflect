#!/bin/bash

# Simple script to print all scenario types

python print_scenario_types.py \
    job_name=print_scenario_types \
    py_func=train \
    scenario_builder=nuplan_mini \
    scenario_filter=training_scenarios_1M \
    splitter=nuplan \
    worker=sequential
