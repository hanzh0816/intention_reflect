"""
Simple script to print all unique scenario types from the dataset.
"""

import os
import hydra
from omegaconf import DictConfig
from collections import Counter
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.utils import set_default_path

set_default_path()


@hydra.main(config_path="./config", config_name="default_training")
def main(cfg: DictConfig) -> None:
    """Print all unique scenario types in the dataset."""

    # Set job_name if not present
    if 'job_name' not in cfg:
        cfg.job_name = 'print_scenario_types'

    print("Loading scenarios from dataset...")

    # Build worker pool
    worker = build_worker(cfg)

    # Load scenarios
    all_scenarios = build_scenarios(cfg, worker, model=None)

    print(f"\nTotal scenarios: {len(all_scenarios)}")

    # Collect scenario types
    scenario_types = [scenario.scenario_type for scenario in all_scenarios]
    type_counts = Counter(scenario_types)

    print(f"\n{'='*60}")
    print(f"Unique Scenario Types: {len(type_counts)}")
    print(f"{'='*60}")

    # Print sorted by count (descending)
    for scenario_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {scenario_type:<40} : {count:>6} scenarios")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
