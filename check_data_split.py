"""
Check the actual train/val/test split in the cache and config
"""
import os
from pathlib import Path
from collections import defaultdict
import yaml

# Load splitter config
splitter_config = "/home/hzh/code/planning/planTF/nuplan-devkit/nuplan/planning/script/config/common/splitter/nuplan.yaml"

print("Analyzing splitter configuration...")
print("=" * 80)

# Count logs in each split
with open(splitter_config, 'r') as f:
    lines = f.readlines()

train_logs = []
val_logs = []
test_logs = []

current_split = None
for line in lines:
    line = line.strip()
    if line == "train:":
        current_split = "train"
    elif line == "val:":
        current_split = "val"
    elif line == "test:":
        current_split = "test"
    elif line.startswith("- ") and current_split:
        log_name = line[2:]  # Remove "- " prefix
        if current_split == "train":
            train_logs.append(log_name)
        elif current_split == "val":
            val_logs.append(log_name)
        elif current_split == "test":
            test_logs.append(log_name)

print(f"Number of logs in train split: {len(train_logs)}")
print(f"Number of logs in val split: {len(val_logs)}")
print(f"Number of logs in test split: {len(test_logs)}")
print()

# Now check cache directory
cache_path = "/data2/hzh/nuplan/exp/cache_plantf_1M"
print(f"Checking cache directory: {cache_path}")
print("=" * 80)

if os.path.exists(cache_path):
    # Count scenarios per log
    scenario_counts = defaultdict(int)

    for root, dirs, files in os.walk(cache_path):
        # Extract log name from path
        # Path structure: cache_path/log_name/scenario_type/...
        parts = Path(root).relative_to(cache_path).parts
        if len(parts) >= 1:
            log_name = parts[0]
            # Count unique scenario directories
            if 'trajectory.gz' in files:
                scenario_counts[log_name] += 1

    # Categorize by split
    train_count = sum(scenario_counts.get(log, 0) for log in train_logs)
    val_count = sum(scenario_counts.get(log, 0) for log in val_logs)
    test_count = sum(scenario_counts.get(log, 0) for log in test_logs)

    print(f"Scenarios in cache (train split): {train_count}")
    print(f"Scenarios in cache (val split): {val_count}")
    print(f"Scenarios in cache (test split): {test_count}")
    print()
    print(f"Train/Val ratio: {train_count}/{val_count} = {train_count/val_count if val_count > 0 else 'N/A':.2f}")
    print()

    # Show top logs with most scenarios
    sorted_logs = sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True)
    print("Top 20 logs by scenario count:")
    for log, count in sorted_logs[:20]:
        split = "train" if log in train_logs else ("val" if log in val_logs else ("test" if log in test_logs else "unknown"))
        print(f"  {log}: {count} scenarios ({split})")
else:
    print(f"Cache path does not exist: {cache_path}")
