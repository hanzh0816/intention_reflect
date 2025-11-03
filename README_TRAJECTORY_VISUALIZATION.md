# Ego Trajectory Visualization

This tool allows you to visualize ego vehicle trajectories from multiple scenarios in the NuPlan dataset.

## Features

- Samples multiple scenarios from the dataset
- Extracts ego vehicle trajectories from each scenario
- **Automatically normalizes all trajectories to start at (0, 0)** - Makes trajectory comparison easier
- Plots all trajectories on the same coordinate system
- Distinguishes trajectories with different colors

## Usage

### Basic Usage

Run the visualization script with the default configuration:

```bash
python visualize_ego_trajectories.py +training=visualize_trajectories
```

### Custom Parameters

You can customize the visualization using command-line arguments:

```bash
# Visualize 20 scenarios instead of 10
python visualize_ego_trajectories.py +training=visualize_trajectories \
    num_scenarios_to_visualize=20

# Limit each trajectory to first 50 points
python visualize_ego_trajectories.py +training=visualize_trajectories \
    num_points_per_trajectory=50

# Specify custom output path
python visualize_ego_trajectories.py +training=visualize_trajectories \
    trajectory_plot_path=my_trajectories.png

# Adjust trajectory time horizons (following nuplan_feature_builder pattern)
python visualize_ego_trajectories.py +training=visualize_trajectories \
    history_horizon=4.0 \
    future_horizon=10.0 \
    sample_interval=0.2

# Combine multiple parameters
python visualize_ego_trajectories.py +training=visualize_trajectories \
    num_scenarios_to_visualize=15 \
    history_horizon=3.0 \
    future_horizon=6.0 \
    trajectory_plot_path=output/trajectories.png
```

### Using Different Scenario Filters

You can use different scenario filters to visualize specific types of scenarios:

```bash
# Use a different scenario filter
python visualize_ego_trajectories.py +training=visualize_trajectories \
    scenario_filter=training_scenarios_1M
```

## Configuration

The main configuration file is located at:
```
config/training/visualize_trajectories.yaml
```

You can modify this file to change default parameters:

```yaml
# Number of scenarios to sample and visualize
num_scenarios_to_visualize: 10

# Optional: limit the number of points per trajectory to plot
# Set to null to plot all points
num_points_per_trajectory: null

# Output path for the trajectory plot
trajectory_plot_path: ego_trajectories.png

# Trajectory extraction parameters (following nuplan_feature_builder defaults)
# These control the time horizon for extracting past and future ego trajectories
history_horizon: 2.0      # Time horizon for past trajectory in seconds
future_horizon: 8.0       # Time horizon for future trajectory in seconds
sample_interval: 0.1      # Time interval between samples in seconds
```

### Trajectory Normalization

All trajectories are automatically normalized to start at the origin (0, 0). This makes it easier to compare trajectory shapes and patterns across different scenarios without being affected by their absolute positions. Each trajectory is shifted so that its first point (the starting position) is at the coordinate system origin.

### Trajectory Extraction Method

The trajectory extraction follows the implementation in `nuplan_feature_builder.py`:

- **Past trajectory**: Extracted using `scenario.get_ego_past_trajectory()` with configurable `history_horizon`
- **Current state**: Retrieved from `scenario.initial_ego_state`
- **Future trajectory**: Extracted using `scenario.get_ego_future_trajectory()` with configurable `future_horizon`

This approach ensures consistency with the training data preparation pipeline and provides better control over the trajectory time range.

## Output

The script generates a PNG image showing:
- Multiple ego trajectories plotted on the same coordinate system
- Different colors for each trajectory
- A legend showing scenario types and tokens
- Grid and axis labels in meters

## Requirements

- Access to NuPlan dataset
- Properly configured NuPlan environment
- Required packages: matplotlib, numpy, hydra, nuplan-devkit

## Troubleshooting

### "No scenarios found"
Make sure your dataset path is correctly configured in the Hydra config files.

### "Failed to extract trajectory"
Some scenarios might have issues. The script will skip failed scenarios and continue with others.

### Model configuration errors
The script may require a valid model configuration. Make sure the model config is properly set in your defaults.
