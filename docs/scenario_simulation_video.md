# Scenario Simulation and Video Visualization

This tool allows you to run nuplan simulation for a specific scenario and generate an ego-centric video visualization with agent IDs.

## Features

- **Uses local config files**: Like `eval.sh`, automatically loads all configuration from `config/local/`
- **Scenario-specific simulation**: Run simulation for a single scenario using its token
- **Inherits all eval settings**: Planner, checkpoint, paths, GPU settings, etc. from local config
- **Three simulation modes**:
  - `open_loop`: Agents follow logged trajectories
  - `closed_loop_nonreactive`: Agents use IDM model
  - `closed_loop_reactive`: Agents use IDM model with reactive metrics
- **Ego-centric video**: Video shows simulation from ego vehicle's perspective (ego always at center, world rotates)
- **Agent IDs**: All agents are labeled with sequential IDs (Agent 1, Agent 2, etc.)
- **ID mapping**: JSON file maps sequential IDs to original track tokens

## Installation

Ensure you have the required dependencies:

```bash
pip install opencv-python matplotlib numpy hydra-core omegaconf
```

## Usage

### Basic Usage

Like `eval.sh`, this script uses local config files to get all configuration (planner, checkpoint, paths, etc.):

```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token <scenario_token> \
  --simulation_mode closed_loop_nonreactive
```

### Command-Line Arguments

**Required Arguments:**
- `--config`: Local config name from `config/local/` (e.g., `251218_eval-plantf`)
- `--scenario_token`: Scenario token to simulate
- `--simulation_mode`: Simulation evaluation mode
  - Choices: `open_loop`, `closed_loop_nonreactive`, `closed_loop_reactive`

**Optional Arguments:**
- `--output_dir`: Output directory for results (default: auto-generated based on config)
- `--video_fps`: Video FPS (default: 10)
- `--video_resolution`: Video resolution as WIDTHxHEIGHT (default: `1920x1080`)
- `--map_radius`: Map display radius in meters (default: 80)
- `--skip_video`: Skip video generation (only run simulation)

### Examples

**Example 1: Using existing eval config for closed-loop nonreactive**
```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token 1a2b3c4d5e6f7g8h \
  --simulation_mode closed_loop_nonreactive
```

**Example 2: Open-loop simulation with custom video settings**
```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token 1a2b3c4d5e6f7g8h \
  --simulation_mode open_loop \
  --video_fps 20 \
  --video_resolution 2560x1440 \
  --map_radius 100
```

**Example 3: Closed-loop reactive with custom output directory**
```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token 1a2b3c4d5e6f7g8h \
  --simulation_mode closed_loop_reactive \
  --output_dir work_dirs/my_custom_output
```

**Example 4: Simulation only (no video)**
```bash
python scripts/simulate_and_visualize.py \
  --config 251218_eval-plantf \
  --scenario_token 1a2b3c4d5e6f7g8h \
  --simulation_mode closed_loop_nonreactive \
  --skip_video
```

## Output Files

After running the script, you will find the following files in the output directory:

### 1. Simulation Results
- `aggregator_metric/`: Aggregated metrics from simulation
- `simulation_logs/`: Detailed simulation logs (pickle files)
- Various metric files and logs

### 2. Video Files
- `simulation_<token>.mp4`: Ego-centric video visualization
- `simulation_<token>_agent_ids.json`: Agent ID mapping file

### 3. Agent ID Mapping Format

The `*_agent_ids.json` file contains a mapping from sequential IDs to track tokens:

```json
{
  "1": "a3f2b1c4e5d6f7a8b9c0d1e2f3a4b5c6",
  "2": "b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2",
  "3": "c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
  ...
}
```

- **Key**: Sequential ID shown in video (e.g., "1", "2", "3")
- **Value**: Full track token (hexadecimal string)

This allows you to:
1. Identify agents in the video by their sequential IDs
2. Look up the original track token for detailed analysis
3. Cross-reference with simulation logs and metrics

## Simulation Modes Explained

### Open Loop (`open_loop`)
- **Observation**: `box_observation` (TracksObservation)
- **Behavior**: Agents follow their logged trajectories from the dataset
- **Use case**: Evaluate planner against ground truth agent behavior
- **Metrics**: Open-loop specific metrics

### Closed Loop Nonreactive (`closed_loop_nonreactive`)
- **Observation**: `idm_agents_observation` (IDMAgents)
- **Behavior**: Agents use Intelligent Driver Model (IDM) for longitudinal control
- **Use case**: Evaluate planner with simulated agent behavior
- **Metrics**: Closed-loop nonreactive metrics

### Closed Loop Reactive (`closed_loop_reactive`)
- **Observation**: `idm_agents_observation` (IDMAgents)
- **Behavior**: Same as nonreactive (IDM model)
- **Use case**: Evaluate planner with reactive metrics
- **Metrics**: Closed-loop reactive metrics (more stringent)

**Note**: The difference between nonreactive and reactive modes is in the metrics used for evaluation, not the agent simulation model.

## Video Visualization Details

### Ego-Centric View
- Ego vehicle is always at the center of the frame (0, 0)
- Ego vehicle always points upward (heading = 0)
- The world rotates around the ego vehicle
- This makes it easy to focus on ego's perspective and interactions

### Visual Elements
- **Ego vehicle**: Green box at center
- **Other vehicles**: Gray boxes with sequential ID labels
- **Lanes**: Light blue filled polygons
- **Lane boundaries**: Dark gray solid lines
- **Centerlines**: Gray dashed lines
- **Agent IDs**: White text with black outline above each vehicle

### Video Specifications
- **Default resolution**: 1920x1080 (Full HD)
- **Default FPS**: 10 (matches typical simulation frequency)
- **Format**: MP4 (H.264 codec)
- **Duration**: Depends on scenario length (typically 10-20 seconds)

## Troubleshooting

### Issue: "No simulation log pickle files found"
**Solution**: The simulation may have failed. Check the simulation logs in the output directory for errors.

### Issue: "Failed to open video writer"
**Solution**: Ensure opencv-python is installed correctly. Try a different codec or resolution.

### Issue: Video is too large/small
**Solution**: Adjust `--video_resolution` and `--map_radius` parameters.

### Issue: Video playback is choppy
**Solution**: Try increasing `--video_fps` or reducing resolution.

### Issue: Agent IDs are not visible
**Solution**: Increase `--map_radius` to show more area, or check that agents are actually present in the scenario.

## Advanced Usage

### Using as a Python Module

You can also use the video generation module directly in your Python code:

```python
from src.utils.simulation_video import generate_video_from_log

# Generate video from existing simulation log
agent_id_map = generate_video_from_log(
    log_path='path/to/simulation_log.pkl',
    output_video_path='output_video.mp4',
    config={
        'fps': 10,
        'resolution': (1920, 1080),
        'map_radius': 80.0,
        'dpi': 100
    }
)

print(f"Generated video with {len(agent_id_map)} agents")
```

### Batch Processing Multiple Scenarios

You can create a shell script to process multiple scenarios:

```bash
#!/bin/bash

SCENARIOS=(
    "1a2b3c4d5e6f7g8h"
    "2b3c4d5e6f7g8h9i"
    "3c4d5e6f7g8h9i0j"
)

for token in "${SCENARIOS[@]}"; do
    echo "Processing scenario: $token"
    python scripts/simulate_and_visualize.py \
        --scenario_token "$token" \
        --simulation_mode closed_loop_nonreactive \
        --output_dir "work_dirs/batch_sim/$token"
done
```

## Performance Considerations

- **Video generation time**: Approximately 1-2 seconds per frame (depends on scene complexity)
- **Memory usage**: ~2-4 GB for typical scenarios
- **Disk space**: Videos are typically 10-50 MB depending on duration and resolution

## References

- [nuPlan Documentation](https://nuplan.org/)
- [nuPlan Simulation Guide](https://nuplan-devkit.readthedocs.io/en/latest/simulation.html)
- [IDM Model](https://en.wikipedia.org/wiki/Intelligent_driver_model)
