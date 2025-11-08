# Intent-Conditioned Trajectory Generation - Integration Guide

## Overview

This document describes the integration of **intent-conditioned trajectory generation** into the PlanTF model. The system classifies ego vehicle intent from expert trajectories during training and uses predicted intents to condition trajectory generation.

## Architecture

### Intent-Conditioned Generation Flow

```
Input Features
    ↓
Encoder
    ↓
    ├──────────────────────────┐
    ↓                          ↓
Intent Prediction          Trajectory Decoding
(M=6 hypotheses)          (conditioned on intent)
    ↓                          ↓
[B,M,5] lateral           Intent Embeddings
[B,M,4] longitudinal           ↓
    ↓                     Conditioned Features
    └──────────────────────────┘
              ↓
    [B, M=6, T, 4] trajectories
```

### Intent Categories

**Lateral Intent** (5 classes):
- `turn_left` (0): Sharp left turn (heading change > 15° + high curvature)
- `turn_right` (1): Sharp right turn (heading change < -15° + high curvature)
- `shift_left` (2): Gentle left lane change
- `shift_right` (3): Gentle right lane change
- `stay_in_lane` (4): Maintain current lane

**Longitudinal Intent** (4 classes):
- `accelerate` (0): Significant speed increase
- `maintain_speed` (1): Constant velocity
- `decelerate` (2): Significant speed decrease
- `stop` (3): Coming to a halt

## Implementation Details

### 1. Intent Classification Utilities

**File**: `src/utils/intent_classification.py`

Core functions for extracting geometric features and classifying intent:

```python
from src.utils.intent_classification import (
    classify_intent_from_states,
    IntentClassificationConfig
)

# Example usage
config = IntentClassificationConfig(
    turn_angle_threshold=15.0,    # degrees
    shift_angle_threshold=5.0,    # degrees
    turn_curvature_threshold=0.05, # 1/meters
    accel_threshold=1.0,          # m/s²
)

lateral, longitudinal = classify_intent_from_states(
    ego_states=ego_states,
    current_idx=0,
    time_horizon=2.0,  # seconds
    config=config
)
```

### 2. Intent Target Builder

**File**: `src/target_builders/intent_target_builder.py`

Computes intent labels from expert trajectories during feature caching:

```python
class IntentTargetBuilder(AbstractTargetBuilder):
    def __init__(
        self,
        time_horizon: float = 2.0,
        sample_interval: float = 0.1,
        config: IntentClassificationConfig = None
    )

    def get_targets(self, scenario: AbstractScenario) -> IntentLabels:
        # Extracts ego trajectory and classifies intent
        # Returns: IntentLabels(lateral_intent=idx, longitudinal_intent=idx)
```

### 3. Model Architecture Updates

**File**: `src/models/planTF/planning_model.py`

New model parameters:
```python
intent_enabled: bool = False           # Enable/disable intent conditioning
intent_time_horizon: float = 2.0       # Time window for intent (2s/4s/8s)
intent_embed_dim: int = 64             # Intent embedding dimension
lateral_classes: int = 5               # Number of lateral classes
longitudinal_classes: int = 4          # Number of longitudinal classes
```

New components:
- `lateral_intent_head`: Predicts M lateral intent hypotheses
- `longitudinal_intent_head`: Predicts M longitudinal intent hypotheses
- `lateral_intent_embed`: Embedding layer for lateral intents
- `longitudinal_intent_embed`: Embedding layer for longitudinal intents
- `intent_fusion`: Fuses lateral + longitudinal embeddings

Forward pass changes:
1. Predict M intent hypotheses (one per trajectory mode)
2. Convert to embeddings using Gumbel-Softmax (training) or argmax (inference)
3. Add intent features to encoded features
4. Decode trajectories conditioned on intent

### 4. Training Loss Updates

**File**: `src/models/planTF/lightning_trainer.py`

Intent classification loss:
```python
# All M modes learn the same expert intent
lateral_loss = CrossEntropy(predicted_lateral, expert_lateral)
longitudinal_loss = CrossEntropy(predicted_longitudinal, expert_longitudinal)
intent_loss = lateral_loss + longitudinal_loss

total_loss = trajectory_loss + intent_loss_weight * intent_loss
```

## Configuration

### Enable Intent-Conditioned Generation

**File**: `config/model/planTF.yaml`

```yaml
# Set intent_enabled to true
intent_enabled: true
intent_time_horizon: 2.0  # Options: 2.0, 4.0, 8.0
intent_embed_dim: 64
lateral_classes: 5
longitudinal_classes: 4
```

**File**: `config/custom_trainer/planTF.yaml`

```yaml
# Adjust loss weight
intent_loss_weight: 1.0  # Recommended range: 0.5 - 2.0
```

## Usage

### 1. Feature Caching with Intent Labels

When intent is enabled, intent labels are automatically computed during caching:

```bash
python run_training.py \
  py_func=cache \
  +training=train_planTF \
  scenario_builder=nuplan \
  cache.cache_path=/path/to/cache \
  scenario_filter=training_scenarios_1M \
  model.intent_enabled=true \
  model.intent_time_horizon=2.0
```

### 2. Training with Intent Conditioning

```bash
python run_training.py \
  py_func=train \
  +training=train_planTF \
  cache.cache_path=/path/to/cache \
  cache.use_cache_without_dataset=true \
  model.intent_enabled=true \
  model.intent_time_horizon=2.0 \
  intent_loss_weight=1.0
```

### 3. Evaluation

During inference, the model:
1. Predicts M intent hypotheses
2. Generates M intent-conditioned trajectories
3. Selects best trajectory based on probability

Intent predictions are available in model output:
```python
output = model(features)
# output["intent"]["lateral"]: [B, M, 5]
# output["intent"]["longitudinal"]: [B, M, 4]
```

## Hyperparameter Tuning

### Time Horizon
- **2.0s** (default): Short-term intent, most stable
- **4.0s**: Medium-term, captures lane changes
- **8.0s**: Long-term, captures complete maneuvers

### Intent Loss Weight
- **0.5**: Prioritize trajectory accuracy
- **1.0** (default): Balanced multi-task learning
- **2.0**: Emphasize intent learning

### Intent Embed Dimension
- **32**: Lightweight, faster training
- **64** (default): Good balance
- **128**: More expressive, slower

## Monitoring Training

Key metrics to monitor:

```
objectives/train_intent_loss          # Overall intent loss
objectives/train_lateral_intent_loss  # Lateral classification accuracy
objectives/train_longitudinal_intent_loss  # Longitudinal classification accuracy
objectives/train_reg_loss             # Trajectory regression loss
```

Expected behavior:
- Intent losses should decrease steadily
- Trajectory losses should remain comparable to baseline
- Classification accuracy should reach >80% within a few epochs

## File Structure

```
src/
├── utils/
│   └── intent_classification.py      # Intent classification utilities
├── features/
│   └── intent_labels.py               # IntentLabels feature class
├── target_builders/
│   ├── __init__.py
│   └── intent_target_builder.py       # Intent target builder
└── models/planTF/
    ├── planning_model.py              # Updated model with intent conditioning
    └── lightning_trainer.py           # Updated training loop with intent loss

config/
├── model/
│   └── planTF.yaml                    # Model config with intent parameters
└── custom_trainer/
    └── planTF.yaml                    # Trainer config with intent_loss_weight
```

## Troubleshooting

### Issue: Intent loss not decreasing

**Solution**:
- Check if intent labels are being loaded correctly
- Verify target_builders includes IntentTargetBuilder
- Increase intent_loss_weight

### Issue: Trajectory performance degraded

**Solution**:
- Reduce intent_loss_weight to 0.5
- Try shorter time_horizon (2.0s instead of 8.0s)
- Check if intent labels distribution is balanced

### Issue: Out of memory during training

**Solution**:
- Reduce intent_embed_dim to 32
- Disable intent temporarily during debugging
- Reduce batch size

## Future Extensions

### 1. Per-mode Intent Targets (Optional)

Currently, all modes learn the same expert intent. To enable diverse intent hypotheses:

```python
# Modify _compute_objectives in lightning_trainer.py
# Use soft labels based on trajectory similarity
weights = compute_mode_weights(trajectory, ego_target)
lateral_loss = weighted_cross_entropy(lateral_logits, lateral_target, weights)
```

### 2. Agent Intent Prediction

Extend to predict intents for surrounding agents:

```python
# In IntentTargetBuilder
def get_targets(self, scenario):
    # Classify ego intent (existing)
    # Classify agent intents (new)
    for agent in tracked_objects:
        agent_intent = classify_agent_intent(agent.trajectory)
```

### 3. Hierarchical Intent

Add high-level maneuver categories:

```python
# Maneuver types: parking, lane_change, turning, cruising
maneuver_intent_head = nn.Linear(dim, num_modes * maneuver_classes)
```

## References

- Intent classification based on: `visualize_short_term_intent.py`
- Geometric features: curvature, heading change, velocity change
- Training framework: NuPlan devkit

## Contact

For questions or issues, please refer to:
- [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) for intent classification details
- PlanTF model documentation
- NuPlan devkit documentation
