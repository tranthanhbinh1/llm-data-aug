# Iterative Loop Architecture

This document describes the complete iterative loop system that combines prompt optimization with trainer evaluation in a continuous cycle.

## Overview

The iterative loop system creates a true feedback loop between lightweight optimization and heavy trainer evaluation:

```
1. Iterative Optimization (fast) → 
2. Similarity Evaluation (built-in) → 
3. [After X cycles] → Trainer Evaluation (heavy) → 
4. [Feedback Loop] → Next Optimization Cycle → 
5. Repeat until convergence or max cycles
```

## Architecture Components

### Jobs

#### 1. `iterative_optimization_job`
- **Purpose**: Fast optimization with similarity feedback
- **Assets**: `iterative_optimization_asset`
- **Duration**: 5-15 minutes
- **Output**: Optimized prompt with similarity metrics

#### 2. `trainer_evaluation_job`
- **Purpose**: Heavy evaluation with multiple ML trainers
- **Assets**: `prompt_asset`, `full_synthetic_data_asset`, `preprocessed_data_asset`, `multi_trainer_scores_asset`
- **Duration**: 20-60 minutes
- **Output**: Comprehensive trainer performance metrics

#### 3. `complete_cycle_job` (NEW)
- **Purpose**: Complete cycle: iterative optimization → trainer evaluation
- **Assets**: 
  - `iterative_optimization_asset` (optimization with similarity feedback)
  - `full_synthetic_data_asset` (uses optimized prompt from iteration)
  - `preprocessed_data_asset` (preprocessing for trainers)
  - `multi_trainer_scores_asset` (trainer evaluation)
- **Duration**: 25-75 minutes
- **Use Case**: Manual runs or simplified workflow

### Sensors

#### 1. `trainer_evaluation_sensor`
- **Monitors**: `iterative_optimization_asset` materializations
- **Triggers**: `trainer_evaluation_job` after X optimization cycles
- **Configuration**: `OPTIMIZATION_ROUNDS_THRESHOLD` (default: 5)
- **Status**: Running by default

#### 2. `optimization_cycle_sensor` (NEW)
- **Monitors**: `trainer_evaluation_job` completions
- **Triggers**: `iterative_optimization_job` after trainer evaluation completes
- **Configuration**: `ENABLE_CONTINUOUS_CYCLES`, `MAX_TOTAL_CYCLES`
- **Status**: Stopped by default (must be manually enabled)

### Assets

#### Core Optimization Assets
- **`iterative_optimization_asset`**: Complete iterative optimization with similarity feedback
- **`prompt_asset`**: Optimized prompt (legacy compatibility)

#### Trainer Evaluation Assets
- **`full_synthetic_data_asset`**: Complete synthetic dataset using iteratively optimized prompt
- **`preprocessed_data_asset`**: Preprocessed data for trainers
- **`multi_trainer_scores_asset`**: Results from multiple ML trainers

## Usage Patterns

### Pattern 1: Sensor-Driven Continuous Loop

**Enable continuous cycling:**
```bash
# Configure environment
export OPTIMIZATION_ROUNDS_THRESHOLD=3    # Trigger trainer eval after 3 optimization cycles
export ENABLE_CONTINUOUS_CYCLES=true      # Enable continuous cycling
export MAX_TOTAL_CYCLES=20                # Stop after 20 total cycles

# Start the optimization cycle sensor
dagster sensor start optimization_cycle_sensor

# Trigger initial optimization to start the loop
dagster job execute iterative_optimization_job
```

**What happens:**
1. Initial optimization runs (similarity feedback, genetic algorithm)
2. After 3 optimization cycles → `trainer_evaluation_sensor` triggers trainer evaluation
3. After trainer evaluation completes → `optimization_cycle_sensor` triggers next optimization
4. Loop continues for up to 20 total cycles
5. Each cycle uses results from previous trainer evaluation as feedback

### Pattern 2: Manual Iterative Control

**Run optimization cycles manually:**
```bash
# Run single optimization cycle
dagster job execute iterative_optimization_job

# After several optimization cycles, run trainer evaluation
dagster job execute trainer_evaluation_job

# Continue with next optimization cycle
dagster job execute iterative_optimization_job
```

### Pattern 3: Complete Cycle Job

**Run optimization + trainer evaluation together:**
```bash
# Single job that includes both optimization and trainer evaluation
dagster job execute complete_cycle_job
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPTIMIZATION_ROUNDS_THRESHOLD` | 5 | Optimization cycles before triggering trainer evaluation |
| `ENABLE_CONTINUOUS_CYCLES` | false | Enable automatic cycling after trainer evaluation |
| `MAX_TOTAL_CYCLES` | 50 | Maximum cycles before automatic stop |

### Job Configuration

#### Iterative Optimization Parameters
```json
{
  "ops": {
    "iterative_optimization_asset": {
      "config": {
        "max_optimization_rounds": 5,
        "similarity_threshold": 0.75,
        "population_size": 4,
        "sentiment": "neutral"
      }
    }
  }
}
```

#### Trainer Evaluation Parameters
```json
{
  "ops": {
    "full_synthetic_data_asset": {
      "config": {
        "sentiment": "neutral",
        "num_samples": null
      }
    }
  }
}
```

## Monitoring and Control

### Sensor Management

```bash
# Check sensor status
dagster sensor list

# Start/stop sensors
dagster sensor start trainer_evaluation_sensor
dagster sensor start optimization_cycle_sensor
dagster sensor stop optimization_cycle_sensor

# View sensor logs
dagster sensor logs trainer_evaluation_sensor
dagster sensor logs optimization_cycle_sensor
```

### Progress Tracking

#### In Dagster UI:
- **Asset Lineage**: View complete optimization → evaluation flow
- **Run History**: Track cycles and performance over time
- **Sensor Logs**: Monitor automatic triggering decisions
- **Asset Metadata**: View similarity scores, trainer metrics, cycle counts

#### Key Metrics to Monitor:
- **Similarity Score Progression**: Track improvement over optimization rounds
- **Trainer Performance**: Monitor accuracy/F1 scores from different models
- **Cycle Efficiency**: Time per optimization vs trainer evaluation
- **Convergence**: Whether similarity thresholds are being reached

## Advanced Configurations

### Custom Cycle Behavior

**Modify `optimization_cycle_sensor` behavior:**
```python
# In optimization_cycle_sensor.py
DEFAULT_MAX_TOTAL_CYCLES = 100  # Increase max cycles
DEFAULT_MINIMUM_INTERVAL_SECONDS = 300  # Wait 5 minutes between checks
```

**Modify trainer evaluation threshold:**
```python
# In trainer_evaluation_sensor.py
DEFAULT_OPTIMIZATION_ROUNDS_THRESHOLD = 3  # More frequent trainer evaluations
```

### Integration with External Systems

**Trigger from external orchestrator:**
```python
import requests

# Trigger optimization cycle via Dagster GraphQL API
mutation = """
mutation {
  launchRun(
    executionParams: {
      selector: { jobName: "iterative_optimization_job" }
    }
  ) {
    __typename
    ... on LaunchRunSuccess {
      run {
        runId
      }
    }
  }
}
"""
```

## Best Practices

### 1. Resource Management
- **CPU/Memory**: Trainer evaluation is memory-intensive, ensure adequate resources
- **Concurrency**: Sensors prevent overlapping jobs automatically
- **Storage**: Monitor disk usage for synthetic data and model checkpoints

### 2. Performance Optimization
- **Caching**: All assets use intelligent caching to avoid recomputation
- **Parallel Processing**: Trainer evaluation runs multiple models in sequence
- **Batch Sizes**: Adjust synthetic data generation batch sizes based on resources

### 3. Monitoring and Alerting
- **Set up alerts** for sensor failures or job timeouts
- **Monitor similarity score trends** to detect optimization plateaus
- **Track trainer performance** to identify when to stop cycles

### 4. Experimentation
- **Use different optimization parameters** for different sentiments
- **Experiment with threshold values** to balance speed vs quality
- **Try different trainer combinations** based on performance requirements

## Troubleshooting

### Common Issues

#### Sensors Not Triggering
```bash
# Check sensor status
dagster sensor list

# Check sensor logs for errors
dagster sensor logs optimization_cycle_sensor

# Ensure environment variables are set
echo $ENABLE_CONTINUOUS_CYCLES
echo $OPTIMIZATION_ROUNDS_THRESHOLD
```

#### Jobs Failing
```bash
# Check recent run failures
dagster run list --status FAILURE

# View specific run logs
dagster run logs <run_id>

# Check asset materialization status
dagster asset list --materialized
```

#### Performance Issues
- **Memory**: Increase Docker/VM memory allocation
- **Disk Space**: Clean up old synthetic data files
- **Network**: Ensure stable API connections for LLM calls

### Recovery Procedures

#### Reset Sensor State
```bash
# Stop sensors
dagster sensor stop optimization_cycle_sensor
dagster sensor stop trainer_evaluation_sensor

# Reset sensor cursors (if needed)
# This requires admin access to Dagster instance storage

# Restart sensors
dagster sensor start trainer_evaluation_sensor
dagster sensor start optimization_cycle_sensor
```

#### Manual Intervention
```bash
# Skip to next phase manually
dagster job execute trainer_evaluation_job  # Force trainer evaluation
dagster job execute iterative_optimization_job  # Force next optimization
```

## Migration Guide

### From Legacy Pipeline
1. **Test new jobs** alongside existing ones
2. **Configure thresholds** based on your quality requirements
3. **Enable sensors gradually** (start with trainer_evaluation_sensor only)
4. **Monitor performance** and adjust parameters
5. **Phase out legacy jobs** once confident in new system

### Rollback Plan
- All legacy jobs (`optimization_job`, `full_pipeline_job`) remain available
- Disable sensors to return to manual control
- Use `complete_cycle_job` for simplified manual iteration 
