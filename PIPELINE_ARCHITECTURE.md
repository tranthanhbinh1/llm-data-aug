# LLM Data Augmentation Pipeline Architecture

## Overview

The pipeline has been restructured to support true iterative prompt optimization with real-time similarity evaluation feedback, separating fast optimization cycles from heavy trainer evaluations.

## Pipeline Structure

### 1. Iterative Optimization Job (`iterative_optimization_job`) - **RECOMMENDED**
**Purpose**: Complete iterative prompt optimization with similarity feedback loop
**Assets**: 
- `iterative_optimization_asset` (internally handles genetic algorithm + similarity evaluation)
**Runtime**: ~5-15 minutes per complete optimization cycle (multiple rounds)
**Frequency**: Run when you need optimized prompts

**Key Features**:
- **True iterative optimization**: Genetic algorithm with real similarity evaluation feedback
- **Configurable stopping criteria**: Similarity threshold and max rounds
- **Built-in feedback loop**: Each genetic algorithm candidate is evaluated using actual similarity scores
- **Progress tracking**: Detailed logs and metrics for each optimization round

### 2. Complete Cycle Job (`complete_cycle_job`)
**Purpose**: Comprehensive cycle including optimization and model evaluation with full dataset
**Assets**:
- `iterative_optimization_asset` → `full_synthetic_data_asset` (complete dataset) → `preprocessed_data_asset` → `multi_trainer_scores_asset`
**Runtime**: ~45-90 minutes per cycle (optimization + evaluation)
**Frequency**: Automatically triggered by optimization_cycle_sensor for continuous improvement

### 3. Legacy Jobs (For Backward Compatibility)
- **`optimization_job`**: Linear prompt optimization (deprecated)
- **`full_pipeline_job`**: Original full pipeline (deprecated)

## Iterative Optimization Process

The `iterative_optimization_asset` implements a sophisticated optimization loop:

1. **Initialize**: Start with initial prompt and configuration
2. **Genetic Algorithm Round**: 
   - Generate prompt candidates using genetic operations
   - **Custom Evaluator**: Each candidate is evaluated by:
     - Generating synthetic data using the candidate prompt
     - Running similarity evaluation on the generated data
     - Assigning fitness score based on similarity score
3. **Selection & Evolution**: Best candidates are selected for next generation
4. **Round Completion**: Check convergence criteria (similarity threshold or max rounds)
5. **Iteration**: If not converged, use best prompt as starting point for next round
6. **Finalization**: Return optimized prompt with metrics

### Configuration Options

```python
class IterativeOptimizationConfig:
    initial_prompt: str = "..."  # Starting prompt
    improvement_request: str = "..."  # What to optimize for
    sentiment: Sentiment = Sentiment.NEUTRAL
    max_optimization_rounds: int = 10  # Max iterative rounds
    similarity_threshold: float = 0.8  # Stop when reached
    population_size: int = 3  # Genetic algorithm population
    num_iterations: int = 2  # GA iterations per round
    num_elites: int = 1  # Elite candidates to preserve
```

## Automated Workflow

The `optimization_cycle_sensor` monitors `complete_cycle_job` completions and automatically triggers the next optimization cycle for continuous improvement.

**Configuration**:
```bash
# Enable continuous cycling (default: false)
export ENABLE_CONTINUOUS_CYCLES=true
# Set maximum cycles before stopping (default: 50)
export MAX_TOTAL_CYCLES=20
```

**Behavior**:
1. Monitor `complete_cycle_job` completions
2. Track total cycles completed
3. When cycle completes: trigger next `iterative_optimization_job`
4. Continue until max cycles reached or manually stopped

## Key Assets

### Current Architecture Assets
- **`iterative_optimization_asset`**: Complete iterative optimization with similarity feedback
- **`full_synthetic_data_asset`**: Generates complete synthetic dataset (no sample limit)
- **`preprocessed_data_asset`**: Data preprocessing for trainers
- **`multi_trainer_scores_asset`**: Trainer evaluation results

### Current Jobs
- **`iterative_optimization_job`**: Fast optimization with similarity feedback
- **`complete_cycle_job`**: Full optimization + trainer evaluation cycle

### Current Sensors
- **`optimization_cycle_sensor`**: Triggers continuous optimization cycles

## Benefits

1. **True Iterative Optimization**: Genetic algorithm with real similarity feedback (not just LLM evaluation)
2. **Quality Feedback Loop**: Each optimization round uses actual data quality metrics
3. **Configurable Convergence**: Stop based on similarity threshold or max rounds
4. **Automated Workflow**: Heavy trainer evaluation triggered automatically
5. **Comprehensive Tracking**: Detailed logs and metrics for each optimization step
6. **Resource Efficiency**: Heavy computations only when needed
7. **Backward Compatibility**: Legacy workflows preserved

## Usage

### Primary Workflow (Recommended)
```bash
# Run iterative optimization with similarity feedback
dagster job execute iterative_optimization_job

# Configure optimization parameters via job config:
{
  "ops": {
    "iterative_optimization_asset": {
      "config": {
        "max_optimization_rounds": 5,
        "similarity_threshold": 0.75,
        "population_size": 4
      }
    }
  }
}
```

### Manual Complete Cycle
```bash
# Force a complete cycle (optimization + evaluation)
dagster job execute complete_cycle_job
```

### Legacy Workflows (Deprecated)
```bash
# Old linear optimization (not recommended)
dagster job execute optimization_job
```

### Configuring Automation
```bash
# Enable continuous cycling
export ENABLE_CONTINUOUS_CYCLES=true
# Set maximum total cycles
export MAX_TOTAL_CYCLES=20
# Start the sensor
dagster sensor start optimization_cycle_sensor
```

## Monitoring

- **Iterative Optimization**: View detailed optimization rounds, similarity scores, and convergence in asset metadata
- **Sensor Status**: Check automation status in Dagster UI under "Sensors"
- **Asset Lineage**: Full tracking of optimization → evaluation flow
- **Performance Metrics**: Execution time, rounds to convergence, similarity improvements

## File Structure

```
src/worker/
├── assets/
│   ├── iterative_optimization.py      # Complete iterative optimization
│   ├── iterative_optimization_v2.py   # Graph-backed optimization alternative
│   ├── full_synthetic_data.py         # Complete synthetic data
│   ├── preprocessed_data.py           # Data preprocessing
│   └── multi_trainer_scores.py        # Trainer evaluation
├── jobs/
│   ├── optimization_job.py            # Fast optimization job
│   └── iterative_loop_job.py          # Complete cycle job
├── sensors/
│   └── optimization_cycle_sensor.py   # Continuous optimization cycling
└── helpers/
    ├── full_data_helper.py            # Complete data generation
    └── score_helper.py                # Similarity evaluation
```

## Migration Guide

### For New Projects
- Use `iterative_optimization_job` for prompt optimization
- Configure similarity thresholds and optimization rounds as needed
- Let the sensor handle trainer evaluations automatically

### For Existing Projects
- **Immediate**: Continue using existing jobs (they still work)
- **Recommended**: Migrate to `iterative_optimization_job` for better optimization quality
- **Gradual**: Test the new approach alongside existing workflows

### Performance Expectations
- **Iterative Optimization**: 5-15 minutes for complete optimization (5-10 rounds)
- **Similarity Evaluation**: Real data quality feedback vs LLM-based evaluation
- **Convergence**: Typically 3-7 rounds to reach good similarity scores
- **Quality**: Significantly improved prompt quality due to real feedback loop
