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

### 2. Heavy Trainer Evaluation Job (`trainer_evaluation_job`)
**Purpose**: Comprehensive model evaluation with full dataset
**Assets**:
- `prompt_asset` → `full_synthetic_data_asset` (complete dataset) → `preprocessed_data_asset` → `multi_trainer_scores_asset`
**Runtime**: ~30-60 minutes per cycle
**Frequency**: Automatically triggered after X iterative optimization runs (configurable)

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

The `trainer_evaluation_sensor` monitors `iterative_optimization_asset` materializations and automatically triggers trainer evaluation after a configurable number of complete optimization cycles.

**Configuration**:
```bash
# Set the threshold via environment variable (default: 5)
export OPTIMIZATION_ROUNDS_THRESHOLD=3
```

**Behavior**:
1. Monitor `iterative_optimization_asset` materializations
2. Count complete optimization cycles since last trainer evaluation
3. When count ≥ threshold: trigger `trainer_evaluation_job`
4. Reset counter and continue monitoring

## Key Assets

### New Assets
- **`iterative_optimization_asset`**: Complete iterative optimization with similarity feedback
- **`full_synthetic_data_asset`**: Generates complete synthetic dataset (no sample limit)

### Modified Components
- **`trainer_evaluation_sensor`**: Now monitors iterative optimization runs
- **`optimization_job`**: Updated to include both legacy and new approaches

### Unchanged Assets
- **`full_synthetic_data_asset`**: Complete dataset for trainer evaluation
- **`preprocessed_data_asset`**: Data preprocessing for trainers
- **`multi_trainer_scores_asset`**: Trainer evaluation results

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

### Manual Trainer Evaluation
```bash
# Force a trainer evaluation
dagster job execute trainer_evaluation_job
```

### Legacy Workflows (Deprecated)
```bash
# Old linear optimization (not recommended)
dagster job execute optimization_job
```

### Configuring Automation
```bash
# Set threshold for automatic trainer evaluation
export OPTIMIZATION_ROUNDS_THRESHOLD=3  # Trigger after 3 optimization cycles
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
│   ├── iterative_optimization.py      # Complete iterative optimization (NEW)
│   ├── full_synthetic_data.py         # Complete synthetic data (EXISTING)
│   ├── preprocessed_data.py           # Data preprocessing (EXISTING)
│   └── multi_trainer_scores.py        # Trainer evaluation (EXISTING)
├── jobs/
│   ├── optimization_job.py            # Updated with iterative job (MODIFIED)
│   ├── trainer_evaluation_job.py      # Heavy evaluation job (EXISTING)
│   └── iterative_optimization_job.py  # Multi-op alternative (NEW)
├── sensors/
│   └── trainer_evaluation_sensor.py   # Updated to monitor iterative optimization (MODIFIED)
└── helpers/
    └── full_data_helper.py            # Complete data generation (EXISTING)
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
