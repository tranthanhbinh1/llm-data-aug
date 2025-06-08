# Prompt Optimization System

A clean, modular implementation of genetic algorithm-based prompt optimization, inspired by the PromptBreeder paper and designed for integration with data augmentation pipelines.

## Overview

This system optimizes prompts using a genetic algorithm approach:

1. **Population Initialization**: Generate diverse prompt variants
2. **Fitness Evaluation**: Score prompts using LLM evaluation or custom evaluators
3. **Selection**: Choose best-performing prompts as parents
4. **Crossover**: Combine successful prompts to create offspring
5. **Evolution**: Iterate until convergence or max generations

## Key Features

- **Modular Design**: Clean separation of concerns with dedicated classes
- **Custom Evaluators**: Plug in your own evaluation functions
- **Async/Concurrent**: Efficient parallel evaluation of candidates
- **Progress Tracking**: Real-time optimization progress
- **Comprehensive Results**: Detailed optimization metadata and history
- **Type Safety**: Full type hints throughout

## Quick Start

```python
import asyncio
from src.prompt_optimization import PromptOptimizer, OptimizationConfig

async def main():
    # Configure optimization
    config = OptimizationConfig(
        population_size=5,
        num_iterations=3,
        num_elites=2,
        threshold=0.9
    )
    
    # Create optimizer
    optimizer = PromptOptimizer(config=config)
    
    # Run optimization
    result = await optimizer.optimize(
        initial_prompt="Generate product reviews",
        improvement_request="Make reviews more diverse and realistic"
    )
    
    print(f"Best prompt: {result.best_prompt}")
    print(f"Score: {result.best_score:.4f}")

asyncio.run(main())
```

## Architecture

### Core Components

1. **`PromptOptimizer`**: Main orchestrator class
2. **`GeneticOperations`**: Handles genetic algorithm operations
3. **`OptimizationConfig`**: Configuration parameters
4. **Data Models**: `PromptCandidate`, `OptimizationResult`

### Data Flow

```
Initial Prompt → Population Init → Evaluation → Selection → Crossover → New Generation
                      ↑                                                        ↓
                      ←←←←←←←←←←←←← Repeat until convergence ←←←←←←←←←←←←←←←←←←
```

## Custom Evaluators

The system supports custom evaluation functions for domain-specific scoring:

```python
async def my_evaluator(
    candidate: PromptCandidate,
    initial_prompt: str,
    improvement_request: str
) -> PromptCandidate:
    """
    Custom evaluator that integrates with your downstream tasks.
    
    Example workflow:
    1. Use candidate.prompt to generate synthetic data
    2. Evaluate data quality (similarity, model performance, etc.)
    3. Return fitness score (0.0 to 1.0)
    """
    if candidate.fitness is not None:
        return candidate
    
    # Your evaluation logic here
    # e.g., generate data, train model, measure performance
    synthetic_data = generate_data_with_prompt(candidate.prompt)
    quality_score = evaluate_data_quality(synthetic_data)
    
    candidate.fitness = quality_score
    candidate.reflection = f"Quality score: {quality_score:.4f}"
    
    return candidate

# Use custom evaluator
result = await optimizer.optimize(
    initial_prompt="...",
    improvement_request="...",
    custom_evaluator=my_evaluator
)
```

## Integration with Data Augmentation Pipeline

For your LLM data augmentation use case:

```python
from src.prompt_optimization import PromptOptimizer
from src.evaluation.similarity_evaluator import SimilarityEvaluator
from src.trainers import CNNBertHybrid, LSTM, PhoBert, SVM

async def data_aug_evaluator(candidate, initial_prompt, improvement_request):
    """Evaluator that measures synthetic data quality."""
    
    if candidate.fitness is not None:
        return candidate
    
    # Generate synthetic data using the candidate prompt
    synthetic_data = await generate_synthetic_data(candidate.prompt)
    
    # Quick evaluation: similarity score (lightweight)
    similarity_evaluator = SimilarityEvaluator()
    similarity_score = similarity_evaluator.evaluate(synthetic_data, original_data)
    
    # For promising candidates, run full model evaluation (heavyweight)
    if similarity_score > 0.7:  # threshold for full evaluation
        model_scores = []
        for model_class in [CNNBertHybrid, LSTM, PhoBert, SVM]:
            model = model_class()
            score = await model.train_and_evaluate(synthetic_data)
            model_scores.append(score)
        
        # Combine similarity and model performance
        model_avg = sum(model_scores) / len(model_scores)
        candidate.fitness = 0.3 * similarity_score + 0.7 * model_avg
    else:
        # Use only similarity for low-quality candidates
        candidate.fitness = similarity_score * 0.5  # penalty for not reaching full eval
    
    return candidate

# Optimize prompts for data augmentation
optimizer = PromptOptimizer()
result = await optimizer.optimize(
    initial_prompt="Generate Vietnamese sentiment analysis examples",
    improvement_request="Create more diverse examples with balanced sentiment distribution",
    custom_evaluator=data_aug_evaluator
)
```

## Configuration Options

```python
config = OptimizationConfig(
    population_size=5,        # Number of candidates per generation
    num_iterations=5,         # Maximum generations
    num_elites=2,            # Top candidates to preserve
    threshold=1.0,           # Fitness threshold for early stopping
    tournament_size=3,       # Tournament selection size
    num_evaluation_samples=3, # Self-consistency samples for LLM evaluation
    model="gemini-2.0-flash", # LLM model for operations
    temperature=1.0,         # Sampling temperature
    max_retries=3           # Retry attempts for failed API calls
)
```

## Progress Tracking

Monitor optimization progress in real-time:

```python
async for message, iteration, progress, best_score in optimizer.optimize_with_progress(
    initial_prompt="...",
    improvement_request="..."
):
    print(f"[{progress*100:.1f}%] Gen {iteration}: {message}")
    if best_score:
        print(f"  Best score: {best_score:.4f}")
```

## Error Handling

The system includes robust error handling:

- **API Failures**: Automatic retries with exponential backoff
- **Evaluation Errors**: Graceful degradation with default scores
- **Population Diversity**: Fallback mechanisms to maintain genetic diversity
- **Resource Management**: Proper cleanup of async resources

## Performance Considerations

- **Concurrent Evaluation**: All candidates evaluated in parallel
- **Early Stopping**: Configurable convergence thresholds
- **Caching**: Elites from previous generations skip re-evaluation

## Comparison with Original Promptimal

| Feature | Original Promptimal | This Implementation |
|---------|-------------------|-------------------|
| **UI Dependencies** | urwid, pyperclip | None (headless) |
| **Integration** | Standalone app | Library/module |
| **Async Support** | Limited | Full async/await |
| **Custom Evaluators** | Basic subprocess | Rich async interface |
| **Progress Tracking** | UI-based | Programmatic callbacks |
| **Error Handling** | Basic | Comprehensive |
| **Type Safety** | Partial | Complete type hints |
| **Modularity** | Monolithic | Clean separation |

## Examples

See `example.py` for complete usage examples:

- Basic optimization with LLM evaluation
- Custom evaluator integration
- Progress tracking
- Error handling patterns

## Dependencies

- `google-genai`: Google AI API client
- `instructor`: Structured output from LLMs
- `pydantic`: Data validation and serialization
- `loguru`: Logging
- `asyncio`: Asynchronous programming

## Future Enhancements

- **Multi-objective Optimization**: Optimize for multiple criteria simultaneously
- **Adaptive Parameters**: Dynamic adjustment of genetic algorithm parameters
- **Population Diversity Metrics**: Ensure genetic diversity maintenance
- **Distributed Evaluation**: Scale evaluation across multiple workers
- **Prompt Templates**: Support for parameterized prompt templates
