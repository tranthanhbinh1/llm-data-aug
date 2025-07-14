# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comparative study of Resampling and LLM-based data augmentation for sentiment analysis tasks. The project implements a genetic algorithm-based prompt optimization system integrated with Dagster pipelines for systematic data augmentation research.

## Development Commands

### Running the Application
```bash
# Start Dagster development server
./start_dagster.sh
# This starts the webserver on port 3004 with the pipeline definitions
```

### Main Entry Points
```bash
# Run the main application (prompt optimization with evaluation)
python src/main.py

# Data preprocessing script
python test.py
```

## Architecture

### Core Components

1. **Prompt Optimization System** (`src/prompt_optimization/`)
   - Genetic algorithm-based prompt optimization inspired by PromptBreeder
   - Clean, modular design with async/concurrent evaluation
   - Custom evaluator support for domain-specific scoring
   - Components: `PromptOptimizer`, `GeneticOperations`, `OptimizationConfig`

2. **Data Augmentation Pipeline** (`src/synthesizer/`)
   - LLM-based text generation using Google GenAI
   - `AugGptRunner` for synthetic data generation
   - Integration with prompt optimization system

3. **ML Models and Training** (`src/trainers/`)
   - Multiple sentiment analysis models: CNN-BERT Hybrid, LSTM, PhoBERT, SVM
   - Unified training interface for comparative studies
   - Vietnamese language support with PhoBERT

4. **Evaluation Framework** (`src/evaluation/`, `src/worker/evaluators/`)
   - Two-tier evaluation: Lightweight (similarity) and Heavyweight (full model training)
   - `LightweightEvaluator`: Fast similarity-based scoring
   - `HeavyweightEvaluator`: Full model training and evaluation
   - `SimilarityEvaluator`: Text similarity assessment

5. **Dagster Pipeline** (`src/worker/`)
   - Asset-based pipeline orchestration
   - Resource management for LLM and synthesizer services
   - Current assets: `optimization_result`

### Data Flow

```
Initial Prompt → Genetic Optimization → Evaluation (Lightweight → Heavyweight) → Best Prompt
                      ↑                                                               ↓
                      ←←←←←←←←←←←←← Iterate until convergence ←←←←←←←←←←←←←←←←←←←
```

### Key Design Patterns

- **Repository Pattern**: Used for data access (`src/repositories/`)
- **Strategy Pattern**: Pluggable evaluators for different assessment strategies
- **Resource Pattern**: Dagster resources for shared LLM and synthesizer instances
- **Type Safety**: Full type hints throughout with Pydantic models

## Environment Variables

Required environment variables:
- `GOOGLE_AI_API_KEY`: Google AI API key for LLM operations

## Data Structure

- Dataset location: `data/cleaned_user_reviews.csv`
- Label mapping: positive=1, neutral=2, negative=0
- Columns expected: "sentence", "sentiment"

## Cursor/IDE Rules

From `.cursor/rules/pytorch-agent.mdc`:
- Use type hints consistently
- Optimize for readability over premature optimization  
- Write modular code with separate files for models, data loading, training, and evaluation
- Follow PEP8 style guide

## Dependencies

Key dependencies from `pyproject.toml`:
- **Dagster**: Pipeline orchestration (`dagster>=1.10.21`)
- **Transformers**: ML models (`transformers>=4.52.4`)
- **Google GenAI**: LLM API (`google-genai>=1.20.0`)
- **Instructor**: Structured LLM outputs (`instructor>=1.8.3`)
- **Pandas**: Data manipulation (`pandas>=2.3.0`)
- **PyTorch**: Deep learning framework (via transformers)
- **VnCoreNLP**: Vietnamese text processing (`vncorenlp>=1.0.3`)

## Testing and Development

The codebase uses a two-phase evaluation approach:
1. **Lightweight**: Quick similarity-based assessment for initial filtering
2. **Heavyweight**: Full model training for final evaluation of promising candidates

When working with the prompt optimization system, use custom evaluators that integrate with your specific data augmentation tasks rather than relying solely on LLM-based evaluation.

## Documentation
- Dagster: https://docs.dagster.io/