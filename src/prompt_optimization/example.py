"""
Example usage of the prompt optimization system.
"""

import asyncio
import os
from dotenv import load_dotenv

from .models import PromptCandidate, OptimizationConfig
from .optimizer import PromptOptimizer

load_dotenv()


async def custom_evaluator_example(
    candidate: PromptCandidate, initial_prompt: str, improvement_request: str
) -> PromptCandidate:
    """
    Example custom evaluator that could integrate with your downstream tasks.

    In your actual implementation, this would:
    1. Use the candidate.prompt to generate synthetic data
    2. Evaluate the synthetic data quality (similarity, model performance, etc.)
    3. Return a fitness score between 0.0 and 1.0
    """
    # Skip if already evaluated
    if candidate.fitness is not None:
        return candidate

    # Placeholder evaluation logic
    # In reality, you'd run your data generation and evaluation pipeline here
    prompt_length = len(candidate.prompt)
    word_count = len(candidate.prompt.split())

    # Simple heuristic: prefer longer, more detailed prompts
    # Replace this with your actual evaluation logic
    candidate.fitness = min(
        1.0, (word_count / 100.0) * 0.8 + (prompt_length / 1000.0) * 0.2
    )
    candidate.reflection = (
        f"Evaluated based on length metrics: {word_count} words, {prompt_length} chars"
    )

    return candidate


async def basic_example():
    """Basic example using default LLM evaluation."""

    # Set up API key
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("Please set GOOGLE_AI_API_KEY environment variable")
        return

    # Configure optimization
    config = OptimizationConfig(
        population_size=3,  # Small for demo
        num_iterations=2,  # Quick demo
        num_elites=1,
        threshold=0.9,
        num_evaluation_samples=2,
        tournament_size=3,
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=3,
    )

    # Create optimizer
    optimizer = PromptOptimizer(api_key=api_key, config=config)

    # Define initial prompt and improvement request
    initial_prompt = """Generate synthetic customer reviews for a product."""

    improvement_request = """Make the prompt more specific about generating diverse, 
    realistic reviews that include both positive and negative sentiment, 
    specific product features, and varied writing styles."""

    print("Starting optimization...")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Improvement request: {improvement_request}")
    print("-" * 50)

    # Run optimization
    result = await optimizer.optimize(
        initial_prompt=initial_prompt, improvement_request=improvement_request
    )

    # Print results
    print("Optimization completed!")
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Total time: {result.execution_time_seconds:.2f}s")
    print(f"Candidates evaluated: {result.total_candidates_evaluated}")


async def custom_evaluator_example_run():
    """Example using a custom evaluator."""

    # Set up API key
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("Please set GOOGLE_AI_API_KEY environment variable")
        return

    # Configure optimization
    config = OptimizationConfig(
        population_size=3,
        num_iterations=2,
        num_elites=1,
        threshold=0.8,
        tournament_size=3,
        num_evaluation_samples=2,
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=3,
    )

    # Create optimizer
    optimizer = PromptOptimizer(api_key=api_key, config=config)

    # Define initial prompt and improvement request
    initial_prompt = """Create training data for sentiment analysis."""

    improvement_request = """Improve the prompt to generate more diverse and 
    balanced training examples with clear sentiment labels."""

    print("Starting optimization with custom evaluator...")
    print(f"Initial prompt: {initial_prompt}")
    print(f"Improvement request: {improvement_request}")
    print("-" * 50)

    # Run optimization with custom evaluator
    result = await optimizer.optimize(
        initial_prompt=initial_prompt,
        improvement_request=improvement_request,
        custom_evaluator=custom_evaluator_example,
    )

    # Print results
    print("Optimization completed!")
    print(f"Best prompt: {result.best_prompt}")
    print(f"Best score: {result.best_score:.4f}")
    print(f"Total time: {result.execution_time_seconds:.2f}s")
    print(f"Candidates evaluated: {result.total_candidates_evaluated}")


async def progress_example():
    """Example showing progress updates."""

    # Set up API key
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("Please set GOOGLE_AI_API_KEY environment variable")
        return

    # Configure optimization
    config = OptimizationConfig(
        population_size=3,
        num_iterations=3,
        num_elites=1,
        threshold=0.9,
        num_evaluation_samples=2,
        tournament_size=3,
        model="gemini-2.0-flash",
        temperature=1.0,
        max_retries=3,
    )

    # Create optimizer
    optimizer = PromptOptimizer(api_key=api_key, config=config)

    initial_prompt = """Generate product descriptions."""
    improvement_request = """Make descriptions more engaging and detailed."""

    print("Starting optimization with progress tracking...")
    print("-" * 50)

    # Track progress
    async for (
        message,
        iteration,
        progress,
        best_score,
    ) in optimizer.optimize_with_progress(
        initial_prompt=initial_prompt, improvement_request=improvement_request
    ):
        score_str = f" (score: {best_score:.4f})" if best_score else ""
        print(f"[{progress * 100:.1f}%] {message}{score_str}")


if __name__ == "__main__":
    print("Prompt Optimization Examples")
    print("=" * 50)

    # Run basic example
    print("\n1. Basic Example (LLM evaluation):")
    asyncio.run(basic_example())

    # Run custom evaluator example
    print("\n2. Custom Evaluator Example:")
    asyncio.run(custom_evaluator_example_run())

    # Run progress example
    print("\n3. Progress Tracking Example:")
    asyncio.run(progress_example())
