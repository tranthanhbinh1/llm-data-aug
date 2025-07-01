"""
Iterative optimization job using multiple ops for fine-grained control.
"""

import dagster as dg
import asyncio
from typing import Dict, Any, List, Tuple

from src.enums import Sentiment
from src.worker.helpers import DataHelper, ScoreHelper
from src.worker.resource import LLMResource, SynthesizerResource
from src.prompt_optimization import PromptOptimizer, OptimizationConfig


@dg.op(
    config_schema={
        "initial_prompt": str,
        "improvement_request": str,
        "sentiment": str,
        "population_size": int,
        "num_iterations": int,
    }
)
def initialize_optimization_op(context: dg.OpExecutionContext) -> Dict[str, Any]:
    """Initialize optimization parameters and state."""
    config = context.op_config

    optimization_state = {
        "current_prompt": config["initial_prompt"],
        "improvement_request": config["improvement_request"],
        "sentiment": config["sentiment"],
        "round": 0,
        "best_score": 0.0,
        "optimization_config": {
            "population_size": config.get("population_size", 3),
            "num_iterations": config.get("num_iterations", 2),
            "num_elites": 1,
            "threshold": 0.8,
            "tournament_size": 3,
            "num_evaluation_samples": 2,
            "model": "gemini-2.0-flash",
            "max_retries": 3,
        },
        "history": [],
    }

    context.log.info("Initialized optimization state")
    return optimization_state


@dg.op
def run_genetic_optimization_op(
    context: dg.OpExecutionContext,
    optimization_state: Dict[str, Any],
    llm: LLMResource,
    synthesizer: SynthesizerResource,
) -> Dict[str, Any]:
    """Run one round of genetic algorithm optimization."""

    async def _run_optimization():
        # Setup optimization config
        config_dict = optimization_state["optimization_config"]
        optimization_config = OptimizationConfig(**config_dict)

        # Create similarity evaluator
        def create_similarity_evaluator():
            async def similarity_evaluator(
                candidate, initial_prompt, improvement_request
            ):
                try:
                    sentiment = Sentiment(optimization_state["sentiment"])

                    # Generate synthetic data
                    data_path = DataHelper.generate_synthetic_data(
                        auggpt_runner=synthesizer.get_synthesizer_instance(),
                        prompt=candidate.prompt,
                        sentiment=sentiment,
                    )

                    # Evaluate similarity
                    score_data = ScoreHelper.evaluate_similarity(
                        data_path=data_path,
                        prompt=candidate.prompt,
                    )

                    candidate.fitness = score_data["similarity_score"]
                    candidate.reflection = (
                        f"Similarity: {score_data['similarity_score']:.4f}"
                    )

                    return candidate

                except Exception as e:
                    context.log.error(f"Similarity evaluation failed: {e}")
                    candidate.fitness = 0.0
                    candidate.reflection = f"Failed: {str(e)}"
                    return candidate

            return similarity_evaluator

        # Run optimization
        optimizer = PromptOptimizer(api_key=llm.api_key, config=optimization_config)

        result = await optimizer.optimize(
            initial_prompt=optimization_state["current_prompt"],
            improvement_request=optimization_state["improvement_request"],
            custom_evaluator=create_similarity_evaluator(),
        )

        return result

    # Run async optimization
    result = asyncio.run(_run_optimization())

    # Update state
    optimization_state["round"] += 1
    optimization_state["history"].append(
        {
            "round": optimization_state["round"],
            "prompt": result.best_prompt,
            "score": result.best_score,
            "iterations": result.total_iterations,
            "candidates_evaluated": result.total_candidates_evaluated,
        }
    )

    # Update current prompt if improved
    if result.best_score > optimization_state["best_score"]:
        optimization_state["current_prompt"] = result.best_prompt
        optimization_state["best_score"] = result.best_score

    context.log.info(
        f"Round {optimization_state['round']} - Score: {result.best_score:.4f}"
    )

    return optimization_state


@dg.op
def check_convergence_op(
    context: dg.OpExecutionContext,
    optimization_state: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    """Check if optimization should continue."""

    max_rounds = 10
    threshold = 0.8

    should_continue = (
        optimization_state["round"] < max_rounds
        and optimization_state["best_score"] < threshold
    )

    context.log.info(
        f"Round {optimization_state['round']}/{max_rounds}, "
        f"Score: {optimization_state['best_score']:.4f}/{threshold}, "
        f"Continue: {should_continue}"
    )

    return should_continue, optimization_state


@dg.op
def finalize_optimization_op(
    context: dg.OpExecutionContext,
    optimization_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Finalize optimization and return results."""

    final_result = {
        "final_prompt": optimization_state["current_prompt"],
        "final_score": optimization_state["best_score"],
        "total_rounds": optimization_state["round"],
        "converged": optimization_state["best_score"] >= 0.8,
        "history": optimization_state["history"],
    }

    context.log.info(
        f"Optimization completed after {optimization_state['round']} rounds"
    )
    context.log.info(f"Final score: {optimization_state['best_score']:.4f}")

    return final_result


# Create the iterative optimization job
@dg.job(
    resource_defs={
        "llm": LLMResource.configure_at_launch(),
        "synthesizer": SynthesizerResource(),
    }
)
def iterative_optimization_job():
    """
    Job that runs iterative optimization using multiple ops.

    Note: This approach requires manual orchestration of the loop
    outside of Dagster, as Dagster jobs are DAGs and don't support
    native loops. For true iterative behavior, use the single asset approach.
    """

    # Initialize
    state = initialize_optimization_op()

    # Run one optimization round
    updated_state = run_genetic_optimization_op(state)

    # Check convergence (in practice, you'd need external orchestration for the loop)
    should_continue, final_state = check_convergence_op(updated_state)

    # Finalize
    result = finalize_optimization_op(final_state)

    return result
