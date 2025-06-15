"""
Prompt optimization asset for Dagster pipeline.
"""

import dagster as dg
import asyncio
from typing import Dict, Any

from src.worker.helpers import PromptHelper
from src.worker.resource import LLMResource


@dg.asset(
    group_name="generation",
    description="Optimized prompt for synthetic data generation",
    metadata={
        "asset_type": "prompt",
        "output_format": "text",
    },
)
def prompt_asset(
    context: dg.AssetExecutionContext,
    llm: LLMResource,
) -> dg.MaterializeResult:
    """Generate optimized prompt using genetic algorithm optimization."""
    # Get configuration from asset context
    config = context.op_execution_context.op_config or {}
    initial_prompt = config.get("initial_prompt", "Generate product reviews")
    improvement_request = config.get("improvement_request", "Make more diverse")
    sentiment = config.get("sentiment", "neutral")

    # Generate cache key and check for existing prompt
    cache_key = PromptHelper.get_cache_key(
        initial_prompt, improvement_request, sentiment
    )
    output_path = PromptHelper.get_output_path(cache_key)

    if PromptHelper.check_cached_prompt(output_path):
        prompt = PromptHelper.load_cached_prompt(output_path)
        context.log.info(f"Using cached prompt from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        # Run optimization asynchronously
        result = asyncio.run(
            PromptHelper.optimize_prompt(
                api_key=llm.api_key,
                initial_prompt=initial_prompt,
                improvement_request=improvement_request,
                config=config.get("optimization", {}),
            )
        )
        prompt = result["prompt"]
        PromptHelper.save_prompt(prompt, output_path)

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "score": result["score"],
            "iterations": result["iterations"],
            "candidates_evaluated": result["candidates_evaluated"],
            "execution_time": result["execution_time"],
        }

    return dg.MaterializeResult(
        asset_key="prompt_asset",
        metadata=result_metadata,
    )
