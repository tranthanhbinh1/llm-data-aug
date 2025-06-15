"""
Synthetic data generation asset for Dagster pipeline.
"""

import dagster as dg
from typing import Literal

from src.worker.helpers import DataHelper
from src.worker.resource import LLMResource


@dg.asset(
    deps=["prompt_asset"],
    group_name="generation",
    description="Synthetic review data generated using optimized prompt",
    metadata={
        "asset_type": "data",
        "output_format": "csv",
    },
)
def synthetic_data_asset(
    context: dg.AssetExecutionContext,
    llm: LLMResource,
    prompt_asset: str,
) -> dg.MaterializeResult:
    """Generate synthetic data using optimized prompt and AugGptRunner."""
    # Get configuration
    config = context.op_execution_context.op_config or {}
    sentiment = config.get("sentiment", "neutral")
    model = config.get("model", "gemini-2.0-flash")

    # Generate cache key and check for existing data
    cache_key = DataHelper.get_cache_key(prompt_asset, sentiment, model)
    output_path = DataHelper.get_output_path(cache_key, sentiment, model)

    if DataHelper.check_cached_data(output_path):
        context.log.info(f"Using cached data from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        # Generate new synthetic data
        from src.synthesizer.aug_gpt_generator import AugGptRunner

        instructor_instance = llm.get_instructor_instance()
        auggpt_runner = AugGptRunner(instructor_instance)

        data_path = DataHelper.generate_synthetic_data(
            auggpt_runner=auggpt_runner,
            prompt=prompt_asset,
            sentiment=sentiment,
            model=model,
        )

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "data_path": data_path,
        }

    # Add data statistics to metadata
    stats = DataHelper.get_data_stats(output_path)
    result_metadata.update({"data_stats": stats})

    return dg.MaterializeResult(
        asset_key="synthetic_data_asset",
        metadata=result_metadata,
    )
