"""
Synthetic data generation asset for Dagster pipeline.
"""

import dagster as dg

from src.worker.helpers import DataHelper
from src.worker.resource import SynthesizerResource
from src.enums import Sentiment


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
    synthesizer: SynthesizerResource,
    prompt_asset: str,
) -> str:
    """Generate synthetic data using optimized prompt and AugGptRunner."""
    # Get configuration
    config = context.op_execution_context.op_config or {}
    sentiment_str = config.get("sentiment", "neutral")
    model = config.get("model", "gemini-2.0-flash")

    # Convert sentiment string to enum
    sentiment = Sentiment(sentiment_str)

    # Generate cache key and check for existing data
    cache_key = DataHelper.get_cache_key(prompt_asset, sentiment)
    output_path = DataHelper.get_output_path(cache_key, sentiment)

    if DataHelper.check_cached_data(output_path):
        context.log.info(f"Using cached data from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        # Generate new synthetic data
        data_path = DataHelper.generate_synthetic_data(
            auggpt_runner=synthesizer.get_synthesizer_instance(),
            prompt=prompt_asset,
            sentiment=sentiment,
        )

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "data_path": data_path,
        }

    # Add data statistics to metadata
    stats = DataHelper.get_data_stats(output_path)
    result_metadata.update({"data_stats": stats})

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    return output_path
