"""
Full synthetic data generation asset for trainer pipeline.
This generates the complete dataset without the 5-sample limit used for similarity evaluation.
"""

import dagster as dg

from src.worker.helpers import FullDataHelper
from src.worker.resource import SynthesizerResource
from src.enums import Sentiment


@dg.asset(
    deps=["prompt_asset"],
    group_name="generation",
    description="Complete synthetic review data generated for trainer evaluation",
    metadata={
        "asset_type": "full_data",
        "output_format": "csv",
    },
)
def full_synthetic_data_asset(
    context: dg.AssetExecutionContext,
    synthesizer: SynthesizerResource,
    prompt_asset: str,
) -> str:
    """Generate complete synthetic data using optimized prompt for trainer evaluation."""
    context.log.info(
        "🏭 Starting full synthetic data generation for trainer evaluation"
    )

    # Get configuration
    config = context.op_execution_context.op_config or {}
    sentiment_str = config.get("sentiment", "neutral")
    num_samples = config.get("num_samples", None)  # Use all samples if not specified

    context.log.info(
        f"📊 Configuration: sentiment={sentiment_str}, num_samples={num_samples or 'all'}"
    )
    context.log.info(f"📝 Using prompt (length: {len(prompt_asset)} chars)")

    # Convert sentiment string to enum
    sentiment = Sentiment(sentiment_str)

    # Generate cache key and check for existing data
    cache_key = FullDataHelper.get_cache_key(prompt_asset, sentiment, num_samples)
    output_path = FullDataHelper.get_output_path(cache_key, sentiment)

    context.log.info(f"🔑 Generated cache key: {cache_key}")
    context.log.info(f"📂 Output path: {output_path}")

    if FullDataHelper.check_cached_data(output_path):
        context.log.info(f"💾 Using cached full dataset from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        context.log.info(
            "🔄 No cached data found - generating new full synthetic dataset"
        )
        context.log.info("⚠️ This is a heavy operation that may take several minutes...")

        # Generate new full synthetic data
        data_path = FullDataHelper.generate_full_synthetic_data(
            auggpt_runner=synthesizer.get_synthesizer_instance(),
            prompt=prompt_asset,
            sentiment=sentiment,
            num_samples=num_samples,
        )

        context.log.info(f"✅ Full synthetic data generation completed")
        context.log.info(f"📁 Generated data saved to: {data_path}")

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "data_path": data_path,
        }

    # Add data statistics to metadata
    context.log.info("📊 Calculating data statistics...")
    stats = FullDataHelper.get_data_stats(output_path)
    result_metadata.update({"data_stats": stats})

    context.log.info(f"📈 Dataset statistics:")
    context.log.info(f"   📋 Total records: {stats.get('total_records', 'N/A')}")
    context.log.info(f"   📊 Average length: {stats.get('average_length', 'N/A')}")
    context.log.info(
        f"   📏 Min/Max length: {stats.get('min_length', 'N/A')}/{stats.get('max_length', 'N/A')}"
    )

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    context.log.info("✅ Full synthetic data asset completed successfully")
    return output_path
