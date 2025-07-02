"""
Full synthetic data generation asset for trainer pipeline.
This generates the complete dataset without the 5-sample limit used for similarity evaluation.
"""

import dagster as dg

from src.worker.helpers import FullDataHelper
from src.worker.resource import SynthesizerResource
from src.enums import Sentiment


@dg.asset(
    deps=["iterative_optimization_asset"],
    group_name="generation",
    description="Complete synthetic review data generated using iteratively optimized prompt",
    metadata={
        "asset_type": "full_data",
        "output_format": "csv",
    },
)
def full_synthetic_data_asset(
    context: dg.AssetExecutionContext,
    synthesizer: SynthesizerResource,
    iterative_optimization_asset: str,
) -> str:
    """Generate complete synthetic data using iteratively optimized prompt for trainer evaluation."""
    context.log.info(
        "🏭 Starting full synthetic data generation for trainer evaluation"
    )
    context.log.info(
        f"📊 Using optimization results from: {iterative_optimization_asset}"
    )

    # Load the optimization results to get the final prompt
    import json

    with open(iterative_optimization_asset, "r") as f:
        optimization_results = json.load(f)

    final_prompt = optimization_results["final_prompt"]
    optimization_sentiment = optimization_results["config"]["sentiment"]

    context.log.info(
        f"📝 Extracted optimized prompt (length: {len(final_prompt)} chars)"
    )
    context.log.info(f"🎯 Optimization converged: {optimization_results['converged']}")
    context.log.info(
        f"📈 Final similarity score: {optimization_results['final_similarity_score']:.4f}"
    )
    sentiment_str = context.run_config.get(
        "sentiment", optimization_sentiment
    )  # Use optimization sentiment as default
    context.log.info(f"🎯 Optimization sentiment: {optimization_sentiment}")
    context.log.info(f"🎯 Sentiment: {sentiment_str}")
    num_samples = context.run_config.get(
        "num_samples", None
    )  # Use all samples if not specified

    context.log.info(
        f"📊 Configuration: sentiment={sentiment_str}, num_samples={num_samples or 'all'}"
    )

    # Convert sentiment string to enum
    sentiment = Sentiment(sentiment_str)

    # Generate cache key and check for existing data
    cache_key = FullDataHelper.get_cache_key(final_prompt, sentiment, num_samples)
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
            prompt=final_prompt,
            sentiment=sentiment,
            num_samples=num_samples,
        )

        context.log.info("✅ Full synthetic data generation completed")
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

    context.log.info("📈 Dataset statistics:")
    context.log.info(f"   📋 Total records: {stats.get('total_records', 'N/A')}")
    context.log.info(f"   📊 Average length: {stats.get('average_length', 'N/A')}")
    context.log.info(
        f"   📏 Min/Max length: {stats.get('min_length', 'N/A')}/{stats.get('max_length', 'N/A')}"
    )

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    context.log.info("✅ Full synthetic data asset completed successfully")
    return output_path
