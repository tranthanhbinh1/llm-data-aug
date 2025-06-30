"""
Similarity evaluation/scoring asset for Dagster pipeline.
"""

import dagster as dg

from src.worker.helpers import ScoreHelper


@dg.asset(
    deps=["synthetic_data_asset", "prompt_asset"],
    group_name="evaluation",
    description="Similarity evaluation score for synthetic data quality",
    metadata={
        "asset_type": "similarity_score",
        "output_format": "json",
    },
)
def score_asset(
    context: dg.AssetExecutionContext,
    synthetic_data_asset: str,
    prompt_asset: str,
) -> str:
    """Evaluate synthetic data quality using similarity analysis."""

    # Generate cache key and check for existing score
    cache_key = ScoreHelper.get_cache_key(synthetic_data_asset, prompt_asset)
    output_path = ScoreHelper.get_score_output_path(cache_key)

    if ScoreHelper.check_cached_score(output_path):
        score_data = ScoreHelper.load_cached_score(output_path)
        context.log.info(f"Using cached similarity score from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
        result_metadata.update(score_data)
    else:
        # Run similarity evaluation
        score_data = ScoreHelper.evaluate_similarity(
            data_path=synthetic_data_asset,
            prompt=prompt_asset,
        )
        ScoreHelper.save_score(score_data, output_path)

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
        }
        result_metadata.update(score_data)

    # Add data info to metadata
    data_info = ScoreHelper.get_data_info(synthetic_data_asset)
    result_metadata.update({"data_info": data_info})

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    return output_path
