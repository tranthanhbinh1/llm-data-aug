"""
Model evaluation/scoring asset for Dagster pipeline.
"""

import dagster as dg

from src.worker.helpers import ScoreHelper


@dg.asset(
    deps=["synthetic_data_asset"],
    group_name="evaluation",
    description="Evaluation score for synthetic data using ML models",
    metadata={
        "asset_type": "score",
        "output_format": "json",
    },
)
def score_asset(
    context: dg.AssetExecutionContext,
    synthetic_data_asset: str,
) -> dg.MaterializeResult:
    """Evaluate synthetic data using trainer models and return weighted F1 score."""
    # Get configuration
    config = context.op_execution_context.op_config or {}
    trainer_type = config.get("trainer_type", "cnn_bert_hybrid")

    # Generate cache key and check for existing score
    cache_key = ScoreHelper.get_cache_key(synthetic_data_asset, trainer_type)
    output_path = ScoreHelper.get_score_output_path(cache_key)

    if ScoreHelper.check_cached_score(output_path):
        score_data = ScoreHelper.load_cached_score(output_path)
        context.log.info(f"Using cached score from {output_path}")
        result_metadata = {"cached": True, "cache_key": cache_key}
        result_metadata.update(score_data)
    else:
        # Run evaluation
        score_data = ScoreHelper.evaluate_with_trainer(
            data_path=synthetic_data_asset,
            trainer_type=trainer_type,
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

    return dg.MaterializeResult(
        asset_key="score_asset",
        metadata=result_metadata,
    )
