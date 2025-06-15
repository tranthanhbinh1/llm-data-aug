import dagster as dg

from src.worker.assets.prompt import prompt_asset
from src.worker.assets.synthetic_data import synthetic_data_asset
from src.worker.assets.score import score_asset


# Create the full pipeline job
full_pipeline_job = dg.define_asset_job(
    name="full_pipeline_job",
    description="Materialize the complete LLM data augmentation pipeline",
    selection=dg.AssetSelection.assets(
        prompt_asset,
        synthetic_data_asset,
        score_asset,
    ),
)


def create_full_pipeline_job(assets):
    """Create a job that materializes the full LLM data augmentation pipeline."""
    return dg.define_asset_job(
        name="full_pipeline_job",
        description="Materialize the complete LLM data augmentation pipeline",
        selection=assets,
    )
