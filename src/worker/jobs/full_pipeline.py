import dagster as dg


def create_full_pipeline_job(assets):
    """Create a job that materializes the full LLM data augmentation pipeline."""
    return dg.define_asset_job(
        name="full_pipeline_job",
        description="Materialize the complete LLM data augmentation pipeline",
        selection=assets,
    )


# This will be updated when assets are implemented
full_pipeline_job = None
