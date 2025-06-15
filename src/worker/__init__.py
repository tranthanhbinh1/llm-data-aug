import dagster as dg

from src.worker.assets.prompt import prompt_asset
from src.worker.assets.synthetic_data import synthetic_data_asset
from src.worker.assets.score import score_asset
from src.worker.resource import LLMResource
from src.worker.jobs.full_pipeline import full_pipeline_job


# Create the main definitions object
defs = dg.Definitions(
    assets=[
        prompt_asset,
        synthetic_data_asset,
        score_asset,
    ],
    jobs=[
        full_pipeline_job,
    ],
    resources={
        "llm": LLMResource(api_key=dg.EnvVar("GOOGLE_AI_API_KEY")),
    },
)


def create_repository():
    """Create the Dagster repository for LLM data augmentation pipeline."""
    return defs
