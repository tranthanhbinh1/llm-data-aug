import dagster as dg
from src.worker.assets.full_synthetic_data import full_synthetic_data_asset
from src.worker.assets.preprocessed_data import preprocessed_data_asset
from src.worker.assets.multi_trainer_scores import multi_trainer_scores_asset
from src.worker.assets.iterative_optimization import iterative_optimization_asset
from src.worker.assets.iterative_optimization_v2 import iterative_optimization_result
from src.worker.resource import LLMResource, SynthesizerResource
from src.worker.jobs.optimization_job import iterative_optimization_job
from src.worker.jobs.iterative_loop_job import complete_cycle_job
from src.worker.sensors.optimization_cycle_sensor import optimization_cycle_sensor


# Create the main definitions object
defs = dg.Definitions(
    assets=[
        iterative_optimization_asset,
        iterative_optimization_result,
        full_synthetic_data_asset,
        preprocessed_data_asset,
        multi_trainer_scores_asset,
    ],
    jobs=[
        iterative_optimization_job,
        complete_cycle_job,
    ],
    sensors=[
        optimization_cycle_sensor,  # Triggers next optimization after complete cycle
    ],
    resources={
        "llm": LLMResource(api_key=dg.EnvVar("GOOGLE_AI_API_KEY")),
        "synthesizer": SynthesizerResource(),
    },
)


def create_repository():
    """Create the Dagster repository for LLM data augmentation pipeline."""
    return defs
