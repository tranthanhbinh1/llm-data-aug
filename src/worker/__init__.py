import dagster as dg
from src.worker.assets.prompt import prompt_asset
from src.worker.assets.synthetic_data import synthetic_data_asset
from src.worker.assets.score import score_asset
from src.worker.assets.full_synthetic_data import full_synthetic_data_asset
from src.worker.assets.preprocessed_data import preprocessed_data_asset
from src.worker.assets.multi_trainer_scores import multi_trainer_scores_asset
from src.worker.assets.iterative_optimization import iterative_optimization_asset
from src.worker.assets.iterative_optimization_v2 import iterative_optimization_result
from src.worker.resource import LLMResource, SynthesizerResource
from src.worker.jobs.optimization_job import iterative_optimization_job
from src.worker.jobs.trainer_evaluation_job import trainer_evaluation_job
from src.worker.jobs.iterative_loop_job import complete_cycle_job
from src.worker.sensors.trainer_evaluation_sensor import trainer_evaluation_sensor
from src.worker.sensors.optimization_cycle_sensor import optimization_cycle_sensor


# Create the main definitions object
defs = dg.Definitions(
    assets=[
        prompt_asset,
        synthetic_data_asset,
        score_asset,
        # Remove the above later
        iterative_optimization_asset,
        iterative_optimization_result,
        full_synthetic_data_asset,
        preprocessed_data_asset,
        multi_trainer_scores_asset,
    ],
    jobs=[
        iterative_optimization_job,
        trainer_evaluation_job,
        complete_cycle_job,
    ],
    sensors=[
        trainer_evaluation_sensor,  # Triggers trainer eval after X optimization cycles
        optimization_cycle_sensor,  # Triggers next optimization after trainer eval
    ],
    resources={
        "llm": LLMResource(api_key=dg.EnvVar("GOOGLE_AI_API_KEY")),
        "synthesizer": SynthesizerResource(),
    },
)


def create_repository():
    """Create the Dagster repository for LLM data augmentation pipeline."""
    return defs
