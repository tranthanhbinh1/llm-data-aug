import dagster as dg

from src.worker.assets.prompt import prompt_asset
from src.worker.assets.synthetic_data import synthetic_data_asset
from src.worker.assets.full_synthetic_data import full_synthetic_data_asset
from src.worker.assets.score import score_asset
from src.worker.assets.preprocessed_data import preprocessed_data_asset
from src.worker.assets.multi_trainer_scores import multi_trainer_scores_asset
from src.worker.assets.iterative_optimization import iterative_optimization_asset
from src.worker.resource import LLMResource, SynthesizerResource
from src.worker.jobs.optimization_job import (
    optimization_job,
    iterative_optimization_job,
)
from src.worker.jobs.trainer_evaluation_job import trainer_evaluation_job
from src.worker.jobs.iterative_loop_job import complete_cycle_job
from src.worker.jobs.full_pipeline import (
    full_pipeline_job,
)  # Keep for backward compatibility
from src.worker.sensors.trainer_evaluation_sensor import trainer_evaluation_sensor
from src.worker.sensors.optimization_cycle_sensor import optimization_cycle_sensor


# Create the main definitions object
defs = dg.Definitions(
    assets=[
        # Core optimization assets
        iterative_optimization_asset,  # New primary optimization asset
        # Legacy assets (for backward compatibility)
        prompt_asset,
        synthetic_data_asset,
        score_asset,
        # Trainer evaluation assets
        full_synthetic_data_asset,
        preprocessed_data_asset,
        multi_trainer_scores_asset,
    ],
    jobs=[
        # Primary workflows
        iterative_optimization_job,  # Optimization with similarity feedback
        trainer_evaluation_job,  # Heavy trainer evaluation
        complete_cycle_job,  # Complete cycle: optimization + trainer evaluation
        # Legacy workflows (for backward compatibility)
        optimization_job,  # Deprecated linear optimization
        full_pipeline_job,  # Legacy full pipeline
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
