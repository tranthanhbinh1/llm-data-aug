"""
Heavy trainer evaluation job for model performance assessment.
"""

import dagster as dg

from src.worker.assets.prompt import prompt_asset
from src.worker.assets.full_synthetic_data import full_synthetic_data_asset
from src.worker.assets.preprocessed_data import preprocessed_data_asset
from src.worker.assets.multi_trainer_scores import multi_trainer_scores_asset


# Create the heavy trainer evaluation job
trainer_evaluation_job = dg.define_asset_job(
    name="trainer_evaluation_job",
    description="Heavy trainer evaluation with full dataset",
    selection=dg.AssetSelection.assets(
        prompt_asset,
        full_synthetic_data_asset,
        preprocessed_data_asset,
        multi_trainer_scores_asset,
    ),
)
