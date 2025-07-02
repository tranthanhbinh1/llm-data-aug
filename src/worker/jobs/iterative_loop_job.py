"""
Complete iterative loop job that combines optimization and trainer evaluation.
This job is designed to work with sensors for continuous iteration:
iterative optimization -> full synthetic data -> trainer eval -> repeat
"""

import dagster as dg

from src.worker.assets.iterative_optimization import iterative_optimization_asset
from src.worker.assets.full_synthetic_data import full_synthetic_data_asset
from src.worker.assets.preprocessed_data import preprocessed_data_asset
from src.worker.assets.multi_trainer_scores import multi_trainer_scores_asset

# Job that runs iterative optimization + trainer evaluation in sequence
complete_cycle_job = dg.define_asset_job(
    name="complete_cycle_job",
    description="Complete cycle: iterative optimization with similarity feedback → trainer evaluation",
    selection=dg.AssetSelection.assets(
        iterative_optimization_asset,  # Runs iterative optimization with similarity feedback
        full_synthetic_data_asset,  # Generates full dataset using optimized prompt
        preprocessed_data_asset,  # Preprocesses data for trainers
        multi_trainer_scores_asset,  # Evaluates with all trainers
    ),
)
