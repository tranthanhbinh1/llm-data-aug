"""
Fast optimization job for prompt optimization cycles.
Now uses iterative optimization with similarity feedback.
"""

import dagster as dg

from src.worker.assets.iterative_optimization import iterative_optimization_asset


# Create the iterative optimization job
iterative_optimization_job = dg.define_asset_job(
    name="iterative_optimization_job",
    description="Iterative prompt optimization with real-time similarity feedback",
    selection=dg.AssetSelection.assets(iterative_optimization_asset),
)

# Keep the old job for backward compatibility (but it's now deprecated)
from src.worker.assets.prompt import prompt_asset
from src.worker.assets.synthetic_data import synthetic_data_asset
from src.worker.assets.score import score_asset

optimization_job = dg.define_asset_job(
    name="optimization_job",
    description="[DEPRECATED] Linear prompt optimization - use iterative_optimization_job instead",
    selection=dg.AssetSelection.assets(
        prompt_asset,
        synthetic_data_asset,
        score_asset,
    ),
)
