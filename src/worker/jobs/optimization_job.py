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
