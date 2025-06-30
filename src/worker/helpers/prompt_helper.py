"""
Helper for prompt optimization assets.
Wraps the existing PromptOptimizer to provide a clean interface for Dagster assets.
"""

import os
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

from src.prompt_optimization import PromptOptimizer, OptimizationConfig
from src.constants import PROJECT_ROOT


class PromptHelper:
    """Helper class for prompt optimization operations."""

    @staticmethod
    def get_cache_key(
        initial_prompt: str, improvement_request: str, sentiment: str
    ) -> str:
        """Generate deterministic cache key for prompt optimization."""
        content = f"{initial_prompt}|{improvement_request}|{sentiment}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def get_output_path(cache_key: str) -> str:
        """Get output file path for prompt."""
        output_dir = Path(PROJECT_ROOT) / "graphs" / "prompts"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"prompt_{cache_key}.txt")

    @staticmethod
    def load_cached_prompt(file_path: str) -> str:
        """Load prompt from cache file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def save_prompt(prompt: str, file_path: str) -> None:
        """Save prompt to cache file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(prompt)

    @staticmethod
    def check_cached_prompt(file_path: str) -> bool:
        """Check if cached prompt exists and is valid."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    async def optimize_prompt(
        api_key: str,
        initial_prompt: str,
        improvement_request: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run prompt optimization and return results with metadata.

        Returns:
            Dict containing optimized prompt, score, and metadata
        """
        if config is None:
            config = {}

        optimization_config = OptimizationConfig(
            population_size=config.get("population_size", 3),
            num_iterations=config.get("num_iterations", 2),
            num_elites=config.get("num_elites", 1),
            threshold=config.get("threshold", 0.9),
            tournament_size=config.get("tournament_size", 3),
            num_evaluation_samples=config.get("num_evaluation_samples", 2),
            model=config.get("model", "gemini-2.0-flash"),
            max_retries=config.get("max_retries", 3),
        )

        optimizer = PromptOptimizer(api_key=api_key, config=optimization_config)

        result = await optimizer.optimize(
            initial_prompt=initial_prompt, improvement_request=improvement_request
        )

        return {
            "prompt": result.best_prompt,
            "score": result.best_score,
            "iterations": result.total_iterations,
            "candidates_evaluated": result.total_candidates_evaluated,
            "execution_time": result.execution_time_seconds,
        }
