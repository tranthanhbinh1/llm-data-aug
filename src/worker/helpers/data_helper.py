"""
Helper for synthetic data generation assets.
Wraps the existing AugGptRunner to provide a clean interface for Dagster assets.
"""

import os
import hashlib
from typing import Dict, Any
from pathlib import Path
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)

from src.synthesizer.aug_gpt import AugGpt
from src.synthesizer.models import Sentiment
from src.constants import PROJECT_ROOT


class DataHelper:
    """Helper class for synthetic data generation operations."""

    @staticmethod
    def get_cache_key(prompt: str, sentiment: Sentiment) -> str:
        """Generate deterministic cache key for synthetic data."""
        content = f"{prompt}|{sentiment}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def get_output_path(cache_key: str, sentiment: Sentiment) -> str:
        """Get output file path for synthetic data."""
        output_dir = Path(PROJECT_ROOT) / "data" / "llm_generated" / sentiment.value
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"auggpt_augmented_user_reviews_{sentiment}_{cache_key}.csv"
        return str(output_dir / filename)

    @staticmethod
    def check_cached_data(file_path: str) -> bool:
        """Check if cached data exists and is valid."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def generate_synthetic_data(
        auggpt_runner: AugGpt,
        prompt: str,
        sentiment: Sentiment,
    ) -> str:
        """
        Generate synthetic data using AugGptRunner.

        Returns:
            Path to generated CSV file
        """
        data_path = auggpt_runner.generate(
            original_sentences=original_sentences,
            system_prompt=prompt,
            sentiment=sentiment,
        )

        return

    @staticmethod
    def get_data_stats(file_path: str) -> Dict[str, Any]:
        """Get statistics about generated data file."""
        import pandas as pd

        if not os.path.exists(file_path):
            return {"exists": False}

        try:
            df = pd.read_csv(file_path)
            return {
                "exists": True,
                "rows": len(df),
                "columns": list(df.columns),
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
