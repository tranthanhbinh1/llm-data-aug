"""
Helper for synthetic data generation assets.
Wraps the existing AugGptRunner to provide a clean interface for Dagster assets.
"""

import os
import hashlib
from typing import Dict, Any, Literal
from pathlib import Path
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)

from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.synthesizer.models import SentimentPrompt
from src.constants import PROJECT_ROOT, NUM_REPHRASED_SENTENCES


class DataHelper:
    """Helper class for synthetic data generation operations."""

    @staticmethod
    def get_cache_key(prompt: str, sentiment: str, model: str) -> str:
        """Generate deterministic cache key for synthetic data."""
        content = f"{prompt}|{sentiment}|{model}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def get_output_path(cache_key: str, sentiment: str, model: str) -> str:
        """Get output file path for synthetic data."""
        output_dir = Path(PROJECT_ROOT) / "data" / "llm_generated" / model
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"auggpt_augmented_user_reviews_{sentiment}_{cache_key}.csv"
        return str(output_dir / filename)

    @staticmethod
    def check_cached_data(file_path: str) -> bool:
        """Check if cached data exists and is valid."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def generate_synthetic_data(
        auggpt_runner: AugGptRunner,
        prompt: str,
        sentiment: Literal["neutral", "negative"],
        model: str = "gemini-2.0-flash",
    ) -> str:
        """
        Generate synthetic data using AugGptRunner.

        Returns:
            Path to generated CSV file
        """
        augmentor_prompt = ChatCompletionSystemMessageParam(
            role="system",
            content=prompt,
        )

        data_path = auggpt_runner.generate_reviews_batch(
            sentiment=sentiment,
            user_prompt=SentimentPrompt.AUG_GPT_PROMPT,
            augmentor_prompt=augmentor_prompt,
            num_to_generate=NUM_REPHRASED_SENTENCES,
            model=model,
        )

        return data_path

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
