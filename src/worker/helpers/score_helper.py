"""
Helper for similarity evaluation/scoring assets.
Wraps the existing SimilarityEvaluator to provide a clean interface for Dagster assets.
"""

import os
import pandas as pd
from typing import Dict, Any
from pathlib import Path

from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.synthesizer.aug_gpt import AugGpt
from src.utils import get_instructor_instance
from src.enums import Sentiment
from src.constants import PROJECT_ROOT


class ScoreHelper:
    """Helper class for similarity evaluation operations."""

    @staticmethod
    def get_cache_key(data_path: str, prompt: str) -> str:
        """Generate cache key for similarity evaluation results."""
        import hashlib

        # Use data file modification time + prompt for cache key
        try:
            mtime = os.path.getmtime(data_path)
            content = f"{data_path}|{prompt}|{mtime}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except OSError:
            # Fallback if file doesn't exist
            content = f"{data_path}|{prompt}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def get_score_output_path(cache_key: str) -> str:
        """Get output file path for similarity score results."""
        output_dir = Path(PROJECT_ROOT) / "graphs" / "similarity_scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"similarity_score_{cache_key}.json")

    @staticmethod
    def load_cached_score(file_path: str) -> Dict[str, Any]:
        """Load cached similarity score from file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_score(score_data: Dict[str, Any], file_path: str) -> None:
        """Save similarity score data to cache file."""
        import json

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, indent=2)

    @staticmethod
    def check_cached_score(file_path: str) -> bool:
        """Check if cached similarity score exists and is valid."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def evaluate_similarity(data_path: str, prompt: str) -> Dict[str, Any]:
        """
        Evaluate synthetic data using similarity analysis.

        Returns:
            Dict containing similarity score and evaluation metadata
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load synthetic data to determine sentiment
        df = pd.read_csv(data_path)

        # Get the predominant sentiment from the data
        sentiment_counts = df.get(
            "sentiment", df.get("Sentiment", pd.Series())
        ).value_counts()
        if sentiment_counts.empty:
            raise ValueError("No sentiment column found in data")

        predominant_sentiment = sentiment_counts.index[0]
        sentiment = Sentiment(predominant_sentiment)

        # Create similarity evaluator
        aug_gpt = AugGpt(get_instructor_instance())
        evaluator = SimiarityEvaluator(data_synthesizer=aug_gpt)

        # Run similarity evaluation
        similarity_score = evaluator.run_evaluation(
            sentiment=sentiment,
            prompt=prompt,
        )

        return {
            "similarity_score": float(similarity_score),
            "sentiment": sentiment.value,
            "data_path": data_path,
            "prompt": prompt[:100] + "..."
            if len(prompt) > 100
            else prompt,  # Truncate for storage
            "metric": "cosine_similarity",
        }

    @staticmethod
    def get_data_info(data_path: str) -> Dict[str, Any]:
        """Get information about the data file."""
        if not os.path.exists(data_path):
            return {"exists": False}

        try:
            df = pd.read_csv(data_path)
            sentiment_col = "sentiment" if "sentiment" in df.columns else "Sentiment"

            if sentiment_col in df.columns:
                sentiment_counts = df[sentiment_col].value_counts().to_dict()
            else:
                sentiment_counts = {}

            return {
                "exists": True,
                "rows": len(df),
                "columns": list(df.columns),
                "sentiment_distribution": sentiment_counts,
                "file_size_mb": os.path.getsize(data_path) / (1024 * 1024),
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
