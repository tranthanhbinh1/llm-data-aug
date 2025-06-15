"""
Helper for evaluation/scoring assets.
Wraps the existing trainer evaluation logic to provide a clean interface for Dagster assets.
"""

import os
from typing import Dict, Any
from pathlib import Path

from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.constants import PROJECT_ROOT


class ScoreHelper:
    """Helper class for model evaluation operations."""

    @staticmethod
    def get_cache_key(data_path: str, trainer_type: str) -> str:
        """Generate cache key for evaluation results."""
        import hashlib

        # Use data file modification time + trainer type for cache key
        try:
            mtime = os.path.getmtime(data_path)
            content = f"{data_path}|{trainer_type}|{mtime}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        except OSError:
            # Fallback if file doesn't exist
            content = f"{data_path}|{trainer_type}"
            return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def get_score_output_path(cache_key: str) -> str:
        """Get output file path for score results."""
        output_dir = Path(PROJECT_ROOT) / "graphs" / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"score_{cache_key}.json")

    @staticmethod
    def load_cached_score(file_path: str) -> Dict[str, Any]:
        """Load cached score from file."""
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_score(score_data: Dict[str, Any], file_path: str) -> None:
        """Save score data to cache file."""
        import json

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(score_data, f, indent=2)

    @staticmethod
    def check_cached_score(file_path: str) -> bool:
        """Check if cached score exists and is valid."""
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    @staticmethod
    def evaluate_with_trainer(
        data_path: str, trainer_type: str = "cnn_bert_hybrid"
    ) -> Dict[str, Any]:
        """
        Evaluate synthetic data using specified trainer.

        Returns:
            Dict containing score and evaluation metadata
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Use CNN-BERT hybrid trainer - create instance like in the main section
        import pandas as pd
        from transformers import AutoModel
        from src.preprocess.text_preprocessor import TextPreprocessor

        bert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        preprocessor = TextPreprocessor(data=pd.read_csv(data_path))

        trainer = CNNBertHybridTrainer(
            bert_model=bert_model,
            preprocessor=preprocessor,
            data_path=data_path,
            freeze_bert=True,
        )
        score = trainer.run_evaluation(data_path)

        return {
            "score": float(score),
            "trainer_type": trainer_type,
            "data_path": data_path,
            "metric": "weighted_f1",
        }

    @staticmethod
    def get_data_info(data_path: str) -> Dict[str, Any]:
        """Get information about the data file."""
        import pandas as pd

        if not os.path.exists(data_path):
            return {"exists": False}

        try:
            df = pd.read_csv(data_path)
            sentiment_counts = df["Sentiment"].value_counts().to_dict()

            return {
                "exists": True,
                "rows": len(df),
                "sentiment_distribution": sentiment_counts,
                "file_size_mb": os.path.getsize(data_path) / (1024 * 1024),
            }
        except Exception as e:
            return {"exists": True, "error": str(e)}
