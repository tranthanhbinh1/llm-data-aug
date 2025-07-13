"""
Heavyweight evaluator for comprehensive prompt evaluation.
"""

from typing import Dict, Any
import dagster as dg
import pandas as pd
from pathlib import Path
import hashlib
import os
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .base import EvaluatorStrategy
from src.enums import Sentiment
from src.worker.helpers import FullDataHelper
from src.worker.resource import SynthesizerResource
from src.preprocess.text_preprocessor import TextPreprocessor
from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.trainers.lstm import BERTLSTMTrainer, BERTLSTMModel
from src.trainers.phoBert import PhoBertTrainer
from src.trainers.svm import SVMTrainer
from src.constants import PROJECT_ROOT


class HeavyweightEvaluator(EvaluatorStrategy):
    """
    Comprehensive evaluation strategy using full datasets and trainer models.

    This evaluator:
    - Generates complete synthetic datasets (all samples)
    - Preprocesses data for trainer consumption
    - Trains and evaluates all 4 models (CNN-BERT, LSTM, PhoBERT, SVM)
    - Calculates composite score from all trainers
    - Completes evaluation in ~15-30 minutes
    """

    def __init__(
        self, context: dg.OpExecutionContext, synthesizer: SynthesizerResource
    ):
        super().__init__(context)
        self.synthesizer = synthesizer

    async def evaluate(
        self, candidate_prompt: str, optimization_state: Dict[str, Any]
    ) -> float:
        """
        Evaluate candidate prompt using comprehensive trainer evaluation.

        Args:
            candidate_prompt: The prompt to evaluate
            optimization_state: Current optimization state

        Returns:
            float: Composite trainer score (0.0 to 1.0)
        """
        try:
            self.context.log.info(
                f"🏋️ Heavyweight evaluation - candidate prompt (length: {len(candidate_prompt)} chars)"
            )
            self.context.log.info(
                "⚠️ This is a heavy operation that may take 15-30 minutes..."
            )

            # Extract sentiment from optimization state
            sentiment = Sentiment(optimization_state["sentiment"])

            # Step 1: Generate full synthetic dataset
            self.context.log.info("📊 Step 1: Generating full synthetic dataset...")
            full_data_path = await self._generate_full_dataset(
                candidate_prompt, sentiment
            )

            # Step 2: Preprocess dataset for trainers
            self.context.log.info("🔧 Step 2: Preprocessing dataset for trainers...")
            preprocessed_data_path = await self._preprocess_dataset(full_data_path)

            # Step 3: Run all trainers and get scores
            self.context.log.info("🎯 Step 3: Running all trainers...")
            trainer_scores = await self._run_all_trainers(preprocessed_data_path)

            # Step 4: Calculate composite score
            self.context.log.info("📈 Step 4: Calculating composite score...")
            composite_score = self._calculate_composite_score(trainer_scores)

            self.context.log.info(
                f"🏆 Heavyweight evaluation complete - Composite Score: {composite_score:.4f}"
            )
            self.context.log.info(f"📋 Individual trainer scores: {trainer_scores}")

            return composite_score

        except Exception as e:
            self.context.log.error(f"❌ Heavyweight evaluation failed: {e}")
            return 0.0

    async def _generate_full_dataset(self, prompt: str, sentiment: Sentiment) -> str:
        """Generate full synthetic dataset using the candidate prompt."""
        data_path = FullDataHelper.generate_full_synthetic_data(
            auggpt_runner=self.synthesizer.get_synthesizer_instance(),
            prompt=prompt,
            sentiment=sentiment,
            num_samples=None,  # Use all samples
        )

        self.context.log.info(f"📁 Generated full dataset at: {data_path}")
        return data_path

    async def _preprocess_dataset(self, data_path: str) -> str:
        """Preprocess dataset for trainer consumption."""
        # Generate cache key for preprocessing
        data_mtime = os.path.getmtime(data_path)
        cache_key = hashlib.sha256(f"{data_path}|{data_mtime}".encode()).hexdigest()[
            :16
        ]

        # Define output path for preprocessed data
        output_dir = Path(PROJECT_ROOT) / "data" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"preprocessed_{cache_key}.csv"

        if output_path.exists():
            self.context.log.info(
                f"💾 Using cached preprocessed data from {output_path}"
            )
        else:
            # Load and preprocess data
            self.context.log.info("🔧 Running preprocessing pipeline...")
            data = pd.read_csv(data_path)

            # Use TextPreprocessor
            preprocessor = TextPreprocessor(data=data)
            preprocessed_data = preprocessor.preprocess()

            # Save to cache
            preprocessed_data.to_csv(output_path, index=False)
            self.context.log.info(f"💾 Preprocessed data saved to {output_path}")

        return str(output_path)

    async def _run_all_trainers(self, preprocessed_data_path: str) -> Dict[str, float]:
        """Run all trainers and return their scores."""
        trainer_scores = {}

        # CNN-BERT Hybrid
        try:
            self.context.log.info("🧠 Evaluating CNN-BERT...")
            bert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
            preprocessor = TextPreprocessor(data=pd.read_csv(preprocessed_data_path))

            trainer = CNNBertHybridTrainer(
                bert_model=bert_model,
                preprocessor=preprocessor,
                data_path=preprocessed_data_path,
                freeze_bert=True,
            )
            score = float(trainer.run_evaluation(preprocessed_data_path))
            trainer_scores["cnn_bert_hybrid"] = score
            self.context.log.info(f"✅ CNN-BERT Score: {score:.4f}")
        except Exception as e:
            self.context.log.error(f"❌ CNN-BERT evaluation failed: {e}")
            trainer_scores["cnn_bert_hybrid"] = 0.0

        # BERT-LSTM
        try:
            self.context.log.info("🔄 Evaluating BERT-LSTM...")
            model = BERTLSTMModel(
                bert_model_name="vinai/phobert-base-v2",
                hidden_dim1=128,
                hidden_dim2=64,
                dense_dim=64,
                output_dim=3,
                dropout_rate=0.5,
                freeze_bert=True,
            )

            preprocessor = TextPreprocessor(data=pd.read_csv(preprocessed_data_path))
            trainer = BERTLSTMTrainer(
                model=model,
                data_path=preprocessed_data_path,
                preprocessor=preprocessor,
            )
            score = float(trainer.run_evaluation(preprocessed_data_path))
            trainer_scores["bert_lstm"] = score
            self.context.log.info(f"✅ BERT-LSTM Score: {score:.4f}")
        except Exception as e:
            self.context.log.error(f"❌ BERT-LSTM evaluation failed: {e}")
            trainer_scores["bert_lstm"] = 0.0

        # PhoBERT
        try:
            self.context.log.info("🤖 Evaluating PhoBERT...")
            model = AutoModelForSequenceClassification.from_pretrained(
                "vinai/phobert-base-v2", num_labels=3
            )
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            preprocessor = TextPreprocessor(data=pd.read_csv(preprocessed_data_path))

            trainer = PhoBertTrainer(
                model=model,
                data_path=preprocessed_data_path,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
            )
            score = float(trainer.run_evaluation(preprocessed_data_path))
            trainer_scores["phobert"] = score
            self.context.log.info(f"✅ PhoBERT Score: {score:.4f}")
        except Exception as e:
            self.context.log.error(f"❌ PhoBERT evaluation failed: {e}")
            trainer_scores["phobert"] = 0.0

        # SVM
        try:
            self.context.log.info("⚡ Evaluating SVM...")
            trainer = SVMTrainer(data_path=preprocessed_data_path)
            score = float(trainer.run_evaluation())
            trainer_scores["svm"] = score
            self.context.log.info(f"✅ SVM Score: {score:.4f}")
        except Exception as e:
            self.context.log.error(f"❌ SVM evaluation failed: {e}")
            trainer_scores["svm"] = 0.0

        return trainer_scores

    def _calculate_composite_score(self, trainer_scores: Dict[str, float]) -> float:
        """Calculate composite score from individual trainer scores."""
        # Filter out failed trainers (score = 0.0)
        valid_scores = [score for score in trainer_scores.values() if score > 0.0]

        if not valid_scores:
            self.context.log.warning("⚠️ No valid trainer scores available")
            return 0.0

        # Use weighted average (you can customize weights here)
        trainer_weights = {
            "cnn_bert_hybrid": 0.3,
            "bert_lstm": 0.25,
            "phobert": 0.3,
            "svm": 0.15,
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for trainer_name, score in trainer_scores.items():
            if score > 0.0:  # Only include successful trainers
                weight = trainer_weights.get(trainer_name, 0.25)
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        composite_score = weighted_sum / total_weight

        self.context.log.info("📊 Composite Score Calculation:")
        self.context.log.info(f"   🏆 Weighted Average: {composite_score:.4f}")
        self.context.log.info(f"   ✅ Successful trainers: {len(valid_scores)}/4")

        return composite_score

    @property
    def evaluation_type(self) -> str:
        return "heavyweight"
