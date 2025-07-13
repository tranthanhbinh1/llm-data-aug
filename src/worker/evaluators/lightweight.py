"""
Lightweight evaluator for fast prompt evaluation.
"""

from typing import Dict, Any
import dagster as dg

from .base import EvaluatorStrategy
from src.enums import Sentiment
from src.worker.helpers import DataHelper, ScoreHelper
from src.worker.resource import SynthesizerResource


class LightweightEvaluator(EvaluatorStrategy):
    """
    Fast evaluation strategy using similarity metrics and small datasets.

    This evaluator:
    - Generates small synthetic datasets (5 samples)
    - Uses similarity evaluation for fitness scoring
    - Completes evaluation in ~30-60 seconds
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
        Evaluate candidate prompt using similarity metrics.

        Args:
            candidate_prompt: The prompt to evaluate
            optimization_state: Current optimization state

        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            self.context.log.info(
                f"🔍 Lightweight evaluation - candidate prompt (length: {len(candidate_prompt)} chars)"
            )

            # Extract sentiment from optimization state
            sentiment = Sentiment(optimization_state["sentiment"])

            # Generate small synthetic dataset
            data_path = DataHelper.generate_synthetic_data(
                auggpt_runner=self.synthesizer.get_synthesizer_instance(),
                prompt=candidate_prompt,
                sentiment=sentiment,
            )

            self.context.log.info(f"📝 Generated small dataset at: {data_path}")

            # Evaluate similarity
            score_data = ScoreHelper.evaluate_similarity(
                data_path=data_path,
                prompt=candidate_prompt,
            )

            similarity_score = score_data["similarity_score"]

            self.context.log.info(
                f"📊 Lightweight evaluation complete - Similarity: {similarity_score:.4f}"
            )

            return similarity_score

        except Exception as e:
            self.context.log.error(f"❌ Lightweight evaluation failed: {e}")
            return 0.0

    @property
    def evaluation_type(self) -> str:
        return "lightweight"
