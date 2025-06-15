from typing import Tuple, Optional
import asyncio
from promptimal.dtos import PromptCandidate, TokenCount
from src.evaluation.strategy_manager import EvaluationStrategyManager


# TODO: watch this one
class PromptimalEvaluatorAdapter:
    """Adapter to connect EvaluationStrategyManager with promptimal's evaluation interface"""

    def __init__(self, strategy_manager: EvaluationStrategyManager):
        self.strategy_manager = strategy_manager
        self.current_iteration = 0

    async def __call__(
        self,
        candidate: PromptCandidate,
        initial_prompt: PromptCandidate,
        improvement_request: str,
        genai_client,  # This parameter is required by promptimal but we don't use it
    ) -> Tuple[PromptCandidate, TokenCount]:
        """
        Evaluate a prompt candidate using the strategy manager.
        This method signature matches what promptimal expects.
        """

        # Skip if already evaluated (elite from previous generation)
        if candidate.fitness is not None:
            return candidate, TokenCount(0, 0)

        try:
            # Use the strategy manager to evaluate the prompt
            score = await self.strategy_manager.evaluate_prompt(
                prompt=candidate.prompt, iteration=self.current_iteration
            )

            # Ensure score is in [0, 1] range expected by promptimal
            candidate.fitness = max(0.0, min(1.0, score))
            candidate.reflection = f"Evaluated using {self.strategy_manager.get_active_strategy(self.current_iteration).value} strategy"  # NOTE: I deleted the manager

        except Exception as e:
            # Fallback to low score if evaluation fails
            candidate.fitness = 0.0
            candidate.reflection = f"Evaluation failed: {str(e)}"

        # No token usage since we're using custom evaluators
        return candidate, TokenCount(0, 0)

    def advance_iteration(self):
        """Call this when moving to the next iteration"""
        self.current_iteration += 1

    def reset_iteration(self):
        """Reset iteration counter"""
        self.current_iteration = 0
