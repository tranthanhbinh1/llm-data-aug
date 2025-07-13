"""
Abstract base class for prompt evaluators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import dagster as dg


class EvaluatorStrategy(ABC):
    """
    Abstract base class for prompt evaluation strategies.

    This allows us to plug in different evaluation methods:
    - Lightweight: Fast evaluation with small datasets and similarity metrics
    - Heavyweight: Full evaluation with complete datasets and trainer models
    """

    def __init__(self, context: dg.OpExecutionContext):
        self.context = context

    @abstractmethod
    async def evaluate(
        self, candidate_prompt: str, optimization_state: Dict[str, Any]
    ) -> float:
        """
        Evaluate a candidate prompt and return a fitness score.

        Args:
            candidate_prompt: The prompt to evaluate
            optimization_state: Current optimization state with configuration

        Returns:
            float: Fitness score (higher is better)
        """
        pass

    @property
    @abstractmethod
    def evaluation_type(self) -> str:
        """Return the type of evaluation (lightweight/heavyweight)"""
        pass
