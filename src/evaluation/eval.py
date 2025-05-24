import asyncio
from typing import Optional, Callable, Tuple, Dict, Any
from loguru import logger
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.repositories.trainer import TrainerEvaluatorRepository

# Promptimal imports
from promptimal.optimizer.main import optimize
from promptimal.dtos import PromptCandidate, TokenCount
from promptimal.app import App


class Evaluator:
    def __init__(
        self,
        trainer_evaluator: TrainerEvaluatorRepository,
        similarity_evaluator: SimiarityEvaluator,
        data_path: Optional[str] = None,
        trainer_config: Optional[Dict[str, Any]] = None,
        evaluator_config: Optional[Dict[str, Any]] = None,
    ):
        self.trainer_evaluator = trainer_evaluator
        self.similarity_evaluator = similarity_evaluator
        self.data_path = data_path
        self.trainer_config = trainer_config or {}
        self.evaluator_config = evaluator_config or {}

    def create_hybrid_evaluator(self) -> Callable:
        """
        Create a hybrid evaluator that combines the trainer and similarity evaluators.
        This evaluator has to keep track of its state and iteration count.
        """

        def hybrid_evaluator(sentiment: str, prompt: str) -> float:
            count = 0
            count += 1
            if count % 2 == 0:
                return self.similarity_evaluator.run_evaluation(sentiment, prompt)
            else:
                # TODO: need to implement the LLM Generation process before running this evaluation
                return self.trainer_evaluator.run_evaluation(sentiment, prompt)

        return hybrid_evaluator
