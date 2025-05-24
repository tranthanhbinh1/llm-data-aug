import os
import asyncio
from typing import Union, Optional, Callable, Tuple, Literal, Dict, Any
from torch.utils.data import DataLoader
from loguru import logger
from src.repositories.trainer import TrainerRepository
from src.constants import DATA_PATH
from torch import nn

# Promptimal imports
from promptimal.optimizer.main import optimize
from promptimal.dtos import PromptCandidate, TokenCount, ProgressStep
from promptimal.app import App


# TODO: Major ovehaul needed to properly implement promptimal and downstream eval suite
class Evaluator:
    def __init__(
        self,
        trainer: Optional[TrainerRepository] = None,
        data_path: Optional[str] = None,
        trainer_config: Optional[Dict[str, Any]] = None,
    ):
        self.trainer = trainer
        self.data_path = data_path
        self.trainer_config = trainer_config or {}

    def create_trainer_evaluator(
        self,
        trainer_class: type[TrainerRepository],
        trainer_kwargs: Dict[str, Any],
        evaluation_method: str = "main",
    ) -> Callable:
        """
        Create an evaluator function that uses a trainer for prompt evaluation.

        Args:
            trainer_class: The trainer class to use (e.g., PhoBertTrainer)
            trainer_kwargs: Arguments to pass to trainer constructor
            evaluation_method: Method name to call on trainer for evaluation

        Returns:
            Async evaluator function compatible with Promptimal
        """

        async def trainer_evaluator(
            candidate: PromptCandidate,
            improvement_request: str,
            initial_prompt: PromptCandidate,
            genai_client,
        ) -> Tuple[PromptCandidate, TokenCount]:
            """
            Evaluate a prompt candidate using the specified trainer.
            """
            if candidate.fitness is not None:
                # Already evaluated
                return candidate, TokenCount(0, 0)

            try:
                logger.info(f"Evaluating prompt with {trainer_class.__name__}")
                logger.info(f"Prompt: {candidate.prompt[:100]}...")

                # Initialize trainer with the prompt-modified data
                # This is where you'd modify your data generation process
                # to use the candidate prompt
                trainer = trainer_class(**trainer_kwargs)

                # Get evaluation score from trainer
                evaluation_method_func = getattr(trainer, evaluation_method)

                # Run trainer evaluation (this should return a score like F1)
                if asyncio.iscoroutinefunction(evaluation_method_func):
                    score = await evaluation_method_func()
                else:
                    # Run in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(None, evaluation_method_func)

                # Normalize score to 0-1 range if needed
                # F1 scores are already in 0-1 range
                candidate.fitness = float(score)

                logger.info(f"Evaluation complete. Score: {candidate.fitness:.4f}")

                # Return with zero token count since we're not using LLM for evaluation
                return candidate, TokenCount(0, 0)

            except Exception as e:
                logger.error(f"Error in trainer evaluation: {str(e)}")
                candidate.fitness = 0.0
                return candidate, TokenCount(0, 0)

        return trainer_evaluator

    def create_custom_evaluator(
        self, evaluation_function: Callable[[str], float]
    ) -> Callable:
        """
        Create an evaluator from a custom evaluation function.

        Args:
            evaluation_function: Function that takes a prompt string and returns a score

        Returns:
            Async evaluator function compatible with Promptimal
        """

        async def custom_evaluator(
            candidate: PromptCandidate,
            improvement_request: str,
            initial_prompt: PromptCandidate,
            genai_client,
        ) -> Tuple[PromptCandidate, TokenCount]:
            if candidate.fitness is not None:
                return candidate, TokenCount(0, 0)

            try:
                logger.info("Evaluating prompt with custom function")

                # Run custom evaluation
                if asyncio.iscoroutinefunction(evaluation_function):
                    score = await evaluation_function(candidate.prompt)
                else:
                    loop = asyncio.get_event_loop()
                    score = await loop.run_in_executor(
                        None, evaluation_function, candidate.prompt
                    )

                candidate.fitness = float(score)
                logger.info(
                    f"Custom evaluation complete. Score: {candidate.fitness:.4f}"
                )

                return candidate, TokenCount(0, 0)

            except Exception as e:
                logger.error(f"Error in custom evaluation: {str(e)}")
                candidate.fitness = 0.0
                return candidate, TokenCount(0, 0)

        return custom_evaluator

    async def optimize_prompt(
        self,
        initial_prompt: str,
        improvement_request: str,
        evaluator_function: Callable,
        population_size: int = 5,
        num_iters: int = 5,
        num_elites: int = 2,
        threshold: float = 1.0,
        api_key: str = "",
    ) -> Tuple[str, float]:
        """
        Run Promptimal optimization with a custom evaluator.

        Args:
            initial_prompt: Starting prompt to optimize
            improvement_request: Description of what to improve
            evaluator_function: Function to evaluate prompt candidates
            population_size: Number of candidates per generation
            num_iters: Maximum number of iterations
            num_elites: Number of top candidates to keep
            threshold: Fitness threshold to stop optimization
            api_key: Google AI API key for prompt generation

        Returns:
            Tuple of (optimized_prompt, final_score)
        """
        logger.info("Starting Promptimal optimization")

        best_prompt = initial_prompt
        best_score = 0.0

        async for step in optimize(
            prompt=initial_prompt,
            improvement_request=improvement_request,
            population_size=population_size,
            num_iters=num_iters,
            num_elites=num_elites,
            threshold=threshold,
            api_key=api_key,
            evaluator=evaluator_function,
        ):
            logger.info(
                f"Step {step.index}: {step.message} "
                f"(Score: {step.best_score:.4f if step.best_score else 'N/A'})"
            )

            if step.best_prompt:
                best_prompt = step.best_prompt
            if step.best_score:
                best_score = step.best_score

        logger.info(f"Optimization complete. Final score: {best_score:.4f}")
        return best_prompt, best_score

    def optimize_with_trainer(
        self,
        initial_prompt: str,
        improvement_request: str,
        trainer_class: type[TrainerRepository],
        trainer_kwargs: Dict[str, Any],
        **optimization_kwargs,
    ) -> Tuple[str, float]:
        """
        Convenient method to optimize prompt using a specific trainer.

        Args:
            initial_prompt: Starting prompt to optimize
            improvement_request: Description of what to improve
            trainer_class: The trainer class to use for evaluation
            trainer_kwargs: Arguments for trainer initialization
            **optimization_kwargs: Additional arguments for optimization

        Returns:
            Tuple of (optimized_prompt, final_score)
        """
        evaluator_func = self.create_trainer_evaluator(
            trainer_class=trainer_class, trainer_kwargs=trainer_kwargs
        )

        return asyncio.run(
            self.optimize_prompt(
                initial_prompt=initial_prompt,
                improvement_request=improvement_request,
                evaluator_function=evaluator_func,
                **optimization_kwargs,
            )
        )

    def run_interactive_optimization(
        self,
        initial_prompt: str,
        improvement_request: str,
        evaluator_function: Callable,
        **optimization_kwargs,
    ) -> Tuple[str, bool]:
        """
        Run Promptimal optimization with interactive UI.

        Args:
            initial_prompt: Starting prompt to optimize
            improvement_request: Description of what to improve
            evaluator_function: Function to evaluate prompt candidates
            **optimization_kwargs: Additional arguments for optimization

        Returns:
            Tuple of (optimized_prompt, was_finished)
        """
        app = App(initial_prompt)

        optimized_prompt, is_finished = app.start(
            improvement_request=improvement_request,
            evaluator=evaluator_function,
            **optimization_kwargs,
        )

        return optimized_prompt, is_finished
