import os
import time
import asyncio
from typing import Optional, Callable, AsyncGenerator
from google import genai
from loguru import logger

from .models import PromptCandidate, OptimizationResult, OptimizationConfig
from .genetic_operations import GeneticOperations


class PromptOptimizer:
    """Main class for prompt optimization using genetic algorithms."""

    def __init__(
        self, api_key: Optional[str] = None, config: Optional[OptimizationConfig] = None
    ):
        self.api_key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not self.api_key:
            raise ValueError("Google AI API key is required")

        self.config = config or OptimizationConfig(
            population_size=5,
            num_iterations=5,
            num_elites=2,
            threshold=1.0,
            tournament_size=3,
            num_evaluation_samples=3,
            model="gemini-2.0-flash",
            max_retries=3,
        )  # Should read this from a config file
        self.genai_client = genai.Client(api_key=self.api_key)
        self.genetic_ops = GeneticOperations(self.genai_client, self.config)

    async def optimize(
        self,
        initial_prompt: str,
        improvement_request: str,
        custom_evaluator: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Optimize a prompt using genetic algorithm.

        Args:
            initial_prompt: The starting prompt to optimize
            improvement_request: Description of what to improve
            custom_evaluator: Optional custom evaluation function
            progress_callback: Optional callback for progress updates

        Returns:
            OptimizationResult with the best prompt and metadata
        """
        start_time = time.time()
        all_candidates = []

        logger.info("Starting prompt optimization")
        logger.info(f"Initial prompt: {initial_prompt}")
        logger.info(f"Improvement request: {improvement_request}")

        # Initialize population
        if progress_callback:
            await progress_callback("Initializing population", 0, 0.1)

        population = await self.genetic_ops.init_population(
            initial_prompt, improvement_request
        )
        all_candidates.extend(population)

        # Initial evaluation
        if progress_callback:
            await progress_callback("Initial evaluation", 0, 0.2)

        population = await self.genetic_ops.evaluate_population(
            population, initial_prompt, improvement_request, custom_evaluator
        )

        # Track best candidate
        best_candidate = max(population, key=lambda c: c.fitness or 0.0)
        convergence_iteration = None

        logger.info(f"Initial best fitness: {best_candidate.fitness}")

        # Evolution loop
        for iteration in range(self.config.num_iterations):
            iteration_start = time.time()

            if progress_callback:
                progress = 0.2 + (iteration / self.config.num_iterations) * 0.8
                await progress_callback(
                    f"Generation {iteration + 1}/{self.config.num_iterations}",
                    iteration + 1,
                    progress,
                )

            logger.info(f"Starting generation {iteration + 1}")

            # Check for convergence
            for candidate in population:
                print(f"Candidate's type: {type(candidate)}")
                print(f"Candidate fitness: {candidate.fitness}")
            current_best = max(population, key=lambda c: c.fitness or 0.0)
            print(f"Current best fitness: {current_best.fitness}")
            print(f"Current best prompt: {current_best.prompt}")
            if current_best.fitness and current_best.fitness >= self.config.threshold:
                logger.info(f"Convergence reached at generation {iteration + 1}")
                convergence_iteration = iteration + 1
                break

            # Selection and reproduction
            elites = self.genetic_ops.select_elites(population)
            offspring = await self.genetic_ops.generate_offspring(
                population, initial_prompt, improvement_request, iteration + 1
            )

            # Create new population
            new_population = elites + offspring
            all_candidates.extend(offspring)

            # Evaluate new population
            new_population = await self.genetic_ops.evaluate_population(
                new_population, initial_prompt, improvement_request, custom_evaluator
            )

            population = new_population

            # Update best candidate
            generation_best = max(population, key=lambda c: c.fitness or 0.0)
            if generation_best.fitness and (
                not best_candidate.fitness
                or generation_best.fitness > best_candidate.fitness
            ):
                best_candidate = generation_best
                logger.info(f"New best fitness: {best_candidate.fitness:.4f}")

            iteration_time = time.time() - iteration_start
            logger.info(
                f"Generation {iteration + 1} completed in {iteration_time:.2f}s"
            )

        execution_time = time.time() - start_time

        # Create result
        result = OptimizationResult(
            best_prompt=best_candidate.prompt,
            best_score=best_candidate.fitness or 0.0,
            initial_prompt=initial_prompt,
            improvement_request=improvement_request,
            total_iterations=self.config.num_iterations,
            total_candidates_evaluated=len(all_candidates),
            execution_time_seconds=execution_time,
            convergence_iteration=convergence_iteration,
            all_candidates=all_candidates,
            metadata=self.config.model_dump(),
        )

        logger.info("Optimization completed")
        logger.info(f"Best prompt: {result.best_prompt}")
        logger.info(f"Best score: {result.best_score:.4f}")
        logger.info(f"Total time: {execution_time:.2f}s")

        return result

    async def optimize_with_progress(
        self,
        initial_prompt: str,
        improvement_request: str,
        custom_evaluator: Optional[Callable] = None,
    ) -> AsyncGenerator[tuple[str, int, float, Optional[float]], None]:
        """
        Optimize with progress updates yielded as generator.

        Yields:
            Tuple of (message, iteration, progress, best_score)
        """
        progress_updates = []

        async def progress_callback(
            message: str,
            iteration: int,
            progress: float,
            best_score: Optional[float] = None,
        ):
            progress_updates.append((message, iteration, progress, best_score))

        # Start optimization in background
        optimization_task = asyncio.create_task(
            self.optimize(
                initial_prompt, improvement_request, custom_evaluator, progress_callback
            )
        )

        # Yield progress updates
        last_update_count = 0
        while not optimization_task.done():
            if len(progress_updates) > last_update_count:
                for update in progress_updates[last_update_count:]:
                    yield update
                last_update_count = len(progress_updates)
            await asyncio.sleep(0.1)

        # Yield any remaining updates
        for update in progress_updates[last_update_count:]:
            yield update
