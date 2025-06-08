import random
import asyncio
from statistics import mean
from typing import List, Callable, Optional
import instructor
from pydantic import BaseModel, Field
from google import genai
from loguru import logger

from .models import PromptCandidate, OptimizationConfig
from .prompts import INIT_POPULATION_PROMPT, EVAL_PROMPT, CROSSOVER_PROMPT


# Pydantic models for structured output
class BetterPrompts(BaseModel):
    prompts: List[str] = Field(
        description="A list of prompts that are better versions of the provided prompt."
    )


class PromptEvaluation(BaseModel):
    evaluation: str = Field(description="Justification for your score.")
    score: float = Field(
        description="A score between 1-10 for the prompt, with 10 being the highest."
    )


class PromptCrossover(BaseModel):
    analysis: str = Field(description="Your step-by-step analysis of the two prompts.")
    prompt: str = Field(description="The combined and improved prompt.")


class GeneticOperations:
    """Handles genetic algorithm operations for prompt optimization."""

    def __init__(self, genai_client: genai.Client, config: OptimizationConfig):
        self.genai_client = genai_client
        self.config = config
        self.instructor_client = instructor.from_genai(genai_client, use_async=True)

    async def init_population(
        self, initial_prompt: str, improvement_request: str
    ) -> List[PromptCandidate]:
        """Initialize a population of candidate prompts."""
        logger.info(f"Initializing population of size {self.config.population_size}")

        system_message = {
            "role": "system",
            "content": INIT_POPULATION_PROMPT.format(
                population_size=self.config.population_size,
                improvement_request=improvement_request,
            ),
        }
        user_message = {
            "role": "user",
            "content": f"Generate {self.config.population_size} better versions of the following prompt:\n\n<prompt>\n{initial_prompt}\n</prompt>",
        }

        try:
            response = await self.instructor_client.chat.completions.create(
                messages=[system_message, user_message],
                model=self.config.model,
                temperature=self.config.temperature,
                response_model=BetterPrompts,
            )

            # Create population with initial prompt + generated prompts
            population = [
                PromptCandidate(
                    prompt=initial_prompt, generation=0, fitness=None, reflection=None
                )
            ]
            for i, prompt in enumerate(response.prompts):
                population.append(
                    PromptCandidate(
                        prompt=prompt, generation=0, fitness=None, reflection=None
                    )
                )

            logger.info(f"Generated {len(population)} candidates")
            return population

        except Exception as e:
            logger.error(f"Error initializing population: {e}")
            # Fallback to just the initial prompt
            return [
                PromptCandidate(
                    prompt=initial_prompt, generation=0, fitness=None, reflection=None
                )
            ]

    async def evaluate_fitness(
        self,
        candidate: PromptCandidate,
        initial_prompt: str,
        improvement_request: str,
        custom_evaluator: Optional[Callable] = None,
    ) -> PromptCandidate:
        """Evaluate fitness of a prompt candidate."""

        # Skip if already evaluated (elite from previous generation)
        if candidate.fitness is not None:
            return candidate

        # Use custom evaluator if provided
        if custom_evaluator:
            try:
                result = await custom_evaluator(
                    candidate, initial_prompt, improvement_request
                )
                if isinstance(result, tuple):
                    return result[0]  # Return just the candidate, ignore token count
                else:
                    candidate.fitness = result
                    return candidate
            except Exception as e:
                logger.error(f"Custom evaluator failed: {e}")
                candidate.fitness = 0.0
                return candidate

        # Default LLM-based evaluation with self-consistency
        return await self._llm_evaluate_fitness(
            candidate, initial_prompt, improvement_request
        )

    async def _llm_evaluate_fitness(
        self, candidate: PromptCandidate, initial_prompt: str, improvement_request: str
    ) -> PromptCandidate:
        """Evaluate fitness using LLM with self-consistency."""

        messages = [
            {
                "role": "system",
                "content": EVAL_PROMPT.format(
                    initial_prompt=initial_prompt,
                    improvement_request=improvement_request,
                ),
            },
            {
                "role": "user",
                "content": f"Evaluate the following prompt:\n\n<prompt>\n{candidate.prompt}\n</prompt>",
            },
        ]

        evaluations = []

        # Generate multiple evaluations for self-consistency
        for _ in range(self.config.num_evaluation_samples):
            try:
                eval_response = await self.instructor_client.chat.completions.create(
                    messages=messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    response_model=PromptEvaluation,
                )

                evaluations.append(eval_response)

            except Exception as e:
                logger.warning(f"Evaluation attempt failed: {e}")
                # Add a default low score for failed evaluations
                evaluations.append(
                    PromptEvaluation(evaluation="Failed to evaluate", score=1.0)
                )

        if evaluations:
            # Normalize score to 0-1 range and take mean
            candidate.fitness = (
                mean(eval_response.score for eval_response in evaluations) / 10.0
            )
            candidate.reflection = evaluations[0].evaluation
        else:
            candidate.fitness = 0.0
            candidate.reflection = "Evaluation failed"

        return candidate

    def select_parent(self, population: List[PromptCandidate]) -> PromptCandidate:
        """Select a parent using tournament selection."""
        tournament = random.sample(
            population, min(self.config.tournament_size, len(population))
        )
        return max(tournament, key=lambda candidate: candidate.fitness or 0.0)

    async def crossover(
        self,
        parent1: PromptCandidate,
        parent2: PromptCandidate,
        initial_prompt: str,
        improvement_request: str,
        generation: int,
    ) -> PromptCandidate:
        """Create offspring by crossing over two parent prompts."""

        system_message = {
            "role": "system",
            "content": CROSSOVER_PROMPT.format(
                initial_prompt=initial_prompt, improvement_request=improvement_request
            ),
        }
        user_message = {
            "role": "user",
            "content": f"Combine the following prompts into a better one:\n\n<prompt_1>\n{parent1.prompt}\n</prompt_1>\n\n<prompt_2>\n{parent2.prompt}\n</prompt_2>",
        }

        try:
            response = await self.instructor_client.chat.completions.create(
                messages=[system_message, user_message],
                model=self.config.model,
                temperature=self.config.temperature,
                response_model=PromptCrossover,
            )

            # Create child candidate
            child = PromptCandidate(
                prompt=response.prompt,
                generation=generation,
                parent_ids=[str(id(parent1)), str(id(parent2))],
                fitness=None,
                reflection=None,
            )

            return child

        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            # Fallback: return a copy of the better parent
            better_parent = (
                parent1 if (parent1.fitness or 0) > (parent2.fitness or 0) else parent2
            )
            child = PromptCandidate(
                prompt=better_parent.prompt,
                generation=generation,
                parent_ids=[str(id(better_parent))],
                fitness=None,
                reflection=None,
            )
            return child

    async def evaluate_population(
        self,
        population: List[PromptCandidate],
        initial_prompt: str,
        improvement_request: str,
        custom_evaluator: Optional[Callable] = None,
    ) -> List[PromptCandidate]:
        """Evaluate fitness for all candidates in population."""

        # Create evaluation tasks
        tasks = [
            self.evaluate_fitness(
                candidate, initial_prompt, improvement_request, custom_evaluator
            )
            for candidate in population
        ]

        # Execute evaluations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluated_population = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Evaluation failed for candidate {i}: {result}")
                # Set a default low fitness for failed evaluations
                population[i].fitness = 0.0
                evaluated_population.append(population[i])
            else:
                evaluated_population.append(result)

        return evaluated_population

    def select_elites(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
        """Select elite candidates for next generation."""
        # Sort by fitness (descending) and take top elites
        sorted_population = sorted(
            population, key=lambda candidate: candidate.fitness or 0.0, reverse=True
        )
        return sorted_population[: self.config.num_elites]

    async def generate_offspring(
        self,
        population: List[PromptCandidate],
        initial_prompt: str,
        improvement_request: str,
        generation: int,
    ) -> List[PromptCandidate]:
        """Generate offspring through crossover."""

        num_offspring = self.config.population_size - self.config.num_elites

        # Create crossover tasks
        tasks = []
        for _ in range(num_offspring):
            parent1 = self.select_parent(population)
            parent2 = self.select_parent(population)
            tasks.append(
                self.crossover(
                    parent1, parent2, initial_prompt, improvement_request, generation
                )
            )

        # Execute crossovers concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        offspring = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Crossover failed for offspring {i}: {result}")
                # Fallback: create a mutated copy of a random parent
                parent = self.select_parent(population)
                child = PromptCandidate(
                    prompt=parent.prompt,
                    generation=generation,
                    parent_ids=[str(id(parent))],
                    fitness=None,
                    reflection=None,
                )
                offspring.append(child)
            else:
                offspring.append(result)

        return offspring
