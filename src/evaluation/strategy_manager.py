from enum import Enum
import time
from typing import Dict, Any, Optional, Callable, List
from src.evaluation.eval import Evaluator
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from pydantic import BaseModel
from loguru import logger as logging


class EvaluationStrategy(Enum):
    """Available evaluation strategies"""

    SIMILARITY = "similarity"
    TRAINER = "trainer"
    HYBRID = "hybrid"
    LLM_JUDGE = "llm_judge"


class EvaluationRule(BaseModel):
    """Defines when to switch evaluation strategies"""

    iteration_start: int
    iteration_end: Optional[int] = None
    strategy: EvaluationStrategy = EvaluationStrategy.SIMILARITY
    frequency: int = 1  # How often to apply this strategy

    def applies_to_iteration(self, iteration: int) -> bool:
        """Check if this rule applies to the given iteration"""
        if iteration < self.iteration_start:
            return False
        if self.iteration_end is not None and iteration > self.iteration_end:
            return False
        return (iteration - self.iteration_start) % self.frequency == 0


class EvaluationStrategyManager:
    """Manages switching between different evaluation strategies during optimization"""

    def __init__(
        self,
        similarity_evaluator: SimiarityEvaluator,
        trainer_evaluator: Any,  # TrainerEvaluatorRepository
        sentiment: str,
        evaluation_rules: List[EvaluationRule],
        llm_judge_evaluator: Optional[Callable] = None,
    ):
        self.similarity_evaluator = similarity_evaluator
        self.trainer_evaluator = trainer_evaluator
        self.sentiment = sentiment
        self.evaluation_rules = sorted(
            evaluation_rules, key=lambda r: r.iteration_start
        )
        self.llm_judge_evaluator = llm_judge_evaluator

        # State tracking
        self.iteration_count = 0
        self.evaluation_history: List[Dict[str, Any]] = []
        self.cached_full_datasets: Dict[str, str] = {}  # prompt -> data_path

    def get_active_strategy(self, iteration: int) -> EvaluationStrategy:
        """Determine which evaluation strategy to use for this iteration"""
        for rule in reversed(self.evaluation_rules):  # Check from most recent rules
            if rule.applies_to_iteration(iteration):
                return rule.strategy

        # Default to similarity if no rules match
        return EvaluationStrategy.SIMILARITY

    async def evaluate_prompt(self, prompt: str, iteration: int) -> float:
        """Evaluate a prompt using the appropriate strategy for this iteration"""
        strategy = self.get_active_strategy(iteration)

        logging.info(
            f"Iteration {iteration}: Using {strategy.value} evaluation strategy"
        )

        start_time = time.time()
        score = await self._execute_strategy(strategy, prompt, iteration)
        execution_time = time.time() - start_time

        # Track evaluation history
        self.evaluation_history.append(
            {
                "iteration": iteration,
                "strategy": strategy.value,
                "prompt": prompt,
                "score": score,
                "execution_time": execution_time,
            }
        )

        logging.info(
            f"Evaluation complete: {strategy.value} strategy gave score {score:.4f} "
            f"in {execution_time:.2f}s"
        )

        return score

    async def _execute_strategy(
        self, strategy: EvaluationStrategy, prompt: str, iteration: int
    ) -> float:
        """Execute the specific evaluation strategy"""

        if strategy == EvaluationStrategy.SIMILARITY:
            return self.similarity_evaluator.run_evaluation(self.sentiment, prompt)

        elif strategy == EvaluationStrategy.TRAINER:
            # Check if we have cached full dataset for this prompt
            if prompt in self.cached_full_datasets:
                data_path = self.cached_full_datasets[prompt]
                logging.info(f"Using cached dataset for prompt: {data_path}")
            else:
                # Generate full synthetic dataset
                evaluator = Evaluator(
                    trainer_evaluator=self.trainer_evaluator,
                    similarity_evaluator=self.similarity_evaluator,
                )
                data_path = evaluator.generate_full_synthetic_data(
                    sentiment=self.sentiment, prompt=prompt
                )
                self.cached_full_datasets[prompt] = data_path
                logging.info(f"Generated new full dataset: {data_path}")

            return self.trainer_evaluator.run_evaluation(data_path=data_path)

        elif strategy == EvaluationStrategy.HYBRID:
            # Use a weighted combination of similarity and trainer evaluations
            sim_score = self.similarity_evaluator.run_evaluation(self.sentiment, prompt)

            # Only run trainer evaluation every few iterations to save compute
            if iteration % 3 == 0:  # Run trainer eval every 3rd iteration
                if prompt not in self.cached_full_datasets:
                    evaluator = Evaluator(
                        trainer_evaluator=self.trainer_evaluator,
                        similarity_evaluator=self.similarity_evaluator,
                    )
                    data_path = evaluator.generate_full_synthetic_data(
                        sentiment=self.sentiment, prompt=prompt
                    )
                    self.cached_full_datasets[prompt] = data_path

                trainer_score = self.trainer_evaluator.run_evaluation(
                    data_path=self.cached_full_datasets[prompt]
                )

                # Weighted combination (adjust weights as needed)
                combined_score = 0.3 * sim_score + 0.7 * trainer_score
                logging.info(
                    f"Hybrid evaluation: similarity={sim_score:.4f}, "
                    f"trainer={trainer_score:.4f}, combined={combined_score:.4f}"
                )
                return combined_score
            else:
                return sim_score

        elif strategy == EvaluationStrategy.LLM_JUDGE:
            if self.llm_judge_evaluator is None:
                logging.warning(
                    "LLM judge evaluator not provided, falling back to similarity"
                )
                return self.similarity_evaluator.run_evaluation(self.sentiment, prompt)
            return await self.llm_judge_evaluator(prompt)

        else:
            raise ValueError(f"Unknown evaluation strategy: {strategy}")

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluations performed"""
        if not self.evaluation_history:
            return {}

        strategies_used = {}
        for eval_record in self.evaluation_history:
            strategy = eval_record["strategy"]
            if strategy not in strategies_used:
                strategies_used[strategy] = {
                    "count": 0,
                    "total_time": 0,
                    "avg_score": 0,
                    "scores": [],
                }
            strategies_used[strategy]["count"] += 1
            strategies_used[strategy]["total_time"] += eval_record["execution_time"]
            strategies_used[strategy]["scores"].append(eval_record["score"])

        # Calculate averages
        for strategy_data in strategies_used.values():
            strategy_data["avg_time"] = (
                strategy_data["total_time"] / strategy_data["count"]
            )
            strategy_data["avg_score"] = sum(strategy_data["scores"]) / len(
                strategy_data["scores"]
            )

        return {
            "total_evaluations": len(self.evaluation_history),
            "strategies_used": strategies_used,
            "evaluation_history": self.evaluation_history,
        }


def create_evaluation_strategy_manager(
    sentiment: str,
    similarity_evaluator: SimiarityEvaluator,
    trainer_evaluator: Any,
    strategy_config: Optional[Dict[str, Any]] = None,
) -> EvaluationStrategyManager:
    """Factory function to create evaluation strategy manager with common configurations"""

    if strategy_config is None:
        # Default configuration: start with similarity, switch to hybrid after iteration 3,
        # use trainer evaluation every 5th iteration starting from iteration 7
        evaluation_rules = [
            EvaluationRule(
                iteration_start=0,
                iteration_end=2,
                strategy=EvaluationStrategy.SIMILARITY,
                frequency=1,
            ),
            EvaluationRule(
                iteration_start=3,
                iteration_end=6,
                strategy=EvaluationStrategy.HYBRID,
                frequency=1,
            ),
            EvaluationRule(
                iteration_start=7,
                strategy=EvaluationStrategy.TRAINER,
                frequency=2,  # Every other iteration
            ),
        ]
    else:
        logging.info(f"Using custom strategy configuration: {strategy_config}")
        # Parse custom configuration
        evaluation_rules = []
        for rule_config in strategy_config.get("rules", []):
            evaluation_rules.append(
                EvaluationRule(
                    iteration_start=rule_config["iteration_start"],
                    iteration_end=rule_config.get("iteration_end"),
                    strategy=EvaluationStrategy(rule_config["strategy"]),
                    frequency=rule_config.get("frequency", 1),
                )
            )

    return EvaluationStrategyManager(
        similarity_evaluator=similarity_evaluator,
        trainer_evaluator=trainer_evaluator,
        sentiment=sentiment,
        evaluation_rules=evaluation_rules,
    )
