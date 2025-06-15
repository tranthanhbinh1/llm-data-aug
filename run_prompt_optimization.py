import argparse
import asyncio
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from src.constants import ORIGINAL_DATASET_PATH
from src.preprocess.text_preprocessor import TextPreprocessor
from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.evaluation.strategy_manager import create_evaluation_strategy_manager
from src.evaluation.promptimal_adapter import PromptimalEvaluatorAdapter
from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.utils import get_instructor_instance
from promptimal.optimizer import optimize


def create_evaluators(sentiment: str):
    """Create the similarity and trainer evaluators"""

    # Create similarity evaluator
    similarity_evaluator = SimiarityEvaluator(
        data_synthesizer=AugGptRunner(get_instructor_instance()),
    )

    # Create trainer evaluator
    trainer_evaluator = CNNBertHybridTrainer(
        bert_model=AutoModel.from_pretrained("vinai/phobert-base-v2"),
        preprocessor=TextPreprocessor(data=pd.read_csv(ORIGINAL_DATASET_PATH)),
        data_path=ORIGINAL_DATASET_PATH,
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze_bert=True,
    )

    return similarity_evaluator, trainer_evaluator


def main():
    parser = argparse.ArgumentParser(
        description="Optimize prompts using sophisticated evaluation strategies"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Initial prompt to optimize"
    )
    parser.add_argument(
        "--sentiment",
        type=str,
        default="neutral",
        choices=["neutral", "negative"],
        help="Sentiment to optimize for",
    )
    parser.add_argument(
        "--improve",
        type=str,
        default="",
        help="Description of what to improve about the prompt",
    )
    parser.add_argument(
        "--num_iters", type=int, default=10, help="Number of optimization iterations"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Population size for genetic algorithm",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Score threshold to stop optimization",
    )
    parser.add_argument(
        "--strategy_config",
        type=str,
        help="Path to JSON file with custom evaluation strategy configuration",
    )

    args = parser.parse_args()

    print(f"🚀 Starting prompt optimization for sentiment: {args.sentiment}")
    print(f"📝 Initial prompt: {args.prompt}")

    # Create evaluators
    similarity_evaluator, trainer_evaluator = create_evaluators(args.sentiment)

    # Load strategy configuration if provided
    strategy_config = None
    if args.strategy_config:
        import json

        with open(args.strategy_config, "r") as f:
            strategy_config = json.load(f)

    # Create evaluation strategy manager
    strategy_manager = create_evaluation_strategy_manager(
        sentiment=args.sentiment,
        similarity_evaluator=similarity_evaluator,
        trainer_evaluator=trainer_evaluator,
        strategy_config=strategy_config,
    )

    # Create promptimal adapter
    evaluator_adapter = PromptimalEvaluatorAdapter(strategy_manager)

    # Run optimization
    async def run_optimization():
        best_prompt = args.prompt
        async for step in optimize(
            prompt=args.prompt,
            improvement_request=args.improve,
            num_iters=args.num_iters,
            population_size=args.population_size,
            threshold=args.threshold,
            evaluator=evaluator_adapter,
        ):
            best_prompt = step.best_prompt
            print(f"Step {step.index}: {step.message} (Score: {step.best_score})")

        return best_prompt

    # Run the optimization
    optimized_prompt = asyncio.run(run_optimization())

    # Print results
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"💡 Optimized prompt: {optimized_prompt}")

    # Print evaluation summary
    summary = strategy_manager.get_evaluation_summary()
    print("\n📊 Evaluation Summary:")
    print(f"   Total evaluations: {summary.get('total_evaluations', 0)}")
    for strategy, data in summary.get("strategies_used", {}).items():
        print(
            f"   {strategy}: {data['count']} times, avg score: {data['avg_score']:.4f}"
        )


if __name__ == "__main__":
    main()
