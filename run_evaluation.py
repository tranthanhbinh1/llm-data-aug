import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from src.constants import ORIGINAL_DATASET_PATH
from src.preprocess.text_preprocessor import TextPreprocessor
import argparse
from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.utils import get_instructor_instance
from src.evaluation.eval import Evaluator
import pandas as pd


class EvaluationCounter:
    """Manages evaluation run counts with persistent storage."""

    def __init__(self, counter_file: str = "evaluation_counter.json"):
        self.counter_file = counter_file
        self.counts = self._load_counts()

    def _load_counts(self) -> dict:
        """Load evaluation counts from file."""
        if os.path.exists(self.counter_file):
            with open(self.counter_file, "r") as file:
                return json.load(file)
        return {}

    def _save_counts(self) -> None:
        """Save evaluation counts to file."""
        with open(self.counter_file, "w") as file:
            json.dump(self.counts, file, indent=2)

    def get_count(self, sentiment: str, prompt_hash: str) -> int:
        """Get current count for a sentiment-prompt combination."""
        key = f"{sentiment}_{prompt_hash}"
        return self.counts.get(key, 0)

    def increment_count(self, sentiment: str, prompt_hash: str) -> int:
        """Increment and return the count for a sentiment-prompt combination."""
        key = f"{sentiment}_{prompt_hash}"
        self.counts[key] = self.counts.get(key, 0) + 1
        self._save_counts()
        return self.counts[key]


def get_prompt_hash(prompt: str) -> str:
    """Generate a simple hash for the prompt to use as identifier."""
    return str(hash(prompt) % 10000)


# Wrapper script to run the evaluator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", type=str, required=False, default="neutral")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--full-eval-threshold",
        type=int,
        required=False,
        default=5,
        help="Number of subset evaluations before switching to full evaluation",
    )
    parser.add_argument(
        "--reset-counter", action="store_true", help="Reset the evaluation counter"
    )
    args = parser.parse_args()

    # Initialize counter
    counter = EvaluationCounter()

    if args.reset_counter:
        if os.path.exists(counter.counter_file):
            os.remove(counter.counter_file)
        print("Evaluation counter reset.")
        exit(0)

    trainer_evaluator = CNNBertHybridTrainer(
        bert_model=AutoModel.from_pretrained("vinai/phobert-base-v2"),
        preprocessor=TextPreprocessor(data=pd.read_csv(ORIGINAL_DATASET_PATH)),
        data_path=ORIGINAL_DATASET_PATH,
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze_bert=True,
    )

    evaluator = Evaluator(
        trainer_evaluator=trainer_evaluator,
        similarity_evaluator=SimiarityEvaluator(
            auggpt_runner=AugGptRunner(get_instructor_instance()),
        ),
    )

    # Get current count and increment
    prompt_hash = get_prompt_hash(args.prompt)
    current_count = counter.increment_count(args.sentiment, prompt_hash)

    print(f"Evaluation run #{current_count} for sentiment: {args.sentiment}")

    # Use hybrid evaluator based on count
    if current_count >= args.full_eval_threshold:
        print(f"Running FULL evaluation (threshold {args.full_eval_threshold} reached)")
        hybrid_eval = evaluator.create_hybrid_evaluator(args.sentiment, args.prompt)
        result = hybrid_eval(current_count)
    else:
        print(f"Running SUBSET evaluation ({current_count}/{args.full_eval_threshold})")
        result = evaluator.evaluate(
            sentiment=args.sentiment,
            prompt=args.prompt,
        )

    print(f"Evaluation result: {result}")
