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
        return {"global_count": 0}

    def _save_counts(self) -> None:
        """Save evaluation counts to file."""
        with open(self.counter_file, "w") as file:
            json.dump(self.counts, file, indent=2)

    def get_global_count(self) -> int:
        """Get current global evaluation count."""
        return self.counts.get("global_count", 0)

    def increment_global_count(self) -> int:
        """Increment and return the global evaluation count."""
        self.counts["global_count"] = self.counts.get("global_count", 0) + 1
        self._save_counts()
        return self.counts["global_count"]


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
        help="Run full evaluation every N evaluations",
    )
    parser.add_argument(
        "--reset-counter", action="store_true", help="Reset the evaluation counter"
    )
    args = parser.parse_args()

    counter = EvaluationCounter()

    if args.reset_counter:
        if os.path.exists(counter.counter_file):
            os.remove(counter.counter_file)
        print("Evaluation counter reset.")
        exit(0)

    # Get current count and increment
    current_count = counter.increment_global_count()
    print(f"Global evaluation run #{current_count} for sentiment: {args.sentiment}")

    # Initialize evaluators
    trainer_evaluator = CNNBertHybridTrainer(
        bert_model=AutoModel.from_pretrained("vinai/phobert-base-v2"),
        preprocessor=TextPreprocessor(data=pd.read_csv(ORIGINAL_DATASET_PATH)),
        data_path=ORIGINAL_DATASET_PATH,
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze_bert=True,
    )

    similarity_evaluator = SimiarityEvaluator(
        auggpt_runner=AugGptRunner(get_instructor_instance()),
    )

    evaluator = Evaluator(
        trainer_evaluator=trainer_evaluator,
        similarity_evaluator=similarity_evaluator,
    )

    # Decide evaluation strategy based on count
    if current_count % args.full_eval_threshold == 0:
        print(
            f"🚀 Running FULL TRAINER evaluation (every {args.full_eval_threshold} evaluations)"
        )

        # Generate full synthetic dataset and run trainer evaluation
        data_path = evaluator.generate_full_synthetic_data(
            sentiment=args.sentiment,
            prompt=args.prompt,
        )
        result = trainer_evaluator.run_evaluation(data_path=data_path)
        print(f"📊 Trainer evaluation result: {result}")

    else:
        print(
            f"⚡ Running SIMILARITY evaluation ({current_count % args.full_eval_threshold}/{args.full_eval_threshold})"
        )

        # Run similarity evaluation only
        result = similarity_evaluator.run_evaluation(
            sentiment=args.sentiment,
            prompt=args.prompt,
        )
        print(f"📊 Similarity evaluation result: {result}")

    print(f"Final result: {result}")
