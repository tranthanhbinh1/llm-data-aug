from dotenv import load_dotenv
import argparse

from src.synthesizer.runner import AugGptRunner
from src.utils import get_instructor_instance
from src.evaluation.prompt_evaluator import PromptEvaluator

if __name__ == "__main__":
    """
    Wrapper script to run the prompt evaluator.
    Usage: python run_evaluation.py --sentiment neutral --prompt "Your prompt here"
    """

    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate prompt performance")
    parser.add_argument(
        "--sentiment",
        type=str,
        default="neutral",
        choices=["neutral", "negative"],
        help="Sentiment to evaluate (default: neutral)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Bạn là một trợ lý hữu ích, có nhiệm vụ diễn đạt lại văn bản và làm cho câu văn trở nên mượt mà hơn.",
        help="System prompt for augmentation",
    )

    args = parser.parse_args()

    evaluator = PromptEvaluator(AugGptRunner(get_instructor_instance()))
    average_cosine_similarity = evaluator.main(
        sentiment=args.sentiment,
        prompt=args.prompt,
    )

    print(f"{average_cosine_similarity:.4f}")
