import asyncio
import pandas as pd
from typing import Optional, Callable, Tuple, Dict, Any
from loguru import logger as logging
from src.constants import LABEL_MAPPING, NUM_REPHRASED_SENTENCES, ORIGINAL_DATASET_PATH
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.repositories.trainer import TrainerEvaluatorRepository

# Promptimal imports
from promptimal.optimizer.main import optimize
from promptimal.dtos import PromptCandidate, TokenCount
from promptimal.app import App
from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.synthesizer.generator import DataGenerator
from src.synthesizer.models import AugmentedUserReviews, SentimentPrompt, UserReviews
from src.utils import get_instructor_instance
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class Evaluator:
    def __init__(
        self,
        trainer_evaluator: TrainerEvaluatorRepository,
        similarity_evaluator: SimiarityEvaluator,
        trainer_config: Optional[Dict[str, Any]] = None,
        evaluator_config: Optional[Dict[str, Any]] = None,
        original_data_path: Optional[str] = ORIGINAL_DATASET_PATH,
        data_generator: AugGptRunner = AugGptRunner(get_instructor_instance()),
    ):
        self.trainer_evaluator = trainer_evaluator
        self.similarity_evaluator = similarity_evaluator
        self.trainer_config = trainer_config or {}
        self.evaluator_config = evaluator_config or {}
        self.original_data = pd.read_csv(original_data_path)
        self.data_generator = data_generator

    def random_split(
        self,
        sentiment: str,
        test_size: float = 0.05,
    ) -> pd.DataFrame:
        data = self.original_data.copy()
        data["Sentiment"] = data["Sentiment"].map(LABEL_MAPPING)

        # Filter first, then sample
        subset = data[data["Sentiment"] == DataGenerator.SENTIMENT_MAPPING[sentiment]]
        sampled_subset = subset.sample(frac=test_size, random_state=42)

        logging.info(f"Total {sentiment} records: {len(subset)}")
        logging.info(f"Sampled subset size: {len(sampled_subset)}")
        return sampled_subset

    def generate_synthetic_data(
        self, sentiment: str, prompt: str
    ) -> tuple[list[AugmentedUserReviews | UserReviews], list[str], list[str]]:
        subset = self.random_split(sentiment=sentiment)
        if subset.empty:
            raise ValueError(f"No {sentiment} records found in the dataset")

        _original_sentences, _original_sentence_prompts = (
            self.data_generator.prepare_original_sentences(
                sentiment=sentiment,
                data=subset,
            )
        )

        if not _original_sentences:
            raise ValueError("No sentences were prepared for generation")

        synthesized_records, original_sentences, failed_sentences = (
            self.data_generator._generate_reviews(
                sentiment=sentiment,
                user_prompt=SentimentPrompt.AUG_GPT_PROMPT,
                augmentor_prompt=ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt,
                ),
                num_to_generate=NUM_REPHRASED_SENTENCES,
                original_sentences=_original_sentences,
                original_sentence_prompts=_original_sentence_prompts,
            )
        )
        # TODO: might need to tweak this return output to make it more straightforward
        return synthesized_records, original_sentences, failed_sentences

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
                synthesized_records, original_sentences, failed_sentences = (
                    self.generate_synthetic_data(sentiment, prompt)
                )
                return self.trainer_evaluator.run_evaluation(
                    sentiment, prompt, synthesized_records, original_sentences
                )

        return hybrid_evaluator
