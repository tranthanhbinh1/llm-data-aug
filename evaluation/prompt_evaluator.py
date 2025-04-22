## This is a script to evaluate the performance of the prompt
## Steps:
## 1. Load the original dataset
## 2. Split the dataset into random subsets
## 3. Randomly select one subset as the test set
## 4. Use the Prompt to generate synthetic data (reviews) with the test set as examples
## 5. Perform pairwise comparison using Cosine Similarity between the original and synthetic data
## 6. Calculate the average similarity score
## 7. Report the average similarity score


from typing import Literal
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from constants import NUM_REPHRASED_SENTENCES
from synthesizer.generator import DataGenerator
from synthesizer.models import (
    AugmentedUserReview,
    SentimentPrompt,
    UserReview,
)
from synthesizer.runner import AugGptRunner


class PromptEvaluator:
    def __init__(
        self,
        auggpt_runner: AugGptRunner,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self._original_data = pd.read_csv("data/cleaned_user_reviews.csv")
        self._auggpt_runner = auggpt_runner

    def random_split(
        self,
        sentiment: Literal["neutral", "negative"],
        test_size: float = 0.05,
    ) -> pd.DataFrame:
        label_mapping = {"Positive": 1, "Neutral": 2, "Negative": 0}
        data = self._original_data.copy()
        data["Sentiment"] = data["Sentiment"].map(label_mapping)

        # Filter first, then sample
        subset = data[data["Sentiment"] == DataGenerator.SENTIMENT_MAPPING[sentiment]]
        sampled_subset = subset.sample(frac=test_size, random_state=42)

        logger.info(f"Total {sentiment} records: {len(subset)}")
        logger.info(f"Sampled subset size: {len(sampled_subset)}")
        return sampled_subset

    def generate_synthetic_data(
        self, sentiment: Literal["neutral", "negative"]
    ) -> tuple[list[str], list[str]]:
        subset = self.random_split(sentiment=sentiment)
        if subset.empty:
            raise ValueError(f"No {sentiment} records found in the dataset")

        _original_sentences, _original_sentence_prompts = (
            self._auggpt_runner.prepare_original_sentences(
                sentiment=sentiment,
                data=subset,
            )
        )

        if not _original_sentences:
            raise ValueError("No sentences were prepared for generation")

        synthesized_sentences, original_sentences, failed_sentences = (
            self._auggpt_runner._generate_reviews(
                sentiment=sentiment,
                user_prompt=SentimentPrompt.AUG_GPT_PROMPT,
                num_to_generate=NUM_REPHRASED_SENTENCES,
                original_sentences=_original_sentences,
                original_sentence_prompts=_original_sentence_prompts,
            )
        )

        if failed_sentences:
            logger.warning(f"Failed to generate for {len(failed_sentences)} sentences")

        _synthesized_records: list[list[AugmentedUserReview] | list[UserReview]] = [
            record.reviews for record in synthesized_sentences
        ]

        synthesized_records: list[str] = [
            review.review
            for review in [
                review for reviews in _synthesized_records for review in reviews
            ]
        ]

        logger.info(f"Generated {len(synthesized_records)} synthetic reviews")
        return original_sentences, synthesized_records

    def evaluate(
        self, original_records: list[str], synthesized_records: list[str]
    ) -> float:
        if not original_records or not synthesized_records:
            raise ValueError("Cannot evaluate empty records")

        logger.info(
            f"Evaluating {len(original_records)} original records against {len(synthesized_records)} synthetic records"
        )

        # Get embeddings
        original_records_embeddings = self.model.encode(original_records)
        synthesized_records_embeddings = self.model.encode(synthesized_records)

        # Convert to tensors
        original_records_tensor = torch.tensor(original_records_embeddings)
        synthesized_records_tensor = torch.tensor(synthesized_records_embeddings)

        # Reshape tensors for pairwise comparison
        original_expanded = original_records_tensor.unsqueeze(1)  # [n, 1, d]
        synthetic_expanded = synthesized_records_tensor.unsqueeze(0)  # [1, m, d]

        # Calculate pairwise similarities
        similarities = torch.nn.functional.cosine_similarity(
            original_expanded, synthetic_expanded, dim=2
        )  # [n, m]

        # Calculate mean similarity
        mean_similarity = similarities.mean().item()
        logger.info(f"Average similarity score: {mean_similarity:.4f}")

        return mean_similarity


if __name__ == "__main__":
    from google import genai
    import instructor
    import os

    instance = instructor.from_genai(
        genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    )
    evaluator = PromptEvaluator(AugGptRunner(instance))

    original_sentences, synthesized_records = evaluator.generate_synthetic_data(
        sentiment="neutral"
    )

    evaluator.evaluate(original_sentences, synthesized_records)
