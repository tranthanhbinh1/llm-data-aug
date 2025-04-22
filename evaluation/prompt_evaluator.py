## This is a script to evaluate the performance of the prompt
## Steps:
## 1. Load the original dataset
## 2. Split the dataset into random subsets
## 3. Randomly select one subset as the test set
## 4. Use the Prompt to generate synthetic data (reviews) with the test set as examples
## 5. Perform pairwise comparison using Cosine Similarity between the original and synthetic data
## 6. Calculate the average similarity score
## 7. Report the average similarity score


from typing import Literal, cast
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from constants import NUM_REPHRASED_SENTENCES
from synthesizer.generator import DataGenerator
from synthesizer.models import (
    AugmentedUserReview,
    AugmentedUserReviews,
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
    ) -> dict[str, list[str]]:
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

        synthesized_records, original_sentences, failed_sentences = (
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

        # Create a mapping between an original sentence and its corresponding records
        _original_sentence_to_records: dict[str, AugmentedUserReviews] = dict()
        for original_sentence, synthesized_record in zip(
            original_sentences, synthesized_records
        ):
            _original_sentence_to_records[original_sentence] = cast(
                AugmentedUserReviews, synthesized_record
            )

        # For each AugmentedUserReviews object in the dict, we extract the reviews
        original_sentence_to_synthesized_reviews: dict[str, list[str]] = {
            original_sentence: [review.review for review in augmented_reviews.reviews]
            for original_sentence, augmented_reviews in _original_sentence_to_records.items()
        }

        logger.info(f"Generated {len(synthesized_records)} synthetic reviews")
        return original_sentence_to_synthesized_reviews

    def create_emebeddings(self, sentence_to_synthesized_reviews: dict[str, list[str]]):
        sentence_to_synthesized_reviews_embeddings: dict[str, torch.Tensor] = dict()
        for sentence, reviews in sentence_to_synthesized_reviews.items():
            embeddings = self.model.encode(reviews)
            sentence_to_synthesized_reviews_embeddings[sentence] = torch.tensor(
                embeddings
            )

        sentence_to_embeddings: dict[str, torch.Tensor] = dict()
        for sentence, _ in sentence_to_synthesized_reviews_embeddings.items():
            embeddings = self.model.encode(sentence)
            sentence_to_embeddings[sentence] = torch.tensor(embeddings)

        return sentence_to_synthesized_reviews_embeddings, sentence_to_embeddings

    def evaluate(
        self,
        sentence_to_synthesized_reviews_embeddings: dict[str, torch.Tensor],
        sentence_to_embeddings: dict[str, torch.Tensor],
    ) -> float:
        cosine_similarity_scores: list[float] = []
        for sentence, original_embedding in sentence_to_embeddings.items():
            # Get synthetic reviews embeddings for this sentence
            synthetic_embeddings = sentence_to_synthesized_reviews_embeddings[sentence]

            # Calculate similarity between original sentence and each synthetic version
            similarities = torch.nn.functional.cosine_similarity(
                synthetic_embeddings,  # Shape: [num_synthetic, embedding_dim]
                original_embedding,  # Shape: [embedding_dim]
                dim=1,  # Compare along embedding dimension
            )

            # Calculate statistics for this sentence
            mean_similarity = similarities.mean().item()
            cosine_similarity_scores.append(mean_similarity)

            # Log detailed statistics for this sentence
            logger.info(f"\nSimilarity stats for sentence: {sentence[:50]}...")
            logger.info(f"  Mean: {mean_similarity:.4f}")
            logger.info(f"  Min: {similarities.min().item():.4f}")
            logger.info(f"  Max: {similarities.max().item():.4f}")
            logger.info(f"  Std: {similarities.std().item():.4f}")

        # Calculate overall mean
        overall_mean = sum(cosine_similarity_scores) / len(cosine_similarity_scores)
        logger.info(f"\nOverall average similarity score: {overall_mean:.4f}")

        return overall_mean


if __name__ == "__main__":
    from google import genai
    import instructor
    import os
    from dotenv import load_dotenv

    load_dotenv()

    instance = instructor.from_genai(
        genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
    )
    evaluator = PromptEvaluator(AugGptRunner(instance))

    sentence_to_synthesized_reviews = evaluator.generate_synthetic_data(
        sentiment="neutral"
    )

    sentence_to_synthesized_reviews_embeddings, sentence_to_embeddings = (
        evaluator.create_emebeddings(sentence_to_synthesized_reviews)
    )

    average_cosine_similarity = evaluator.evaluate(
        sentence_to_synthesized_reviews_embeddings, sentence_to_embeddings
    )

    logger.info(f"Average cosine similarity: {average_cosine_similarity}")
