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

from ..constants import LABEL_MAPPING, NUM_REPHRASED_SENTENCES
from synthesizer.generator import DataGenerator
from synthesizer.models import (
    AugmentedUserReviews,
    SentimentPrompt,
)
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
)
from synthesizer.runner import AugGptRunner
from ..utils import get_instructor_instance
from ..constants import ORIGINAL_DATASET_PATH


class PromptEvaluator:
    def __init__(
        self,
        auggpt_runner: AugGptRunner,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self._original_data = pd.read_csv(ORIGINAL_DATASET_PATH)
        self._auggpt_runner = auggpt_runner

    def random_split(
        self,
        sentiment: Literal["neutral", "negative"],
        test_size: float = 0.05,
    ) -> pd.DataFrame:
        data = self._original_data.copy()
        data["Sentiment"] = data["Sentiment"].map(LABEL_MAPPING)

        # Filter first, then sample
        subset = data[data["Sentiment"] == DataGenerator.SENTIMENT_MAPPING[sentiment]]
        sampled_subset = subset.sample(frac=test_size, random_state=42)

        logger.info(f"Total {sentiment} records: {len(subset)}")
        logger.info(f"Sampled subset size: {len(sampled_subset)}")
        return sampled_subset

    def generate_synthetic_data(
        self, sentiment: Literal["neutral", "negative"], prompt: str
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
                augmentor_prompt=ChatCompletionSystemMessageParam(
                    role="system",
                    content=prompt,
                ),
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

            # NOTE: exclude the 'perfect' score
            similarities = similarities[similarities < 1]
            # print("Similarities after excluding perfect score: ", similarities)
            if len(similarities) == 0:
                logger.warning(
                    "Similarities reduced to 0 after excluding perfect score"
                )
                continue
            # Calculate statistics for this sentence
            mean_similarity = similarities.mean().item()
            cosine_similarity_scores.append(mean_similarity)

            # Log detailed statistics for this sentence
            # logger.info(f"\nSimilarity stats for sentence: {sentence[:50]}...")
            # logger.info(f"  Mean: {mean_similarity:.4f}")
            # logger.info(f"  Min: {similarities.min().item():.4f}")
            # logger.info(f"  Max: {similarities.max().item():.4f}")
            # logger.info(f"  Std: {similarities.std().item():.4f}")

        # Calculate overall mean
        overall_mean = sum(cosine_similarity_scores) / len(cosine_similarity_scores)
        # logger.info(f"\nOverall average similarity score: {overall_mean:.4f}")

        return overall_mean

    def main(
        self,
        sentiment: Literal["neutral", "negative"],
        prompt: str,
    ):
        sentence_to_synthesized_reviews = self.generate_synthetic_data(
            sentiment=sentiment, prompt=prompt
        )
        sentence_to_synthesized_reviews_embeddings, sentence_to_embeddings = (
            self.create_emebeddings(sentence_to_synthesized_reviews)
        )
        return self.evaluate(
            sentence_to_synthesized_reviews_embeddings, sentence_to_embeddings
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    evaluator = PromptEvaluator(AugGptRunner(get_instructor_instance()))
    average_cosine_similarity = evaluator.main(
        sentiment="neutral",
        prompt="Bạn là một trợ lý hữu ích, có nhiệm vụ diễn đạt lại văn bản và làm cho câu văn trở nên mượt mà hơn.",
    )

    logger.info(f"Average cosine similarity: {average_cosine_similarity}")
