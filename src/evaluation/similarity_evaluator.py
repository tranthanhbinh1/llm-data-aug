## This is a script to evaluate the performance of the prompt
## Steps:
## 1. Load the original dataset
## 2. Split the dataset into random subsets
## 3. Randomly select one subset as the test set
## 4. Use the Prompt to generate synthetic data (reviews) with the test set as examples
## 5. Perform pairwise comparison using Cosine Similarity between the original and synthetic data
## 6. Calculate the average similarity score
## 7. Report the average similarity score


import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.enums import Sentiment
from src.repositories.evaluator import EvaluatorRepository

from ..synthesizer.models import AugmentedSentencesBatch
from ..utils import get_instructor_instance
from ..constants import ORIGINAL_DATASET_PATH
from ..synthesizer.aug_gpt import AugGpt


class SimiarityEvaluator(EvaluatorRepository):
    def __init__(
        self,
        data_synthesizer: AugGpt,
        data_path: str = ORIGINAL_DATASET_PATH,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self._original_data = pd.read_csv(data_path)
        self._data_synthesizer = data_synthesizer

    def random_split(
        self,
        sentiment: Sentiment,
        test_size: float = 0.05,
    ) -> pd.DataFrame:
        data = self._data_synthesizer.prepare_original_sentences(
            sentiment=sentiment,
            data=self._original_data,
        )
        sampled_subset = data.sample(frac=test_size, random_state=42)
        return sampled_subset

    # TODO: maybe this function should be detached
    def generate_synthetic_data(
        self, sentiment: Sentiment, prompt: str
    ) -> dict[str, list[str]]:
        subset = self.random_split(sentiment=sentiment)
        if subset.empty:
            raise ValueError(f"No {sentiment} records found in the dataset")

        original_sentences_df = self._data_synthesizer.prepare_original_sentences(
            sentiment=sentiment,
            data=subset,
        )

        original_sentences_and_augmented_sentences: list[
            tuple[str, AugmentedSentencesBatch]
        ] = self._data_synthesizer.generate(
            sentiment=sentiment,
            original_sentences=original_sentences_df["sentence"].tolist(),
            system_prompt=prompt,
        )

        # Create a mapping between an original sentence and its corresponding records
        _original_sentence_to_records: dict[str, AugmentedSentencesBatch] = dict()
        for (
            original_sentence,
            augmented_sentences,
        ) in original_sentences_and_augmented_sentences:
            _original_sentence_to_records[original_sentence] = augmented_sentences

        # For each AugmentedUserReviews object in the dict, we extract the reviews
        original_sentence_to_synthesized_sentences: dict[str, list[str]] = {
            original_sentence: [
                review.sentence for review in augmented_reviews.sentences
            ]
            for original_sentence, augmented_reviews in _original_sentence_to_records.items()
        }

        logger.info(
            f"Generated {len(original_sentence_to_synthesized_sentences)} synthetic sentences"
        )
        return original_sentence_to_synthesized_sentences

    def create_emebeddings(
        self, sentence_to_synthesized_sentences: dict[str, list[str]]
    ):
        sentence_to_synthesized_sentences_embeddings: dict[str, torch.Tensor] = dict()
        for sentence, sentences in sentence_to_synthesized_sentences.items():
            embeddings = self.model.encode(sentences)
            sentence_to_synthesized_sentences_embeddings[sentence] = torch.tensor(
                embeddings
            )

        sentence_to_embeddings: dict[str, torch.Tensor] = dict()
        for sentence, _ in sentence_to_synthesized_sentences_embeddings.items():
            embeddings = self.model.encode(sentence)
            sentence_to_embeddings[sentence] = torch.tensor(embeddings)

        return sentence_to_synthesized_sentences_embeddings, sentence_to_embeddings

    def evaluate(
        self,
        sentence_to_synthesized_sentences_embeddings: dict[str, torch.Tensor],
        sentence_to_embeddings: dict[str, torch.Tensor],
    ) -> float:
        cosine_similarity_scores: list[float] = []
        for sentence, original_embedding in sentence_to_embeddings.items():
            # Get synthetic reviews embeddings for this sentence
            synthetic_embeddings = sentence_to_synthesized_sentences_embeddings[
                sentence
            ]

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

    def run_evaluation(
        self,
        sentiment: Sentiment,
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

    evaluator = SimiarityEvaluator(AugGpt(get_instructor_instance()))
    average_cosine_similarity = evaluator.run_evaluation(
        sentiment=Sentiment.NEUTRAL,
        prompt="Bạn là một trợ lý hữu ích, có nhiệm vụ diễn đạt lại văn bản và làm cho câu văn trở nên mượt mà hơn.",
    )

    logger.info(f"Average cosine similarity: {average_cosine_similarity}")
