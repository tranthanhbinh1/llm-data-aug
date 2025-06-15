from typing import cast
import instructor
from loguru import logger as logging
import pandas as pd

from src.constants import NUM_REPHRASED_SENTENCES, ORIGINAL_DATASET_PATH
from src.enums import Sentiment
from src.synthesizer.models import AugmentedSentencesBatch
from src.utils import get_instructor_instance


class AugGpt:
    """https://arxiv.org/abs/2302.13007"""

    def __init__(self, instructor: instructor.AsyncInstructor):
        self.instructor = instructor
        self.response_model = AugmentedSentencesBatch

    @staticmethod
    def prepare_original_sentences(
        sentiment: Sentiment, data: pd.DataFrame = pd.read_csv(ORIGINAL_DATASET_PATH)
    ) -> pd.DataFrame:
        _data = data.copy()

        subset = _data.where(_data.sentiment == sentiment)

        return subset

    def _generate_augmented_sentences(
        self, sentiment: Sentiment, original_sentence: str, system_prompt: str
    ) -> AugmentedSentencesBatch:
        """Generate augmented sentences for an original sentence, while keeping the sentiment the same as the original sentence"""
        augmented_sentences = self.instructor.messages.create(
            model="gemini-2.0-flash",
            strict=False,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Câu văn gốc: {original_sentence}",
                },
                {
                    "role": "user",
                    "content": f"Từ câu văn gốc được cung cấp, hãy tạo ra {NUM_REPHRASED_SENTENCES} phiên bản diễn đạt lại mà vẫn giữ nguyên ý nghĩa và ngôn ngữ gốc của câu văn",
                },
            ],
            response_model=AugmentedSentencesBatch,
        )

        return cast(AugmentedSentencesBatch, augmented_sentences)

    def generate_augmented_sentences_batch(
        self, sentiment: Sentiment, original_sentences: list[str], system_prompt: str
    ) -> list[tuple[str, AugmentedSentencesBatch]]:
        """Generate augmented sentences for a list of original sentences, while keeping the sentiment the same as the original sentences"""

        original_and_augmented_sentences: list[tuple[str, AugmentedSentencesBatch]] = []

        for original_sentence in original_sentences:
            logging.info(f"Generating augmented sentences for '{original_sentence}'")
            augmented_sentences = self._generate_augmented_sentences(
                sentiment, original_sentence, system_prompt
            )
            original_and_augmented_sentences.append(
                (original_sentence, augmented_sentences)
            )

        return original_and_augmented_sentences

    def save_augmented_sentences(
        self,
        original_and_augmented_sentences: list[tuple[str, AugmentedSentencesBatch]],
        sentiment: Sentiment,  # TODO: we might need to use sentiment here to validate the output of LLM
    ) -> None:
        """Save the augmented sentences to a CSV file"""
        # Flatten the data for better DataFrame handling
        flattened_data = []
        for original_sentence, augmented_reviews in original_and_augmented_sentences:
            for review in augmented_reviews.sentences:
                flattened_data.append(
                    {
                        "original_sentence": original_sentence,
                        "augmented_sentence": review.sentence,
                        "sentiment": review.sentiment.value,
                    }
                )

        df = pd.DataFrame(flattened_data)
        logging.info(f"Generated {len(df)} augmented sentences")
        logging.info(df.to_markdown(index=False))
        # return df.to_csv(f"data/llm_generated/aug_gpt_{sentiment}.csv", index=False)

    def generate(
        self,
        sentiment: Sentiment,
        original_sentences: list[str],
        system_prompt: str,
    ) -> list[tuple[str, AugmentedSentencesBatch]]:
        """Generate augmented sentences for a list of original sentences, while keeping the sentiment the same as the original sentences, and save the results to a CSV file"""
        original_and_augmented_sentences = self.generate_augmented_sentences_batch(
            sentiment, original_sentences, system_prompt
        )
        # Save
        self.save_augmented_sentences(original_and_augmented_sentences, sentiment)

        return original_and_augmented_sentences


if __name__ == "__main__":
    _instructor = get_instructor_instance()
    aug_gpt = AugGpt(_instructor)
    aug_gpt.generate(
        sentiment=Sentiment.POSITIVE,
        original_sentences=[
            "cũng tốt. đồ ăn ngon , đợi không quá lâu. gần ngay bờ hồ hoàn kiếm",
            "đồ ăn khá tệ, phục vụ chậm.",
        ],
        system_prompt="Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là tạo ra các phiên bản câu được bổ sung và làm phong phú hơn từ một câu gốc được cung cấp.",
    )
