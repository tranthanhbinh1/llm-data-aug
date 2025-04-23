from typing import Literal, Optional
import instructor
from openai import OpenAI

from constants import NUM_REPHRASED_SENTENCES
from .generator import DataGenerator
from .models import AugmentedUserReviews, SentimentPrompt, UserReviews
from instructor import Instructor
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from google import genai
import pandas as pd
from loguru import logger
import time
import os
from dotenv import load_dotenv


class AugGptRunner:
    """https://arxiv.org/abs/2302.13007"""

    BASE_AUGMENTOR_PROMPT = ChatCompletionSystemMessageParam(
        role="system",
        content="Bạn là một trợ lý hữu ích, có nhiệm vụ diễn đạt lại văn bản và làm cho câu văn trở nên mượt mà hơn.",
    )

    def __init__(self, instructor: Instructor):
        self.instructor = instructor
        self.response_model = AugmentedUserReviews
        self.data_generator = DataGenerator(instructor, self.response_model)

    @classmethod
    def prepare_original_sentences(
        cls,
        sentiment: Literal["neutral", "negative"],  # The minority classes
        data: pd.DataFrame = pd.read_csv(
            "/home/tb24/projects/llm-data-aug/data/cleaned_user_reviews.csv"
        ),
    ) -> tuple[list[str], list[ChatCompletionUserMessageParam]]:
        """Get examples from the original dataset for each LLM call"""
        label_mapping = {"Positive": 1, "Neutral": 2, "Negative": 0}

        # Make a copy to avoid modifying the input
        data = data.copy()

        # Check if Sentiment is already mapped
        if data["Sentiment"].dtype == "object":
            logger.info("Mapping sentiment labels to numeric values")
            data["Sentiment"] = data["Sentiment"].map(label_mapping)

        target_sentiment = DataGenerator.SENTIMENT_MAPPING[sentiment]
        logger.info(f"Filtering for sentiment {sentiment} (value: {target_sentiment})")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Sentiment value counts:\n{data['Sentiment'].value_counts()}")

        # Data of the minority classes
        minority_data = data[data["Sentiment"] == target_sentiment][
            ["Review", "Sentiment"]
        ].to_dict(orient="records")

        logger.info(f"Found {len(minority_data)} records for sentiment {sentiment}")

        if not minority_data:
            logger.warning(f"No records found for sentiment {sentiment}")
            return [], []

        # List of original sentences
        original_sentences = [sentence["Review"] for sentence in minority_data]
        original_sentence_prompts = [
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Review: {sentence['Review']}. Sentiment: {sentence['Sentiment']}",
            )
            for sentence in minority_data
        ]

        return original_sentences, original_sentence_prompts

    def _generate_reviews(
        self,
        sentiment: Literal["neutral", "negative"],
        user_prompt: SentimentPrompt,
        augmentor_prompt: ChatCompletionSystemMessageParam = BASE_AUGMENTOR_PROMPT,
        model: str = "gemini-2.0-flash",
        num_to_generate: int = 6,
        _: ChatCompletionSystemMessageParam = DataGenerator.BASE_SYSTEM_PROMPT,
        index: Optional[int] = 0,
        original_sentences: Optional[list[str]] = None,
        original_sentence_prompts: Optional[
            list[ChatCompletionUserMessageParam]
        ] = None,
    ) -> tuple[list[AugmentedUserReviews | UserReviews], list[str], list[str]]:
        if original_sentences is None or original_sentence_prompts is None:
            original_sentences, original_sentence_prompts = (
                self.prepare_original_sentences(sentiment=sentiment)
            )

        if not original_sentences or not original_sentence_prompts:
            logger.warning("No sentences to process")
            return [], [], []

        failed_original_sentences = []
        batched_records: list[AugmentedUserReviews | UserReviews] = []
        _hit_count = 0

        # Generate rephrased sentences for each original sentence
        for original_sentence, original_sentence_prompt in zip(
            original_sentences, original_sentence_prompts
        ):
            if index and original_sentences.index(original_sentence) < index:
                logger.info(
                    f"Skipping sentence {original_sentences.index(original_sentence)} because continue from {index}"
                )
                continue
            try:
                logger.info(f"Generating reviews for sentence: {original_sentence}")
                logger.info(
                    f"Sentence number: {original_sentences.index(original_sentence)}"
                )
                generated_samples: AugmentedUserReviews | UserReviews = (
                    self.data_generator.generate_reviews(
                        [original_sentence_prompt],
                        user_prompt,
                        model,
                        num_to_generate,
                        augmentor_prompt,
                    )
                )
                logger.info(f"Generated reviews: {generated_samples}")
                batched_records.append(generated_samples)
                _hit_count += DataGenerator.MAX_RETRIES
                logger.info(f"Hit count (including retries): {_hit_count}")
            except Exception as e:
                failed_original_sentences.append(original_sentence)
                if "429" in str(e):
                    logger.error("Rate limit reached, sleeping for 60 seconds")
                    time.sleep(60)
                    continue
                logger.error(f"Error generating reviews: {e}")
                continue

        return batched_records, original_sentences[index:], failed_original_sentences

    def save_generated_reviews(
        self,
        batched_records: list[AugmentedUserReviews | UserReviews],
        original_sentences: list[str],
        failed_sentences: list[str],
        model: str,
        sentiment: str,
    ) -> None:
        # Save the successful generations
        self.data_generator.save_reviews(
            batched_records,
            original_sentences,
            f"data/llm_generated/{model}/auggpt_augmented_user_reviews_{sentiment}.csv",
        )
        logger.info(f"Saved {len(batched_records)} sentences")
        logger.info(pd.DataFrame(batched_records).head().to_markdown())

        # Save the failed original sentences
        pd.DataFrame(failed_sentences).to_csv(
            f"data/llm_generated/{model}/auggpt_failed_original_sentences_{sentiment}.csv",
            index=False,
        )

    def generate_reviews_batch(
        self,
        sentiment: Literal["neutral", "negative"],
        user_prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        num_to_generate: int = NUM_REPHRASED_SENTENCES,
        _: ChatCompletionSystemMessageParam = DataGenerator.BASE_SYSTEM_PROMPT,
        index: Optional[int] = 0,
    ) -> list[AugmentedUserReviews | UserReviews]:
        batched_records, original_sentences, failed_sentences = self._generate_reviews(
            sentiment=sentiment,
            user_prompt=user_prompt,
            model=model,
            num_to_generate=num_to_generate,
            _=_,
            index=index,
        )

        self.save_generated_reviews(
            batched_records,
            original_sentences,
            failed_sentences,
            model,
            sentiment,
        )

        return batched_records


if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser()
    # Continue from a specific index
    parser.add_argument("--continue_from", type=int, default=0)
    parser.add_argument("--sentiment", type=str, default="neutral")
    parser.add_argument("--num_to_generate", type=int, default=6)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    args = parser.parse_args()
    if args.model == "gemini-2.0-flash":
        instance = instructor.from_genai(
            genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY"))
        )
    else:
        instance = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )

    runner = AugGptRunner(instance)
    augmented_reviews = runner.generate_reviews_batch(
        sentiment=args.sentiment,
        user_prompt=SentimentPrompt.AUG_GPT_PROMPT,
        num_to_generate=args.num_to_generate,
        model=args.model,
        index=args.continue_from,
    )
