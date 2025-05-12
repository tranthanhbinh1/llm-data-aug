import time
import instructor
from instructor import Instructor
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai import OpenAI
from pydantic import BaseModel

from src.constants import LABEL_MAPPING
from .models import AugmentedUserReviews, SentimentPrompt, UserReviews
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, cast
from loguru import logger

load_dotenv()


class DataGenerator:
    """Generate synthetic data from the original dataset using LLM without any specific strategy"""

    SENTIMENT_MAPPING = {"positive": 1, "neutral": 2, "negative": 0}
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    MAX_RETRIES = 3
    BASE_SYSTEM_PROMPT = ChatCompletionSystemMessageParam(
        role="system",
        content="""You are a helpful assistant that generates user reviews for a product or service.
                You are proficient in Vietnamese, and will be asked to generate reviews in Vietnamese.
                """,
    )

    def __init__(
        self,
        instructor: Instructor,
        response_model: type[BaseModel],
    ):
        self.instructor = instructor
        self.response_model = response_model

    @classmethod
    def get_target_number(
        cls, sentiment: Literal["positive", "neutral", "negative"]
    ) -> int:
        """Get the target number generate reviews"""
        original_data = pd.read_csv(
            os.path.join(cls.DATA_PATH, "cleaned_user_reviews.csv")
        )
        original_data["Sentiment"] = original_data["Sentiment"].str.lower()
        logger.info(original_data.head().to_markdown())

        sentiment_counts = original_data["Sentiment"].value_counts()
        logger.info(f"Sentiment counts: {sentiment_counts}")
        current_count = sentiment_counts.to_dict().get(sentiment)
        if current_count is None:
            raise ValueError(f"Sentiment {sentiment} not found in data")
        logger.info(f"Current count for {sentiment}: {current_count}")
        return max(sentiment_counts) - current_count

    @classmethod
    def generate_examples(
        cls,
        sentiment: Literal["positive", "neutral", "negative"],
        num_examples: int = 5,
    ) -> list[ChatCompletionUserMessageParam]:
        """Get examples from the original dataset for each LLM call"""
        data = pd.read_csv(os.path.join(cls.DATA_PATH, "cleaned_user_reviews.csv"))
        data["Sentiment"] = data["Sentiment"].map(LABEL_MAPPING)
        review_examples = (
            data[["Review", "Sentiment"]]
            .query(f"Sentiment == {cls.SENTIMENT_MAPPING[sentiment]}")
            .sample(n=num_examples)
            .to_dict(orient="records")
        )
        few_shot_examples = [
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Review: {example['Review']}. Sentiment: {example['Sentiment']}",
            )
            for example in review_examples
        ]

        return few_shot_examples

    def generate_reviews(
        self,
        examples: list[ChatCompletionUserMessageParam],
        user_prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        batch_size: int = 15,
        system_prompt: ChatCompletionSystemMessageParam = BASE_SYSTEM_PROMPT,
    ) -> AugmentedUserReviews | UserReviews:
        resp = self.instructor.messages.create(
            messages=[
                system_prompt,
                ChatCompletionUserMessageParam(
                    role="user",
                    content=user_prompt.value.format(batch_size=batch_size),
                ),
            ]
            + examples,
            model=model,
            max_retries=self.MAX_RETRIES,
            response_model=self.response_model,
        )

        return cast(AugmentedUserReviews | UserReviews, resp)

    def generate_reviews_batch(
        self,
        examples: list[ChatCompletionUserMessageParam],
        sentiment: Literal["positive", "neutral", "negative"],
        prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        num_examples: int = 5,
        batch_size: int = 15,
        system_prompt: ChatCompletionSystemMessageParam = BASE_SYSTEM_PROMPT,
    ) -> list[AugmentedUserReviews | UserReviews]:
        target_batches = self.get_target_number(sentiment) // batch_size
        logger.info(f"Target batches: {target_batches}")

        batched_records: list[AugmentedUserReviews | UserReviews] = []
        _hit_count = 0
        while len(batched_records) < target_batches:
            try:
                examples = self.generate_examples(
                    num_examples=num_examples, sentiment=sentiment
                )
                logger.info(f"Generating reviews for batch {len(batched_records) + 1}")
                generated_samples: AugmentedUserReviews | UserReviews = (
                    self.generate_reviews(
                        examples, prompt, model, batch_size, system_prompt
                    )
                )
                logger.info(f"Generated reviews: {generated_samples}")
                batched_records.append(generated_samples)
                _hit_count += self.MAX_RETRIES
                logger.info(f"Hit count (including retries): {_hit_count}")
            except Exception as e:
                if "429" in str(e):
                    logger.error("Rate limit reached, sleeping for 60 seconds")
                    time.sleep(60)
                    continue
                logger.error(f"Error generating reviews: {e}")
                continue
        return batched_records

    def save_reviews(
        self,
        batched_records: list[AugmentedUserReviews | UserReviews],
        original_sentences: list[str],
        path: str,
    ):
        """Save reviews to CSV in the proper format with Review and Sentiment columns."""
        all_reviews = []
        all_sentiments = []
        all_original_reviews = []

        for record, original_sentence in zip(batched_records, original_sentences):
            record_dict = record.model_dump()
            for review in record_dict["reviews"]:
                all_reviews.append(review["review"])
                all_sentiments.append(review["sentiment"])

            # Because each sentence create 6 new reviews, we need to repeat the original sentence 6 times
            for _ in range(6):
                all_original_reviews.append(original_sentence)

        df = pd.DataFrame(
            {
                "Review": all_reviews,
                "Sentiment": all_sentiments,
                "OriginalReview": all_original_reviews,
            }
        )
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} reviews to {path}")


if __name__ == "__main__":
    sentiment_prompt_mapping = {
        "negative": SentimentPrompt.NEGATIVE,
        "neutral": SentimentPrompt.NEUTRAL,
        "positive": SentimentPrompt.POSITIVE,
    }
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", type=str, default="negative")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    args = parser.parse_args()

    instance = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="local",
        ),
        mode=instructor.Mode.JSON,
    )

    generator = DataGenerator(instance, UserReviews)
    examples: list[ChatCompletionUserMessageParam] = generator.generate_examples(
        num_examples=5, sentiment=args.sentiment
    )
    reviews: list[AugmentedUserReviews | UserReviews] = (
        generator.generate_reviews_batch(
            examples=examples,
            sentiment=args.sentiment,
            num_examples=args.num_examples,
            prompt=sentiment_prompt_mapping[args.sentiment],
            model=args.model,
        )
    )
    generator.save_reviews(
        reviews, f"data/llm_generated/{args.sentiment}_user_reviews.csv"
    )

    # NOTE: Hit Count calculation is not correct, need to set Retry of Instructor to 1 if we want to accurately calculate the hit count.
