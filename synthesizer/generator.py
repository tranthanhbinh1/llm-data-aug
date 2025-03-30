import time
import instructor
import google.generativeai as genai
from .models import UserReview
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Literal, cast
from loguru import logger

load_dotenv()


class DataGenerator:
    SENTIMENT_MAPPING = {"positive": 1, "neutral": 2, "negative": 0}
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    RATE_LIMIT = 15

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-2.0-flash",
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )

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
    ) -> list[dict[str, str]]:
        review_examples = (
            pd.read_csv("data/class_samples/user_review_class_samples.csv")[
                ["Review", "Sentiment"]
            ]
            .query(f"Sentiment == {cls.SENTIMENT_MAPPING[sentiment]}")
            .head(num_examples)
            .to_dict(orient="records")
        )

        few_shot_examples = [
            {
                "role": "user",
                "content": f"Review: {example['Review']}. Sentiment: {example['Sentiment']}",
            }
            for example in review_examples
        ]

        return few_shot_examples

    def generate_reviews(self, examples: list[dict[str, str]]) -> UserReview:
        resp = self.client.messages.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates user reviews for a product or service.
                        You are proficient in Vietnamese, and will be asked to generate reviews in Vietnamese.
                        Please do not include trailing characters.
                        """,
                },
                {
                    "role": "user",
                    "content": "Hãy tạo ra các bình luận tương tự nói về trải nghiệm: Trong vai là một khách hàng vừa trải qua một một trải nghiệm tồi tệ, ko hài lòng về chất lượng dịch vụ của McDonald. Hãy tạo, generate ra các bình luận tương tự nói về trải nghiệm đó.",
                },
                *examples,
            ],
            response_model=UserReview,
        )

        return cast(UserReview, resp)

    def generate_reviews_batch(
        self,
        examples: list[dict[str, str]],
        sentiment: Literal["positive", "neutral", "negative"],
        num_examples: int = 5,
    ) -> list[UserReview]:
        """We are being rate-limited to 15 requests per minute, so we need to generate reviews in batches."""
        examples = self.generate_examples(
            num_examples=num_examples, sentiment=sentiment
        )
        target_number = self.get_target_number(sentiment)
        logger.info(f"Target number: {target_number}")
        reviews = []
        _hit_count = 0
        while len(reviews) < target_number:
            try:
                reviews.extend(
                    [self.generate_reviews(examples) for _ in range(num_examples)]
                )
                _hit_count += 1
                logger.info(f"Generated {len(reviews)} reviews")
                logger.info(f"Hit count: {_hit_count}")
                logger.info(f"Generated reviews: {reviews[-1]}")
            except Exception as e:
                logger.error(f"Error generating reviews: {e}")
                continue
            if _hit_count >= self.RATE_LIMIT:
                logger.info("Rate limit reached, sleeping for 60 seconds")
                time.sleep(60)
                _hit_count = 0
        return reviews

    def save_reviews(self, reviews: list[UserReview], path: str):
        pd.DataFrame(reviews).to_csv(path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", type=str, default="negative")
    args = parser.parse_args()

    generator = DataGenerator()
    examples: list[dict[str, str]] = generator.generate_examples(
        num_examples=5, sentiment=args.sentiment
    )
    reviews: list[UserReview] = generator.generate_reviews_batch(
        examples=examples, sentiment=args.sentiment, num_examples=5
    )
    generator.save_reviews(
        reviews, f"data/llm_generated/{args.sentiment}_user_reviews.csv"
    )

    # NOTE: Hit Count calculation is not correct, need to set Retry of Instructor to 1 if we want to accurately calculate the hit count.
