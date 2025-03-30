import time
import instructor
from instructor import Instructor, AsyncInstructor
from google import genai
from openai import OpenAI
from .models import SentimentPrompt, UserReviews
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
    MAX_RETRIES = 3

    def __init__(self, instructor: AsyncInstructor):
        self.instructor = instructor

    def get_client(self):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.client: AsyncInstructor = instructor.from_genai(
            client, mode=instructor.Mode.GENAI_TOOLS
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
        """Get examples from the original dataset for each LLM call"""
        label_mapping = {"Positive": 1, "Neutral": 2, "Negative": 0}

        data = pd.read_csv("data/cleaned_user_reviews.csv")
        data["Sentiment"] = data["Sentiment"].map(label_mapping)
        review_examples = (
            data[["Review", "Sentiment"]]
            .query(f"Sentiment == {cls.SENTIMENT_MAPPING[sentiment]}")
            .sample(n=num_examples)
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

    def generate_reviews(
        self,
        examples: list[dict[str, str]],
        prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        batch_size: int = 15,
    ) -> UserReviews:
        resp = self.instructor.messages.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates user reviews for a product or service.
                        You are proficient in Vietnamese, and will be asked to generate reviews in Vietnamese.
                        """,
                },
                # TODO: change the prompt once we want to generate Neutral, Positive reviews
                {
                    "role": "user",
                    "content": prompt.value.format(batch_size=batch_size),
                },
                *examples,
            ],
            model=model,
            max_retries=self.MAX_RETRIES,
            response_model=UserReviews,
        )

        return cast(UserReviews, resp)

    def generate_reviews_batch(
        self,
        examples: list[dict[str, str]],
        sentiment: Literal["positive", "neutral", "negative"],
        prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        num_examples: int = 5,
        batch_size: int = 15,
    ) -> list[UserReviews]:
        target_batches = self.get_target_number(sentiment) // batch_size
        logger.info(f"Target batches: {target_batches}")

        batched_records: list[UserReviews] = []
        _hit_count = 0
        while len(batched_records) < target_batches:
            try:
                examples = self.generate_examples(
                    num_examples=num_examples, sentiment=sentiment
                )
                logger.info(f"Generating reviews for batch {len(batched_records) + 1}")
                generated_samples = self.generate_reviews(
                    examples, prompt, model, batch_size
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
            # Test
            break
        return batched_records

    def save_reviews(self, batched_records: list[UserReviews], path: str):
        batched_df = []
        for rec in batched_records:
            _df = pd.DataFrame(rec.model_dump())
            batched_df.append(_df)
        batched_df = pd.concat(batched_df)
        batched_df.to_csv(path, index=False)


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

    # client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    # instance = instructor.from_genai(client, mode=instructor.Mode.GENAI_TOOLS)

    instance = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        ),
        mode=instructor.Mode.JSON,
    )

    generator = DataGenerator(instance)
    examples: list[dict[str, str]] = generator.generate_examples(
        num_examples=5, sentiment=args.sentiment
    )
    reviews: list[UserReviews] = generator.generate_reviews_batch(
        examples=examples,
        sentiment=args.sentiment,
        num_examples=args.num_examples,
        prompt=sentiment_prompt_mapping[args.sentiment],
        model=args.model,
    )
    generator.save_reviews(
        reviews, f"data/llm_generated/{args.sentiment}_user_reviews.csv"
    )

    # NOTE: Hit Count calculation is not correct, need to set Retry of Instructor to 1 if we want to accurately calculate the hit count.
