from typing import Literal, Optional
import instructor
from openai import OpenAI
from .generator import DataGenerator
from .models import AugmentedUserReviews, SentimentPrompt
from instructor import AsyncInstructor
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
import pandas as pd
from loguru import logger
import time


class AugGptRunner:
    """https://arxiv.org/abs/2302.13007"""

    BASE_AUGMENTOR_PROMPT = (
        "You are a helpful assistant that rephrase text and make sentence smooth."
    )

    def __init__(self, instructor: AsyncInstructor):
        self.instructor = instructor
        self.response_model = AugmentedUserReviews
        self.data_generator = DataGenerator(instructor, self.response_model)

    @classmethod
    def prepare_original_sentences(
        cls,
        sentiment: Literal["neutral", "negative"],  # The minority classes
    ) -> list[ChatCompletionUserMessageParam]:
        """Get examples from the original dataset for each LLM call"""
        label_mapping = {"Positive": 1, "Neutral": 2, "Negative": 0}

        data = pd.read_csv("data/cleaned_user_reviews.csv")
        data["Sentiment"] = data["Sentiment"].map(label_mapping)

        # Data of the minority classes
        minority_data = (
            data[["Review", "Sentiment"]]
            .query(f"Sentiment == {DataGenerator.SENTIMENT_MAPPING[sentiment]}")
            .to_dict(orient="records")
        )

        logger.info(minority_data)

        # List of original sentences
        original_sentences = [
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Review: {sentence['Review']}. Sentiment: {sentence['Sentiment']}",
            )
            for sentence in minority_data
        ]

        return original_sentences

    def generate_reviews_batch(
        self,
        sentiment: Literal["neutral", "negative"],
        user_prompt: SentimentPrompt,
        model: str = "gemini-2.0-flash",
        num_to_generate: int = 6,
        system_prompt: ChatCompletionSystemMessageParam = DataGenerator.BASE_SYSTEM_PROMPT,
        index: Optional[int] = 0,
    ) -> list[AugmentedUserReviews]:
        # With each original sentence, we call the LLM to generate a number of augmented sentences
        original_sentences = self.prepare_original_sentences(sentiment)
        failed_original_sentences = []
        batched_records: list[AugmentedUserReviews] = []
        _hit_count = 0
        # Batching the amount of original sentences to generate

        for original_sentence in original_sentences:
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
                generated_samples = self.data_generator.generate_reviews(
                    [original_sentence],
                    user_prompt,
                    model,
                    num_to_generate,
                    system_prompt,
                )
                logger.info(f"Generated reviews: {generated_samples}")
                batched_records.append(generated_samples)
                _hit_count += DataGenerator.MAX_RETRIES
                logger.info(f"Hit count (including retries): {_hit_count}")
            except Exception as e:
                failed_original_sentences.append(original_sentence)
                # Save the failed original sentence
                if "429" in str(e):
                    logger.error("Rate limit reached, sleeping for 60 seconds")
                    time.sleep(60)
                    continue
                logger.error(f"Error generating reviews: {e}")
                continue
            # Save in each 50 sentences
            if original_sentences.index(original_sentence) % 10 == 0:
                self.data_generator.save_reviews(
                    batched_records,
                    f"data/llm_generated/auggpt_augmented_user_reviews_{sentiment}_{original_sentences.index(original_sentence)}.csv",
                )
                logger.info(
                    f"Saved {original_sentences.index(original_sentence)} sentences"
                )
                batched_records = []

                # Save the failed original sentences
                pd.DataFrame(failed_original_sentences).to_csv(
                    f"data/llm_generated/auggpt_failed_original_sentences_{sentiment}.csv",
                    index=False,
                )
                failed_original_sentences = []

        return batched_records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Continue from a specific index
    parser.add_argument("--continue_from", type=int, default=0)
    parser.add_argument("--sentiment", type=str, default="neutral")
    parser.add_argument("--num_to_generate", type=int, default=6)
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    args = parser.parse_args()

    instance = instructor.from_openai(
        OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
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
