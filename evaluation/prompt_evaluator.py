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
from sentence_transformers import SentenceTransformer

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

    def random_split(self, test_size: float = 0.2) -> pd.DataFrame:
        return self._original_data.sample(frac=test_size)

    def generate_synthetic_data(self, test_set: pd.DataFrame) -> pd.DataFrame:
        pass

    def evaluate(
        self, original_records: list[str], synthesized_records: list[str]
    ) -> float:
        original_records_embeddings = self.model.encode(original_records)
        original_records_tensor = torch.tensor(original_records_embeddings)
        synthesized_records_embeddings = self.model.encode(synthesized_records)
        synthesized_records_tensor = torch.tensor(synthesized_records_embeddings)

        # Calculate the average similarity score
        similarity_scores = torch.nn.functional.cosine_similarity(
            original_records_tensor, synthesized_records_tensor
        )

        return similarity_scores.mean().item()
