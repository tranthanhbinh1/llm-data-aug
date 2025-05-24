import os
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
from loguru import logger as logging
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse

from src.constants import DATA_PATH, SEED
from src.repositories.trainer import TrainerEvaluatorRepository


class SVMTrainer(TrainerEvaluatorRepository):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer
        | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            "vinai/phobert-base-v2"
        ),
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = pd.read_csv(data_path)
        initial_rows = len(self.data)

        self.data = self.data.dropna(subset=["Review", "Sentiment"])
        self.data = self.data.reset_index(drop=True)

        removed_rows = initial_rows - len(self.data)
        if removed_rows > 0:
            logging.warning(f"Removed {removed_rows} rows containing NaN values")

        random.seed(SEED)
        np.random.seed(SEED)

    def _prepare_data(self):
        # First split the data
        train_data, val_data = train_test_split(
            self.data, test_size=0.2, random_state=SEED
        )

        # Log dataset sizes
        logging.info(f"Train set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")

        # Get raw text for TF-IDF vectorization
        X_train = train_data["tokenized_text"].tolist()
        X_val = val_data["tokenized_text"].tolist()

        # Convert labels as needed
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_data["Sentiment"])
        y_val = label_encoder.transform(val_data["Sentiment"])

        # Log label distribution
        train_label_counts = np.bincount(y_train)
        logging.info(f"Training label distribution: {train_label_counts}")

        return (X_train, y_train), (X_val, y_val)

    def load_data(self):
        # self._words_processing()
        return self._prepare_data()

    def train(self, X_train, y_train, X_val, y_val):
        # Create pipeline with TF-IDF vectorizer and SVM classifier
        logging.info("Training SVM model with TF-IDF features...")

        svc_model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ("clf", SVC(kernel="linear", C=1.0, probability=True)),
            ]
        )
        svc_model.fit(X_train, y_train)
        y_pred_svc = svc_model.predict(X_val)

        # Calculate and log metrics
        accuracy_svc = accuracy_score(y_val, y_pred_svc)
        logging.info(f"SVC Model Accuracy: {accuracy_svc:.4f}")

        # Log prediction distribution
        pred_counts = np.bincount(np.array(y_pred_svc, dtype=int), minlength=3)
        logging.info(f"Prediction distribution: {pred_counts}")

        logging.info("\n" + str(classification_report(y_val, y_pred_svc)))
        weighted_f1 = f1_score(y_val, y_pred_svc, average="weighted")
        logging.info(f"F1 Score: {weighted_f1:.4f}")

        return weighted_f1

    def run_evaluation(self):
        """
        Run the complete training pipeline from data loading to evaluation

        Returns:
            float: F1 score from validation
        """
        # Load and prepare data
        (X_train, y_train), (X_val, y_val) = self.load_data()

        # Train model and get F1 score
        weighted_f1 = self.train(X_train, y_train, X_val, y_val)

        return weighted_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(
            DATA_PATH,
            "llm_generated/gemini-2.0-flash/auggpt_upsampled_user_reviews_cleaned.csv",
        ),
    )
    args = parser.parse_args()

    svm_trainer = SVMTrainer(data_path=args.data_path)
    weighted_f1 = svm_trainer.run_evaluation()

    print(weighted_f1)
