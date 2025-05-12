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

from src.constants import DATA_PATH
from src.utils import (
    expand_abbr,
    remove_non_alphanumeric,
    remove_special_characters,
    normalize_repeated_words,
    tokenize_text,
    abbr,
)


class SVMTrainer:
    SEED = 42

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

        random.seed(self.SEED)
        np.random.seed(self.SEED)

    def _words_processing(self):
        self.data["Review"] = self.data["Review"].apply(str.lower)
        self.data["Review"] = self.data["Review"].apply(remove_non_alphanumeric)
        self.data["Review"] = self.data["Review"].apply(lambda x: expand_abbr(x, abbr))
        self.data["Review"] = self.data["Review"].apply(remove_special_characters)
        self.data["Review"] = self.data["Review"].apply(normalize_repeated_words)
        self.data["tokenized_text"] = self.data["Review"].apply(tokenize_text)

    def _prepare_data(self):
        # First split the data
        train_data, val_data = train_test_split(
            self.data, test_size=0.2, random_state=self.SEED
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
        self._words_processing()
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
        f1 = f1_score(y_val, y_pred_svc, average="weighted")
        logging.info(f"F1 Score: {f1:.4f}")

        return f1


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

    (X_train, y_train), (X_val, y_val) = svm_trainer.load_data()
    f1 = svm_trainer.train(X_train, y_train, X_val, y_val)

    print(f1)
