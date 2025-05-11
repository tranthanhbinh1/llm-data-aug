# TODO: implement SVM trainer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModel
from loguru import logger as logging
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class SVMTrainer:
    SEED = 42

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer
        | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("vinai/phobert-base"),
    ):
        self.tokenizer = tokenizer

        self.data = pd.read_csv(data_path)
        initial_rows = len(self.data)

        self.data = self.data.dropna(subset=["Review", "Sentiment"])
        self.data = self.data.reset_index(drop=True)

        removed_rows = initial_rows - len(self.data)
        if removed_rows > 0:
            logging.warning(f"Removed {removed_rows} rows containing NaN values")

        random.seed(self.SEED)
        np.random.seed(self.SEED)

    def _prepare_data(self):
        # Split the data
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=self.SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=self.SEED
        )

        # Convert text to sequences using tokenizer
        X_train = [
            self.tokenizer.encode(text, add_special_tokens=True)
            for text in train_data["Review"]
        ]
        X_val = [
            self.tokenizer.encode(text, add_special_tokens=True)
            for text in val_data["Review"]
        ]
        X_test = [
            self.tokenizer.encode(text, add_special_tokens=True)
            for text in test_data["Review"]
        ]

        # Log max token ID for debugging
        max_token_id_train = max([max(seq) if seq else 0 for seq in X_train])
        max_token_id_val = max([max(seq) if seq else 0 for seq in X_val])
        max_token_id_test = max([max(seq) if seq else 0 for seq in X_test])
        logging.info(f"Max token ID in train: {max_token_id_train}")
        logging.info(f"Max token ID in val: {max_token_id_val}")
        logging.info(f"Max token ID in test: {max_token_id_test}")

        # Convert labels to integers using LabelEncoder
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_data["Sentiment"])
        y_val = label_encoder.transform(val_data["Sentiment"])
        y_test = label_encoder.transform(test_data["Sentiment"])

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
