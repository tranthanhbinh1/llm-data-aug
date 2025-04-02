import os
from regex import P
import torch
import pandas as pd
import numpy as np
from underthesea import sent_tokenize, word_tokenize, text_normalize
import emoji
from vncorenlp import VnCoreNLP
import collections
import emoji_vietnamese as ev
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import re
from dataloaders.train_test_split import DataScenario
from dataloaders.custom_dataset import CustomDataset
from utils import save_classification_report
from loguru import logger
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


class Evaluator:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def prepare_data(self, data_scenario: DataScenario):
        train_data = pd.read_csv(
            os.path.join(
                f"data/{data_scenario.lower()}",
                f"train_{data_scenario.lower()}_user_reviews.csv",
            )
        )
        val_data = pd.read_csv(
            os.path.join(
                f"data/{data_scenario.lower()}",
                f"val_{data_scenario.lower()}_user_reviews.csv",
            )
        )
        test_data = pd.read_csv(
            os.path.join(
                f"data/{data_scenario.lower()}",
                f"test_{data_scenario.lower()}_user_reviews.csv",
            )
        )

        # Detached the data frames to train texts, lables, val texts, val labels, test texts, test labels
        train_texts = train_data["text"].tolist()
        train_labels = train_data["label"].tolist()
        val_texts = val_data["text"].tolist()
        val_labels = val_data["label"].tolist()
        test_texts = test_data["text"].tolist()
        test_labels = test_data["label"].tolist()

        return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

    def load_model(self, model_name: str):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(self.PROJECT_ROOT, "models", model_name)
        )
        model.to(self.DEVICE)
        return model

    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        data_scenario: DataScenario,
    ):
        # Evaluation on test set
        model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.DEVICE)
                attention_mask = batch["attention_mask"].to(self.DEVICE)
                labels = batch["labels"].to(self.DEVICE)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                logits = outputs.logits

                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert predictions and true labels to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate classification report
        target_names = ["Label 0", "Label 1", "Label 2"]  # Specify label names
        logger.info(
            classification_report(true_labels, predictions, target_names=target_names)
        )
        logger.info("Accuracy: {}".format(accuracy_score(true_labels, predictions)))

        # save_classification_report(
        #     true_labels, predictions, target_names, self.PROJECT_ROOT, data_scenario
        # )

        return true_labels, predictions

    def run_evaluation(
        self,
        model_name: str,
        data_scenario: DataScenario,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int,
    ):
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
            self.prepare_data(data_scenario)
        )
        test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        model = self.load_model(model_name)
        true_labels, predictions = self.evaluate(model, test_loader, data_scenario)

        return true_labels, predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_scenario", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_length", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    evaluator = Evaluator()
    evaluator.run_evaluation(
        args.model_name, args.data_scenario, tokenizer, args.max_length
    )
