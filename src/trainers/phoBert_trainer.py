import numpy as np
from sklearn.metrics import classification_report
from torch.optim import Optimizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoModelForSequenceClassification
from ..dataloaders.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import pandas as pd
from loguru import logger
import torch
from tqdm import tqdm
import os
from src.utils import save_classification_report
import pickle
import joblib


class PhoBertTrainer:
    """
    Trainer for PhoBert model. Only use PhoBERT v2.
    """

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _load_data(train_data_path: str, val_data_path: str, test_data_path: str):
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)

        # Detached the dataframes to train texts, lables, val texts, val labels, test texts, test labels
        train_texts = train_data["text"].tolist()
        train_labels = train_data["label"].tolist()
        val_texts = val_data["text"].tolist()
        val_labels = val_data["label"].tolist()
        test_texts = test_data["text"].tolist()
        test_labels = test_data["label"].tolist()

        return (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels),
        )

    def train(
        self,
        train_tuple: tuple[list[str], list[str]],
        val_tuple: tuple[list[str], list[str]],
        epochs: int,
        batch_size: int,
        max_length: int,
        optimizer: Optimizer,
    ):
        train_texts, train_labels = train_tuple
        val_texts, val_labels = val_tuple

        train_dataset = CustomDataset(
            train_texts, train_labels, self.tokenizer, max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = CustomDataset(val_texts, val_labels, self.tokenizer, max_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.to(self.device)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Evaluation
            self.model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_predictions / total_predictions

            logger.info(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

    def evaluate(
        self,
        test_tuple: tuple[list[str], list[str]],
        batch_size: int,
        max_length: int,
        scenario: str,
        project_root: str = os.getcwd(),
    ):
        """Perform evaluation on the test set"""
        test_texts, test_labels = test_tuple

        test_dataset = CustomDataset(
            test_texts, test_labels, self.tokenizer, max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
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
        report = classification_report(
            true_labels, predictions, target_names=target_names
        )
        logger.info(report)

        # Save the classification report
        save_classification_report(
            true_labels, predictions, target_names, project_root, scenario
        )

    def save(self, project_root: str, scenario: str):
        pickle_path = os.path.join(
            project_root, "models", f"{scenario.lower()}_phobert_pickle.pkl"
        )
        with open(pickle_path, "wb") as file:
            pickle.dump(self.model, file)

        joblib_path = os.path.join(
            project_root, "models", f"{scenario.lower()}_phobert_joblib.pkl"
        )
        joblib.dump(self.model, joblib_path)

        self.model.save_pretrained(
            os.path.join(
                project_root, "models", f"{scenario.lower()}_phobert_fine_tuned"
            )
        )

        logger.info(f"Model saved to {pickle_path} and {joblib_path}")

    @staticmethod
    def load(project_root: str, scenario: str, device: torch.device):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(
                project_root, "models", f"{scenario.lower()}_phobert_fine_tuned"
            )
        )
        model.to(device)
        return model

    def run(
        self,
        epochs: int,
        batch_size: int,
        max_length: int,
        optimizer: Optimizer,
        scenario: str,
    ):
        train_tuple, val_tuple, test_tuple = self._load_data(
            "data/train.csv", "data/val.csv", "data/test.csv"
        )

        self.train(train_tuple, val_tuple, epochs, batch_size, max_length, optimizer)
        self.evaluate(test_tuple, batch_size, max_length, scenario)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
