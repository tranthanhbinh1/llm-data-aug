"""Model training and evaluation utilities."""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AdamW
from sklearn.metrics import classification_report, confusion_matrix


class SentimentTrainer:
    """Trainer class for sentiment analysis model."""

    def __init__(
        self, model: PreTrainedModel, device: torch.device, learning_rate: float = 2e-5
    ):
        """
        Initialize the trainer.

        Args:
            model: The pre-trained model to fine-tune
            device: Device to use for training
            learning_rate: Learning rate for optimization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (train_loss, val_loss, val_accuracy)
        """
        self.model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        # Validation
        val_loss, val_accuracy = self.evaluate(val_loader)
        avg_train_loss = train_loss / len(train_loader)

        return avg_train_loss, val_loss, val_accuracy

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset.

        Args:
            data_loader: DataLoader for prediction

        Returns:
            Tuple of (predictions, true_labels)
        """
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                _, predicted = torch.max(outputs.logits, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        return np.array(predictions), np.array(true_labels)

    def get_metrics(
        self, predictions: np.ndarray, true_labels: np.ndarray, target_names: List[str]
    ) -> Dict:
        """
        Calculate classification metrics.

        Args:
            predictions: Model predictions
            true_labels: True labels
            target_names: Names of the target classes

        Returns:
            Dictionary containing classification report and confusion matrix
        """
        return {
            "classification_report": classification_report(
                true_labels, predictions, target_names=target_names
            ),
            "confusion_matrix": confusion_matrix(true_labels, predictions),
        }
