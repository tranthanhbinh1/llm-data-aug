"""Main training script for Vietnamese sentiment analysis."""

import os
import torch
import pandas as pd
from typing import Tuple
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocessing import (
    preprocess_text,
    build_abbreviation_dict,
    load_stopwords,
    remove_stopwords,
)
from dataset import SentimentDataset
from model import SentimentTrainer
from visualization import plot_confusion_matrix, plot_roc_curves, plot_training_history


def load_data(data_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(data_path)
    df["emoji to text"] = df["emoji to text"].astype(str)
    return df


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.2
) -> Tuple[list, list, list, list, list, list]:
    """Split data into train, validation, and test sets."""
    texts = df["emoji to text"].tolist()
    labels = df["Sentiment"].tolist()

    # First split into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )

    # Then split train into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=42
    )

    return (train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)


def main():
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Set paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data")
    models_path = os.path.join(project_root, "models")

    # Load and prepare data
    df = load_data(os.path.join(data_path, "cleaned_user_reviews.csv"))
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = (
        prepare_data(df)
    )

    # Initialize tokenizer and model
    model_name = "vinai/phobert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Create datasets and dataloaders
    batch_size = 32
    max_length = 128

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize trainer
    trainer = SentimentTrainer(model, device)

    # Training loop
    epochs = 10
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        train_loss, val_loss, val_accuracy = trainer.train_epoch(
            train_loader, val_loader
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    predictions, true_labels = trainer.predict(test_loader)
    metrics = trainer.get_metrics(
        predictions, true_labels, target_names=["Negative", "Positive", "Neutral"]
    )

    print("\nTest Set Metrics:")
    print(metrics["classification_report"])

    # Plot confusion matrix and ROC curves
    plot_confusion_matrix(
        metrics["confusion_matrix"], target_names=["Negative", "Positive", "Neutral"]
    )
    plot_roc_curves(
        true_labels, predictions, target_names=["Negative", "Positive", "Neutral"]
    )

    # Save the model
    model_save_path = os.path.join(models_path, "fine_tuned_phobert_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\nModel saved to {model_save_path}")


if __name__ == "__main__":
    main()
