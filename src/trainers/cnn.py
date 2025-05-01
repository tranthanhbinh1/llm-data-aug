import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os
import pandas as pd
from loguru import logger
import pickle
import joblib
from torch.nn.utils.rnn import pad_sequence


class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, 128, kernel_size=5)
        self.relu = nn.ReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        conv_out = self.relu(self.conv1d(embedded))  # [batch_size, 128, seq_len-4]
        pooled = self.global_max_pool(conv_out)  # [batch_size, 128, 1]
        pooled = pooled.squeeze(-1)  # [batch_size, 128]
        fc1_out = self.relu(self.fc1(pooled))  # [batch_size, 64]
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)  # [batch_size, num_classes]
        return logits


class CNNTrainer:
    """
    Trainer for CNN text classification model
    """

    def __init__(self, model, tokenizer, vocab, device):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.device = device
        self.label_mapping = {"Positive": 0, "Neutral": 2, "Negative": 1}

    @staticmethod
    def _load_data(train_data_path, val_data_path, test_data_path):
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        test_data = pd.read_csv(test_data_path)

        # Map sentiment values
        for df in [train_data, val_data, test_data]:
            df["Sentiment"].replace(
                {0: "Positive", 1: "Negative", 2: "Neutral"}, inplace=True
            )

        # Extract texts and labels
        train_texts = train_data["emoji to text"].tolist()
        train_labels = train_data["Sentiment"].tolist()
        val_texts = val_data["emoji to text"].tolist()
        val_labels = val_data["Sentiment"].tolist()
        test_texts = test_data["emoji to text"].tolist()
        test_labels = test_data["Sentiment"].tolist()

        return (
            (train_texts, train_labels),
            (val_texts, val_labels),
            (test_texts, test_labels),
        )

    def yield_tokens(self, data_iter):
        for text in data_iter:
            yield self.tokenizer(text)

    def text_to_tensor(self, text_series):
        sequences = []
        for text in text_series:
            sequence = [self.vocab[token] for token in self.tokenizer(text)]
            sequences.append(sequence)
        return sequences

    def prepare_data(self, texts, labels):
        # Convert text to sequences
        sequences = self.text_to_tensor(texts)

        # Converting sequences to tensors and padding
        tensors = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)

        # Convert labels to tensor
        label_indices = [self.label_mapping[label] for label in labels]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return padded, label_tensor

    def train(
        self,
        train_tuple,
        val_tuple,
        epochs,
        batch_size,
        optimizer,
        criterion=nn.CrossEntropyLoss(),
    ):
        train_texts, train_labels = train_tuple
        val_texts, val_labels = val_tuple

        # Prepare data
        X_train, y_train = self.prepare_data(train_texts, train_labels)
        X_val, y_val = self.prepare_data(val_texts, val_labels)

        # Create DataLoader
        train_dataset = TensorDataset(X_train.to(self.device), y_train.to(self.device))
        val_dataset = TensorDataset(X_val.to(self.device), y_val.to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Move model to device
        self.model.to(self.device)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = correct / total
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

    def evaluate(self, test_tuple, batch_size, scenario, project_root=os.getcwd()):
        test_texts, test_labels = test_tuple

        # Prepare data
        X_test, y_test = self.prepare_data(test_texts, test_labels)

        # Create DataLoader
        test_dataset = TensorDataset(X_test.to(self.device), y_test.to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Evaluation
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate classification report
        target_names = ["Positive", "Negative", "Neutral"]
        report = classification_report(
            true_labels, predictions, target_names=target_names
        )
        logger.info(report)

        # Save the classification report if utils function exists
        try:
            from src.utils import save_classification_report

            save_classification_report(
                true_labels, predictions, target_names, project_root, scenario
            )
        except ImportError:
            logger.warning("save_classification_report not found in utils")

    def save(self, project_root, scenario):
        """Save the model to disk"""
        os.makedirs(os.path.join(project_root, "models"), exist_ok=True)

        model_path = os.path.join(project_root, "models", f"{scenario.lower()}_cnn.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save with pickle for compatibility
        pickle_path = os.path.join(
            project_root, "models", f"{scenario.lower()}_cnn_pickle.pkl"
        )
        with open(pickle_path, "wb") as file:
            pickle.dump(self.model, file)

        logger.info(f"Model saved to {model_path} and {pickle_path}")

    @staticmethod
    def load(project_root, scenario, vocab_size, embedding_dim, num_classes, device):
        """Load a saved model"""
        model = CNNModel(vocab_size, embedding_dim, num_classes)
        model.load_state_dict(
            torch.load(
                os.path.join(project_root, "models", f"{scenario.lower()}_cnn.pt"),
                map_location=device,
            )
        )
        model.to(device)
        return model

    def run(
        self,
        epochs,
        batch_size,
        optimizer,
        scenario,
        train_data_path="data/train.csv",
        val_data_path="data/val.csv",
        test_data_path="data/test.csv",
    ):
        """Run the full training and evaluation pipeline"""
        train_tuple, val_tuple, test_tuple = self._load_data(
            train_data_path, val_data_path, test_data_path
        )

        self.train(train_tuple, val_tuple, epochs, batch_size, optimizer)
        self.evaluate(test_tuple, batch_size, scenario)
        self.save(os.getcwd(), scenario)


# Example usage
if __name__ == "__main__":
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get tokenizer
    tokenizer = get_tokenizer("basic_english")

    # Build vocabulary (replace this with proper loading)
    # Placeholder, need to load data first
    df = pd.read_csv("data/train.csv")

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(df["emoji to text"]), specials=["<unk>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    # Create model and trainer
    model = CNNModel(len(vocab), embedding_dim=50, num_classes=3)
    trainer = CNNTrainer(model, tokenizer, vocab, device)

    # Train and evaluate
    optimizer = optim.Adam(model.parameters())
    trainer.run(epochs=5, batch_size=32, optimizer=optimizer, scenario="cnn_baseline")
