import os
from typing import Optional
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
import torch.nn as nn
import random
from loguru import logger as logging
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from src.constants import DATA_PATH, LABEL_MAPPING, ORIGINAL_DATASET_PATH


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim1,
        hidden_dim2,
        dense_dim,
        output_dim,
        dropout_rate,
        seq_length: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim2, dense_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take only the last hidden state for classification
        x = torch.relu_(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class LSTMTrainer:
    SEED = 42

    def __init__(
        self,
        model: LSTMModel,
        data_path: str,
        tokenizer: PreTrainedTokenizer
        | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("vinai/phobert-base"),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.model = model
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
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True

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

    def load_data(self):
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self._prepare_data()

        # Convert sequences to tensors
        X_train_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X_train]
        X_val_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X_val]
        X_test_tensor = [torch.tensor(seq, dtype=torch.long) for seq in X_test]

        # Convert labels to tensors
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Pad sequences
        X_train_padded = pad_sequence(X_train_tensor, batch_first=True)
        X_val_padded = pad_sequence(X_val_tensor, batch_first=True)
        X_test_padded = pad_sequence(X_test_tensor, batch_first=True)

        return (
            (X_train_padded, y_train_tensor),
            (X_val_padded, y_val_tensor),
            (X_test_padded, y_test_tensor),
        )

    def train(
        self, optimizer: Optimizer, criterion: nn.Module, train_loader: DataLoader
    ):
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_x, batch_y in progress_bar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(1, num_batches)  # Return average loss

    @torch.no_grad()
    def eval(
        self,
        criterion: nn.Module,
        val_loader: DataLoader,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ):
        self.model.eval()
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        outputs = self.model(X_val)
        val_loss = criterion(outputs, y_val)

        return val_loss

    def training_loop(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int = 10,
        patience: int = 10,
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        self.model.to(self.device)
        criterion = criterion.to(self.device)  # Move criterion to device

        epoch_progress = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_progress:
            train_loss = self.train(optimizer, criterion, train_loader)
            val_loss = self.eval(criterion, val_loader, X_val, y_val)

            # Update progress bar with metrics
            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                patience=patience_counter,
            )

            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    @torch.no_grad()
    def evaluate(self, X_val: torch.Tensor, y_val: torch.Tensor):
        self.model.eval()
        self.model.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        logging.info("Evaluating model...")
        outputs = self.model(X_val)
        _, preds = torch.max(outputs, dim=1)

        # Count predictions per class
        pred_counts = np.bincount(preds.cpu().numpy(), minlength=len(LABEL_MAPPING))
        logging.info(f"Predictions per class: {pred_counts}")

        accuracy = accuracy_score(y_val.cpu().numpy(), preds.cpu().numpy())
        logging.info(f"Accuracy: {accuracy:.4f}")

        logging.info(classification_report(y_val.cpu().numpy(), preds.cpu().numpy()))

        return accuracy


if __name__ == "__main__":
    # Create model instance
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    vocab_size = len(tokenizer.get_vocab())  # Get actual vocab size from tokenizer
    logging.info(f"Tokenizer vocabulary size: {vocab_size}")

    embedding_dim = 50
    hidden_dim1 = 128
    hidden_dim2 = 64
    dense_dim = 64
    output_dim = len(LABEL_MAPPING)
    dropout_rate = 0.5

    lstm = LSTMModel(
        vocab_size,
        embedding_dim,
        hidden_dim1,
        hidden_dim2,
        dense_dim,
        output_dim,
        dropout_rate,
    )

    trainer = LSTMTrainer(
        model=lstm,
        data_path=os.path.join(
            DATA_PATH,
            "llm_generated/gemini-2.0-flash/auggpt_upsampled_user_reviews_cleaned.csv",
        ),
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
    )

    (
        (X_train_padded, y_train_tensor),
        (X_val_padded, y_val_tensor),
        (X_test_padded, y_test_tensor),
    ) = trainer.load_data()

    train_loader = DataLoader(
        TensorDataset(X_train_padded, y_train_tensor),
        batch_size=16,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val_padded, y_val_tensor),
        batch_size=16,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)

    trainer.training_loop(
        optimizer,
        criterion,
        train_loader,
        val_loader,
        X_val_padded,
        y_val_tensor,
        epochs=10,
    )

    trainer.evaluate(X_val_padded, y_val_tensor)
