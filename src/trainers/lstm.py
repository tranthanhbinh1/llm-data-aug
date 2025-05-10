import time
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from loguru import logger as logging
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from src.constants import LABEL_MAPPING


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
        seq_length,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(
            embedding_dim, hidden_dim1, batch_first=True, return_sequences=True
        )
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim2, dense_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(dense_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return F.log_softmax(x, dim=1)


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
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = pd.read_csv(data_path)
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)
        torch.backends.cudnn.deterministic = True

    def _prepare_data(self):
        # TODO: this process needs standardization
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=self.SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=self.SEED
        )

        X_train = train_data["Review"].tolist()
        y_train = train_data["Sentiment"].tolist()

        X_val = val_data["Review"].tolist()
        y_val = val_data["Sentiment"].tolist()

        X_test = test_data["Review"].tolist()
        y_test = test_data["Sentiment"].tolist()

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _split_sentences_and_labels(self):
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=self.SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=self.SEED
        )

        train_sentences = train_data["Review"].to_list()
        train_labels = train_data["Sentiment"].to_list()

        val_sentences = val_data["Review"].to_list()
        val_labels = val_data["Sentiment"].to_list()

        test_sentences = test_data["Review"].to_list()
        test_labels = test_data["Sentiment"].to_list()

        return (
            (train_sentences, train_labels),
            (
                val_sentences,
                val_labels,
            ),
            (test_sentences, test_labels),
        )

    def train(
        self, optimizer: Optimizer, criterion: nn.Module, train_loader: DataLoader
    ):
        self.model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def eval(
        self,
        criterion: nn.Module,
        val_loader: DataLoader,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ):
        val_loss = 0
        self.model.eval()

        outputs = self.model(X_val.to(self.device))
        val_loss = criterion(outputs, y_val.to(self.device))

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

        for epoch in range(epochs):
            train_loss = self.train(optimizer, criterion, train_loader)
            val_loss = self.eval(criterion, val_loader, X_val, y_val)

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
        X_val.to(self.device)
        y_val.to(self.device)

        outputs = self.model(X_val)
        _, preds = torch.max(outputs, dim=1)
        accuracy = accuracy_score(y_val, preds.cpu().numpy())
        logging.info(f"Accuracy: {accuracy:.4f}")

        return accuracy


if __name__ == "__main__":
    # Create model instance
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 50
    seq_length = X_padded.shape[1]
    hidden_dim1 = 128
    hidden_dim2 = 64
    dense_dim = 64
    output_dim = len(LABEL_MAPPING)
    dropout_rate = 0.5

    model = LSTMModel(
        vocab_size,
        embedding_dim,
        hidden_dim1,
        hidden_dim2,
        dense_dim,
        output_dim,
        dropout_rate,
        seq_length,
    )
