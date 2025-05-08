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

        train_sentences = list(train_data["Review"].values)
        train_labels = list(train_data["Sentiment"].values)

        val_sentences = list(val_data["Review"].values)
        val_labels = list(val_data["Sentiment"].values)

        test_sentences = list(test_data["Review"].values)
        test_labels = list(test_data["Sentiment"].values)

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
