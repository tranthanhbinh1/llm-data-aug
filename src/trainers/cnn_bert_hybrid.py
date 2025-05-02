import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


class CNN(nn.Module):
    def __init__(
        self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx
    ):
        super().__init__()

        self.fc_input = nn.Linear(embedding_dim, embedding_dim)

        self.conv_0 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=n_filters,
            kernel_size=filter_sizes[0],
        )

        self.conv_1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=n_filters,
            kernel_size=filter_sizes[1],
        )

        self.conv_2 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=n_filters,
            kernel_size=filter_sizes[2],
        )

        self.conv_3 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=n_filters,
            kernel_size=filter_sizes[3],
        )

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded):
        # embedded = [batch size, sent len, emb dim]
        embedded = self.fc_input(encoded)
        # print(embedded.shape)

        embedded = embedded.permute(0, 2, 1)
        # print(embedded.shape)

        # embedded = [batch size, emb dim, sent len]

        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

        # pooled_n = [batch size, n_fibatlters]

        cat = self.dropout(
            torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1).cuda()
        )

        # cat = [batch size, n_filters * len(filter_sizes)]

        result = self.fc(cat)

        # print(result.shape)

        return result


class CNNBertHybridTrainer:
    def __init__(
        self,
        bert_model,
        data_path: str,
        tokenizer: PreTrainedTokenizer
        | PreTrainedTokenizerFast = AutoTokenizer.from_pretrained("vinai/phobert-base"),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.data = pd.read_csv(data_path)

    def _prepare_data(self):
        # TODO: this process needs standardization
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42
        )

        X_train = train_data["Review"].tolist()
        y_train = train_data["Sentiment"].tolist()

        X_val = val_data["Review"].tolist()
        y_val = val_data["Sentiment"].tolist()

        X_test = test_data["Review"].tolist()
        y_test = test_data["Sentiment"].tolist()

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _split_sentences_and_labales(self):
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42
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

    def _tokenize(
        self,
        sentences: list,
        labels: list,
    ):
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)

        sentence_index, input_ids, attention_masks, encoded_label_tensors = (
            self._encoder_generator(self.tokenizer, sentences, encoded_labels)
        )

        return (
            sentence_index,
            input_ids,
            attention_masks,
            encoded_label_tensors,
        )

    @staticmethod
    def _encoder_generator(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, sentences, labels
    ):
        sent_index = []
        input_ids = []
        attention_masks = []

        for index, sent in enumerate(sentences):
            sent_index.append(index)

            encoded_dict = tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=20,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoded_dict["input_ids"])

            attention_masks.append(encoded_dict["attention_mask"])

        input_ids = torch.cat(input_ids, dim=0).cuda()
        attention_masks = torch.cat(attention_masks, dim=0).cuda()
        labels = torch.tensor(labels).cuda()
        sent_index = torch.tensor(sent_index).cuda()

        # Sentence index, token ids, attention masks, and labels
        return sent_index, input_ids, attention_masks, labels

    @staticmethod
    def _categorical_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(
            dim=1, keepdim=True
        )  # get the index of the max probability
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.FloatTensor([y.shape[0]]).cuda()

    def train(
        self,
        cnn_model: CNN,
        train_data_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
    ):
        epoch_loss = 0
        epoch_acc = 0

        self.bert_model.train()
        cnn_model.train()

        for batch in tqdm(train_data_loader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            optimizer.zero_grad()

            embedded = self.bert_model(b_input_ids, b_input_mask)[0]

            predictions = cnn_model(embedded)

            loss = criterion(predictions, b_labels)

            acc = self._categorical_accuracy(predictions, b_labels)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(train_data_loader), epoch_acc / len(train_data_loader)

    # TODO: Eval, Training Loop, Saving
