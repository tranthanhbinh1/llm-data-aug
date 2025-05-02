import pandas as pd
import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder


class CNNBertHybridTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _load_data(self, data_path: str):
        data = pd.read_csv(data_path)

        # TODO: this process needs standardization
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
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

    def _split_sentences_and_labales(self, data_path: str):
        data = pd.read_csv(data_path)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=42
        )

        train_sentences = list(train_data["emoji to text"].values)
        train_labels = list(train_data["Sentiment"].values)

        val_sentences = list(val_data["emoji to text"].values)
        val_labels = list(val_data["Sentiment"].values)

        test_sentences = list(test_data["emoji to text"].values)
        test_labels = list(test_data["Sentiment"].values)

        return (
            (train_sentences, train_labels),
            (
                val_sentences,
                val_labels,
            ),
            (test_sentences, test_labels),
        )

    def _tokenize(self, train_sentences: list, train_labels: list, val_labels: list):
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # Encode labels
        le = LabelEncoder()
        encoded_train_labels = le.fit_transform(train_labels)
        encoded_val_labels = le.transform(val_labels)

        train_encodings = tokenizer(
            train_sentences,
            truncation=True,
            padding=True,
        )

    @staticmethod
    def _encoder_generator(tokenizer, sentences, labels):
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
