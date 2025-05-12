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
import argparse
import os

from src.constants import DATA_PATH


class CNN(nn.Module):
    def __init__(
        self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx
    ):
        super().__init__()

        # Project BERT embeddings to the desired dimension
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
        # encoded = [batch size, seq len, bert_dim]
        # Project embeddings
        embedded = self.fc_input(encoded)
        # Permute for CNN [batch size, emb dim, seq len]
        embedded = embedded.permute(0, 2, 1)

        # Apply convolutions
        conved_0 = F.relu(self.conv_0(embedded))
        conved_1 = F.relu(self.conv_1(embedded))
        conved_2 = F.relu(self.conv_2(embedded))
        conved_3 = F.relu(self.conv_3(embedded))

        # Max pool each conv output
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)

        # Concatenate pooled outputs
        cat = self.dropout(
            torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1).cuda()
        )

        # Final linear layer
        return self.fc(cat)


class CNNBertHybridTrainer:
    SEED = 42

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
        self.bert_model.to(self.device)
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

    # NOTE: This is redundant, we should use the _prepare_data method instead
    def _split_sentences_and_labels(self):
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=self.SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=self.SEED
        )

        train_sentences = train_data["Review"].tolist()
        train_labels = train_data["Sentiment"].tolist()

        val_sentences = val_data["Review"].tolist()
        val_labels = val_data["Sentiment"].tolist()

        test_sentences = test_data["Review"].tolist()
        test_labels = test_data["Sentiment"].tolist()

        return (
            (train_sentences, train_labels),
            (
                val_sentences,
                val_labels,
            ),
            (test_sentences, test_labels),
        )

    def encode_tokenize(
        self,
        sentences: list,
        labels: list,
    ):
        # Encode labels
        self.le = LabelEncoder()
        encoded_labels = self.le.fit_transform(labels)

        sentence_index, input_ids, attention_masks, encoded_label_tensors = (
            self.__encoder_generator(
                self.tokenizer, sentences, encoded_labels, self.device
            )
        )

        return (
            sentence_index,
            input_ids,
            attention_masks,
            encoded_label_tensors,
        )

    @staticmethod
    def __encoder_generator(
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        sentences,
        labels,
        device,
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

        input_ids = torch.cat(input_ids, dim=0).to(device)
        attention_masks = torch.cat(attention_masks, dim=0).to(device)
        labels = torch.tensor(labels).to(device)
        sent_index = torch.tensor(sent_index).to(device)

        logging.info("Original: ", sentences[0])
        logging.info("Token IDs:", input_ids[0])

        # Sentence index, token ids, attention masks, and labels
        return sent_index, input_ids, attention_masks, labels

    def _create_loaders(
        self,
        inputs_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        encoded_label_tensors: torch.Tensor,
        batch_size: int = 128,
    ):
        train_dataset = TensorDataset(
            inputs_ids, attention_masks, encoded_label_tensors
        )
        train_sampler = RandomSampler(train_dataset)
        train_data_loader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=batch_size
        )

        return train_data_loader

    @staticmethod
    def _categorical_accuracy(preds, y, device):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """
        max_preds = preds.argmax(
            dim=1, keepdim=True
        )  # get the index of the max probability
        correct = max_preds.squeeze(1).eq(y)
        return correct.sum() / torch.FloatTensor([y.shape[0]]).to(device)

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
        cnn_model.to(self.device)
        cnn_model.train()

        for batch in tqdm(train_data_loader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            optimizer.zero_grad()

            # Get BERT output
            embeddings = self.bert_model(b_input_ids, b_input_mask)[0]

            predictions = cnn_model(embeddings)

            loss = criterion(predictions, b_labels)

            acc = self._categorical_accuracy(predictions, b_labels, self.device)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(train_data_loader), epoch_acc / len(train_data_loader)

    @staticmethod
    def _predictions_labels(preds, labels):
        pred = np.argmax(preds, axis=1).flatten()
        label = labels.flatten()
        return pred, label

    @torch.no_grad()
    def eval(
        self,
        cnn_model: CNN,
        val_data_loader: DataLoader,
        criterion: nn.Module,
    ):
        epoch_loss = 0

        all_true_labels = []
        all_pred_labels = []

        self.bert_model.eval()
        cnn_model.to(self.device)
        cnn_model.eval()

        for batch in tqdm(val_data_loader):
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_labels = batch[2].to(self.device)

            # Get BERT output with hidden states
            embeddings = self.bert_model(b_input_ids, b_input_mask)[0]

            predictions = cnn_model(embeddings)

            loss = criterion(predictions, b_labels)
            epoch_loss += loss.item()

            predictions = predictions.detach().cpu().numpy()

            label_ids = b_labels.to("cpu").numpy()

            pred, true = self._predictions_labels(predictions, label_ids)

            all_pred_labels.extend(pred)
            all_true_labels.extend(true)

        logging.info(classification_report(all_pred_labels, all_true_labels))
        avg_val_accuracy = accuracy_score(all_pred_labels, all_true_labels)
        weighted_f1_score = f1_score(
            all_pred_labels, all_true_labels, average="weighted"
        )

        avg_val_loss = epoch_loss / len(val_data_loader)

        logging.info("accuracy = {0:.2f}".format(avg_val_accuracy))

        return (
            avg_val_loss,
            avg_val_accuracy,
            weighted_f1_score,
        )

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def training_loop(
        self,
        cnn_model: CNN,
        train_data_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        val_data_loader: DataLoader,
        epochs: int = 10,
    ):
        best_weighted_f1 = float("0")

        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = self.train(
                cnn_model, train_data_loader, optimizer, criterion
            )
            valid_loss, valid_acc, weighted_f1 = self.eval(
                cnn_model, val_data_loader, criterion
            )
            end_time = time.time()

            if weighted_f1 > best_weighted_f1:
                best_weighted_f1 = weighted_f1
                # torch.save(cnn_model.state_dict(), "best_model.pth")

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            logging.info(
                f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
            )
            logging.info(
                f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
            )

            logging.info(
                f"Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc * 100:.2f}% | Weighted F1: {weighted_f1:.3f}"
            )

        return best_weighted_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(
            DATA_PATH,
            "llm_generated/gemini-2.0-flash/auggpt_upsampled_user_reviews_cleaned.csv",
        ),
    )
    args = parser.parse_args()

    # Initialize BERT with output_hidden_states=True
    bert_model = AutoModel.from_pretrained(
        "vinai/phobert-base-v2",
    )

    trainer = CNNBertHybridTrainer(
        bert_model=bert_model,
        data_path=args.data_path,
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer._prepare_data()

    # Create sentences and labels
    (
        (train_sentences, train_labels),
        (val_sentences, val_labels),
        (test_sentences, test_labels),
    ) = trainer._split_sentences_and_labels()

    # Create indexs, ids and masks
    (
        train_sent_index,
        train_input_ids,
        train_attention_masks,
        train_encoded_label_tensors,
    ) = trainer.encode_tokenize(train_sentences, train_labels)
    (
        val_sent_index,
        val_input_ids,
        val_attention_masks,
        val_encoded_label_tensors,
    ) = trainer.encode_tokenize(val_sentences, val_labels)
    (
        test_sent_index,
        test_input_ids,
        test_attention_masks,
        test_encoded_label_tensors,
    ) = trainer.encode_tokenize(test_sentences, test_labels)

    # Create loaders
    train_data_loader = trainer._create_loaders(
        train_input_ids, train_attention_masks, train_encoded_label_tensors, 128
    )

    test_data_loader = trainer._create_loaders(
        test_input_ids, test_attention_masks, test_encoded_label_tensors, 128
    )

    val_data_loader = trainer._create_loaders(
        val_input_ids, val_attention_masks, val_encoded_label_tensors, 128
    )

    EMBEDDING_DIM = 768  # BERT's hidden size
    N_FILTERS = 32
    FILTER_SIZES = [1, 2, 3, 5]
    OUTPUT_DIM = len(trainer.le.classes_)
    DROPOUT = 0.1
    PAD_IDX = trainer.tokenizer.pad_token_id

    cnn = CNN(EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    model_parameters = list(trainer.bert_model.parameters()) + list(cnn.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    weighted_f1_score = trainer.training_loop(
        cnn,
        train_data_loader,
        optimizer,
        criterion,
        val_data_loader,
        epochs=10,
    )

    print(weighted_f1_score)
