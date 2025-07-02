from datetime import datetime
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch
import torch.nn as nn
import random
from loguru import logger as logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from src.constants import LABEL_MAPPING, ORIGINAL_DATASET_PATH, PROJECT_ROOT, SEED

from src.repositories.preprocessor import PreprocessorRepository
from src.repositories.trainer import TrainerEvaluatorRepository


class BERTLSTMModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        hidden_dim1: int,
        hidden_dim2: int,
        dense_dim: int,
        output_dim: int,
        dropout_rate: float,
        freeze_bert: bool = True,
    ):
        super().__init__()
        # Load pretrained BERT model
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Get embedding dimension from BERT model
        embedding_dim = self.bert.config.hidden_size

        # Freeze BERT weights if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # LSTM layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)

        # Classification layers
        self.dense1 = nn.Linear(hidden_dim2, dense_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(dense_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # Pass through LSTM layers
        lstm1_output, _ = self.lstm1(sequence_output)
        lstm2_output, _ = self.lstm2(lstm1_output)

        # Get last hidden state from second LSTM
        last_hidden = lstm2_output[:, -1, :]

        # Classification head
        x = torch.relu_(self.dense1(last_hidden))
        x = self.dropout(x)
        x = self.dense2(x)

        return x


class BERTLSTMTrainer(TrainerEvaluatorRepository):
    def __init__(
        self,
        model: BERTLSTMModel,
        data_path: str,
        preprocessor: PreprocessorRepository,
        tokenizer_name: str = "vinai/phobert-base-v2",
        max_length: int = 128,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.device = device
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.preprocessor = preprocessor
        # Load data
        self.data = pd.read_csv(data_path)
        initial_rows = len(self.data)
        self.data = self.data.dropna(subset=["sentence", "sentiment"])
        self.data = self.data.reset_index(drop=True)

        removed_rows = initial_rows - len(self.data)
        if removed_rows > 0:
            logging.warning(f"Removed {removed_rows} rows containing NaN values")

        # Set random seeds for reproducibility
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

    def _prepare_data(self, column_name: str):
        self.data = self.preprocessor.preprocess(column_name)
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=SEED
        )

        logging.info(f"Train set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")
        logging.info(f"Test set size: {len(test_data)}")

        train_encodings = self.tokenizer(
            train_data["tokenized_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        val_encodings = self.tokenizer(
            val_data["tokenized_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        test_encodings = self.tokenizer(
            test_data["tokenized_text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        logging.info(f"Unique labels in train data: {train_data['sentiment'].unique()}")

        train_sentiment_mapped = [
            LABEL_MAPPING[label] for label in train_data["sentiment"]
        ]
        val_sentiment_mapped = [LABEL_MAPPING[label] for label in val_data["sentiment"]]
        test_sentiment_mapped = [
            LABEL_MAPPING[label] for label in test_data["sentiment"]
        ]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_sentiment_mapped)
        y_val = label_encoder.transform(val_sentiment_mapped)
        y_test = label_encoder.transform(test_sentiment_mapped)

        train_label_counts = np.bincount(y_train)
        logging.info(f"Training label distribution by class: {train_label_counts}")
        logging.info(f"Label encoder classes: {label_encoder.classes_}")
        logging.info(
            f"Final label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
        )

        return (
            (train_encodings, y_train),
            (val_encodings, y_val),
            (test_encodings, y_test),
        )

    def load_data(self, column_name: str):
        (train_encodings, y_train), (val_encodings, y_val), (test_encodings, y_test) = (
            self._prepare_data(column_name)
        )

        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(
            train_encodings.input_ids, train_encodings.attention_mask, y_train_tensor
        )

        val_dataset = TensorDataset(
            val_encodings.input_ids, val_encodings.attention_mask, y_val_tensor
        )

        test_dataset = TensorDataset(
            test_encodings.input_ids, test_encodings.attention_mask, y_test_tensor
        )

        return train_dataset, val_dataset, test_dataset

    def train(
        self, optimizer: Optimizer, criterion: nn.Module, train_loader: DataLoader
    ):
        self.model.train()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch_input_ids, batch_attention_mask, batch_labels in progress_bar:
            # Move tensors to device
            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_input_ids, batch_attention_mask)

            loss = criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Training accuracy: {train_acc:.4f}")

        pred_counts = np.bincount(all_preds, minlength=3)  # 3 classes for sentiment
        logging.info(f"Prediction distribution: {pred_counts}")

        return total_loss / max(1, num_batches)  # Return average loss

    @torch.no_grad()
    def evaluate(self, criterion: nn.Module, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_input_ids, batch_attention_mask, batch_labels in data_loader:
            batch_input_ids = batch_input_ids.to(self.device)
            batch_attention_mask = batch_attention_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            outputs = self.model(batch_input_ids, batch_attention_mask)

            # Calculate loss
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(data_loader)

        pred_counts = np.bincount(all_preds, minlength=3)  # 3 classes for sentiment

        return avg_loss, accuracy, all_preds, all_labels, pred_counts

    def training_loop(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 5,
    ):
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        self.model.to(self.device)
        criterion = criterion.to(self.device)

        epoch_progress = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_progress:
            # Training phase
            train_loss = self.train(optimizer, criterion, train_loader)

            val_loss, val_acc, _, _, pred_counts = self.evaluate(criterion, val_loader)

            epoch_progress.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
                patience=patience_counter,
            )

            logging.info(
                f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Pred Dist: {pred_counts}"
            )

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                torch.save(
                    self.model.state_dict(),
                    f"{PROJECT_ROOT}/models/best_lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
                )
                logging.info("Model improved, saved checkpoint.")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logging.info("Restored best model from checkpoint.")

    def final_evaluate(self, test_loader: DataLoader):
        logging.info("Evaluating model on test set...")

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)

        test_loss, test_acc, test_preds, test_labels, pred_counts = self.evaluate(
            criterion, test_loader
        )

        weighted_f1_score = f1_score(test_labels, test_preds, average="weighted")

        logging.info(f"Test accuracy: {test_acc:.4f}")
        logging.info(f"Test loss: {test_loss:.4f}")
        logging.info(f"Predictions per class: {pred_counts}")
        logging.info(f"F1 Score: {weighted_f1_score:.4f}")
        logging.info("\n" + str(classification_report(test_labels, test_preds)))

        return float(weighted_f1_score)

    def run_training(self, column_name: str, batch_size=128, epochs=10, patience=3):
        """
        Run the complete training pipeline from data loading to evaluation

        Args:
            batch_size: Size of batches for training
            epochs: Number of training epochs
            patience: Number of epochs to wait before early stopping

        Returns:
            float: Weighted F1 score from test set evaluation
        """
        train_dataset, val_dataset, test_dataset = self.load_data(column_name)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
        )

        # Setup training
        criterion = nn.CrossEntropyLoss()

        # Check if BERT is frozen
        freeze_bert = not any(p.requires_grad for p in self.model.bert.parameters())

        if not freeze_bert:
            # Parameters with different learning rates
            bert_params = list(self.model.bert.parameters())
            non_bert_params = (
                list(self.model.lstm1.parameters())
                + list(self.model.lstm2.parameters())
                + list(self.model.dense1.parameters())
                + list(self.model.dropout.parameters())
                + list(self.model.dense2.parameters())
            )

            optimizer = torch.optim.AdamW(
                [
                    {"params": bert_params, "lr": 2e-5},  # Lower learning rate for BERT
                    {
                        "params": non_bert_params,
                        "lr": 1e-3,
                    },  # Higher learning rate for LSTM layers
                ],
                weight_decay=1e-5,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-3, weight_decay=1e-5
            )

        self.training_loop(
            optimizer,
            criterion,
            train_loader,
            val_loader,
            epochs=epochs,
            patience=patience,
        )

        weighted_f1_score = self.final_evaluate(test_loader)

        return weighted_f1_score

    def run_evaluation(self, data_path: str, column_name: str = "sentence") -> float:
        """Run evaluation on the model using the given sentiment and prompt.

        Args:
            sentiment: The sentiment to evaluate on
            prompt: The prompt to use for evaluation

        Returns:
            float: The weighted F1 score from the evaluation
        """
        # Configuration
        bert_model_name = "vinai/phobert-base-v2"
        hidden_dim1 = 128
        hidden_dim2 = 64
        dense_dim = 64
        output_dim = 3  # Number of sentiment classes
        dropout_rate = 0.5
        freeze_bert = True  # Freeze BERT weights

        model = BERTLSTMModel(
            bert_model_name=bert_model_name,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dense_dim=dense_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            freeze_bert=freeze_bert,
        )

        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"BERT parameters frozen: {freeze_bert}")

        # TODO: dirty import, fix later
        from src.preprocess.text_preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor(data=pd.read_csv(data_path))
        trainer = BERTLSTMTrainer(
            model=model,
            data_path=data_path,
            tokenizer_name=bert_model_name,
            max_length=128,
            preprocessor=preprocessor,
        )

        return float(
            trainer.run_training(column_name, batch_size=128, epochs=10, patience=3)
        )


if __name__ == "__main__":
    import argparse
    from src.preprocess.text_preprocessor import TextPreprocessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=ORIGINAL_DATASET_PATH)
    args = parser.parse_args()

    trainer = BERTLSTMTrainer(
        model=BERTLSTMModel(
            bert_model_name="vinai/phobert-base-v2",
            hidden_dim1=128,
            hidden_dim2=64,
            dense_dim=64,
            output_dim=3,  # Number of sentiment classes
            dropout_rate=0.5,
            freeze_bert=True,
        ),
        data_path=args.data_path,
        preprocessor=TextPreprocessor(data=pd.read_csv(args.data_path)),
    )
    weighted_f1 = trainer.run_evaluation(args.data_path, args.column_name)
    print(weighted_f1)
