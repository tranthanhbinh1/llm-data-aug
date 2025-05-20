import numpy as np
from sklearn.metrics import classification_report, f1_score
from torch.optim import Optimizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.repositories.preprocessor import PreprocessorRepository
from ..dataloaders.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import pandas as pd
from loguru import logger
import torch
from tqdm import tqdm
import os
from src.utils import save_classification_report
import pickle
import joblib
from sklearn.model_selection import train_test_split
from src.repositories.trainer import TrainerRepository
from loguru import logger as logging
from src.constants import SEED, DATA_PATH
import argparse


# TODO: needs fixing
class PhoBertTrainer(TrainerRepository):
    """
    Trainer for PhoBert model. Only use PhoBERT v2.
    """

    def __init__(
        self,
        model,
        data_path: str,
        preprocessor: PreprocessorRepository,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.data = pd.read_csv(data_path)
        self.preprocessor = preprocessor

    def _prepare_data(self):
        self.data = self.preprocessor.preprocess()
        # Split data into train, validation and test sets
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=SEED
        )

        # Extract sentences and labels
        train_sentences = train_data["tokenized_text"].tolist()
        train_labels = train_data["Sentiment"].tolist()

        val_sentences = val_data["tokenized_text"].tolist()
        val_labels = val_data["Sentiment"].tolist()

        test_sentences = test_data["tokenized_text"].tolist()
        test_labels = test_data["Sentiment"].tolist()

        # Log dataset sizes
        logging.info(f"Train set size: {len(train_sentences)}")
        logging.info(f"Validation set size: {len(val_sentences)}")
        logging.info(f"Test set size: {len(test_sentences)}")

        return (
            (train_sentences, train_labels),
            (val_sentences, val_labels),
            (test_sentences, test_labels),
        )

    def load_data(self):
        self.data = self.preprocessor.preprocess()
        train_data, test_data = train_test_split(
            self.data, test_size=0.2, random_state=SEED
        )
        train_data, val_data = train_test_split(
            train_data, test_size=0.2, random_state=SEED
        )

        # Detached the dataframes to train texts, lables, val texts, val labels, test texts, test labels
        train_sentences = train_data["tokenized_text"].tolist()
        train_labels = train_data["Sentiment"].tolist()
        val_sentences = val_data["tokenized_text"].tolist()
        val_labels = val_data["Sentiment"].tolist()
        test_sentences = test_data["tokenized_text"].tolist()
        test_labels = test_data["Sentiment"].tolist()

        return (
            (train_sentences, train_labels),
            (val_sentences, val_labels),
            (test_sentences, test_labels),
        )

    def train(
        self,
        train_tuple: tuple[list[str], list[str]],
        val_tuple: tuple[list[str], list[str]],
        epochs: int,
        batch_size: int,
        max_length: int,
        optimizer: Optimizer,
    ):
        train_sentences, train_labels = train_tuple
        val_sentences, val_labels = val_tuple

        train_dataset = CustomDataset(
            train_sentences, train_labels, self.tokenizer, max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = CustomDataset(
            val_sentences, val_labels, self.tokenizer, max_length
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.to(self.device)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Evaluation
            self.model.eval()
            val_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                    logits = outputs.logits

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct_predictions / total_predictions

            logger.info(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

    def evaluate(
        self,
        test_tuple: tuple[list[str], list[str]],
        batch_size: int,
        max_length: int,
    ):
        """Perform evaluation on the test set"""
        test_texts, test_labels = test_tuple

        test_dataset = CustomDataset(
            test_texts, test_labels, self.tokenizer, max_length
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                logits = outputs.logits

                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Convert predictions and true labels to numpy arrays
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Calculate classification report
        target_names = ["Label 0", "Label 1", "Label 2"]  # Specify label names
        report = classification_report(
            true_labels, predictions, target_names=target_names
        )
        logger.info(report)

        # TODO: Save the classification report

        weighted_f1 = f1_score(true_labels, predictions, average="weighted")
        return weighted_f1

    def save(self, project_root: str, scenario: str):
        pickle_path = os.path.join(
            project_root, "models", f"{scenario.lower()}_phobert_pickle.pkl"
        )
        with open(pickle_path, "wb") as file:
            pickle.dump(self.model, file)

        joblib_path = os.path.join(
            project_root, "models", f"{scenario.lower()}_phobert_joblib.pkl"
        )
        joblib.dump(self.model, joblib_path)

        self.model.save_pretrained(
            os.path.join(
                project_root, "models", f"{scenario.lower()}_phobert_fine_tuned"
            )
        )

        logger.info(f"Model saved to {pickle_path} and {joblib_path}")

    @staticmethod
    def load(project_root: str, scenario: str, device: torch.device):
        model = AutoModelForSequenceClassification.from_pretrained(
            os.path.join(
                project_root, "models", f"{scenario.lower()}_phobert_fine_tuned"
            )
        )
        model.to(device)
        return model

    def main(
        self,
        epochs: int,
        batch_size: int,
        max_length: int,
        optimizer: Optimizer,
    ):
        train_tuple, val_tuple, test_tuple = self.load_data()

        self.train(train_tuple, val_tuple, epochs, batch_size, max_length, optimizer)
        self.evaluate(test_tuple, batch_size, max_length)


if __name__ == "__main__":
    from src.preprocess import TextPreprocessor

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

    preprocessor = TextPreprocessor(data=pd.read_csv(args.data_path))

    model = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base-v2")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    trainer = PhoBertTrainer(
        model=model,
        data_path=args.data_path,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        device=device,
    )

    trainer.main(
        epochs=10,
        batch_size=16,
        max_length=128,
        optimizer=optimizer,
    )

    # TODO: needs fixing
