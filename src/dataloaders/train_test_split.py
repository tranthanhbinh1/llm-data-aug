import pandas as pd
from sklearn.model_selection import train_test_split
import os
from enum import StrEnum


class DataScenario(StrEnum):
    """
    Enum for the different data scenarios.
    """

    ORIGINAL = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "cleaned_user_reviews.csv",
    )
    UPSAMPLED = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "upsampled",
    )
    DOWNSAMPLED = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "downsampled",
    )

    LLM_GENERATED = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "llm_generated",
    )


class DatasetType(StrEnum):
    """
    Enum for the different dataset types.
    """

    TRAIN = "train"
    TEST = "test"
    VAL = "val"


class TrainTestSplit:
    """
    Class for splitting the data into training and testing sets.
    """

    @staticmethod
    def load_data(data_scenario: DataScenario):
        # Have a name for the file
        file_name = data_scenario.name.lower() + "_user_reviews.csv"
        return pd.read_csv(os.path.join(data_scenario.value, file_name))

    @staticmethod
    def custom_train_test_split(
        texts: list[str],
        labels: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        # Split the data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        return train_texts, test_texts, train_labels, test_labels

    @staticmethod
    def save_data(
        df: pd.DataFrame,
        data_scenario: DataScenario,
        dataset_type: DatasetType,
    ):
        file_name = os.path.join(
            data_scenario.value,
            f"{dataset_type}_{data_scenario.name.lower()}_user_reviews.csv",
        )
        return df.to_csv(file_name, index=False)

    @staticmethod
    def run_train_test_split(
        data_scenario: DataScenario,
        test_size: float = 0.2,
        random_state: int = 42,
        texts_column: str = "emoji to text",  # NOTE: This column has corporated emoji meanings and text
        labels_column: str = "Sentiment",
    ):
        df = TrainTestSplit.load_data(data_scenario)

        # Split the data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = (
            TrainTestSplit.custom_train_test_split(
                df[texts_column].tolist(),
                df[labels_column].tolist(),
                test_size=test_size,
                random_state=random_state,
            )
        )

        # Split the training data into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=test_size, random_state=random_state
        )

        train_data = pd.DataFrame({"text": train_texts, "label": train_labels})
        val_data = pd.DataFrame({"text": val_texts, "label": val_labels})
        test_data = pd.DataFrame({"text": test_texts, "label": test_labels})

        # Save the data
        TrainTestSplit.save_data(train_data, data_scenario, DatasetType.TRAIN)
        TrainTestSplit.save_data(val_data, data_scenario, DatasetType.VAL)
        TrainTestSplit.save_data(test_data, data_scenario, DatasetType.TEST)
