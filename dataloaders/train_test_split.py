from typing import Literal
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
        file_name = data_scenario.value + data_scenario.name.lower() + ".csv"
        return pd.read_csv(file_name)

    @staticmethod
    def custom_train_test_split(
        df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Split the data into training and testing sets
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train_df, test_df

    @staticmethod
    def save_data(
        df: pd.DataFrame,
        data_scenario: DataScenario,
        dataset_type: DatasetType,
    ):
        file_name = (
            data_scenario.value
            + f"{dataset_type}"
            + data_scenario.name.lower()
            + ".csv"
        )
        df.to_csv(file_name, index=False)

    @staticmethod
    def run_train_test_split(
        data_scenario: DataScenario, test_size: float = 0.2, random_state: int = 42
    ):
        df = TrainTestSplit.load_data(data_scenario)
        train_df, test_df = TrainTestSplit.custom_train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        TrainTestSplit.save_data(train_df, data_scenario, DatasetType.TRAIN)
        TrainTestSplit.save_data(test_df, data_scenario, DatasetType.TEST)
