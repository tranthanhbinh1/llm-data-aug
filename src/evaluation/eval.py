import os
from typing import Union
from torch.utils.data import DataLoader
from loguru import logger
from src.trainers import SVMTrainer, CNNBertHybridTrainer, BERTLSTMTrainer
from src.constants import DATA_PATH
from torch import nn


# TODO: Major ovehaul needed to properly implement promptimal and downstream eval suite
class Evaluator:
    def __init__(
        self, trainer: Union[SVMTrainer, CNNBertHybridTrainer, BERTLSTMTrainer]
    ):
        self.trainer = trainer

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> float:
        pass
