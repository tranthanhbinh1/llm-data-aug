from abc import ABC, abstractmethod


class TrainerRepository(ABC):
    @abstractmethod
    def load_data(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()
