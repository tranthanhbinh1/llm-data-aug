from abc import ABC, abstractmethod


class TrainerEvaluatorRepository(ABC):
    @abstractmethod
    def load_data(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def run_evaluation(self):
        raise NotImplementedError()
