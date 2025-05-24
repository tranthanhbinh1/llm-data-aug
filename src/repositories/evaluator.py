from abc import ABC, abstractmethod


class EvaluatorRepository(ABC):
    @abstractmethod
    def evaluate(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def run_evaluation(self, sentiment: str, prompt: str) -> float:
        raise NotImplementedError()
