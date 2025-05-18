from abc import ABC, abstractmethod
import pandas as pd


class PreprocessorRepository(ABC):
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        raise NotImplementedError()
