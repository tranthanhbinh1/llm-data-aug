from abc import ABC, abstractmethod
import pandas as pd


class PreprocessorRepository(ABC):
    @abstractmethod
    def preprocess(self, column_name: str) -> pd.DataFrame:
        raise NotImplementedError()
