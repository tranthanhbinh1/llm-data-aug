from src.repositories.preprocessor import PreprocessorRepository
import pandas as pd
import os
import re
from src.constants import DATA_PATH, PROJECT_ROOT
from vncorenlp import VnCoreNLP


# TODO: Needs to make this a Singleton, because a new VnCoreNLP instance is created for each instance of TextPreprocessor
class TextPreprocessor(PreprocessorRepository):
    def __init__(self, data: pd.DataFrame):
        self.data = data
        # Đường dẫn đến tệp chứa danh sách các từ viết tắt và định nghĩa tương ứng
        self.abbreviations_path = os.path.join(DATA_PATH, "abbreviate.txt")
        self.abbr = self.build_dictionary_from_file(self.abbreviations_path)

        # Initialize VnCoreNLP
        self.vncorenlp_path = os.path.join(
            PROJECT_ROOT, "notebooks", "VnCoreNLP", "VnCoreNLP-1.2.jar"
        )  # Thay đường dẫn đến VnCoreNLP.jar tại đây
        self.vncorenlp = VnCoreNLP(self.vncorenlp_path)

    # Hàm normalize_repeated_words
    def normalize_repeated_words(self, text):
        # Sử dụng biểu thức chính quy để tìm và thay thế các từ viết kéo dài
        normalized_text = re.sub(r"(\w)(\1{2,})", r"\1", text)
        return normalized_text

    # Hàm remove_non_alphanumeric
    def remove_non_alphanumeric(self, string):
        allowed_characters = r"[^\w\sA-Za-zÀÁẮẤẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-z0-9.,\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.]+"
        return re.sub(allowed_characters, "", string)

    # Hàm xử lý các ký tự đặc biệt
    def remove_special_characters(self, text):
        special_characters = (
            r"[\x00-\x1F\x7F"
            + r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            + r"¢£¥€©®™“”‘’–\/‒—ñàáâäçßæøÿ]"
        )
        clean_text = re.sub(r"\.\.\.", "...", str(text))
        clean_text = re.sub(special_characters, "", clean_text)
        return clean_text

    # Hàm build_dictionary_from_file
    def build_dictionary_from_file(self, file_path):
        abbreviation_dict = {}

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 2:
                        abbreviation, full_form = map(str.strip, parts)
                        abbreviation_dict[abbreviation] = full_form

        return abbreviation_dict

    # Function to expand abbreviations in a given text
    def expand_abbr(self, text, abbr_dict):
        return " ".join(abbr_dict.get(word, word) for word in text.split())

    # Tokenize text in each row of the DataFrame
    def tokenize_text(self, text):
        tokens = self.vncorenlp.tokenize(text)
        return " ".join(" ".join(sentence) for sentence in tokens)

    def preprocess(self) -> pd.DataFrame:
        # Remove float objects
        self.data = self.data[self.data["Review"].apply(lambda x: isinstance(x, str))]
        self.data["Review"] = self.data["Review"].apply(
            str.lower
        )  # Chuyển đổi văn bản thành chữ thường trước khi xử lý
        self.data["Review"] = self.data["Review"].apply(self.remove_non_alphanumeric)
        self.data["Review"] = self.data["Review"].apply(
            lambda x: self.expand_abbr(x, self.abbr)
        )
        self.data["Review"] = self.data["Review"].apply(self.remove_special_characters)
        self.data["Review"] = self.data["Review"].apply(self.normalize_repeated_words)
        self.data["tokenized_text"] = self.data["Review"].apply(self.tokenize_text)

        return self.data
