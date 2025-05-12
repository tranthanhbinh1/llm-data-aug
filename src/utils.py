import os
import re
from .constants import DATA_PATH, PROJECT_ROOT
from vncorenlp import VnCoreNLP

import pandas as pd
from sklearn.metrics import classification_report

import instructor
from google import genai


def save_classification_report(
    true_labels, predictions, target_names, project_root, data_scenario
):
    # Save the classification report
    report = classification_report(true_labels, predictions, target_names=target_names)
    classification_report_df = pd.DataFrame(report).transpose()
    classification_report_df.to_csv(
        os.path.join(
            project_root,
            "reports",
            f"{data_scenario.name.lower()}_phobert_v1_classification_report.csv",
        )
    )


def get_instructor_instance():
    return instructor.from_genai(genai.Client(api_key=os.getenv("GOOGLE_AI_API_KEY")))


# Hàm normalize_repeated_words
def normalize_repeated_words(text):
    # Sử dụng biểu thức chính quy để tìm và thay thế các từ viết kéo dài
    normalized_text = re.sub(r"(\w)(\1{2,})", r"\1", text)
    return normalized_text


# Hàm remove_non_alphanumeric
def remove_non_alphanumeric(string):
    allowed_characters = r"[^\w\sA-Za-zÀÁẮẤẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-z0-9.,\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.]+"
    return re.sub(allowed_characters, "", string)


# Hàm xử lý các ký tự đặc biệt
def remove_special_characters(text):
    special_characters = (
        r"[\x00-\x1F\x7F"
        + r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        + r"¢£¥€©®™“”‘’–\/‒—ñàáâäçßæøÿ]"
    )
    clean_text = re.sub(r"\.\.\.", "...", str(text))
    clean_text = re.sub(special_characters, "", clean_text)
    return clean_text


# Hàm build_dictionary_from_file
def build_dictionary_from_file(file_path):
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
def expand_abbr(text, abbr_dict):
    return " ".join(abbr_dict.get(word, word) for word in text.split())


# Đường dẫn đến tệp chứa danh sách các từ viết tắt và định nghĩa tương ứng
abbreviations_path = os.path.join(DATA_PATH, "abbreviate.txt")
abbr = build_dictionary_from_file(abbreviations_path)

# Initialize VnCoreNLP
vncorenlp_path = os.path.join(
    PROJECT_ROOT, "notebooks", "VnCoreNLP", "VnCoreNLP-1.2.jar"
)  # Thay đường dẫn đến VnCoreNLP.jar tại đây
vncorenlp = VnCoreNLP(vncorenlp_path)


# Tokenize text in each row of the DataFrame
def tokenize_text(text):
    tokens = vncorenlp.tokenize(text)
    return " ".join(" ".join(sentence) for sentence in tokens)
