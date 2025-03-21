"""Data preprocessing utilities for Vietnamese text analysis."""

import re
import pandas as pd
from typing import Dict, List
from vncorenlp import VnCoreNLP
from underthesea import text_normalize


def normalize_repeated_words(text: str) -> str:
    """Normalize words with repeated characters."""
    return re.sub(r"(\w)(\1{2,})", r"\1", text)


def remove_non_alphanumeric(text: str) -> str:
    """Remove non-alphanumeric characters while preserving Vietnamese characters."""
    allowed_chars = r"[^\w\sA-Za-zÀÁẮẤẰẦẢẲẨÃẴẪẠẶẬĐEÊÉẾÈỀẺỂẼỄẸỆIÍÌỈĨỊOÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢUƯÚỨÙỪỦỬŨỮỤỰYÝỲỶỸỴa-z0-9.,\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF.]+"
    return re.sub(allowed_chars, "", text)


def remove_special_characters(text: str) -> str:
    """Remove special characters while preserving essential punctuation."""
    special_chars = (
        r"[\x00-\x1F\x7F" + r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + r'¢£¥€©®™""'
        "–\/‒—ñàáâäçßæøÿ]"
    )
    clean_text = re.sub(r"\.\.\.", "...", str(text))
    return re.sub(special_chars, "", clean_text)


def build_abbreviation_dict(file_path: str) -> Dict[str, str]:
    """Build dictionary from abbreviation file."""
    abbreviation_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(",")
                if len(parts) == 2:
                    abbr, full = map(str.strip, parts)
                    abbreviation_dict[abbr] = full
    return abbreviation_dict


def expand_abbreviations(text: str, abbr_dict: Dict[str, str]) -> str:
    """Expand abbreviations in text using provided dictionary."""
    return " ".join(abbr_dict.get(word, word) for word in text.split())


def preprocess_text(text: str, vncorenlp_path: str, abbr_dict: Dict[str, str]) -> str:
    """Apply full preprocessing pipeline to text."""
    # Initialize VnCoreNLP
    vncore_nlp = VnCoreNLP(vncorenlp_path)

    # Apply preprocessing steps
    text = text.lower()
    text = remove_non_alphanumeric(text)
    text = expand_abbreviations(text, abbr_dict)
    text = remove_special_characters(text)
    text = normalize_repeated_words(text)

    # Tokenize text
    tokens = vncore_nlp.tokenize(text)
    return " ".join(" ".join(sentence) for sentence in tokens)


def load_stopwords(file_path: str) -> set:
    """Load stopwords from file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return set(word.strip() for word in file.readlines())


def remove_stopwords(text: str, stop_words: set) -> str:
    """Remove stopwords from text."""
    words = text.split()
    return " ".join(word for word in words if word.lower() not in stop_words)
