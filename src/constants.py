import os

NUM_REPHRASED_SENTENCES = 6

SEED = 42

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

ORIGINAL_DATASET_PATH = os.path.join(DATA_PATH, "cleaned_user_reviews.csv")

LABEL_MAPPING = {
    "positive": 1,
    "neutral": 2,
    "negative": 0,
}
