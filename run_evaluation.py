from transformers import AutoTokenizer, AutoModel
import torch
from src.constants import ORIGINAL_DATASET_PATH
from src.preprocess.text_preprocessor import TextPreprocessor
import argparse
from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.utils import get_instructor_instance
from src.evaluation.eval import Evaluator
import pandas as pd

# Wrapper script to run the evaluator
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", type=str, required=False, default="neutral")
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    trainer_evaluator = CNNBertHybridTrainer(
        bert_model=AutoModel.from_pretrained("vinai/phobert-base-v2"),
        preprocessor=TextPreprocessor(data=pd.read_csv(ORIGINAL_DATASET_PATH)),
        data_path=ORIGINAL_DATASET_PATH,
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze_bert=True,
    )
    evaluator = Evaluator(
        trainer_evaluator=trainer_evaluator,
        similarity_evaluator=SimiarityEvaluator(
            auggpt_runner=AugGptRunner(get_instructor_instance()),
        ),
    )
    result = evaluator.evaluate(
        sentiment=args.sentiment,
        prompt=args.prompt,
    )

    print(result)
