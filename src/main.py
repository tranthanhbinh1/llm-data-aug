from src.evaluation.eval import Evaluator
from src.evaluation.similarity_evaluator import SimiarityEvaluator
from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.preprocess.text_preprocessor import TextPreprocessor
from src.constants import ORIGINAL_DATASET_PATH
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from src.synthesizer.aug_gpt_generator import AugGptRunner
from src.utils import get_instructor_instance
from promptimal.app import App
from promptimal.promptimal import generate_evaluator


def main():
    trainer_evaluator = CNNBertHybridTrainer(
        bert_model=AutoModel.from_pretrained("vinai/phobert-base-v2"),
        preprocessor=TextPreprocessor(data=pd.read_csv(ORIGINAL_DATASET_PATH)),
        data_path=ORIGINAL_DATASET_PATH,
        tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base-v2"),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze_bert=True,
    )
    # evaluator = Evaluator(
    #     trainer_evaluator=trainer_evaluator,
    #     similarity_evaluator=SimiarityEvaluator(
    #         auggpt_runner=AugGptRunner(get_instructor_instance()),
    #     ),
    # )

    app = App(
        init_prompt="""
        Từ đánh giá/mẫu gốc được cung cấp, hãy tạo ra 6 phiên bản diễn đạt lại mà vẫn giữ nguyên ý nghĩa gốc.
        """
    )
    optimized_prompt, is_finished = app.start(
        improvement_request="Viết lại câu văn bằng ngôn ngữ đơn giản, dễ hiểu hơn cho người đọc phổ thông, giữ nguyên ý nghĩa",
        num_iters=5,
        population_size=5,
        threshold=1.0,
        api_key="",
        evaluator=generate_evaluator(
            evaluator_path="/home/tb24/projects/llm-data-aug/run_evaluation.py",
            evaluator_python_path=None,
        ),
    )

    if is_finished:
        print(f"Optimized prompt: {optimized_prompt}")
    else:
        print("Optimization was interrupted")


if __name__ == "__main__":
    main()
