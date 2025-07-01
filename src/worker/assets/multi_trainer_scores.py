"""
Multi-trainer evaluation asset that runs multiple trainers sequentially.
"""

import dagster as dg
import json
import time
from pathlib import Path

from src.trainers.cnn_bert_hybrid import CNNBertHybridTrainer
from src.trainers.lstm import BERTLSTMTrainer, BERTLSTMModel
from src.trainers.phoBert import PhoBertTrainer
from src.trainers.svm import SVMTrainer
from src.constants import PROJECT_ROOT


@dg.asset(
    deps=["preprocessed_data_asset"],
    group_name="evaluation",
    description="Sequential evaluation using multiple trainers",
    metadata={
        "asset_type": "multi_scores",
        "output_format": "json",
    },
)
def multi_trainer_scores_asset(
    context: dg.AssetExecutionContext,
    preprocessed_data_asset: str,
) -> str:
    """Evaluate preprocessed data using multiple trainers sequentially."""
    context.log.info("🎯 Starting multi-trainer evaluation process")
    context.log.info(f"📊 Using preprocessed data from: {preprocessed_data_asset}")

    # Generate cache key
    import hashlib
    import os

    data_mtime = os.path.getmtime(preprocessed_data_asset)
    cache_key = hashlib.sha256(
        f"{preprocessed_data_asset}|{data_mtime}".encode()
    ).hexdigest()[:16]

    context.log.info(f"🔑 Generated cache key: {cache_key}")

    output_dir = Path(PROJECT_ROOT) / "graphs" / "multi_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"multi_scores_{cache_key}.json"

    context.log.info(f"📂 Output path: {output_path}")

    if output_path.exists():
        context.log.info(f"💾 Using cached multi-trainer scores from {output_path}")
        with open(output_path, "r") as f:
            scores = json.load(f)

        # Log cached results summary
        successful_count = sum(
            1 for k, v in scores.items() if k.endswith("_score") and v is not None
        )
        total_count = sum(1 for k in scores.keys() if k.endswith("_score"))
        context.log.info(
            f"📈 Cached results: {successful_count}/{total_count} trainers successful"
        )

        result_metadata = {"cached": True, "cache_key": cache_key}
        result_metadata.update(scores)
    else:
        context.log.info(
            "🔄 No cached results found - running new multi-trainer evaluation"
        )
        context.log.info("⚠️ This is a heavy operation that may take 10-30 minutes...")

        # Track overall timing
        overall_start_time = time.time()

        # Define trainer evaluation functions
        def evaluate_cnn_bert():
            try:
                from transformers import AutoModel
                import pandas as pd
                from src.preprocess.text_preprocessor import TextPreprocessor

                context.log.info("Evaluating CNN-BERT...")
                bert_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
                preprocessor = TextPreprocessor(
                    data=pd.read_csv(preprocessed_data_asset)
                )

                trainer = CNNBertHybridTrainer(
                    bert_model=bert_model,
                    preprocessor=preprocessor,
                    data_path=preprocessed_data_asset,
                    freeze_bert=True,
                )
                score = float(trainer.run_evaluation(preprocessed_data_asset))
                context.log.info(f"CNN-BERT evaluation successful. Score: {score}")
                return {"cnn_bert_hybrid": score}
            except Exception as e:
                context.log.error(f"CNN-BERT evaluation failed: {e}")
                return {"cnn_bert_hybrid": None}

        def evaluate_lstm():
            try:
                import pandas as pd
                from src.preprocess.text_preprocessor import TextPreprocessor

                context.log.info("Evaluating BERT-LSTM...")
                model = BERTLSTMModel(
                    bert_model_name="vinai/phobert-base-v2",
                    hidden_dim1=128,
                    hidden_dim2=64,
                    dense_dim=64,
                    output_dim=3,
                    dropout_rate=0.5,
                    freeze_bert=True,
                )

                preprocessor = TextPreprocessor(
                    data=pd.read_csv(preprocessed_data_asset)
                )
                trainer = BERTLSTMTrainer(
                    model=model,
                    data_path=preprocessed_data_asset,
                    preprocessor=preprocessor,
                )
                score = float(trainer.run_evaluation(preprocessed_data_asset))
                context.log.info(f"BERT-LSTM evaluation successful. Score: {score}")
                return {"bert_lstm": score}
            except Exception as e:
                context.log.error(f"BERT-LSTM evaluation failed: {e}")
                return {"bert_lstm": None}

        def evaluate_phobert():
            try:
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
                import pandas as pd
                from src.preprocess.text_preprocessor import TextPreprocessor

                context.log.info("Evaluating PhoBERT...")
                model = AutoModelForSequenceClassification.from_pretrained(
                    "vinai/phobert-base-v2", num_labels=3
                )
                tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
                preprocessor = TextPreprocessor(
                    data=pd.read_csv(preprocessed_data_asset)
                )

                trainer = PhoBertTrainer(
                    model=model,
                    data_path=preprocessed_data_asset,
                    preprocessor=preprocessor,
                    tokenizer=tokenizer,
                )
                score = float(trainer.run_evaluation(preprocessed_data_asset))
                context.log.info(f"PhoBERT evaluation successful. Score: {score}")
                return {"phobert": score}
            except Exception as e:
                context.log.error(f"PhoBERT evaluation failed: {e}")
                return {"phobert": None}

        def evaluate_svm():
            try:
                context.log.info("Evaluating SVM...")
                trainer = SVMTrainer(data_path=preprocessed_data_asset)
                score = float(trainer.run_evaluation())
                context.log.info(f"SVM evaluation successful. Score: {score}")
                return {"svm": score}
            except Exception as e:
                context.log.error(f"SVM evaluation failed: {e}")
                return {"svm": None}

        # Run evaluations sequentially
        context.log.info("Starting sequential trainer evaluations...")
        scores = {}
        scores.update(evaluate_cnn_bert())
        scores.update(evaluate_lstm())
        scores.update(evaluate_phobert())
        scores.update(evaluate_svm())

        # Add metadata
        scores.update(
            {
                "cache_key": cache_key,
                "data_path": preprocessed_data_asset,
                "successful_trainers": sum(1 for v in scores.values() if v is not None),
                "failed_trainers": sum(1 for v in scores.values() if v is None),
            }
        )

        # Save results
        with open(output_path, "w") as f:
            json.dump(scores, f, indent=2)

        result_metadata = {"cached": False}
        result_metadata.update(scores)

        context.log.info(f"Sequential evaluation completed. Results: {scores}")

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    return str(output_path)
