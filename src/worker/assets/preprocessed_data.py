"""
Preprocessed data asset for trainer pipeline.
"""

import dagster as dg
import pandas as pd
from pathlib import Path
from src.preprocess.text_preprocessor import TextPreprocessor
from src.constants import PROJECT_ROOT


@dg.asset(
    deps=["synthetic_data_asset"],
    group_name="preprocessing",
    description="Preprocessed data ready for trainer consumption",
    metadata={
        "asset_type": "preprocessed_data",
        "output_format": "csv",
    },
)
def preprocessed_data_asset(
    context: dg.AssetExecutionContext,
    synthetic_data_asset: str,
) -> str:
    """Preprocess synthetic data for trainer consumption with caching."""

    # Generate cache key based on input data
    import hashlib
    import os

    data_mtime = os.path.getmtime(synthetic_data_asset)
    cache_key = hashlib.sha256(
        f"{synthetic_data_asset}|{data_mtime}".encode()
    ).hexdigest()[:16]

    # Define output path
    output_dir = Path(PROJECT_ROOT) / "data" / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"preprocessed_{cache_key}.csv"

    if output_path.exists():
        context.log.info(f"Using cached preprocessed data from {output_path}")
        preprocessed_data = pd.read_csv(output_path)
        result_metadata = {"cached": True, "cache_key": cache_key}
    else:
        # Load and preprocess data
        context.log.info("Running preprocessing pipeline...")
        data = pd.read_csv(synthetic_data_asset)

        # Use TextPreprocessor (expensive VnCoreNLP step)
        preprocessor = TextPreprocessor(data=data)
        context.log.info("Preprocessing data...")
        context.log.info(f"Data: {data.head().to_markdown()}")
        preprocessed_data = preprocessor.preprocess()

        # Save to cache
        preprocessed_data.to_csv(output_path, index=False)
        context.log.info(f"Preprocessed data saved to {output_path}")

        result_metadata = {
            "cached": False,
            "cache_key": cache_key,
            "rows_processed": len(preprocessed_data),
        }

    # Add data stats
    result_metadata.update(
        {
            "output_path": str(output_path),
            "rows": len(preprocessed_data),
            "columns": list(preprocessed_data.columns),
        }
    )

    # Add metadata to context
    context.add_output_metadata(metadata=result_metadata)

    return str(output_path)
