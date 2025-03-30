import pandas as pd
import ast
from pathlib import Path
from loguru import logger


def transform_csv(input_path: str, output_path: str):
    """Transform the incorrectly formatted CSV back to proper format with two columns."""
    # Read the CSV file
    df = pd.read_csv(input_path)

    # Initialize lists to store transformed data
    all_reviews = []
    all_sentiments = []

    # Process each row
    for row in df.iloc[:, 0]:  # Assuming the data is in the first column
        try:
            # Convert string representation of list to actual list
            reviews_list = ast.literal_eval(row)

            # Extract reviews and sentiments
            for review_dict in reviews_list:
                all_reviews.append(review_dict["review"])
                all_sentiments.append(review_dict["sentiment"])
        except Exception as e:
            logger.error(f"Error processing row: {e}")
            continue

    # Create new DataFrame with proper format
    transformed_df = pd.DataFrame({"Review": all_reviews, "Sentiment": all_sentiments})

    # Save to CSV
    transformed_df.to_csv(output_path, index=False)
    logger.info(f"Transformed data saved to {output_path}")
    logger.info(f"Total reviews processed: {len(transformed_df)}")


if __name__ == "__main__":
    # Example usage
    input_path = "data/llm_generated/neutral_user_reviews.csv"
    output_path = "data/llm_generated/neutral_user_reviews_transformed.csv"
    transform_csv(input_path, output_path)
