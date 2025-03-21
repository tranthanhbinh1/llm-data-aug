# Vietnamese Sentiment Analysis

A deep learning project for sentiment analysis of Vietnamese text using PhoBERT.

## Project Structure

```
.
├── data/                  # Data directory
│   ├── abbreviate.txt    # Abbreviation dictionary
│   ├── cleaned_user_reviews.csv  # Preprocessed dataset
│   └── vietnamese-stopwords.txt  # Vietnamese stopwords
├── models/               # Saved models directory
├── notebooks/           # Jupyter notebooks for exploration
├── src/                 # Source code
│   ├── preprocessing.py # Text preprocessing utilities
│   ├── dataset.py      # Dataset and data loading utilities
│   ├── model.py        # Model training and evaluation
│   ├── visualization.py # Visualization utilities
│   └── train.py        # Main training script
└── README.md           # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download VnCoreNLP:
```bash
mkdir -p notebooks/VnCoreNLP
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar -O notebooks/VnCoreNLP/VnCoreNLP-1.2.jar
```

## Data Preprocessing

The project uses several preprocessing steps:
- Text normalization
- Abbreviation expansion
- Special character removal
- Vietnamese tokenization
- Stopword removal

## Training

To train the model:

```bash
python src/train.py
```

This will:
1. Load and preprocess the data
2. Train a PhoBERT model for sentiment analysis
3. Save the trained model
4. Generate evaluation metrics and visualizations

## Model Architecture

The project uses PhoBERT-base-v2, a Vietnamese language model pretrained on a large corpus. The model is fine-tuned for sentiment analysis with three classes:
- Positive
- Negative
- Neutral

## Results

The model achieves the following metrics on the test set:
- Accuracy: [To be filled after training]
- F1-Score: [To be filled after training]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch/PhoBERT) for the PhoBERT model
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) for Vietnamese text processing
