"""Visualization utilities for sentiment analysis results."""

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_wordcloud(text: str, width: int = 800, height: int = 400) -> None:
    """
    Create and display a word cloud from text.

    Args:
        text: Text to create word cloud from
        width: Width of the word cloud image
        height: Height of the word cloud image
    """
    wordcloud = WordCloud(
        width=width, height=height, background_color="white"
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    target_names: List[str],
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: Confusion matrix to plot
        target_names: Names of the target classes
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    plt.show()


def plot_roc_curves(
    true_labels: np.ndarray, predictions: np.ndarray, target_names: List[str]
) -> None:
    """
    Plot ROC curves for each class.

    Args:
        true_labels: True labels
        predictions: Model predictions
        target_names: Names of the target classes
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curve and ROC area for each class
    for i in range(len(target_names)):
        fpr[i], tpr[i], _ = roc_curve(true_labels == i, predictions == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(len(target_names)):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve (AUC = {roc_auc[i]:.2f}) for {target_names[i]}",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]], title: str = "Training History"
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary containing training metrics
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))
    for metric_name, metric_values in history.items():
        plt.plot(metric_values, label=metric_name)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
