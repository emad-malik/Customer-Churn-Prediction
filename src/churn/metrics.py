"""churn prediction evaluation metrics.

BASELINE §5: Evaluation Protocol — Nested Cross Validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    BASELINE §5 – Select the decision threshold on inner validation data
    that maximises F1 score.

    Returns the threshold with the highest F1.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has one fewer element than precisions/recalls
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full suite of metrics used in the paper.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of float  (positive-class probabilities)
    threshold : float
        Decision threshold applied to y_prob.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc, threshold
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "threshold": threshold,
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-fold metric dicts into a summary DataFrame with
    mean ± std for each metric.
    """
    df = pd.DataFrame(fold_metrics)
    summary = pd.concat(
        [df.mean().rename("mean"), df.std().rename("std")], axis=1
    )
    return summary


def print_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str = "Model",
) -> None:
    """Pretty-print classification report and confusion matrix."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    print(f"\n{'=' * 60}")
    print(f"  {model_name}  (threshold={threshold:.3f})")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=["No Churn", "Churn"]))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'=' * 60}\n")


def summarise_folds(fold_metrics: list[dict]) -> dict:
    """
    Summarize fold metrics into mean and std for each metric.
    
    Parameters
    ----------
    fold_metrics : list of dict
        List of metric dicts from each fold
        
    Returns
    -------
    dict : keys are metric names, values are {'mean': float, 'std': float}
    """
    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in df.columns:
        summary[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }
    return summary


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """
    Plot confusion matrix for the given predictions.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    threshold : float
        Decision threshold
    model_name : str
        Name of the model for the title
    save_path : str, optional
        Path to save the figure
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    
    # Add counts in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j],
                    ha="center", va="center", color="black", fontsize=12)
    
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"{model_name}\nConfusion Matrix (threshold={threshold:.3f})", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: str | None = None,
) -> None:
    """
    Plot ROC and Precision-Recall curves side-by-side.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_prob : array-like
        Predicted probabilities
    model_name : str
        Name of the model for the title
    save_path : str, optional
        Path to save the figure
    """
    # Compute curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # ROC curve
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.set_title("ROC Curve", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    
    # PR curve
    ax2.plot(recalls, precisions, color="green", lw=2, label=f"PR (AUC={pr_auc:.3f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.set_title("Precision-Recall Curve", fontsize=11)
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    
    fig.suptitle(f"{model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()

