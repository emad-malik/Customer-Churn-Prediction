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
    confusion_matrix,
    classification_report,
)


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
