"""
Concept drift detection using Population Stability Index (PSI).

PSI interpretation (standard thresholds):
  < 0.10  — no significant change
  0.10–0.20 — moderate change, monitor closely
  > 0.20  — significant shift, investigate / retrain

We also compute prediction drift (change in output score distribution)
and feature-level drift for a ranked report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
)


# ── PSI core ─────────────────────────────────────────────────────────────────

def _psi_single(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute PSI for a single continuous feature.
    `expected` = reference (training) distribution.
    `actual`   = current (production) distribution.
    """
    # Use quantiles of expected to define bucket edges
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  -= eps
    breakpoints[-1] += eps

    exp_counts = np.histogram(expected, bins=breakpoints)[0]
    act_counts = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / (exp_counts.sum() + eps)
    act_pct = act_counts / (act_counts.sum() + eps)

    # Replace zeros to avoid log(0)
    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def compute_feature_psi(
    X_ref: pd.DataFrame,
    X_cur: pd.DataFrame,
    buckets: int = 10,
) -> pd.Series:
    """
    Compute per-feature PSI between reference and current DataFrames.
    Returns a Series sorted descending by PSI value.
    """
    common_cols = X_ref.columns.intersection(X_cur.columns)
    scores = {}
    for col in common_cols:
        try:
            scores[col] = _psi_single(
                X_ref[col].values.astype(float),
                X_cur[col].values.astype(float),
                buckets=buckets,
            )
        except Exception:
            scores[col] = np.nan
    return pd.Series(scores).sort_values(ascending=False)


def compute_prediction_psi(
    proba_ref: np.ndarray,
    proba_cur: np.ndarray,
    buckets: int = 10,
) -> float:
    """PSI on the model output (churn probability) distribution."""
    return _psi_single(proba_ref, proba_cur, buckets=buckets)


def overall_psi(feature_psi: pd.Series) -> float:
    """Mean PSI across all features — a single scalar drift signal."""
    return float(feature_psi.dropna().mean())


# ── Performance degradation analysis ─────────────────────────────────────────

def performance_comparison(
    y_ref: np.ndarray,
    proba_ref: np.ndarray,
    y_cur: np.ndarray,
    proba_cur: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Compare classification performance between reference and current splits.
    Returns a dict with keys 'reference' and 'current', each containing
    accuracy, f1, roc_auc, pr_auc.
    """
    def _metrics(y, p):
        pred = (p >= threshold).astype(int)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "f1":       float(f1_score(y, pred, zero_division=0)),
            "roc_auc":  float(roc_auc_score(y, p)),
            "pr_auc":   float(average_precision_score(y, p)),
        }

    return {
        "reference": _metrics(y_ref, proba_ref),
        "current":   _metrics(y_cur, proba_cur),
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_drift_report(
    feature_psi: pd.Series,
    perf_comparison: dict,
    prediction_psi: float,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Three-panel drift report:
      1. Top-N feature PSI bar chart
      2. Performance degradation grouped bar chart
      3. Prediction score distribution PSI annotation
    """
    top_n = min(15, len(feature_psi))
    top_features = feature_psi.head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Feature PSI
    colors = [
        "#d62728" if v > 0.2 else "#ff7f0e" if v > 0.1 else "#2ca02c"
        for v in top_features.values
    ]
    axes[0].barh(top_features.index[::-1], top_features.values[::-1], color=colors[::-1])
    axes[0].axvline(0.10, color="orange", linestyle="--", lw=1.5, label="Moderate (0.10)")
    axes[0].axvline(0.20, color="red",    linestyle="--", lw=1.5, label="Significant (0.20)")
    axes[0].set_xlabel("PSI")
    axes[0].set_title("Feature-level PSI (top features)")
    axes[0].legend(fontsize=8)

    # Panel 2: Performance comparison
    metric_keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
    ref_vals = [perf_comparison["reference"][k] for k in metric_keys]
    cur_vals = [perf_comparison["current"][k]   for k in metric_keys]
    x = np.arange(len(metric_keys))
    w = 0.35
    axes[1].bar(x - w/2, ref_vals, w, label="Reference (train split)", color="#1f77b4")
    axes[1].bar(x + w/2, cur_vals, w, label="Current (drift split)",   color="#ff7f0e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Accuracy", "F1", "ROC-AUC", "PR-AUC"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Performance: Reference vs Drifted Data")
    axes[1].legend(fontsize=8)
    axes[1].axhline(0.85, color="red", linestyle="--", lw=1, label="Alert threshold")

    # Panel 3: PSI summary text
    axes[2].axis("off")
    summary_lines = [
        f"Prediction PSI : {prediction_psi:.4f}",
        f"Mean Feature PSI: {feature_psi.mean():.4f}",
        "",
        "Status:",
        ("SIGNIFICANT DRIFT" if prediction_psi > 0.2 else
         "MODERATE DRIFT"   if prediction_psi > 0.1 else
         "STABLE"),
        "",
        "Performance delta:",
    ]
    for k in metric_keys:
        delta = perf_comparison["current"][k] - perf_comparison["reference"][k]
        arrow = "▼" if delta < 0 else "▲"
        summary_lines.append(f"  {k:>10}: {arrow} {abs(delta):.4f}")

    axes[2].text(
        0.05, 0.95, "\n".join(summary_lines),
        transform=axes[2].transAxes,
        fontsize=11, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )
    axes[2].set_title("Drift Summary")

    fig.suptitle("Concept Drift Report", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig