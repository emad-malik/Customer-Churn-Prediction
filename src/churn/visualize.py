"""churn prediction visualizations.

BASELINE §7 – Explainability:
  - SHAP beeswarm plots (global feature importance)
  - SHAP dependence plots for Tenure + TechTickets
  - Partial Dependence Plots (PDP) + ICE curves
  - Nested CV results summary chart
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

from sklearn.inspection import PartialDependenceDisplay


# Use non-interactive backend so plots can be saved from headless scripts
matplotlib.use("Agg")

OUTPUT_DIR = pathlib.Path("outputs") / "plots"


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _get_explainer(model: Any, X_background: pd.DataFrame):
    """Auto-select the best SHAP explainer for the given model type."""
    model_type = type(model).__name__

    if model_type in ("XGBClassifier", "RandomForestClassifier"):
        return shap.TreeExplainer(model)

    if model_type in ("LogisticRegression",):
        return shap.LinearExplainer(model, X_background)

    if hasattr(model, "named_steps"):
        # Pipeline — try to unwrap
        last_step = list(model.named_steps.values())[-1]
        return _get_explainer(last_step, X_background)

    predict_fn = (
        model.predict_proba if hasattr(model, "predict_proba")
        else model.predict
    )
    background = shap.sample(X_background, min(100, len(X_background)), random_state=42)
    return shap.KernelExplainer(predict_fn, background)


def shap_beeswarm(
    model: Any,
    X: pd.DataFrame,
    model_name: str = "Model",
    max_display: int = 20,
    out_dir: pathlib.Path | str | None = None,
) -> None:
    """
    BASELINE §7 – SHAP beeswarm plot for global feature importance.
    Saves the figure to `out_dir/<model_name>_shap_beeswarm.png`.
    """
    out_dir = _ensure_dir(pathlib.Path(out_dir or OUTPUT_DIR))

    explainer = _get_explainer(model, X)
    shap_values = explainer(X)

    # For multi-output explainers, take positive class slice
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        sv = shap_values[..., 1]
    else:
        sv = shap_values

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(sv, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm — {model_name}", fontsize=13, pad=12)
    plt.tight_layout()
    fname = out_dir / f"{model_name.lower().replace(' ', '_')}_shap_beeswarm.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def shap_dependence(
    model: Any,
    X: pd.DataFrame,
    features: list[str],
    model_name: str = "Model",
    out_dir: pathlib.Path | str | None = None,
) -> None:
    """
    BASELINE §7 – SHAP dependence plots for specified features.
    Saves one figure per feature.
    """
    out_dir = _ensure_dir(pathlib.Path(out_dir or OUTPUT_DIR))

    explainer = _get_explainer(model, X)
    shap_values = explainer(X)

    if isinstance(shap_values, list):
        sv_arr = shap_values[1].values
    elif hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        sv_arr = shap_values.values[:, :, 1]
    else:
        sv_arr = shap_values.values if hasattr(shap_values, "values") else shap_values

    for feat in features:
        if feat not in X.columns:
            print(f"  [warn] Feature '{feat}' not in X — skipping dependence plot.")
            continue
        feat_idx = X.columns.get_loc(feat)
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            feat_idx,
            sv_arr,
            X,
            feature_names=X.columns.tolist(),
            show=False,
            ax=plt.gca(),
        )
        plt.title(f"SHAP Dependence: {feat} — {model_name}", fontsize=12)
        plt.tight_layout()
        slug = feat.lower().replace(" ", "_")
        fname = out_dir / f"{model_name.lower().replace(' ', '_')}_shap_dep_{slug}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Partial Dependence + ICE
# ---------------------------------------------------------------------------

def pdp_ice(
    model: Any,
    X: pd.DataFrame,
    features: list[str],
    model_name: str = "Model",
    out_dir: pathlib.Path | str | None = None,
) -> None:
    """
    BASELINE §7 – Partial Dependence Plots + ICE curves for specified features.
    """
    out_dir = _ensure_dir(pathlib.Path(out_dir or OUTPUT_DIR))

    present = [f for f in features if f in X.columns]
    if not present:
        print(f"  [warn] None of {features} found in X — skipping PDP/ICE.")
        return

    for feat in present:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PDP
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feat],
            kind="average",
            ax=axes[0],
            random_state=42,
        )
        axes[0].set_title(f"PDP: {feat}", fontsize=12)

        # ICE
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            features=[feat],
            kind="individual",
            subsample=200,
            ax=axes[1],
            random_state=42,
            alpha=0.05,
        )
        axes[1].set_title(f"ICE: {feat}", fontsize=12)

        plt.suptitle(f"{model_name} — PDP & ICE", fontsize=13)
        plt.tight_layout()
        slug = feat.lower().replace(" ", "_")
        fname = out_dir / f"{model_name.lower().replace(' ', '_')}_pdp_ice_{slug}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Nested CV summary bar chart
# ---------------------------------------------------------------------------

def plot_cv_results(
    results: dict[str, dict],
    metric: str = "pr_auc",
    out_dir: pathlib.Path | str | None = None,
) -> None:
    """
    Bar chart comparing mean ± std of `metric` across all models.

    Parameters
    ----------
    results : {model_name: nested_cv_return_dict}
    metric  : metric key to plot (default "pr_auc")
    """
    out_dir = _ensure_dir(pathlib.Path(out_dir or OUTPUT_DIR))

    names, means, stds = [], [], []
    for model_name, res in results.items():
        vals = [m[metric] for m in res["fold_metrics"]]
        names.append(model_name)
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(metric.upper().replace("_", " "))
    ax.set_title(f"Nested CV Results — {metric.upper().replace('_', ' ')}")
    ax.set_ylim(0, 1)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fname = out_dir / f"cv_results_{metric}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")
