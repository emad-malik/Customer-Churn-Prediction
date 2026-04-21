"""
scripts/benchmark.py
=====================
Run the full benchmark and produce the final results table comparing:
  Logistic Regression, SVC, Random Forest, XGBoost, MLP,
  LightGBM (+ Borderline-SMOTE), Stacking Ensemble.

Reads pre-computed nested-CV JSON results if they exist (fast path),
otherwise runs the full evaluation (slow path).

Usage
-----
    python scripts/benchmark.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python scripts/benchmark.py --from-json   # load existing results
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

RESULTS_DIR = Path("outputs/results")
PLOTS_DIR   = Path("outputs/plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = [
    "logistic_regression",
    "svc",
    "random_forest",
    "xgboost",
    "mlp",
    "lightgbm",
    "stacking_ensemble",
]

# Paper's reported numbers for comparison (Table 3 of Zerine et al. 2026)
PAPER_RESULTS = {
    "logistic_regression": {"accuracy": 0.86, "precision": 0.72, "recall": 0.58, "f1": 0.64, "roc_auc": 0.88, "pr_auc": 0.69},
    "svc":                 {"accuracy": 0.84, "precision": 0.67, "recall": 0.52, "f1": 0.59, "roc_auc": 0.86, "pr_auc": 0.65},
    "random_forest":       {"accuracy": 0.87, "precision": 0.74, "recall": 0.60, "f1": 0.66, "roc_auc": 0.89, "pr_auc": 0.71},
    "xgboost":             {"accuracy": 0.89, "precision": 0.76, "recall": 0.64, "f1": 0.69, "roc_auc": 0.91, "pr_auc": 0.74},
    "mlp":                 {"accuracy": 0.92, "precision": 0.88, "recall": 0.82, "f1": 0.85, "roc_auc": 0.95, "pr_auc": 0.82},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--from-json", action="store_true",
                   help="Load existing JSON results instead of re-running CV")
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--no-feature-selection", action="store_true")
    return p.parse_args()


def load_from_json() -> dict[str, dict]:
    """Load pre-computed results from JSON files."""
    results = {}
    for name in MODEL_NAMES:
        path = RESULTS_DIR / f"{name}_cv_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            results[name] = data.get("summary", {})
        else:
            print(f"  [missing] {path}")
    return results


def run_full_benchmark(args) -> dict[str, dict]:
    """Run the full nested-CV benchmark for all models."""
    from src.churn.dataset import prepare
    from src.churn.trainer import nested_cv
    from src.churn.metrics import summarise_folds

    print(f"Loading data: {args.data}")
    X, y = prepare(args.data, engineer=True)

    summaries = {}
    xgb_params = lgbm_params = None

    for model_name in MODEL_NAMES:
        print(f"\n{'─'*50}")
        print(f"  {model_name}")
        print(f"{'─'*50}")

        kwargs = {}
        if model_name == "stacking_ensemble":
            kwargs = {"xgb_params": xgb_params or {}, "lgbm_params": lgbm_params or {}}

        cv_out = nested_cv(
            X=X, y=y,
            model_name=model_name,
            outer_splits=5,
            inner_splits=3,
            n_iter=args.n_iter,
            use_feature_selection=not args.no_feature_selection,
            model_kwargs=kwargs,
        )
        summary = summarise_folds(cv_out["fold_metrics"])
        summaries[model_name] = summary

        # Capture best params for downstream stacking
        if model_name == "xgboost" and cv_out["best_params_per_fold"]:
            xgb_params = cv_out["best_params_per_fold"][0]
        if model_name == "lightgbm" and cv_out["best_params_per_fold"]:
            lgbm_params = cv_out["best_params_per_fold"][0]

        # Save JSON
        with open(RESULTS_DIR / f"{model_name}_cv_results.json", "w") as f:
            json.dump({
                "summary":         summary,
                "fold_metrics":    cv_out["fold_metrics"],
                "fold_thresholds": cv_out["fold_thresholds"],
            }, f, indent=2)

    return summaries


def build_comparison_dataframe(
    our_results: dict[str, dict],
) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []

    for model in MODEL_NAMES:
        if model not in our_results:
            continue
        s = our_results[model]
        row = {"Model": model.replace("_", " ").title(), "Source": "Ours"}
        for m in metrics:
            if m in s:
                row[m.upper()] = f"{s[m]['mean']:.4f}±{s[m]['std']:.4f}"
            else:
                row[m.upper()] = "—"
        rows.append(row)

    # Add paper baselines
    for model, paper in PAPER_RESULTS.items():
        row = {"Model": model.replace("_", " ").title() + " (paper)", "Source": "Paper"}
        for m in metrics:
            row[m.upper()] = f"{paper[m]:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def plot_results_comparison(
    our_results: dict[str, dict],
    save_path: str,
) -> None:
    metrics   = ["accuracy", "f1", "roc_auc", "pr_auc"]
    col_names = ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]
    models    = [m for m in MODEL_NAMES if m in our_results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, metric, col_name in zip(axes, metrics, col_names):
        our_means = [our_results[m][metric]["mean"] for m in models if metric in our_results[m]]
        our_stds  = [our_results[m][metric]["std"]  for m in models if metric in our_results[m]]
        paper_vals = [PAPER_RESULTS[m][metric] if m in PAPER_RESULTS else np.nan for m in models]
        labels = [m.replace("_", "\n").title() for m in models]

        x = np.arange(len(models))
        w = 0.35

        ax.bar(x - w/2, our_means, w, yerr=our_stds, capsize=3,
               label="Ours", color="#1f77b4", alpha=0.85)
        ax.bar(x + w/2, paper_vals, w,
               label="Paper", color="#ff7f0e", alpha=0.7,
               hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(col_name)
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=7)
        ax.axhline(0.85, color="red", ls="--", lw=0.8, alpha=0.5)

    fig.suptitle("Benchmark: Our Pipeline vs. Zerine et al. 2026", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison chart saved: {save_path}")


def print_latex_table(df: pd.DataFrame) -> None:
    """Print a LaTeX-ready table for the paper."""
    print("\n" + "=" * 70)
    print("LaTeX Table")
    print("=" * 70)
    print(df.to_latex(index=False, escape=False))


def main():
    args = parse_args()

    if args.from_json:
        print("Loading results from existing JSON files...")
        our_results = load_from_json()
        if not our_results:
            print("No JSON results found. Run train_all.py first.")
            sys.exit(1)
    else:
        our_results = run_full_benchmark(args)

    df = build_comparison_dataframe(our_results)

    # Print to console
    print("\n" + "=" * 90)
    print("FULL BENCHMARK RESULTS")
    print("=" * 90)
    print(df.to_string(index=False))

    # Save CSV
    csv_path = str(RESULTS_DIR / "benchmark_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Comparison chart
    chart_path = str(PLOTS_DIR / "benchmark_comparison.png")
    try:
        plot_results_comparison(our_results, chart_path)
    except Exception as e:
        print(f"Chart skipped: {e}")

    # LaTeX
    print_latex_table(df)


if __name__ == "__main__":
    main()