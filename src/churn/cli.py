"""CLI entry points — train and evaluate models."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from src.churn.dataset import build_features, load_raw
from src.churn.metrics import aggregate_fold_metrics, print_report
from src.churn.model import MODEL_REGISTRY
from src.churn.trainer import (
    nested_cv,
    train_final_model,
    save_model,
    load_model,
)
from src.churn.visualize import (
    shap_beeswarm,
    shap_dependence,
    pdp_ice,
    plot_cv_results,
)

# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA = pathlib.Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODELS_DIR = pathlib.Path("outputs") / "models"
DEFAULT_RESULTS_DIR = pathlib.Path("outputs") / "results"
DEFAULT_PLOTS_DIR = pathlib.Path("outputs") / "plots"

EXPLAINABILITY_FEATURES = ["tenure", "TechTickets"]


# ---------------------------------------------------------------------------
# train_main
# ---------------------------------------------------------------------------

def train_main() -> None:
    """Entry point: churn-train (and python scripts/train.py)."""
    parser = argparse.ArgumentParser(
        description="Train churn prediction models with nested cross-validation."
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Path to the Telco churn CSV file.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=list(MODEL_REGISTRY.keys()),
        help="Which models to train (default: all).",
    )
    parser.add_argument(
        "--outer-folds", type=int, default=5,
        help="Number of outer CV folds (default: 5).",
    )
    parser.add_argument(
        "--inner-folds", type=int, default=3,
        help="Number of inner CV folds (default: 3).",
    )
    parser.add_argument(
        "--n-iter", type=int, default=30,
        help="RandomizedSearchCV iterations (default: 30).",
    )
    parser.add_argument(
        "--no-feature-selection", action="store_true",
        help="Skip the three-layer feature selection step.",
    )
    parser.add_argument(
        "--save-models", action="store_true",
        help="Retrain final model on full data and persist to disk.",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write results JSON files.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    args = parser.parse_args()

    # -- Load and preprocess --
    print(f"Loading data from: {args.data}")
    raw = load_raw(args.data)

    # For tree/boosting models we use ordered contract encoding;
    # for linear/MLP we use OHE. Because nested_cv is model-specific we
    # rebuild features inside the loop with the appropriate encoding.
    # Here we build once with ordered=True as the baseline dataset;
    # linear/MLP will override during evaluation.
    X_tree, y = build_features(raw, drop_first=False, contract_ordered=True)
    X_linear, _ = build_features(raw, drop_first=True, contract_ordered=False)

    print(f"Dataset shape (tree/boosting encoding): {X_tree.shape}")
    print(f"Class balance: {y.value_counts().to_dict()}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for model_name in args.models:
        # Select appropriate feature matrix
        linear_models = {"logistic_regression", "svc", "mlp"}
        X = X_linear if model_name in linear_models else X_tree

        cv_result = nested_cv(
            X=X,
            y=y,
            model_name=model_name,
            outer_splits=args.outer_folds,
            inner_splits=args.inner_folds,
            n_iter=args.n_iter,
            random_state=args.seed,
            use_feature_selection=not args.no_feature_selection,
        )
        all_results[model_name] = cv_result

        # Summary table
        summary = aggregate_fold_metrics(cv_result["fold_metrics"])
        print(f"\n--- {model_name.upper()} — Nested CV Summary ---")
        print(summary.to_string(float_format="{:.4f}".format))

        # Save per-model results
        result_path = out_dir / f"{model_name}_cv_results.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "fold_metrics": cv_result["fold_metrics"],
                    "fold_thresholds": cv_result["fold_thresholds"],
                    "best_params_per_fold": [
                        {k: str(v) for k, v in p.items()}
                        for p in cv_result["best_params_per_fold"]
                    ],
                },
                f,
                indent=2,
            )
        print(f"  Results written to: {result_path}")

        # -- Optional: save final model --
        if args.save_models:
            DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            # Use median-index best params as final params
            med_idx = len(cv_result["best_params_per_fold"]) // 2
            final_params = cv_result["best_params_per_fold"][med_idx]
            stable_feats = cv_result["selected_features_per_fold"][med_idx]
            final_model = train_final_model(X, y, model_name, final_params, stable_feats)
            model_path = str(DEFAULT_MODELS_DIR / f"{model_name}.joblib")
            save_model(final_model, model_path)

    # -- CV comparison plot --
    print("\nGenerating CV results comparison charts ...")
    plot_cv_results(all_results, metric="pr_auc",  out_dir=DEFAULT_PLOTS_DIR)
    plot_cv_results(all_results, metric="roc_auc", out_dir=DEFAULT_PLOTS_DIR)
    plot_cv_results(all_results, metric="f1",      out_dir=DEFAULT_PLOTS_DIR)

    print("\n✓ Training complete.")


# ---------------------------------------------------------------------------
# evaluate_main
# ---------------------------------------------------------------------------

def evaluate_main() -> None:
    """Entry point: churn-evaluate (and python scripts/evaluate.py)."""
    parser = argparse.ArgumentParser(
        description="Evaluate a saved churn model and generate explainability plots."
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to the .joblib model file produced by churn-train --save-models.",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA),
        help="Path to the Telco churn CSV file.",
    )
    parser.add_argument(
        "--model-name", default="model",
        help="Display name for the model (used in plot titles).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for classification (default: 0.5).",
    )
    parser.add_argument(
        "--contract-ordered", action="store_true", default=False,
        help="Use ordered Contract encoding (for tree models).",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_PLOTS_DIR),
        help="Directory to save explainability plots.",
    )
    parser.add_argument(
        "--shap-samples", type=int, default=500,
        help="Max number of samples passed to SHAP explainer (default: 500).",
    )
    args = parser.parse_args()

    # -- Load data & model --
    print(f"Loading data from: {args.data}")
    raw = load_raw(args.data)
    is_tree = args.contract_ordered
    X, y = build_features(
        raw,
        drop_first=(not is_tree),
        contract_ordered=is_tree,
    )

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    # -- Metrics on full dataset (indicative, not a nested-CV estimate) --
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)

    print_report(y, y_prob, args.threshold, model_name=args.model_name)

    # -- SHAP plots --
    print("\nGenerating SHAP beeswarm plot ...")
    sample_X = X.sample(min(args.shap_samples, len(X)), random_state=42)
    shap_beeswarm(model, sample_X, model_name=args.model_name, out_dir=args.out_dir)

    print("\nGenerating SHAP dependence plots ...")
    dep_feats = [f for f in EXPLAINABILITY_FEATURES if f in X.columns]
    shap_dependence(model, sample_X, dep_feats, model_name=args.model_name, out_dir=args.out_dir)

    # -- PDP / ICE --
    print("\nGenerating PDP + ICE curves ...")
    pdp_ice(model, sample_X, dep_feats, model_name=args.model_name, out_dir=args.out_dir)

    print("\n✓ Evaluation complete. Plots saved to:", args.out_dir)
