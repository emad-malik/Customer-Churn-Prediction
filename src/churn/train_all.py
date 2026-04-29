"""
churn/train_all.py
==================
Train all seven models, log every experiment to MLflow,
register the best model in the registry.

Usage
-----
    churn-train-all
    churn-train-all --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    churn-train-all --no-feature-selection --n-iter 10
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import joblib

from churn.dataset import prepare

from churn.trainer import nested_cv, train_final_model, stability_selection
from churn.metrics import summarise_folds, plot_confusion_matrix, plot_roc_pr_curves
from churn.tracking import setup_mlflow, log_cv_results, register_model

warnings.filterwarnings("ignore")


# ── Config ───────────────────────────────────────────────────────────────────

MODELS_TO_TRAIN = [
    "logistic_regression",
    "svc",
    "random_forest",
    "xgboost",
    "mlp",
    "lightgbm",
    "stacking_ensemble",
]

OUTPUT_DIR   = Path("outputs")
RESULTS_DIR  = OUTPUT_DIR / "results"
MODELS_DIR   = OUTPUT_DIR / "models"
PLOTS_DIR    = OUTPUT_DIR / "plots"

for d in [RESULTS_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train all churn models")
    p.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--experiment", default="churn_prediction")
    p.add_argument("--n-iter", type=int, default=30)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--no-feature-selection", action="store_true")
    p.add_argument("--models", nargs="+", default=MODELS_TO_TRAIN)
    p.add_argument("--registry-model-name", default="churn_best_model")
    p.add_argument("--register-best", action="store_true", default=True)
    return p.parse_args()


def train_single(
    model_name: str,
    X, y,
    args: argparse.Namespace,
    xgb_best_params: dict | None = None,
    lgbm_best_params: dict | None = None,
) -> tuple[dict, str]:
    """
    Run nested CV for one model, log to MLflow.
    Returns (summary, run_id).
    """
    model_kwargs = {}
    if model_name == "stacking_ensemble":
        model_kwargs = {
            "xgb_params":  xgb_best_params or {},
            "lgbm_params": lgbm_best_params or {},
        }

    results = nested_cv(
        X=X, y=y,
        model_name=model_name,
        outer_splits=args.outer_folds,
        inner_splits=args.inner_folds,
        n_iter=args.n_iter,
        use_feature_selection=not args.no_feature_selection,
        model_kwargs=model_kwargs,
    )

    summary = summarise_folds(results["fold_metrics"])

    # Print summary table
    print(f"\n--- {model_name.upper()} — Nested CV Summary ---")
    print(f"{'':>12}{'mean':>8}{'std':>8}")
    for metric, stats in summary.items():
        if metric == "threshold":
            continue
        print(f"  {metric:<12}{stats['mean']:>8.4f}{stats['std']:>8.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────
    oof_probs = results["oof_probs"]
    thresh    = float(np.median(results["fold_thresholds"]))

    cm_path  = str(PLOTS_DIR / f"{model_name}_confusion_matrix.png")
    roc_path = str(PLOTS_DIR / f"{model_name}_roc_pr.png")

    plot_confusion_matrix(y, oof_probs, thresh, model_name.replace("_", " ").title(),
                          save_path=cm_path)
    plot_roc_pr_curves(y, oof_probs, model_name.replace("_", " ").title(),
                       save_path=roc_path)

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = str(RESULTS_DIR / f"{model_name}_cv_results.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "summary":    summary,
                "fold_metrics":    results["fold_metrics"],
                "fold_thresholds": results["fold_thresholds"],
            },
            f, indent=2,
        )

    # ── MLflow ────────────────────────────────────────────────────────────
    mean_params = {}
    if results["best_params_per_fold"]:
        all_keys = set().union(*[p.keys() for p in results["best_params_per_fold"]])
        for k in all_keys:
            vals = [p[k] for p in results["best_params_per_fold"] if k in p]
            try:
                mean_params[k] = float(np.median([float(v) for v in vals]))
            except (TypeError, ValueError):
                mean_params[k] = str(vals[0])

    run_id = log_cv_results(
        model_name=model_name,
        fold_metrics=results["fold_metrics"],
        summary=summary,
        params=mean_params,
        tags={"smote": str(model_name in {"lightgbm"}),
              "n_iter": str(args.n_iter)},
        artifact_paths=[cm_path, roc_path, json_path],
    )

    print(f"  MLflow run_id: {run_id}")
    return summary, run_id, results


def main():
    args = parse_args()

    print(f"\nLoading data from: {args.data}")
    X, y = prepare(args.data, engineer=True)
    print(f"Dataset shape: {X.shape}  |  Churn rate: {y.mean():.2%}")

    setup_mlflow(args.mlflow_uri, args.experiment)

    all_summaries: dict[str, dict] = {}
    all_run_ids:   dict[str, str]  = {}
    failed_models: dict[str, str] = {}
    best_params_by_model: dict[str, dict] = {}

    xgb_best_params  = None
    lgbm_best_params = None

    for model_name in args.models:
        print(f"\n{'═'*60}")
        print(f"  Training: {model_name}")
        print(f"{'═'*60}")

        try:
            summary, run_id, results = train_single(
                model_name, X, y, args,
                xgb_best_params=xgb_best_params,
                lgbm_best_params=lgbm_best_params,
            )
            all_summaries[model_name] = summary
            all_run_ids[model_name]   = run_id

            if results["best_params_per_fold"]:
                best_params_by_model[model_name] = results["best_params_per_fold"][0]

            if model_name == "xgboost" and results["best_params_per_fold"]:
                xgb_best_params = results["best_params_per_fold"][0]
            if model_name == "lightgbm" and results["best_params_per_fold"]:
                lgbm_best_params = results["best_params_per_fold"][0]
        except Exception as e:
            failed_models[model_name] = str(e)
            print(f"Training failed for {model_name}: {e}")
            continue

    # ── Benchmark comparison plot ────────────────────────────────────────────
    if all_summaries:
        try:
            from churn.metrics import plot_benchmark_table
            bench_path = str(PLOTS_DIR / "benchmark_comparison.png")
            plot_benchmark_table(all_summaries, save_path=bench_path)
            print(f"\nBenchmark table saved: {bench_path}")
        except Exception as e:
            print(f"\nBenchmark plot skipped: {e}")

    # ── Select and register best model ──────────────────────────────────────
    if args.register_best and all_summaries:
        best_name = max(
            all_summaries,
            key=lambda m: all_summaries[m].get("pr_auc", {}).get("mean", 0),
        )
        print(f"\nBest model by PR-AUC: {best_name}")

        stable_feats = stability_selection(X, y) if not args.no_feature_selection else X.columns.tolist()

        model_kwargs = {}
        if best_name == "stacking_ensemble":
            model_kwargs = {
                "xgb_params":  xgb_best_params or {},
                "lgbm_params": lgbm_best_params or {},
            }

        best_params = best_params_by_model.get(best_name, {})

        final_model = train_final_model(
            X=X, y=y,
            model_name=best_name,
            stable_features=stable_feats,
            best_params=best_params,
            model_kwargs=model_kwargs,
        )

        model_path = str(MODELS_DIR / f"{best_name}.joblib")
        joblib.dump(final_model, model_path)
        print(f"Final model saved: {model_path}")

        try:
            register_model(
                run_id=all_run_ids[best_name],
                model_name=args.registry_model_name,
                estimator=final_model,
                X_sample=X[stable_feats].head(10),
                stage="Production",
                description=(
                    f"Best model from benchmark: {best_name}. "
                    f"PR-AUC={all_summaries[best_name]['pr_auc']['mean']:.4f}"
                ),
            )
            print(f"Model registered in MLflow registry as '{args.registry_model_name}' (Production)")
        except Exception as e:
            print(f"MLflow registration skipped (not connected): {e}")

    if failed_models:
        print("\nFailed models summary:")
        for name, err in failed_models.items():
            print(f"  - {name}: {err}")

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
