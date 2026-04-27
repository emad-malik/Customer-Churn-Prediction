"""
scripts/drift_simulation.py
============================
Simulate concept drift by splitting the dataset temporally:
  - First 70% → "production reference" data (model was trained on this)
  - Last 30%  → "new incoming data" with shifted distribution

Steps
-----
1. Load the production model from MLflow registry (or local fallback).
2. Compute per-feature PSI between reference and current splits.
3. Compute prediction distribution PSI.
4. Report performance degradation.
5. Log all drift metrics + plots to MLflow.
6. Write a JSON cache that the FastAPI /drift endpoint serves.
7. Push updated accuracy + PSI gauges to the API so Prometheus / Grafana
   reflect the drift without waiting for a scrape cycle.

Usage
-----
    python scripts/drift_simulation.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from src.churn.dataset import prepare, temporal_split
from src.churn.drfit import (
    compute_feature_psi,
    compute_prediction_psi,
    performance_comparison,
    plot_drift_report,
    overall_psi,
)
from src.churn.tracking import setup_mlflow, log_drift_results

OUTPUT_DIR   = Path("outputs")
RESULTS_DIR  = OUTPUT_DIR / "results"
PLOTS_DIR    = OUTPUT_DIR / "plots"

for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--model-name", default="churn_stacking_ensemble")
    p.add_argument("--model-stage", default="Production")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--api-url", default="http://localhost:8000",
                   help="FastAPI base URL for pushing live gauge updates")
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def load_model(args):
    """Try MLflow registry first, fall back to local joblib."""
    import mlflow.sklearn
    mlflow.set_tracking_uri(args.mlflow_uri)
    try:
        uri = f"models:/{args.model_name}/{args.model_stage}"
        model = mlflow.sklearn.load_model(uri)
        print(f"Loaded model from MLflow: {uri}")
        return model
    except Exception as e:
        print(f"MLflow load failed ({e}). Trying local fallback...")

    for path in [
        f"outputs/models/{args.model_name}.joblib",
        "outputs/models/stacking_ensemble.joblib",
        "outputs/models/lightgbm.joblib",
        "outputs/models/xgboost.joblib",
    ]:
        if os.path.isfile(path):
            print(f"Loaded model from local file: {path}")
            return joblib.load(path)

    raise FileNotFoundError("No trained model found. Run train_all.py first.")


def main():
    args = parse_args()

    print(f"\nLoading data: {args.data}")
    X, y = prepare(args.data, engineer=True)

    print(f"Temporal split: {args.train_frac:.0%} reference / {1-args.train_frac:.0%} current")
    X_ref, y_ref, X_cur, y_cur = temporal_split(X, y, train_frac=args.train_frac)
    print(f"  Reference: {len(X_ref)} rows  |  churn rate: {y_ref.mean():.2%}")
    print(f"  Current  : {len(X_cur)} rows  |  churn rate: {y_cur.mean():.2%}")

    model = load_model(args)

    # ── Probabilities ────────────────────────────────────────────────────────
    def safe_predict_proba(m, Xd):
        cols = list(Xd.columns)
        try:
            cols = list(m.feature_names_in_)
        except AttributeError:
            pass
        Xd = Xd.reindex(columns=cols, fill_value=0)
        return m.predict_proba(Xd)[:, 1]

    proba_ref = safe_predict_proba(model, X_ref)
    proba_cur = safe_predict_proba(model, X_cur)

    # ── PSI ──────────────────────────────────────────────────────────────────
    print("\nComputing feature-level PSI...")
    feature_psi = compute_feature_psi(X_ref, X_cur)

    # Persist full feature PSI list for reporting
    feature_psi_json = RESULTS_DIR / "feature_psi_full.json"
    feature_psi_csv = RESULTS_DIR / "feature_psi_full.csv"
    feature_psi.to_json(feature_psi_json, indent=2)
    feature_psi.to_frame(name="psi").to_csv(feature_psi_csv, index_label="feature")

    pred_psi = compute_prediction_psi(proba_ref, proba_cur)
    mean_psi = overall_psi(feature_psi)

    print(f"  Prediction PSI   : {pred_psi:.4f}")
    print(f"  Mean Feature PSI : {mean_psi:.4f}")

    # Interpret
    if pred_psi > 0.2:
        status = "SIGNIFICANT DRIFT 🔴"
    elif pred_psi > 0.1:
        status = "MODERATE DRIFT 🟠"
    else:
        status = "STABLE 🟢"
    print(f"  Status: {status}")

    # ── Performance comparison ────────────────────────────────────────────────
    print("\nPerformance comparison (reference vs. drifted):")
    perf = performance_comparison(
        y_ref.values, proba_ref,
        y_cur.values, proba_cur,
        threshold=args.threshold,
    )
    for split, metrics in perf.items():
        row = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  {split:>10}: {row}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    drift_plot = str(PLOTS_DIR / "drift_report.png")
    plot_drift_report(feature_psi, perf, pred_psi, save_path=drift_plot)
    print(f"\nDrift report saved: {drift_plot}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    setup_mlflow(args.mlflow_uri, "churn_drift")
    run_id = log_drift_results(
        model_name=args.model_name,
        feature_psi=feature_psi,
        prediction_psi=pred_psi,
        perf_comparison=perf,
        artifact_paths=[drift_plot],
    )
    print(f"MLflow drift run: {run_id}")

    # ── Cache for FastAPI /drift endpoint ────────────────────────────────────
    cache = {
        "prediction_psi":    pred_psi,
        "mean_feature_psi":  mean_psi,
        "status":            status,
        "performance":       perf,
        "top_drifted_features": feature_psi.head(10).to_dict(),
    }
    cache_path = str(RESULTS_DIR / "drift_cache.json")
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Drift cache written: {cache_path}")

    # ── Push gauges to live API (best-effort) ────────────────────────────────
    cur_accuracy = perf["current"]["accuracy"]
    try:
        resp = requests.post(
            f"{args.api_url}/metrics/update",
            params={"accuracy": cur_accuracy, "psi": pred_psi},
            timeout=3,
        )
        if resp.ok:
            print(f"Live gauge update pushed (accuracy={cur_accuracy:.4f}, psi={pred_psi:.4f})")
    except Exception:
        print("API gauge push skipped (API not running).")

    print("\n✓ Drift simulation complete.")


if __name__ == "__main__":
    main()