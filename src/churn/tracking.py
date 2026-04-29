"""MLflow logging helpers — metrics, params, artifacts, model registration."""

from __future__ import annotations

import os
import json
import tempfile
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import pandas as pd


# ── Setup ────────────────────────────────────────────────────────────────────

def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str = "churn_prediction",
) -> str:
    """
    Configure MLflow tracking URI and create/retrieve the experiment.
    Returns the experiment ID.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(experiment_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id


# ── Run helpers ───────────────────────────────────────────────────────────────

def log_cv_results(
    model_name: str,
    fold_metrics: list[dict],
    summary: dict,
    params: dict | None = None,
    tags: dict | None = None,
    artifact_paths: list[str] | None = None,
) -> str:
    """
    Log a complete nested-CV experiment to MLflow.
    Returns the MLflow run_id.
    """
    with mlflow.start_run(run_name=model_name) as run:

        # Tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("eval_protocol", "nested_5fold_cv")
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, v)

        # Params
        for k, v in (params or {}).items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        # Summary metrics (mean values)
        for metric, stats in summary.items():
            mlflow.log_metric(f"{metric}_mean", stats["mean"])
            mlflow.log_metric(f"{metric}_std",  stats["std"])

        # Per-fold metrics as a step series
        for fold_idx, fold_m in enumerate(fold_metrics):
            for metric, val in fold_m.items():
                mlflow.log_metric(f"fold_{metric}", val, step=fold_idx)

        # Artifacts (plots, JSON, etc.)
        for path in (artifact_paths or []):
            if os.path.isfile(path):
                mlflow.log_artifact(path)

        return run.info.run_id


def log_drift_results(
    model_name: str,
    feature_psi: "pd.Series",
    prediction_psi: float,
    perf_comparison: dict,
    artifact_paths: list[str] | None = None,
) -> str:
    """Log drift detection results to MLflow."""
    with mlflow.start_run(run_name=f"{model_name}_drift") as run:
        mlflow.set_tag("run_type", "drift_detection")
        mlflow.set_tag("model", model_name)

        mlflow.log_metric("prediction_psi", prediction_psi)
        mlflow.log_metric("mean_feature_psi", float(feature_psi.mean()))
        mlflow.log_metric("max_feature_psi",  float(feature_psi.max()))

        for split in ("reference", "current"):
            for metric, val in perf_comparison[split].items():
                mlflow.log_metric(f"{split}_{metric}", val)

        # Log delta metrics
        for metric in perf_comparison["reference"]:
            delta = (
                perf_comparison["current"][metric]
                - perf_comparison["reference"][metric]
            )
            mlflow.log_metric(f"delta_{metric}", delta)

        # Log feature PSI as a JSON artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_feature_psi.json", delete=False
        ) as f:
            json.dump(feature_psi.to_dict(), f, indent=2)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="drift")
        os.unlink(tmp_path)

        for path in (artifact_paths or []):
            if os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path="drift")

        return run.info.run_id


def register_model(
    run_id: str,
    model_name: str,
    estimator,
    X_sample: "pd.DataFrame",
    stage: str = "Staging",
    description: str = "",
) -> None:
    """
    Log a sklearn-compatible estimator to MLflow and register it in the
    Model Registry under `model_name`, promoting to `stage`.
    """
    client = MlflowClient()

    with mlflow.start_run(run_id=run_id):
        signature = infer_signature(X_sample, estimator.predict_proba(X_sample)[:, 1])
        mlflow.sklearn.log_model(
            sk_model=estimator,
            artifact_path="model",
            signature=signature,
            registered_model_name=model_name,
        )

    # Transition to requested stage
    versions = client.get_latest_versions(model_name)
    if versions:
        latest = versions[-1]
        client.transition_model_version_stage(
            name=model_name,
            version=latest.version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),
        )
        if description:
            client.update_model_version(
                name=model_name,
                version=latest.version,
                description=description,
            )


def load_production_model(model_name: str, stage: str = "Production"):
    """Load the production-stage model from the MLflow registry."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.sklearn.load_model(model_uri)