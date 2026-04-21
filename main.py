"""
FastAPI inference server for the Churn prediction model.

Endpoints
---------
POST /predict         — single-record or batch prediction
POST /predict/batch   — explicit batch endpoint
GET  /health          — liveness/readiness probe
GET  /metrics         — Prometheus metrics scrape target
GET  /drift           — latest drift summary (reads cached values)
GET  /model/info      — loaded model metadata
"""

from __future__ import annotations

import os
import time
import logging
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import mlflow.sklearn

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

logger = logging.getLogger("churn_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "info").upper())


# ── Prometheus metrics ────────────────────────────────────────────────────────

PRED_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of predictions served",
    ["predicted_class", "model_version"],
)

LATENCY_HIST = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

ACCURACY_GAUGE = Gauge(
    "churn_model_accuracy_gauge",
    "Rolling accuracy (updated via /feedback endpoint)",
)

PSI_GAUGE = Gauge(
    "churn_psi_score_gauge",
    "Latest PSI drift score (updated by drift_simulation script)",
)

CHURN_RATE_GAUGE = Gauge(
    "churn_output_rate_gauge",
    "Rolling fraction of positive (churn) predictions",
)

REQUESTS_IN_FLIGHT = Gauge(
    "churn_requests_in_flight",
    "Number of requests currently being processed",
)

# Rolling window for live churn-rate tracking
_pred_window: list[int] = []
_window_lock = threading.Lock()
_WINDOW_SIZE = 200


def _update_churn_rate(prediction: int) -> None:
    global _pred_window
    with _window_lock:
        _pred_window.append(prediction)
        if len(_pred_window) > _WINDOW_SIZE:
            _pred_window.pop(0)
        rate = sum(_pred_window) / len(_pred_window)
    CHURN_RATE_GAUGE.set(rate)


# ── Model loading ─────────────────────────────────────────────────────────────

_model = None
_model_meta: dict = {}


def _load_model() -> None:
    global _model, _model_meta
    model_name  = os.getenv("MODEL_NAME",  "churn_best_model")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    tracking    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking)
    try:
        uri = f"models:/{model_name}/{model_stage}"
        _model = mlflow.sklearn.load_model(uri)
        _model_meta = {
            "source": "mlflow_registry",
            "name":   model_name,
            "stage":  model_stage,
            "uri":    uri,
        }
        logger.info("Model loaded from MLflow registry: %s/%s", model_name, model_stage)
    except Exception as e:
        logger.warning("MLflow registry load failed (%s). Falling back to local joblib.", e)
        _load_local_fallback()


def _load_local_fallback() -> None:
    """Load a locally serialised model if MLflow is unavailable."""
    global _model, _model_meta
    import joblib
    paths = [
        "outputs/models/logistic_regression.joblib",
        "outputs/models/random_forest.joblib",
        "outputs/models/stacking_ensemble.joblib",
        "outputs/models/lightgbm.joblib",
        "outputs/models/xgboost.joblib",
        "outputs/models/mlp.joblib",
        "outputs/models/svc.joblib",
    ]
    for path in paths:
        if os.path.isfile(path):
            _model = joblib.load(path)
            _model_meta = {"source": "local", "path": path}
            logger.info("Model loaded from local file: %s", path)
            return
    logger.error("No model found — /predict will return 503.")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    # Seed accuracy gauge so Grafana panel doesn't show 0
    ACCURACY_GAUGE.set(float(os.getenv("INITIAL_ACCURACY", "0.90")))
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="MLOps inference server — predictions, metrics, drift.",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / response schemas ────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Feature vector for a single customer. All fields optional for flexibility."""
    tenure:              float = Field(default=0,   ge=0,   description="Months as customer")
    MonthlyCharges:      float = Field(default=0,   ge=0)
    TotalCharges:        float = Field(default=0,   ge=0)
    SeniorCitizen:       int   = Field(default=0,   ge=0, le=1)
    Contract:            int   = Field(default=0,   ge=0, le=2,
                                       description="0=M2M, 1=1yr, 2=2yr")
    PaperlessBilling:    int   = Field(default=0,   ge=0, le=1)
    # Add-ons / encoded categoricals — defaults to 0 (not subscribed)
    InternetService_DSL:          int = Field(default=0)
    InternetService_Fiber_optic:  int = Field(default=0)
    OnlineSecurity_Yes:           int = Field(default=0)
    OnlineBackup_Yes:             int = Field(default=0)
    DeviceProtection_Yes:         int = Field(default=0)
    TechSupport_Yes:              int = Field(default=0)
    StreamingTV_Yes:              int = Field(default=0)
    StreamingMovies_Yes:          int = Field(default=0)
    # Engineered features
    tenure_bin:          int   = Field(default=0)
    high_charge_flag:    int   = Field(default=0)
    charges_per_tenure:  float = Field(default=0.0)
    tickets_capped:      float = Field(default=0.0)
    tickets_sq:          float = Field(default=0.0)

    class Config:
        extra = "allow"   # Allow extra fields so new features don't break old clients


class PredictionResponse(BaseModel):
    customer_id:       Optional[str] = None
    churn_probability: float
    churn_prediction:  int
    confidence:        str
    model_version:     str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]
    customer_ids: Optional[list[str]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": _model_meta}


@app.get("/model/info")
async def model_info():
    return _model_meta


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerFeatures, request: Request):
    """Predict churn probability for a single customer."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    REQUESTS_IN_FLIGHT.inc()
    t0 = time.perf_counter()
    try:
        row = pd.DataFrame([customer.model_dump()])
        row = row.reindex(columns=_get_feature_cols(row), fill_value=0)

        prob = float(_model.predict_proba(row)[0, 1])
        pred = int(prob >= 0.5)

    finally:
        latency = time.perf_counter() - t0
        LATENCY_HIST.observe(latency)
        REQUESTS_IN_FLIGHT.dec()

    version = _model_meta.get("stage", "unknown")
    PRED_COUNTER.labels(predicted_class=str(pred), model_version=version).inc()
    _update_churn_rate(pred)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        confidence="high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.1 else "low",
        model_version=version,
    )


@app.post("/predict/batch")
async def predict_batch(batch: BatchRequest):
    """Batch prediction endpoint."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    REQUESTS_IN_FLIGHT.inc()
    t0 = time.perf_counter()
    try:
        rows = pd.DataFrame([c.model_dump() for c in batch.customers])
        rows = rows.reindex(columns=_get_feature_cols(rows), fill_value=0)
        probs = _model.predict_proba(rows)[:, 1]
    finally:
        LATENCY_HIST.observe(time.perf_counter() - t0)
        REQUESTS_IN_FLIGHT.dec()

    version = _model_meta.get("stage", "unknown")
    responses = []
    for i, (prob, pred_row) in enumerate(zip(probs, batch.customers)):
        pred = int(prob >= 0.5)
        PRED_COUNTER.labels(predicted_class=str(pred), model_version=version).inc()
        _update_churn_rate(pred)
        responses.append({
            "customer_id": batch.customer_ids[i] if batch.customer_ids else str(i),
            "churn_probability": round(float(prob), 4),
            "churn_prediction":  pred,
        })

    return {"predictions": responses, "count": len(responses)}


@app.get("/metrics")
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
async def drift_summary():
    """Return the latest cached drift scores (written by drift_simulation.py)."""
    cache_path = "outputs/results/drift_cache.json"
    if os.path.isfile(cache_path):
        import json
        with open(cache_path) as f:
            return json.load(f)
    return {"psi": PSI_GAUGE._value.get(), "status": "no drift run yet"}


@app.post("/metrics/update")
async def update_gauges(accuracy: float, psi: float):
    """
    Allow the drift_simulation script to push live accuracy and PSI
    into Prometheus gauges via the API (simulates real-time monitoring).
    """
    ACCURACY_GAUGE.set(accuracy)
    PSI_GAUGE.set(psi)
    return {"updated": True}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return column list aligned with what the loaded model expects.
    Falls back to the DataFrame's own columns if feature_names_in_ is absent.
    """
    try:
        return list(_model.feature_names_in_)
    except AttributeError:
        pass
    try:
        # StackingClassifier / Pipeline
        inner = _model
        while hasattr(inner, "steps"):
            inner = inner.steps[-1][1]
        return list(inner.feature_names_in_)
    except AttributeError:
        return list(df.columns)