<div align="center">

# Customer Churn Prediction

**Production-Grade MLOps Pipeline on Telco Data**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package_manager-5C4EE5)](https://docs.astral.sh/uv/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.52.0-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-11.0-F46800?logo=grafana&logoColor=white)](https://grafana.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end MLOps system for telecom customer churn prediction. Covers the full production lifecycle: nested cross-validation training, MLflow experiment tracking, FastAPI inference serving, drift simulation, and live observability via Prometheus, Grafana and Pushgateway.

</div>

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [CLI Reference](#cli-reference)
- [Service URLs](#service-urls)
- [Monitoring](#monitoring)
- [Inference API](#inference-api)
- [Project Structure](#project-structure)
- [License](#license)

---

## Key Features

- **Seven-model benchmark:** Logistic Regression, SVC, Random Forest, XGBoost, MLP, LightGBM (with Borderline-SMOTE) and Stacking Ensemble (XGBoost + LightGBM to LR meta-learner)
- **Nested cross-validation:** 5-outer / 3-inner fold protocol with three-layer feature selection (filter, embedded and stability) and threshold optimisation per fold
- **MLflow tracking:** Full experiment logging of per-fold metrics, hyperparameters and model artifacts with model registry integration
- **FastAPI inference:** REST API with Prometheus instrumentation, MLflow registry load and local joblib fallback
- **Drift simulation:** Temporal train/test split with PSI reporting per feature and prediction distribution
- **Observability:** Two Grafana dashboards for inference monitoring and training metrics fed by Prometheus and Pushgateway respectively
- **CI/CD:** GitHub Actions pipeline covering lint (ruff), test (pytest) and Docker build for both training and inference images on every push to `main`
- **Decoupled training:** Training runs as a separate Docker service or CLI command; inference container does not depend on training being in the same process

---

## Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │                     GitHub Actions CI/CD                     │
        │         push → lint → test → build inference + training      │
        └──────────────────────────────┬───────────────────────────────┘
                                       │
              ┌────────────────────────▼────────────────────────┐
              │                 Docker Compose                  │
              │                                                 │
              │  ┌─────────────┐    ┌──────────────────────┐    │
              │  │  MLflow     │    │  FastAPI (inference) │    │
              │  │  :5000      │    │  :8000               │    │
              │  └──────┬──────┘    └──────────┬───────────┘    │
              │         │                      │                │
              │         │ model registry       │ /metrics       │
              │         │                      │                │
              │  ┌──────▼──────┐    ┌──────────▼───────────┐    │
              │  │ Pushgateway │    │    Prometheus        │    │
              │  │  :9091      ├───►│    :9090             │    │
              │  └─────────────┘    └──────────┬───────────┘    │
              │                               │                 │
              │                    ┌──────────▼───────────┐     │
              │                    │    Grafana           │     │
              │                    │    :3001             │     │
              │                    └──────────────────────┘     │
              └─────────────────────────────────────────────────┘
                                       │
                          ┌────────────▼──────────────┐
                          │     ./outputs (volume)    │
                          │  models/  results/ plots/ │
                          └───────────────────────────┘
```

**Training flow (decoupled):**
```
uv run churn-train
```

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

---

## Getting Started

```bash
git clone https://github.com/emad-malik/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
uv sync
```

### 1. Start the full stack

```bash
docker compose up --build
```

Services start in order: MLflow → FastAPI → Prometheus → Pushgateway → Grafana.

### 2. Train the model

```bash
uv run churn-train
```

This trains the **Stacking Ensemble** by default, saves the model to `outputs/models/`, logs the run to MLflow and pushes CV metrics to Pushgateway.

### 3. Verify health

```bash
curl http://localhost:8000/health
```

---

## CLI Reference

All commands are available after `uv sync`:

| Command | Description |
|:--------|:------------|
| `uv run churn-train` | Train stacking ensemble (default), save model, log to MLflow |
| `uv run churn-train --models logistic_regression xgboost` | Train specific models |
| `uv run churn-train-all` | Train all 7 models, register best in MLflow |
| `uv run churn-evaluate --model-path outputs/models/stacking_ensemble.joblib` | Evaluate saved model with SHAP plots |
| `uv run churn-benchmark` | Run cross-model benchmark comparison |
| `uv run churn-drift-simulation` | Simulate concept drift and generate PSI report |

**Key flags for `churn-train`:**

```bash
uv run churn-train \
  --models stacking_ensemble \   # model to train (default: stacking_ensemble)
  --outer-folds 5 \              # outer CV folds (default: 5)
  --inner-folds 3 \              # inner CV folds (default: 3)
  --seed 42                      # random seed (default: 42)
```

---

## Service URLs

| Service | URL | Credentials |
|:--------|:----|:------------|
| FastAPI Swagger | http://localhost:8000/docs | — |
| FastAPI Health | http://localhost:8000/health | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Pushgateway | http://localhost:9091 | — |
| Grafana | http://localhost:3001 | `admin` / `admin` |

---

## Monitoring

Grafana at `http://localhost:3001`. Two dashboards provisioned automatically:

| Dashboard | Source | Panels |
|:----------|:-------|:-------|
| Churn MLOps Monitoring | Prometheus ← FastAPI | Prediction count, churn rate, request latency, drift scores |
| Churn Model Training Metrics | Prometheus ← Pushgateway | PR-AUC, ROC-AUC, F1, Accuracy, Precision, Recall (mean ± std) |

Training metrics appear in Grafana immediately after `uv run churn-train` completes.

Alert rules in `monitoring/alert_rules.yml` fire on:
- API latency above threshold
- Model not loaded (health check failing)

---

## Inference API

FastAPI server on port 8000.

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Service health and model load status |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/predict` | POST | Single customer churn prediction |
| `/drift` | GET | Latest drift simulation cache |

---

## Project Structure

```
Customer-Churn-Prediction/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── docker/
│   ├── inference/
│   │   ├── app.py
│   │   └── Dockerfile
│   └── training/
│       └── Dockerfile
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── grafana/
│       ├── dashboards/
│       │   ├── churn-mlops.json
│       │   └── training-metrics.json
│       └── datasources/
│           └── prometheus.yml
├── outputs/
│   ├── models/
│   ├── results/
│   └── plots/
├── src/
│   └── churn/
│       ├── cli.py                  # CLI entry points
│       ├── dataset.py              # data loading and feature engineering
│       ├── model.py                # model registry (7 models)
│       ├── trainer.py              # nested CV training loop
│       ├── metrics.py              # evaluation metrics and plots
│       ├── tracking.py             # MLflow logging helpers
│       ├── visualize.py            # SHAP and PDP plots
│       ├── drift.py                # PSI drift computation
│       ├── drift_simulation.py     # drift simulation CLI
│       ├── train_all.py            # train all models CLI
│       └── benchmark.py            # benchmark comparison CLI
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## License

For educational and research use.