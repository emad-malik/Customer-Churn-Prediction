<div align="center">

# One Step Ahead

**Predicting and Preventing Customer Churn in Telecom**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![uv](https://img.shields.io/badge/uv-package_manager-5C4EE5)](https://docs.astral.sh/uv/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13.0-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-2.52.0-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io)
[![Grafana](https://img.shields.io/badge/Grafana-11.0-F46800?logo=grafana&logoColor=white)](https://grafana.com)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%2B%20ECR-FF9900?logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end MLOps system for telecom customer churn prediction. Covers the full production lifecycle: nested cross-validation training, MLflow experiment tracking, FastAPI inference serving, live observability via Prometheus and Grafana, and cloud deployment on AWS with CI/CD.

[Live Frontend](https://customer-churn-prediction-vite-ui.vercel.app/) · [API Docs](http://34.235.199.95:8000/docs) · [Grafana](http://34.235.199.95:3001) · [MLflow](http://34.235.199.95:5000)

</div>

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Cloud Deployment](#cloud-deployment)
- [Prerequisites](#prerequisites)
- [Getting Started (Local)](#getting-started-local)
- [Getting Started (AWS)](#getting-started-aws)
- [CLI Reference](#cli-reference)
- [Service URLs](#service-urls)
- [Monitoring](#monitoring)
- [Inference API](#inference-api)
- [Frontend](#frontend)
- [Project Structure](#project-structure)
- [License](#license)

---

## Key Features

- **Seven-model benchmark:** Logistic Regression, SVC, Random Forest, XGBoost, MLP, LightGBM (with Borderline-SMOTE) and Stacking Ensemble (XGBoost + LightGBM + ExtraTrees to LR meta-learner)
- **Nested cross-validation:** 5-outer / 3-inner fold protocol with three-layer feature selection (filter, embedded and stability) and threshold optimisation per fold
- **MLflow tracking:** Full experiment logging of per-fold metrics, hyperparameters and model artifacts with model registry integration
- **FastAPI inference:** REST API with CORS, Prometheus instrumentation, MLflow registry load and local joblib fallback
- **Drift simulation:** Temporal train/test split with PSI reporting per feature and prediction distribution
- **Observability:** Two Grafana dashboards for inference monitoring and training metrics fed by Prometheus and Pushgateway respectively
- **CI/CD:** GitHub Actions pipeline: lint (ruff), test (pytest), build Docker images, push to Amazon ECR and deploy to EC2 on every push to `main`
- **Cloud deployment:** Docker images hosted on Amazon ECR, served from a `t3.2xlarge` EC2 instance with Elastic IP
- **Live frontend:** React app on Vercel calling the FastAPI inference endpoint
- **Decoupled training:** Training runs as a separate Docker service or CLI command; inference container does not depend on training being in the same process

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      GitHub Actions CI/CD                           │
  │     push → lint → test → build images → push ECR → deploy EC2       │
  └────────────────────────────┬────────────────────────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │              Amazon ECR                   │
         │   churn-inference:latest                  │
         │   churn-training:latest                   │
         └─────────────────────┬─────────────────────┘
                               │ docker pull
         ┌─────────────────────▼─────────────────────┐
         │         EC2 (t3.2xlarge) + Elastic IP     │
         │              Docker Compose               │
         │                                           │
         │  ┌────────────┐    ┌─────────────────────┐│
         │  │  MLflow    │    │  FastAPI (inference)││
         │  │  :5000     │    │  :8000              ││
         │  └─────┬──────┘    └──────────┬──────────┘│
         │        │                      │           │
         │        │ model registry       │ /metrics  │
         │        │                      │           │
         │  ┌─────▼───────┐    ┌─────────▼──────────┐│
         │  │ Pushgateway │    │   Prometheus       ││
         │  │  :9091      ├───►│   :9090            ││
         │  └─────────────┘    └──────────┬─────────┘│
         │                               │           │
         │                    ┌──────────▼─────────┐ │
         │                    │   Grafana          │ │
         │                    │   :3001            │ │
         │                    └────────────────────┘ │
         └───────────────────────────────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │       Vercel (Frontend)         │
              │  React UI → POST /predict       │
              └─────────────────────────────────┘
```

---

## Cloud Deployment

The production stack runs on AWS:

| Component | Service | Details |
|:----------|:--------|:--------|
| Container registry | Amazon ECR | `churn-inference` and `churn-training` repos |
| Compute | EC2 `t3.2xlarge` | 8 vCPU, 32GB RAM, 100GB gp3, Elastic IP |
| CI/CD | GitHub Actions | Build, push to ECR, SSH deploy on every push to `main` |
| Frontend | Vercel | React app deployed from separate repo |
| IAM | `github-actions` user | `AmazonEC2ContainerRegistryPowerUser` for CI/CD |
| IAM | `ec2-ecr-pull` role | `AmazonEC2ContainerRegistryReadOnly` attached to EC2 |

---

## Prerequisites

**Local development:**
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

**Cloud deployment:**
- AWS account with ECR, EC2 and IAM configured
- GitHub repository secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`

---

## Getting Started (Local)

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

## Getting Started (AWS)

After the CI/CD pipeline runs successfully (all green on GitHub Actions):

### 1. SSH into EC2

```bash
ssh -i "Customer-Churn.pem" ubuntu@34.235.199.95
cd /home/ubuntu/app
```

### 2. Verify the stack is running

```bash
docker-compose ps
```

### 3. Train the model

```bash
docker-compose --profile training up training
docker-compose restart fastapi_app
```

### 4. Verify health

```bash
curl http://34.235.199.95:8000/health
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
  --models stacking_ensemble \
  --outer-folds 5 \
  --inner-folds 3 \
  --seed 42
```

---

## Service URLs

**Local:**

| Service | URL | Credentials |
|:--------|:----|:------------|
| FastAPI Swagger | http://localhost:8000/docs | - |
| FastAPI Health | http://localhost:8000/health | - |
| MLflow | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |
| Pushgateway | http://localhost:9091 | - |
| Grafana | http://localhost:3001 | `admin` / `admin` |

**Production (AWS):**

| Service | URL | Credentials |
|:--------|:----|:------------|
| FastAPI Swagger | http://34.235.199.95:8000/docs | - |
| FastAPI Health | http://34.235.199.95:8000/health | - |
| MLflow | http://34.235.199.95:5000 | - |
| Prometheus | http://34.235.199.95:9090 | - |
| Pushgateway | http://34.235.199.95:9091 | - |
| Grafana | http://34.235.199.95:3001 | `admin` / `admin` |

---

## Monitoring

Grafana at port 3001. Two dashboards provisioned automatically:

| Dashboard | Source | Panels |
|:----------|:-------|:-------|
| Churn MLOps Monitoring | Prometheus ← FastAPI | Prediction count, churn rate, request latency, drift scores |
| Churn Model Training Metrics | Prometheus ← Pushgateway | PR-AUC, ROC-AUC, F1, Accuracy, Precision, Recall (mean ± std) |

Training metrics appear in Grafana immediately after training completes.

Alert rules in `monitoring/alert_rules.yml` fire on:
- API latency above threshold
- Model not loaded (health check failing)
- Churn rate spiking above 50%
- PSI drift score exceeding 0.2

---

## Inference API

FastAPI server on port 8000.

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/health` | GET | Service health and model load status |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/predict` | POST | Single customer churn prediction |
| `/predict/batch` | POST | Batch churn prediction |
| `/drift` | GET | Latest drift simulation cache |
| `/model/info` | GET | Loaded model metadata |
| `/metrics/update` | POST | Push accuracy and PSI gauge updates |

---

## Frontend

A React frontend deployed on Vercel provides a visual interface for the prediction API.

**Live:** https://customer-churn-prediction-vite-ui.vercel.app/

The UI lets you configure all 36 model features (tenure, contract type, internet service, payment method, add-ons) and returns churn probability, prediction, confidence level and detected risk factors.

Built with Vite + React, deployed on Vercel. Source in a separate repository.

---

## Project Structure

```
Customer-Churn-Prediction/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # Lint → Test → Build → ECR → EC2
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
│           └── datasource.yml
├── outputs/
│   ├── models/
│   ├── results/
│   └── plots/
├── src/
│   └── churn/
│       ├── cli.py                 # CLI entry points
│       ├── dataset.py             # Data loading and feature engineering
│       ├── model.py               # Model registry (7 models)
│       ├── trainer.py             # Nested CV training loop
│       ├── metrics.py             # Evaluation metrics and plots
│       ├── tracking.py            # MLflow logging helpers
│       ├── visualize.py           # SHAP and PDP plots
│       ├── drift.py               # PSI drift computation
│       ├── drift_simulation.py    # Drift simulation CLI
│       ├── train_all.py           # Train all models CLI
│       └── benchmark.py           # Benchmark comparison CLI
├── tests/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## License

For educational and research use.