# Telecom Churn MLOps Pipeline

This project extends the Zerine et al. (2026) churn benchmark into a complete MLOps pipeline with training, experiment tracking, model serving, monitoring, and drift simulation.

## Features

- Seven-model benchmark training
  - Logistic Regression
  - SVC
  - Random Forest
  - XGBoost
  - MLP
  - LightGBM with Borderline-SMOTE
  - Stacking Ensemble (XGBoost + LightGBM with Logistic Regression meta-learner)
- MLflow tracking and model registry integration
- FastAPI inference service with Prometheus metrics
- Grafana dashboard for live model observability
- Drift simulation workflow with PSI reporting

## Architecture

- Training pipeline
  - Runs nested CV and logs metrics and artifacts to MLflow
  - Selects best model at runtime using PR-AUC
  - Registers the best model in MLflow registry
- Serving pipeline
  - FastAPI serves prediction endpoints
  - Loads model from MLflow registry; falls back to local model artifacts when needed
- Monitoring pipeline
  - Prometheus scrapes FastAPI and service metrics
  - Grafana visualizes prediction count, class split, and latency

## Prerequisites

- Windows, Linux, or macOS
- Docker and Docker Compose
- Python 3.14
- uv package manager (recommended) or pip

## Project Setup

### 1) Install Python dependencies

Using uv (recommended):

```bash
uv sync
```

If you use pip instead:

```bash
py -3.14 -m pip install -e .
```

Windows CMD venv activation:

```cmd
cd /d C:\Users\emaad\Downloads\Telco-Churn-Prediction
.\.venv\Scripts\activate.bat
```

### 2) Train all models and log runs

```bash
python scripts/train_all.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

What this produces:

- MLflow runs for each model
- Local model files in outputs/models
- CV results in outputs/results
- Plots in outputs/plots

### 3) Start monitoring and serving stack

```bash
docker-compose up -d --build
```

### 4) Confirm services are healthy

```bash
docker-compose ps --all
```

Expected healthy services:

- mlflow
- fastapi_app
- prometheus
- grafana

## Service URLs

| Service    | URL                        | Credentials   |
|------------|----------------------------|---------------|
| FastAPI    | http://localhost:8000/docs | none          |
| MLflow     | http://localhost:5000      | none          |
| Prometheus | http://localhost:9090      | none          |
| Grafana    | http://localhost:3001      | admin / admin |

Note: Grafana host port is 3001 because 3000 may already be occupied.

## Testing Predictions from PowerShell

Important: Use Invoke-RestMethod with Method Post. If you send GET to /predict you will receive 405 Method Not Allowed.

### Single prediction request

```powershell
$url = "http://localhost:8000/predict"
$body = @{
  tenure = 12
  MonthlyCharges = 65.5
  TotalCharges = 786.5
  SeniorCitizen = 0
  Contract = 0
  PaperlessBilling = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $body
```

### Burst test (10 POST requests)

```powershell
$url = "http://localhost:8000/predict"

1..10 | ForEach-Object {
  $body = @{
    tenure = Get-Random -Minimum 0 -Maximum 73
    MonthlyCharges = [math]::Round((Get-Random -Minimum 20 -Maximum 121) + (Get-Random), 2)
    TotalCharges = [math]::Round((Get-Random -Minimum 1 -Maximum 73) * (Get-Random -Minimum 20 -Maximum 121), 2)
    SeniorCitizen = Get-Random -Minimum 0 -Maximum 2
    Contract = Get-Random -Minimum 0 -Maximum 3
    PaperlessBilling = Get-Random -Minimum 0 -Maximum 2
  } | ConvertTo-Json

  Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $body | Out-Null
}
```

## Monitoring Validation

### Prometheus checks

Open Prometheus and run:

- sum(churn_predictions_total)
- sum by (predicted_class) (churn_predictions_total)
- sum(rate(churn_predictions_total[5m]))
- histogram_quantile(0.95, sum by (le) (rate(churn_prediction_latency_seconds_bucket[5m])))

### Grafana checks

1. Open Grafana at http://localhost:3001
2. Open dashboard: Churn MLOps Monitoring
3. Set time range to Last 15 minutes
4. Click Refresh

If panels look empty, first generate new prediction traffic and refresh again.

## Typical End-to-End Run Order

1. Install dependencies (uv sync)
2. Train models (python scripts/train_all.py ...)
3. Start stack (docker-compose up -d --build)
4. Verify health (docker-compose ps --all)
5. Send prediction traffic
6. Inspect MLflow runs
7. Inspect Prometheus metrics
8. Inspect Grafana dashboard
9. Run benchmark and drift scripts if needed

## Additional Scripts

```bash
python scripts/benchmark.py
python scripts/drift_simulation.py
```

## Troubleshooting

### 1) 405 Method Not Allowed on /predict

Cause: GET was sent to /predict.

Fix: send POST with JSON body using Invoke-RestMethod.

### 2) 503 Model not loaded from /health

Possible causes:

- API container started before model files were available
- MLflow model artifact path unavailable inside API container

Fix:

```bash
docker-compose up -d --build fastapi_app
```

Then confirm:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health -Method Get
```

### 3) Grafana dashboard appears unresponsive

Checks:

- Ensure Grafana container is healthy
- Ensure dashboard time window includes recent traffic
- Ensure prediction traffic was sent recently
- Re-login and hard refresh browser if session token issues occur

### 4) Verify all containers

```bash
docker-compose ps --all
docker-compose logs fastapi_app --tail 120
docker-compose logs grafana --tail 120
docker-compose logs prometheus --tail 120
```

## Notes

- model_version may appear as unknown in prediction responses when local model fallback is used.
- MLflow can still be fully usable for run tracking even if serving falls back to local artifacts.