# ── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Use system Python, don't let uv download its own
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# System deps for LightGBM / XGBoost compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies only (not the project) — cached layer
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy application source
COPY . /app

# Install the project itself in non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


# ── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Runtime lib for LightGBM OpenMP + health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the virtual environment from the builder (no uv, no build tools)
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/                    ./src/
COPY docker/inference/app.py ./app.py

# Non-root user for security
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Use the venv's Python/binaries directly
ENV PATH="/app/.venv/bin:$PATH" \
    MLFLOW_TRACKING_URI=http://mlflow:5000 \
    MODEL_NAME=churn_stacking_ensemble \
    MODEL_STAGE=Production \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info"]
