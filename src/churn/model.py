"""Churn prediction models.

BASELINE §6 – Model Configurations and Hyperparameters.

Each builder function returns:
  (estimator, param_grid)

where `estimator` is a sklearn-compatible Pipeline/estimator and
`param_grid` is a dict ready to be fed into RandomizedSearchCV or GridSearchCV.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# 1. Logistic Regression (Elastic Net)
# ---------------------------------------------------------------------------

def build_logistic_regression() -> tuple:
    """
    BASELINE §6 – Logistic Regression (Elastic Net):
      C in [1e-4, 1e4] (log scale), l1_ratio in [0, 1].
      Class weights balanced to handle imbalance.
    """
    estimator = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    param_grid = {
        "C":        loguniform(1e-4, 1e4),
        "l1_ratio": uniform(0, 1),
    }
    return estimator, param_grid


# ---------------------------------------------------------------------------
# 2. Support Vector Classifier (RBF)
# ---------------------------------------------------------------------------

def build_svc() -> tuple:
    """
    BASELINE §6 – SVC (RBF kernel):
      C in [1e2, 1e3] (log scale), gamma in [1e-4, 1].
      StandardScaler included in the pipeline.
      Outputs calibrated with CalibratedClassifierCV (Platt/sigmoid).
    """
    base_svc = SVC(kernel="rbf", probability=False, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_svc, method="sigmoid", cv=3)

    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    calibrated),
    ])
    param_grid = {
        "clf__estimator__C":     loguniform(1e2, 1e3),
        "clf__estimator__gamma": loguniform(1e-4, 1),
    }
    return estimator, param_grid


# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------

def build_random_forest() -> tuple:
    """
    BASELINE §6 – Random Forest:
      n_estimators in [300, 500], max_depth in [5, 40],
      min_samples_leaf in [1, 10].
      Class weights balanced.
    """
    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    param_grid = {
        "n_estimators":     randint(300, 501),
        "max_depth":        randint(5, 41),
        "min_samples_leaf": randint(1, 11),
    }
    return estimator, param_grid


# ---------------------------------------------------------------------------
# 4. XGBoost
# ---------------------------------------------------------------------------

def build_xgboost(scale_pos_weight: float = 2.5) -> tuple:
    """
    BASELINE §6 – XGBoost:
      n_estimators in [200, 300], learning_rate in [0.01, 0.30],
      max_depth in [3, 10], min_child_weight in [1, 10].
      scale_pos_weight handles class imbalance.
      Early stopping is applied on an inner validation slice.
    """
    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    param_grid = {
        "n_estimators":     randint(200, 301),
        "learning_rate":    loguniform(0.01, 0.30),
        "max_depth":        randint(3, 11),
        "min_child_weight": randint(1, 11),
    }
    return estimator, param_grid


# ---------------------------------------------------------------------------
# 5. Multilayer Perceptron (MLP)
# ---------------------------------------------------------------------------

def build_mlp() -> tuple:
    """
    BASELINE §6 – MLP:
      Hidden layer sizes in {(128,), (256,), (128, 64)}.
      Learning rate in [1e-4, 1e-2] (log scale).
      Alpha (weight decay) in [1e-5, 1e-2] (log scale).
      Adam optimizer, mini-batches, early stopping.
    """
    estimator = MLPClassifier(
        activation="relu",
        solver="adam",
        batch_size=64,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=500,
        random_state=42,
    )
    param_grid = {
        "hidden_layer_sizes": [(128,), (256,), (128, 64)],
        "learning_rate_init": loguniform(1e-4, 1e-2),
        "alpha":              loguniform(1e-5, 1e-2),
    }
    return estimator, param_grid


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": build_logistic_regression,
    "svc":                 build_svc,
    "random_forest":       build_random_forest,
    "xgboost":             build_xgboost,
    "mlp":                 build_mlp,
}


def get_model(name: str) -> tuple:
    """Return (estimator, param_grid) for the named model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
