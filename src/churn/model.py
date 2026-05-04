"""Model registry — all five baseline models plus LightGBM and Stacking Ensemble.

Each builder returns (estimator, param_grid) where estimator is a
sklearn-compatible object and param_grid feeds RandomizedSearchCV.
"""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ── 1. Logistic Regression (Elastic Net) ────────────────────────────────────

def build_logistic_regression() -> tuple:
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )),
    ])
    param_grid = {
        "clf__C":        loguniform(1e-4, 1e4),
        "clf__l1_ratio": uniform(0, 1),
    }
    return estimator, param_grid


# ── 2. SVC (RBF) ─────────────────────────────────────────────────────────────

def build_svc() -> tuple:
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


# ── 3. Random Forest ──────────────────────────────────────────────────────────

def build_random_forest() -> tuple:
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


# ── 4. XGBoost ────────────────────────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = 2.5) -> tuple:
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


# ── 5. MLP ───────────────────────────────────────────────────────────────────

def build_mlp() -> tuple:
    """MLP wrapped in StandardScaler pipeline. sample_weight handled in trainer."""
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            activation="relu",
            solver="adam",
            batch_size=64,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=500,
            random_state=42,
        )),
    ])
    param_grid = {
        "clf__hidden_layer_sizes": [(128,), (256,), (128, 64)],
        "clf__learning_rate_init": loguniform(1e-4, 1e-2),
        "clf__alpha":              loguniform(1e-5, 1e-2),
    }
    return estimator, param_grid


# ── 6. LightGBM ───────────────────────────────────────────────────────────────

def build_lightgbm(class_weight: str = "balanced") -> tuple:
    """
    LightGBM with dart boosting type for better generalisation on small-medium
    tabular datasets. class_weight='balanced' handles the 3:1 imbalance natively.
    Borderline-SMOTE oversampling is applied *before* this model is fitted
    in the training script (not inside the pipeline) to avoid leakage.
    """
    estimator = LGBMClassifier(
        boosting_type="dart",
        objective="binary",
        metric="average_precision",
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    param_grid = {
        "n_estimators":    randint(200, 600),
        "learning_rate":   loguniform(0.005, 0.3),
        "max_depth":       randint(3, 12),
        "num_leaves":      randint(20, 120),
        "min_child_samples": randint(10, 60),
        "subsample":       uniform(0.6, 0.4),       # [0.6, 1.0]
        "colsample_bytree": uniform(0.5, 0.5),      # [0.5, 1.0]
        "reg_alpha":       loguniform(1e-4, 1.0),
        "reg_lambda":      loguniform(1e-4, 1.0),
    }
    return estimator, param_grid


# ── 7. Stacking Ensemble ──────────────────────────────────────────────────────

def build_stacking_ensemble(
    xgb_params: dict | None = None,
    lgbm_params: dict | None = None,
) -> tuple:
    """
        Stacking ensemble:
            Base learners : XGBoost + LightGBM + ExtraTrees
      Meta-learner  : Logistic Regression (with StandardScaler)
      passthrough   : True → meta-learner also sees original features

    param_grid is empty because the base learners are pre-tuned before
    constructing the stack. Call build_stacking_ensemble(xgb_params, lgbm_params)
    with the best params from each model's nested CV.
    """
    _xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": 2.5,
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        **(xgb_params or {}),
    }
    _lgbm_params = {
        "boosting_type": "dart",
        "objective": "binary",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
        **(lgbm_params or {}),
    }

    extra = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    base_learners = [
        ("xgb",  XGBClassifier(**_xgb_params)),
        ("lgbm", LGBMClassifier(**_lgbm_params)),
        ("extra", extra),
    ]

    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )),
    ])

    estimator = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=-1,
    )
    return estimator, {}


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, callable] = {
    "logistic_regression": build_logistic_regression,
    "svc":                 build_svc,
    "random_forest":       build_random_forest,
    "xgboost":             build_xgboost,
    "mlp":                 build_mlp,
    "lightgbm":            build_lightgbm,
    "stacking_ensemble":   build_stacking_ensemble,
}


def get_model(name: str, **kwargs) -> tuple:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)