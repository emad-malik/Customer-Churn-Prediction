"""
Nested cross-validation trainer — extended for:
  - LightGBM with Borderline-SMOTE oversampling
  - Stacking Ensemble (XGBoost + LightGBM → LR meta-learner)
  - sample_weight passthrough for MLP
  - Pipeline-safe clone() instead of set_params()
"""

from __future__ import annotations

import warnings
import inspect
from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.feature_selection import (
    SelectKBest, VarianceThreshold, chi2, f_classif, RFE,
)
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import BorderlineSMOTE

from churn.metrics import compute_metrics, find_best_threshold
from churn.model import get_model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Models that may use sample_weight when supported by estimator.fit() ─────
_MAY_USE_SAMPLE_WEIGHT = {"mlp"}

# ── Models that require SMOTE oversampling (applied only to training fold) ───
_NEEDS_SMOTE = {"lightgbm", "stacking_ensemble"}

# ── Models with no hyperparameter search (pre-tuned stacking) ────────────────
_NO_SEARCH = {"stacking_ensemble"}


def _supports_sample_weight(estimator: Any) -> tuple[bool, str | None]:
    """Return (supported, fit_param_key) for sample_weight passthrough."""
    # Pipeline case: pass as "<step>__sample_weight"
    if hasattr(estimator, "steps") and getattr(estimator, "steps", None):
        step_name, final_est = estimator.steps[-1]
        try:
            sig = inspect.signature(final_est.fit)
            if "sample_weight" in sig.parameters:
                return True, f"{step_name}__sample_weight"
        except (TypeError, ValueError):
            return False, None
        return False, None

    # Plain estimator case
    try:
        sig = inspect.signature(estimator.fit)
        if "sample_weight" in sig.parameters:
            return True, "sample_weight"
    except (TypeError, ValueError):
        return False, None
    return False, None


def _make_fit_params(model_name: str, y_train: pd.Series, estimator: Any) -> dict:
    if model_name not in _MAY_USE_SAMPLE_WEIGHT:
        return {}

    supports_sw, key = _supports_sample_weight(estimator)
    if not supports_sw or key is None:
        return {}

    sw = compute_sample_weight("balanced", y_train)
    return {key: sw}


# ── Feature selection ────────────────────────────────────────────────────────

def _filter_stage(X: pd.DataFrame, y: pd.Series, k_frac: float = 0.8) -> list[str]:
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    kept = X.columns[vt.get_support()].tolist()
    Xk = X[kept]

    cat_cols  = [c for c in Xk.columns if Xk[c].nunique() <= 2]
    cont_cols = [c for c in Xk.columns if c not in cat_cols]

    k_total = max(1, int(len(kept) * k_frac))
    k_cont  = max(1, int(k_total * len(cont_cols) / max(len(kept), 1)))
    k_cat   = max(1, k_total - k_cont)
    selected = []

    if cont_cols:
        k_cont = min(k_cont, len(cont_cols))
        Xc = RobustScaler().fit_transform(Xk[cont_cols])
        fs = SelectKBest(f_classif, k=k_cont)
        fs.fit(Xc, y)
        selected += [cont_cols[i] for i in fs.get_support(indices=True)]

    if cat_cols:
        k_cat = min(k_cat, len(cat_cols))
        Xcat = Xk[cat_cols].clip(lower=0)
        fs = SelectKBest(chi2, k=k_cat)
        fs.fit(Xcat, y)
        selected += [cat_cols[i] for i in fs.get_support(indices=True)]

    return list(set(selected))


def _embedded_stage(X: pd.DataFrame, y: pd.Series) -> list[str]:
    Xs = RobustScaler().fit_transform(X)

    en = ElasticNet(l1_ratio=0.9, alpha=0.01, max_iter=5000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en.fit(Xs, y)
    en_selected = X.columns[en.coef_ != 0].tolist()

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    rf_selected = X.columns[perm.importances_mean > 0].tolist()

    n_keep = max(1, int(len(X.columns) * 0.7))
    lsvc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    rfe  = RFE(lsvc, n_features_to_select=n_keep, step=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfe.fit(Xs, y)
    rfe_selected = X.columns[rfe.support_].tolist()

    return list(set(en_selected) | set(rf_selected) | set(rfe_selected))


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_appearances: int = 3,
) -> list[str]:
    counts = {c: 0 for c in X.columns}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, _ in skf.split(X, y):
        Xf, yf = X.iloc[train_idx], y.iloc[train_idx]
        filter_cols = _filter_stage(Xf, yf)
        embed_cols  = _embedded_stage(Xf[filter_cols], yf)
        for col in embed_cols:
            counts[col] += 1

    stable = [c for c, cnt in counts.items() if cnt >= min_appearances]
    if not stable:
        stable = [c for c, cnt in counts.items() if cnt > 0]
    return stable


# ── Borderline-SMOTE ─────────────────────────────────────────────────────────

def apply_borderline_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply Borderline-SMOTE to the training fold.
    Must be called AFTER feature selection and ONLY on training data.
    """
    smote = BorderlineSMOTE(
        kind="borderline-1",
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ── Nested CV ────────────────────────────────────────────────────────────────

def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 30,
    random_state: int = 42,
    use_feature_selection: bool = True,
    use_smote: bool | None = None,       # None → auto-detect from _NEEDS_SMOTE
    model_kwargs: dict | None = None,    # extra kwargs forwarded to get_model()
) -> dict[str, Any]:
    """
    Full nested-CV loop supporting all seven model types.

    Returns
    -------
    dict with:
      fold_metrics, fold_thresholds,
      selected_features_per_fold, best_params_per_fold,
      oof_probs (for stacking meta-feature generation)
    """
    apply_smote = use_smote if use_smote is not None else (model_name in _NEEDS_SMOTE)

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1)

    fold_metrics: list[dict] = []
    fold_thresholds: list[float] = []
    selected_features_per_fold: list[list] = []
    best_params_per_fold: list[dict] = []
    oof_probs = np.zeros(len(y))

    print(f"\n>>> Nested CV — {model_name.upper()}")
    print(f"    outer={outer_splits}-fold, inner={inner_splits}-fold, "
          f"n_iter={n_iter}, smote={apply_smote}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"  Fold {fold_idx}/{outer_splits} ...", end=" ", flush=True)

        X_train = X.iloc[train_idx].copy()
        X_test  = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test  = y.iloc[test_idx].copy()

        # Feature selection (inside outer training set)
        if use_feature_selection:
            stable_feats = stability_selection(X_train, y_train)
        else:
            stable_feats = X_train.columns.tolist()

        selected_features_per_fold.append(stable_feats)
        X_tr_sel = X_train[stable_feats]
        X_te_sel = X_test[stable_feats]

        # Borderline-SMOTE on training fold only
        if apply_smote:
            X_tr_sel, y_train = apply_borderline_smote(
                X_tr_sel, y_train, random_state=random_state + fold_idx
            )

        mkw = model_kwargs or {}
        estimator, param_grid = get_model(model_name, **mkw)
        fit_params = _make_fit_params(model_name, y_train, estimator)

        # Hyperparameter search (skip for stacking — already tuned)
        if model_name in _NO_SEARCH or not param_grid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_tr_sel, y_train, **fit_params)
            best_model = estimator
            best_params_per_fold.append({})
        else:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring="average_precision",
                refit=True,
                cv=inner_cv,
                random_state=random_state + fold_idx,
                n_jobs=-1,
                error_score=0.0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_tr_sel, y_train, **fit_params)
            best_params_per_fold.append(search.best_params_)
            best_model = search.best_estimator_

        # Threshold selection via inner OOF probabilities
        inner_oof = np.zeros(len(y_train))
        for in_tr, in_val in inner_cv.split(X_tr_sel, y_train):
            Xi_tr  = X_tr_sel.iloc[in_tr] if hasattr(X_tr_sel, "iloc") else X_tr_sel[in_tr]
            Xi_val = X_tr_sel.iloc[in_val] if hasattr(X_tr_sel, "iloc") else X_tr_sel[in_val]
            yi_tr  = y_train.iloc[in_tr] if hasattr(y_train, "iloc") else y_train[in_tr]

            est_clone = clone(best_model)
            inner_fp  = _make_fit_params(model_name, yi_tr, est_clone)

            if apply_smote:
                Xi_tr, yi_tr = apply_borderline_smote(
                    pd.DataFrame(Xi_tr, columns=stable_feats),
                    pd.Series(yi_tr),
                    random_state=random_state,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est_clone.fit(Xi_tr, yi_tr, **inner_fp)

            if hasattr(est_clone, "predict_proba"):
                inner_oof[in_val] = est_clone.predict_proba(Xi_val)[:, 1]
            else:
                raw = est_clone.decision_function(Xi_val)
                inner_oof[in_val] = 1 / (1 + np.exp(-raw))

        threshold = find_best_threshold(y_train, inner_oof)
        fold_thresholds.append(threshold)

        # Evaluate on outer test
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_te_sel)[:, 1]
        else:
            raw = best_model.decision_function(X_te_sel)
            y_prob = 1 / (1 + np.exp(-raw))

        oof_probs[test_idx] = y_prob
        metrics = compute_metrics(y_test, y_prob, threshold=threshold)
        fold_metrics.append(metrics)

        print(
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"thr={threshold:.3f}"
        )

    return {
        "fold_metrics":               fold_metrics,
        "fold_thresholds":            fold_thresholds,
        "selected_features_per_fold": selected_features_per_fold,
        "best_params_per_fold":       best_params_per_fold,
        "oof_probs":                  oof_probs,
    }


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    stable_features: list[str],
    best_params: dict | None = None,
    apply_smote: bool | None = None,
    model_kwargs: dict | None = None,
) -> Any:
    """
    Refit on the full dataset using the best configuration.
    """
    _smote = apply_smote if apply_smote is not None else (model_name in _NEEDS_SMOTE)
    mkw = model_kwargs or {}

    estimator, _ = get_model(model_name, **mkw)
    if best_params:
        try:
            estimator.set_params(**best_params)
        except Exception:
            pass

    X_sel = X[stable_features]
    if _smote:
        X_sel, y = apply_borderline_smote(X_sel, y)

    fit_params = _make_fit_params(model_name, y, estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_sel, y, **fit_params)

    return estimator