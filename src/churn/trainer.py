"""churn prediction training loop.

BASELINE §4–5 – Feature Selection Protocol + Nested Cross Validation.

Three-layer feature selection:
  1. Filter Stage  : variance threshold + univariate F-test / chi-squared
  2. Embedded Stage: ElasticNet + permutation importance (RandomForest)
  3. Wrapper Stage : RFE with LinearSVC
  4. Stability     : keep features selected in >= 3 of 5 inner folds

Nested CV:
  Outer: 5-fold stratified (unbiased performance estimate)
  Inner: 3-fold stratified (hyperparameter tuning, optimises PR AUC)

v2 changes vs v1:
  - MLP class-imbalance fix: compute_sample_weight('balanced') passed as
    fit param so the cross-entropy loss accounts for the minority class.
    MLPClassifier silently ignores class_weight= so this was a silent bug.
  - Pipeline-aware cloning: use sklearn.base.clone() instead of manually
    calling set_params(), which broke when estimators were Pipelines.
  - LR / SVC feature-selection inputs no longer double-scaled: the models
    themselves now own their StandardScaler step, so _filter_stage and
    _embedded_stage operate on the raw (OHE + numeric) feature matrix.
"""

from __future__ import annotations

import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


from sklearn.base import clone
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from churn.metrics import compute_metrics, find_best_threshold
from churn.model import get_model

# Suppress convergence warnings during hyperparameter search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Helper: does a model need sample_weight for class-imbalance correction?
# ---------------------------------------------------------------------------

_NEEDS_SAMPLE_WEIGHT = {"mlp"}   # models whose sklearn class ignores class_weight


def _make_fit_params(model_name: str, y_train: pd.Series) -> dict:
    """
    Return a fit_params dict for RandomizedSearchCV / manual fit calls.
    For MLP (and any future model in _NEEDS_SAMPLE_WEIGHT) we compute
    balanced sample weights so the cross-entropy loss treats both classes
    proportionally despite the ~3:1 imbalance.

    The key format is  "<last_pipeline_step_name>__sample_weight"
    which sklearn's Pipeline.fit() routes to that step's fit() call.
    """
    if model_name in _NEEDS_SAMPLE_WEIGHT:
        sw = compute_sample_weight("balanced", y_train)
        # All models in model.py use Pipeline with final step named "clf"
        return {"clf__sample_weight": sw}
    return {}


# ---------------------------------------------------------------------------
# Feature selection helpers
# ---------------------------------------------------------------------------

def _filter_stage(X: pd.DataFrame, y: pd.Series, k_frac: float = 0.8) -> list[str]:
    """
    Filter Stage (BASELINE §4):
      1. Remove constant / quasi-constant columns (variance < 0.01)
      2. Retain top-k% features via:
         - F-test for scaled continuous features
         - Chi-squared (or mutual information) for binary/categorical features
    """
    # 1. Variance filter
    vt = VarianceThreshold(threshold=0.01)
    vt.fit(X)
    kept = X.columns[vt.get_support()].tolist()
    Xk = X[kept]

    # 2. Determine continuous vs categorical columns
    cat_cols = [c for c in Xk.columns if Xk[c].nunique() <= 2]
    cont_cols = [c for c in Xk.columns if c not in cat_cols]

    k_total = max(1, int(len(kept) * k_frac))
    k_cont = max(1, int(k_total * len(cont_cols) / max(len(kept), 1)))
    k_cat  = max(1, k_total - k_cont)

    selected = []

    if cont_cols:
        k_cont = min(k_cont, len(cont_cols))
        scaler = RobustScaler()
        Xc_scaled = scaler.fit_transform(Xk[cont_cols])
        fs_cont = SelectKBest(f_classif, k=k_cont)
        fs_cont.fit(Xc_scaled, y)
        selected += [cont_cols[i] for i in fs_cont.get_support(indices=True)]

    if cat_cols:
        k_cat = min(k_cat, len(cat_cols))
        Xcat = Xk[cat_cols].clip(lower=0)
        fs_cat = SelectKBest(chi2, k=k_cat)
        fs_cat.fit(Xcat, y)
        selected += [cat_cols[i] for i in fs_cat.get_support(indices=True)]

    return list(set(selected))


def _embedded_stage(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """
    Embedded / Wrapper Stage (BASELINE §4):
      1. ElasticNet (high L1 ratio) zeros out redundant features.
      2. Permutation importance on a fast Random Forest.
      3. RFE with LinearSVC.
    Returns the union of selected features.
    """
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    # -- ElasticNet --
    en = ElasticNet(l1_ratio=0.9, alpha=0.01, max_iter=5000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        en.fit(Xs, y)
    en_selected = X.columns[en.coef_ != 0].tolist()

    # -- Permutation importance on a fast Random Forest --
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    rf_selected = X.columns[perm.importances_mean > 0].tolist()

    # -- RFE with LinearSVC --
    n_keep = max(1, int(len(X.columns) * 0.7))
    lsvc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    rfe = RFE(lsvc, n_features_to_select=n_keep, step=1)
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
    """
    BASELINE §4 Stability Selection.

    Run the two-stage filter inside each of `n_splits` stratified folds
    of the *training* data and keep features selected in at least
    `min_appearances` folds.
    """
    counts: dict[str, int] = {c: 0 for c in X.columns}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, _ in skf.split(X, y):
        Xf, yf = X.iloc[train_idx], y.iloc[train_idx]

        filter_cols = _filter_stage(Xf, yf)
        Xf2 = Xf[filter_cols]
        embed_cols = _embedded_stage(Xf2, yf)

        for col in embed_cols:
            counts[col] += 1

    stable = [col for col, cnt in counts.items() if cnt >= min_appearances]
    if len(stable) == 0:
        stable = [col for col, cnt in counts.items() if cnt > 0]
    return stable


# ---------------------------------------------------------------------------
# Nested Cross Validation
# ---------------------------------------------------------------------------

def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_iter: int = 30,
    random_state: int = 42,
    use_feature_selection: bool = True,
) -> dict[str, Any]:
    """
    BASELINE §5 – Full nested cross-validation loop.

    Outer loop : 5-fold stratified → unbiased performance estimates.
    Inner loop : 3-fold stratified → hyperparameter tuning (PR AUC primary,
                 ROC AUC as tiebreaker).
    Threshold  : selected on inner validation fold (maximises F1), applied
                 unchanged to outer test fold.
    """
    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=random_state + 1
    )

    fold_metrics: list[dict] = []
    fold_thresholds: list[float] = []
    selected_features_per_fold: list[list[str]] = []
    best_params_per_fold: list[dict] = []

    print(f"\n>>> Nested CV for model: {model_name.upper()}")
    print(f"    outer={outer_splits}-fold, inner={inner_splits}-fold, n_iter={n_iter}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"  Outer fold {fold_idx}/{outer_splits} ...", end=" ", flush=True)

        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

        # ---- Feature selection (entirely inside the outer training set) ----
        if use_feature_selection:
            stable_feats = stability_selection(X_train, y_train)
        else:
            stable_feats = X_train.columns.tolist()

        selected_features_per_fold.append(stable_feats)
        X_tr_sel = X_train[stable_feats]
        X_te_sel = X_test[stable_feats]

        # ---- Build fit_params for this model (handles MLP sample_weight) ----
        fit_params = _make_fit_params(model_name, y_train)

        # ---- Hyperparameter search on inner CV ----
        estimator, param_grid = get_model(model_name)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="average_precision",   # PR AUC (primary)
            refit=True,
            cv=inner_cv,
            random_state=random_state + fold_idx,
            n_jobs=-1,
            error_score=0.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # NOTE: fit_params routes sample_weight through the pipeline to the
            # clf step for MLP. For other models fit_params is empty ({}).
            # RandomizedSearchCV passes fit_params to each inner fit() call.
            search.fit(X_tr_sel, y_train, **fit_params)

        best_params_per_fold.append(search.best_params_)
        best_model = search.best_estimator_

        # ---- Threshold selection on inner validation predictions ----
        # Re-run inner CV to collect OOF probabilities for threshold tuning.
        # Use clone() so we get a fresh unfitted copy with the best params.
        inner_oof_probs = np.zeros(len(y_train))

        for in_tr_idx, in_val_idx in inner_cv.split(X_tr_sel, y_train):
            Xi_tr  = X_tr_sel.iloc[in_tr_idx]
            Xi_val = X_tr_sel.iloc[in_val_idx]
            yi_tr  = y_train.iloc[in_tr_idx]

            # Clone the best estimator (handles Pipeline correctly)
            est_clone = clone(best_model)

            # Build sample_weight for this inner training slice
            inner_fit_params: dict = {}
            if model_name in _NEEDS_SAMPLE_WEIGHT:
                sw_inner = compute_sample_weight("balanced", yi_tr)
                inner_fit_params["clf__sample_weight"] = sw_inner

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est_clone.fit(Xi_tr, yi_tr, **inner_fit_params)

            if hasattr(est_clone, "predict_proba"):
                inner_oof_probs[in_val_idx] = est_clone.predict_proba(Xi_val)[:, 1]
            elif hasattr(est_clone, "decision_function"):
                raw = est_clone.decision_function(Xi_val)
                inner_oof_probs[in_val_idx] = 1 / (1 + np.exp(-raw))
            else:
                inner_oof_probs[in_val_idx] = 0.5

        threshold = find_best_threshold(y_train, inner_oof_probs)
        fold_thresholds.append(threshold)

        # ---- Evaluate on outer test fold (threshold fixed) ----
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_te_sel)[:, 1]
        elif hasattr(best_model, "decision_function"):
            raw = best_model.decision_function(X_te_sel)
            y_prob = 1 / (1 + np.exp(-raw))
        else:
            y_prob = best_model.predict(X_te_sel).astype(float)

        metrics = compute_metrics(y_test, y_prob, threshold=threshold)
        fold_metrics.append(metrics)

        print(
            f"PR AUC={metrics['pr_auc']:.4f}  "
            f"ROC AUC={metrics['roc_auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"threshold={threshold:.3f}"
        )

    return {
        "fold_metrics":               fold_metrics,
        "fold_thresholds":            fold_thresholds,
        "selected_features_per_fold": selected_features_per_fold,
        "best_params_per_fold":       best_params_per_fold,
    }


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    best_params: dict,
    stable_features: list[str],
) -> Any:
    """
    Retrain the best model on the *full* dataset using the best hyperparameters
    found in nested CV (using the median-fold params as a heuristic).
    """
    estimator, _ = get_model(model_name)
    try:
        estimator.set_params(**best_params)
    except Exception:
        pass

    X_sel = X[stable_features]
    fit_params = _make_fit_params(model_name, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_sel, y, **fit_params)

    return estimator


def save_model(model: Any, path: str) -> None:
    """Persist a trained model to disk with joblib."""
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str) -> Any:
    """Load a persisted model from disk."""
    return joblib.load(path)