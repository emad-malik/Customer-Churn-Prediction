"""Data loading, cleaning, and feature engineering for the Telco churn dataset."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path



# ── Columns ─────────────────────────────────────────────────────────────────

DROP_COLS = ["customerID"]

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling", "Churn",
]

ORDINAL_MAP = {
    "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
}

NOMINAL_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV and return with consistent dtypes."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # TotalCharges is sometimes blank for brand-new customers (tenure=0)
    df["TotalCharges"] = (
        df["TotalCharges"]
        .astype(str)
        .str.strip()
        .replace("", "0")
        .astype(float)
    )
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply binary mapping, ordinal encoding, and one-hot encoding.
    Returns a fully numeric DataFrame ready for modelling.
    """
    df = df.copy()

    # Normalize string values to avoid category mismatches
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop identifiers
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Binary: Yes/No and gender → 1/0
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
            df[col] = df[col].map(mapping).fillna(df[col])

    # Ordinal
    for col, mapping in ORDINAL_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # One-hot encode any remaining object columns (excluding target)
    object_cols = [c for c in df.select_dtypes("object").columns if c != "Churn"]
    if object_cols:
        df = pd.get_dummies(df, columns=object_cols, drop_first=True)

    return df.infer_objects()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering based on the paper's §3.2:
      - Tenure bins (early life vs. protected zone)
      - High charge flag
      - Contract × Charge interaction
      - Technical ticket features (capped + quadratic)
      - Charges per tenure proxy
    """
    df = df.copy()

    # Tenure bins: 0-6, 7-12, 13-24, 25-36, >36
    bins = [-1, 6, 12, 24, 36, float("inf")]
    labels = [0, 1, 2, 3, 4]
    df["tenure_bin"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(int)

    # High charge flag (above 75th percentile of training data)
    q75 = df["MonthlyCharges"].quantile(0.75)
    df["high_charge_flag"] = (df["MonthlyCharges"] > q75).astype(int)

    # Contract × Charge interaction (only if Contract is ordinal-encoded)
    if "Contract" in df.columns:
        df["contract_x_charge"] = df["Contract"] * df["MonthlyCharges"]

    # Technical ticket features — numTechTickets column may exist in
    # expanded datasets; create a synthetic proxy from TechSupport otherwise
    if "numTechTickets" not in df.columns:
        tech_col = [c for c in df.columns if "TechSupport" in c]
        if tech_col:
            df["numTechTickets"] = df[tech_col[0]]  # binary proxy
        else:
            df["numTechTickets"] = 0

    df["tickets_capped"] = df["numTechTickets"].clip(upper=5)
    df["tickets_sq"] = df["tickets_capped"] ** 2

    # Charges per tenure (avoid division by zero for new customers)
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Fiber interaction
    fiber_col = [c for c in df.columns if "InternetService_Fiber" in c]
    if fiber_col:
        df["fiber_x_charge"] = df[fiber_col[0]] * df["MonthlyCharges"]

    return df


def prepare(
    path: str | Path,
    engineer: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: load → encode → (optionally) engineer features.
    Returns (X, y).
    """
    df = load_raw(path)
    df = encode(df)
    if engineer:
        df = engineer_features(df)

    y = df.pop("Churn").astype(int)
    X = df.select_dtypes(include=[np.number, "bool"]).fillna(0)
    return X, y

def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.70,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Simulate concept drift by treating the first `train_frac` of rows as
    'production data' and the remainder as 'new incoming data'.
    The dataset is NOT shuffled — row order acts as a time proxy.
    """
    split = int(len(X) * train_frac)
    return (
        X.iloc[:split].copy(), y.iloc[:split].copy(),
        X.iloc[split:].copy(), y.iloc[split:].copy(),
    )


def build_features(
    df: pd.DataFrame,
    drop_first: bool = True,
    contract_ordered: bool = True,
    engineer: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Legacy API wrapper — encode and engineer features with flexible options.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe
    drop_first : bool, default=True
        Whether to drop the first category in one-hot encoding
    contract_ordered : bool, default=True
        Whether to use ordinal encoding for Contract (vs. one-hot)
    engineer : bool, default=True
        Whether to apply feature engineering
        
    Returns
    -------
    X, y : (pd.DataFrame, pd.Series)
    """
    df = df.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    
    # Binary encoding
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
            df[col] = df[col].map(mapping).fillna(df[col])
    
    # Ordinal encoding for Contract
    if contract_ordered:
        for col, mapping in ORDINAL_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
    
    # One-hot encoding (excluding Contract if ordinal)
    nominal = [c for c in NOMINAL_COLS if c in df.columns]
    if not contract_ordered and "Contract" in df.columns:
        nominal.append("Contract")
    if nominal:
        df = pd.get_dummies(df, columns=nominal, drop_first=drop_first)
    
    # Engineer features if requested
    if engineer:
        df = engineer_features(df)
    
    # Extract target and return numeric features
    y = df.pop("Churn").astype(int) if "Churn" in df.columns else None
    X = df.select_dtypes(include=[np.number]).fillna(0)
    
    return (X, y) if y is not None else (X, None)