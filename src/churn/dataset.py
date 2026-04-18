"""churn prediction dataset and preprocessing.

Implements BASELINE.md Sections 1–3:
  1. Data Acquisition and Initial Cleaning
  2. Preprocessing and Encoding
  3. Feature Engineering
"""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
BINARY_YES_NO = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns that need Yes/No → 1/0 but may also have "No internet service" / "No phone service"
ADDON_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NOMINAL_CATS = ["gender", "InternetService", "PaymentMethod"]

CONTRACT_ORDER = {"Month-to-month": 0, "One year": 1, "Two year": 2}

CONTINUOUS_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Tenure bins: 0-6, 7-12, 13-24, 25-36, 36+
TENURE_BINS = [0, 6, 12, 24, 36, np.inf]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-36", "36+"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: str | pathlib.Path) -> pd.DataFrame:
    """Read the CSV and return a raw DataFrame."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    BASELINE §1 – Initial cleaning.

    * Trim whitespace in TotalCharges, coerce blanks → 0.
    * Drop customerID.
    * Drop any obvious data-leakage identifiers.
    """
    df = df.copy()

    # Fix TotalCharges: blank entries for brand-new customers (tenure == 0)
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Drop identifier
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    BASELINE §2 – Map Yes/No fields to 1/0.

    Add-on service columns may contain "No internet service" or
    "No phone service"; treat these the same as "No" → 0.
    """
    df = df.copy()

    for col in ADDON_COLS:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: 1 if str(x).strip().lower() == "yes" else 0
            )

    simple_binary = [c for c in BINARY_YES_NO if c in df.columns and c not in ADDON_COLS]
    for col in simple_binary:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # gender: Female → 0, Male → 1
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})

    return df


def encode_contract(df: pd.DataFrame, ordered: bool = True) -> pd.DataFrame:
    """
    BASELINE §2 – Contract encoding.

    * ordered=True  → ordinal integers (for trees / boosting)
    * ordered=False → one-hot (for linear / MLP)
    """
    df = df.copy()
    if ordered:
        df["Contract"] = df["Contract"].map(CONTRACT_ORDER).astype(int)
    else:
        ohe = pd.get_dummies(df["Contract"], prefix="Contract", drop_first=True)
        df = pd.concat([df.drop(columns=["Contract"]), ohe], axis=1)
    return df


def one_hot_nominals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    BASELINE §2 – One-hot encode nominal categoricals.

    * drop_first=True  for linear / MLP (avoid perfect multicollinearity)
    * drop_first=False for trees / boosting
    """
    df = df.copy()
    cols = [c for c in NOMINAL_CATS if c in df.columns]
    for col in cols:
        ohe = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
        df = pd.concat([df.drop(columns=[col]), ohe], axis=1)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BASELINE §3 – Feature engineering.

    * Tenure Bins (ordinal 0-4)
    * High Charge Flag (within-sample upper quartile — caller must pass the
      quartile value when computing in-fold; here we use the full-sample
      quartile for the first pass)
    * Contract × MonthlyCharges interaction
    * Fiber flag + Fiber × MonthlyCharges interaction
    * NumAdminTickets proxy (not present in standard dataset → set to 0)
    * NumTechTickets proxy (TechSupport inverse as ordinal) capped at 5
    * TechTickets² quadratic term
    * ChargesPerTenure = TotalCharges / max(tenure, 1)
    """
    df = df.copy()

    # -- Tenure Bins --
    df["TenureBin"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=[0, 1, 2, 3, 4],
        right=True,
        include_lowest=True,
    ).astype(int)

    # -- High Charge Flag --
    q75 = df["MonthlyCharges"].quantile(0.75)
    df["HighChargeFlag"] = (df["MonthlyCharges"] > q75).astype(int)

    # -- Contract × MonthlyCharges --
    # If Contract was already encoded to int (ordered), use it directly;
    # otherwise fall back to a numeric proxy.
    if "Contract" in df.columns and pd.api.types.is_integer_dtype(df["Contract"]):
        df["ContractByCharge"] = df["Contract"] * df["MonthlyCharges"]
    else:
        # one-hot branch: use TenureBin as proxy for term length
        df["ContractByCharge"] = df["TenureBin"] * df["MonthlyCharges"]

    # -- Fiber flag --
    if "InternetService" in df.columns:
        df["FiberFlag"] = (df["InternetService"] == "Fiber optic").astype(int)
    elif "InternetService_Fiber optic" in df.columns:
        df["FiberFlag"] = df["InternetService_Fiber optic"].astype(int)
    else:
        df["FiberFlag"] = 0

    df["FiberByCharge"] = df["FiberFlag"] * df["MonthlyCharges"]

    # -- Tech Tickets proxy --
    # The standard dataset has TechSupport (binary 0/1 after encoding).
    # We create a 0-5 scale: no support → 5 tickets, support → 0.
    if "TechSupport" in df.columns and pd.api.types.is_integer_dtype(df["TechSupport"]):
        df["TechTickets"] = (1 - df["TechSupport"]) * 5
    else:
        df["TechTickets"] = 0

    df["TechTickets"] = df["TechTickets"].clip(upper=5)
    df["TechTickets2"] = df["TechTickets"] ** 2

    # -- Charges per Tenure --
    df["ChargesPerTenure"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    return df


def build_features(
    df: pd.DataFrame,
    drop_first: bool = True,
    contract_ordered: bool = True,
    high_charge_q75: float | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full preprocessing pipeline:
      clean → encode binary → encode contract → one-hot nominals
      → engineer features → split X / y.

    Parameters
    ----------
    drop_first : bool
        Drop reference OHE level (True for linear/MLP, False for trees).
    contract_ordered : bool
        Use ordinal contract encoding (True for trees, False for linear/MLP).
    high_charge_q75 : float, optional
        Precomputed upper quartile for HighChargeFlag (use fold's training
        quartile to avoid leakage). If None, computed from `df`.
    """
    df = clean(df)
    df = encode_binary(df)
    df = encode_contract(df, ordered=contract_ordered)
    df = one_hot_nominals(df, drop_first=drop_first)
    df = add_engineered_features(df)

    # Override HighChargeFlag with fold-specific quartile if provided
    if high_charge_q75 is not None:
        df["HighChargeFlag"] = (df["MonthlyCharges"] > high_charge_q75).astype(int)
        df["FiberByCharge"] = df["FiberFlag"] * df["MonthlyCharges"]

    y = df.pop("Churn").astype(int)
    X = df.copy()
    return X, y
