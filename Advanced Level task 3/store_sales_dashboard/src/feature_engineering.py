"""
feature_engineering.py
-----------------------
Creates time-series features, lag features, and rolling
statistics from the master DataFrame.

All functions are pure (no side-effects) and work on
copies of the input DataFrame.
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# TIME FEATURES
# ──────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from the 'date' column.

    Features added
    --------------
    year, month, day, day_of_week, week_of_year,
    quarter, is_weekend, day_of_year
    """
    df = df.copy()
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["day"]          = df["date"].dt.day
    df["day_of_week"]  = df["date"].dt.dayofweek       # 0=Mon, 6=Sun
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year"]  = df["date"].dt.dayofyear
    return df


# ──────────────────────────────────────────────
# LAG FEATURES
# ──────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None
) -> pd.DataFrame:
    """
    Add lag features on the 'sales' column.
    Lags are computed per (store_nbr, family) group.

    Parameters
    ----------
    df   : Master DataFrame (must be sorted by date within group)
    lags : List of lag days (default: [7, 14, 28])
    """
    if lags is None:
        lags = [7, 14, 28]

    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    for lag in lags:
        col = f"sales_lag_{lag}"
        df[col] = (
            df.groupby(["store_nbr", "family"])["sales"]
            .shift(lag)
        )
    return df


# ──────────────────────────────────────────────
# ROLLING STATISTICS
# ──────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None
) -> pd.DataFrame:
    """
    Add rolling mean and rolling std features on 'sales'.

    Parameters
    ----------
    df      : Master DataFrame (must be sorted by date within group)
    windows : List of rolling windows in days (default: [7, 14])
    """
    if windows is None:
        windows = [7, 14]

    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    grouped = df.groupby(["store_nbr", "family"])["sales"]

    for window in windows:
        df[f"sales_roll_mean_{window}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window).mean())
        )
        df[f"sales_roll_std_{window}"] = (
            grouped.transform(lambda x: x.shift(1).rolling(window).std())
        )
    return df


# ──────────────────────────────────────────────
# PROMOTION FEATURES
# ──────────────────────────────────────────────

def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling promotion count features.
    Counts number of promoted days in last 7 / 14 days.
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    grouped = df.groupby(["store_nbr", "family"])["onpromotion"]

    for window in [7, 14]:
        df[f"promo_roll_{window}"] = (
            grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            )
        )
    return df


# ──────────────────────────────────────────────
# FULL FEATURE PIPELINE
# ──────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Steps
    -----
    1. Time features
    2. Lag features
    3. Rolling statistics
    4. Promotion features
    5. Drop rows with NaN lags (first N days)

    Returns
    -------
    pd.DataFrame
        Feature-rich DataFrame, NaN rows dropped, ready for modeling.
    """
    df = add_time_features(df)
    df = add_lag_features(df, lags=[7, 14, 28])
    df = add_rolling_features(df, windows=[7, 14])
    df = add_promo_features(df)

    # Drop NaN only on columns used by the model + the target.
    # This avoids removing rows just because optional columns
    # (e.g. oil_price) are missing.
    drop_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    df = df.dropna(subset=drop_cols).reset_index(drop=True)
    return df


# ──────────────────────────────────────────────
# FEATURE COLUMNS (used by model)
# ──────────────────────────────────────────────

FEATURE_COLS = [
    # Time
    "year", "month", "day", "day_of_week", "week_of_year",
    "quarter", "is_weekend", "day_of_year",
    # Store & Product
    "store_nbr", "family_enc", "store_type_enc", "cluster",
    # Promotion
    "onpromotion", "promo_roll_7", "promo_roll_14",
    # Holiday
    "is_holiday",
    # Lag features
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    # Rolling stats
    "sales_roll_mean_7", "sales_roll_mean_14",
    "sales_roll_std_7", "sales_roll_std_14",
]

TARGET_COL = "sales"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return only feature columns that actually exist in df.
    Safely ignores missing optional columns (e.g. oil_price).
    """
    return [c for c in FEATURE_COLS if c in df.columns]


def build_X_y(df: pd.DataFrame):
    """
    Split engineered DataFrame into X (features) and y (target).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    feat_cols = get_feature_cols(df)
    X = df[feat_cols]
    y = df[TARGET_COL]
    return X, y


# ──────────────────────────────────────────────
# SINGLE-ROW PREDICTION BUILDER
# ──────────────────────────────────────────────

def build_prediction_row(
    store_nbr: int,
    family_enc: int,
    store_type_enc: int,
    cluster: int,
    date: pd.Timestamp,
    onpromotion: int,
    is_holiday: int,
    sales_lag_7: float,
    sales_lag_14: float,
    sales_lag_28: float,
    sales_roll_mean_7: float,
    sales_roll_mean_14: float,
    sales_roll_std_7: float,
    sales_roll_std_14: float,
    promo_roll_7: float,
    promo_roll_14: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for prediction from user inputs.
    All feature names match FEATURE_COLS exactly.
    """
    row = {
        "year":               date.year,
        "month":              date.month,
        "day":                date.day,
        "day_of_week":        date.dayofweek,
        "week_of_year":       date.isocalendar()[1],
        "quarter":            date.quarter,
        "is_weekend":         int(date.dayofweek >= 5),
        "day_of_year":        date.timetuple().tm_yday,
        "store_nbr":          store_nbr,
        "family_enc":         family_enc,
        "store_type_enc":     store_type_enc,
        "cluster":            cluster,
        "onpromotion":        onpromotion,
        "promo_roll_7":       promo_roll_7,
        "promo_roll_14":      promo_roll_14,
        "is_holiday":         is_holiday,
        "sales_lag_7":        sales_lag_7,
        "sales_lag_14":       sales_lag_14,
        "sales_lag_28":       sales_lag_28,
        "sales_roll_mean_7":  sales_roll_mean_7,
        "sales_roll_mean_14": sales_roll_mean_14,
        "sales_roll_std_7":   sales_roll_std_7,
        "sales_roll_std_14":  sales_roll_std_14,
    }
    return pd.DataFrame([row])
