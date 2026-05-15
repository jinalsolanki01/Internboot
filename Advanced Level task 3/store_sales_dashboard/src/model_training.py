"""
model_training.py
-----------------
Trains a GradientBoostingRegressor on the engineered features,
evaluates it with time-series-aware splits, and provides
save / load helpers using joblib.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.feature_engineering import build_X_y, get_feature_cols


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sales_model.pkl")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.pkl")


# ──────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────

def get_model(model_type: str = "gradient_boosting"):
    """
    Return a fresh (unfitted) sklearn estimator.

    Supported model_type values
    ---------------------------
    'gradient_boosting'  : GradientBoostingRegressor (default, best accuracy)
    'random_forest'      : RandomForestRegressor (faster training)
    'ridge'              : Ridge regression (fastest, baseline)
    """
    models = {
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
            verbose=0,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        ),
        "ridge": Ridge(alpha=1.0),
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            f"Choose from: {list(models.keys())}"
        )
    return models[model_type]


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    model_type: str = "gradient_boosting",
    sample_size: int = 200_000,
) -> tuple:
    """
    Train the regression model on the feature-engineered dataset.

    Parameters
    ----------
    df          : Feature-engineered master DataFrame
    model_type  : Which estimator to use
    sample_size : Cap rows for faster training (0 = use all)

    Returns
    -------
    model   : Fitted sklearn estimator
    metrics : dict with MAE, RMSE, R2 on hold-out fold
    feature_cols : list of column names used
    """
    # ── Sample for speed (optional)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        df = df.sort_values("date")       # keep time order after sample

    X, y = build_X_y(df)
    feature_cols = get_feature_cols(df)

    # ── Time-series-aware train/test split (last 20% as test)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # ── Fit
    model = get_model(model_type)
    model.fit(X_train, y_train)

    # ── Evaluate
    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = _compute_metrics(y_test, y_pred)

    return model, metrics, feature_cols


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R² metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def cross_validate_model(
    df: pd.DataFrame,
    model_type: str = "gradient_boosting",
    n_splits: int = 3,
) -> list[dict]:
    """
    Run TimeSeriesSplit cross-validation.
    Returns list of metric dicts per fold.
    """
    X, y = build_X_y(df)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = get_model(model_type)
        model.fit(X_tr, y_tr)
        y_pred = np.clip(model.predict(X_te), 0, None)
        m = _compute_metrics(y_te, y_pred)
        m["fold"] = fold
        fold_metrics.append(m)

    return fold_metrics


# ──────────────────────────────────────────────
# FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def get_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances.
    Works for tree-based models only (GB, RF).
    Returns empty DataFrame for Ridge.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    imp = model.feature_importances_
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": imp}
    ).sort_values("importance", ascending=False)
    return fi


# ──────────────────────────────────────────────
# SAVE / LOAD
# ──────────────────────────────────────────────

def save_model(model, metrics: dict, feature_cols: list[str]) -> None:
    """
    Persist trained model and metadata to disk using joblib.

    Saved files
    -----------
    models/sales_model.pkl  : Fitted estimator
    models/model_meta.pkl   : Metrics + feature column list
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump({"metrics": metrics, "feature_cols": feature_cols}, META_PATH)


def load_model():
    """
    Load saved model + metadata from disk.

    Returns
    -------
    model        : Fitted sklearn estimator
    metrics      : dict (MAE, RMSE, R2)
    feature_cols : list of feature column names

    Raises
    ------
    FileNotFoundError if model files don't exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. "
            "Please train the model first using the 'Train Model' button."
        )
    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)
    return model, meta["metrics"], meta["feature_cols"]


def model_exists() -> bool:
    """Check whether a saved model exists on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


# ──────────────────────────────────────────────
# BATCH PREDICTION
# ──────────────────────────────────────────────

def predict_store_family(
    model,
    df_feat: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run batch prediction for a store+family slice.

    Parameters
    ----------
    model        : Trained sklearn estimator
    df_feat      : Feature-engineered slice (store+family)
    feature_cols : Columns to use for prediction

    Returns
    -------
    pd.DataFrame with columns: date, actual, predicted
    """
    cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[cols]
    y_pred = np.clip(model.predict(X), 0, None)
    result = df_feat[["date", "sales"]].copy()
    result.columns = ["date", "actual"]
    result["predicted"] = y_pred
    return result.reset_index(drop=True)
