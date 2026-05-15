"""
preprocessing.py
----------------
Handles all data loading, cleaning, and merging for the
Store Sales Forecasting dashboard.

Supports:
  - train.csv       (required)
  - stores.csv      (required)
  - holidays_events.csv (required)
  - oil.csv         (optional)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ──────────────────────────────────────────────
# LOADERS
# ──────────────────────────────────────────────

def _path(filename: str) -> str:
    """Return full path to a data file."""
    return os.path.join(DATA_DIR, filename)


@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    """
    Load train.csv.
    Columns: date, store_nbr, family, sales, onpromotion
    """
    path = _path("train.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"train.csv not found at: {path}\n"
            "Please download from Kaggle and place in the /data folder."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["sales"] = df["sales"].clip(lower=0)       # no negative sales
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_stores() -> pd.DataFrame:
    """
    Load stores.csv.
    Columns: store_nbr, city, state, type, cluster
    """
    path = _path("stores.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"stores.csv not found at: {path}\n"
            "Please download from Kaggle and place in the /data folder."
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_holidays() -> pd.DataFrame:
    """
    Load holidays_events.csv.
    Columns: date, type, locale, locale_name, description, transferred
    """
    path = _path("holidays_events.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"holidays_events.csv not found at: {path}\n"
            "Please download from Kaggle and place in the /data folder."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    # Flag: national holidays only (broadest impact)
    df["is_holiday"] = (
        (df["type"].isin(["Holiday", "Transfer", "Bridge"])) &
        (df["locale"] == "National")
    ).astype(int)
    return df[["date", "is_holiday"]].drop_duplicates("date")


@st.cache_data(show_spinner=False)
def load_oil() -> pd.DataFrame | None:
    """
    Load oil.csv (optional).
    Columns: date, dcoilwtico
    Returns None if file is missing.
    """
    path = _path("oil.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"dcoilwtico": "oil_price"})
    # Forward-fill weekend/holiday gaps
    df = df.set_index("date").resample("D").asfreq()
    df["oil_price"] = df["oil_price"].ffill().bfill()
    return df.reset_index()


# ──────────────────────────────────────────────
# MERGE & CLEAN
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_master_df() -> pd.DataFrame:
    """
    Merge all required (and optional) datasets into one
    clean master DataFrame ready for feature engineering.

    Returns
    -------
    pd.DataFrame
        Merged, cleaned master dataset.
    """
    train = load_train()
    stores = load_stores()
    holidays = load_holidays()
    oil = load_oil()

    # ── Merge stores metadata
    df = train.merge(stores, on="store_nbr", how="left")

    # ── Merge holiday flag
    df = df.merge(holidays, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # ── Merge oil price (optional)
    if oil is not None:
        df = df.merge(oil, on="date", how="left")
        df["oil_price"] = df["oil_price"].ffill().bfill()
    else:
        df["oil_price"] = np.nan

    # ── Encode store type (A–E → 0–4)
    df["store_type_enc"] = df["type"].astype("category").cat.codes

    # ── Encode product family
    df["family_enc"] = df["family"].astype("category").cat.codes

    # ── Sort chronologically
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def get_store_list(df: pd.DataFrame) -> list[int]:
    """Return sorted list of unique store numbers."""
    return sorted(df["store_nbr"].unique().tolist())


def get_family_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique product families."""
    return sorted(df["family"].unique().tolist())


def filter_store_family(
    df: pd.DataFrame,
    store_nbr: int,
    family: str
) -> pd.DataFrame:
    """
    Filter master dataframe for a specific store + product family.
    Returns a time-indexed DataFrame sorted by date.
    """
    mask = (df["store_nbr"] == store_nbr) & (df["family"] == family)
    return df[mask].sort_values("date").reset_index(drop=True)
