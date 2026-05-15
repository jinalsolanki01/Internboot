"""
visualization.py
----------------
All Matplotlib / Seaborn chart functions for the dashboard.
Each function returns a Matplotlib Figure object so Streamlit
can render it with st.pyplot().

Design language
---------------
- Background : #F9FAFB (off-white)
- Cards/plot  : #FFFFFF
- Accent      : #2563EB (blue)
- Secondary   : #10B981 (green)
- Muted text  : #6B7280
- Grid lines  : #E5E7EB
"""

import matplotlib
matplotlib.use("Agg")    # non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

# ── Global style
ACCENT   = "#2563EB"
GREEN    = "#10B981"
ORANGE   = "#F59E0B"
RED      = "#EF4444"
MUTED    = "#6B7280"
BG       = "#F9FAFB"
CARD     = "#FFFFFF"
GRID     = "#E5E7EB"
TEXT     = "#111827"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    GRID,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def _save_tight(fig: plt.Figure) -> plt.Figure:
    """Apply tight layout and return figure."""
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# 1. ACTUAL vs PREDICTED LINE CHART
# ──────────────────────────────────────────────

def plot_actual_vs_predicted(
    result_df: pd.DataFrame,
    store_nbr: int,
    family: str,
    last_n_days: int = 90,
) -> plt.Figure:
    """
    Line chart: Actual sales vs Predicted sales over time.

    Parameters
    ----------
    result_df   : DataFrame with columns [date, actual, predicted]
    store_nbr   : Store number (for title)
    family      : Product family (for title)
    last_n_days : How many recent days to display
    """
    df = result_df.tail(last_n_days).copy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["date"], df["actual"],    color=MUTED,   linewidth=1.5,
            label="Actual", alpha=0.85)
    ax.plot(df["date"], df["predicted"], color=ACCENT,  linewidth=2,
            label="Predicted", linestyle="--")

    ax.fill_between(df["date"], df["actual"], df["predicted"],
                    alpha=0.08, color=ACCENT)

    ax.set_title(
        f"Actual vs Predicted Sales  ·  Store {store_nbr}  ·  {family}",
        fontsize=12, fontweight="bold", color=TEXT, pad=12
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Sales", fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 2. SALES TREND (Aggregated)
# ──────────────────────────────────────────────

def plot_sales_trend(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
    freq: str = "W",
) -> plt.Figure:
    """
    Area chart: Aggregated sales trend over time.

    Parameters
    ----------
    df        : Filtered (store, family) DataFrame with [date, sales]
    store_nbr : Store number
    family    : Product family
    freq      : Resample frequency ('W'=weekly, 'M'=monthly)
    """
    trend = (
        df.set_index("date")["sales"]
        .resample(freq)
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(trend["date"], trend["sales"],
                    color=ACCENT, alpha=0.15)
    ax.plot(trend["date"], trend["sales"],
            color=ACCENT, linewidth=2.2)

    # Rolling average overlay
    rolling = trend["sales"].rolling(4, min_periods=1).mean()
    ax.plot(trend["date"], rolling,
            color=GREEN, linewidth=1.5, linestyle="--", label="4-period MA")

    freq_label = "Weekly" if freq == "W" else "Monthly"
    ax.set_title(
        f"{freq_label} Sales Trend  ·  Store {store_nbr}  ·  {family}",
        fontsize=12, fontweight="bold", color=TEXT, pad=12
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Total Sales", fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 3. MONTHLY SALES HEATMAP
# ──────────────────────────────────────────────

def plot_monthly_heatmap(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
) -> plt.Figure:
    """
    Heatmap: Average daily sales by month × year.
    """
    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.pivot_table(
        values="sales",
        index="year",
        columns="month",
        aggfunc="mean",
    )
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_labels[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, max(3, len(pivot) * 0.9)))
    sns.heatmap(
        pivot, ax=ax,
        cmap="Blues", annot=True, fmt=".0f",
        linewidths=0.5, linecolor=GRID,
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": 8},
    )
    ax.set_title(
        f"Avg Daily Sales by Month  ·  Store {store_nbr}  ·  {family}",
        fontsize=12, fontweight="bold", color=TEXT, pad=12
    )
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Year",  fontsize=10)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 4. FEATURE IMPORTANCE BAR CHART
# ──────────────────────────────────────────────

def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    fi_df : DataFrame with columns [feature, importance]
    top_n : How many features to show
    """
    df = fi_df.head(top_n).copy()
    colors = [ACCENT if i == 0 else "#93C5FD" for i in range(len(df))]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.42)))
    bars = ax.barh(df["feature"][::-1], df["importance"][::-1],
                   color=colors[::-1], edgecolor="white", height=0.65)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", ha="left",
                fontsize=8, color=MUTED)

    ax.set_title("Feature Importance (Top 15)",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 5. RESIDUALS DISTRIBUTION
# ──────────────────────────────────────────────

def plot_residuals(result_df: pd.DataFrame) -> plt.Figure:
    """
    Histogram + KDE of prediction residuals.
    """
    residuals = result_df["actual"] - result_df["predicted"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(residuals, bins=50, color=ACCENT, alpha=0.4,
            edgecolor="white", density=True, label="Residual dist.")
    residuals.plot.kde(ax=ax, color=ACCENT, linewidth=2)
    ax.axvline(0, color=RED, linewidth=1.5, linestyle="--", label="Zero error")

    ax.set_title("Residuals Distribution (Actual − Predicted)",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Residual", fontsize=10)
    ax.set_ylabel("Density",  fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 6. DAY-OF-WEEK SALES PATTERN
# ──────────────────────────────────────────────

def plot_dow_pattern(df: pd.DataFrame, store_nbr: int, family: str) -> plt.Figure:
    """
    Bar chart: Average sales by day of week.
    """
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    avg = df.groupby("dow")["sales"].mean().reindex(range(7))

    colors = [GREEN if d >= 5 else ACCENT for d in range(7)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(dow_labels, avg.values, color=colors, edgecolor="white",
           width=0.65)
    ax.set_title(
        f"Avg Sales by Day of Week  ·  Store {store_nbr}  ·  {family}",
        fontsize=12, fontweight="bold", color=TEXT, pad=12
    )
    ax.set_xlabel("Day of Week", fontsize=10)
    ax.set_ylabel("Avg Sales",   fontsize=10)
    ax.grid(axis="x", visible=False)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 7. SCATTER: ACTUAL vs PREDICTED
# ──────────────────────────────────────────────

def plot_scatter_actual_predicted(result_df: pd.DataFrame) -> plt.Figure:
    """
    Scatter plot: Actual (x) vs Predicted (y) with a perfect-fit line.
    """
    sample = result_df.sample(min(2000, len(result_df)), random_state=42)
    lo = min(sample["actual"].min(), sample["predicted"].min())
    hi = max(sample["actual"].max(), sample["predicted"].max())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample["actual"], sample["predicted"],
               alpha=0.25, s=10, color=ACCENT, label="Predictions")
    ax.plot([lo, hi], [lo, hi], color=RED, linewidth=1.5,
            linestyle="--", label="Perfect fit")

    ax.set_title("Actual vs Predicted (Scatter)",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Actual Sales",    fontsize=10)
    ax.set_ylabel("Predicted Sales", fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    return _save_tight(fig)


# ──────────────────────────────────────────────
# 8. TOP STORES BY SALES
# ──────────────────────────────────────────────

def plot_top_stores(master_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Horizontal bar: Top-N stores by total sales.
    """
    top = (
        master_df.groupby("store_nbr")["sales"]
        .sum()
        .nlargest(top_n)
        .sort_values()
    )
    colors = [ACCENT if i == len(top) - 1 else "#93C5FD"
              for i in range(len(top))]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.5)))
    ax.barh([f"Store {s}" for s in top.index], top.values,
            color=colors, edgecolor="white", height=0.65)
    ax.set_title(f"Top {top_n} Stores by Total Sales",
                 fontsize=12, fontweight="bold", color=TEXT, pad=12)
    ax.set_xlabel("Total Sales", fontsize=10)
    ax.grid(axis="y", visible=False)
    return _save_tight(fig)
