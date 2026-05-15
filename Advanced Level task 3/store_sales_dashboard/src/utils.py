"""
utils.py
--------
CSS injection + utility helpers for the Store Sales Dashboard.

The CUSTOM_CSS block aggressively overrides every Streamlit element
to enforce the light design system. This works alongside
.streamlit/config.toml which sets base="light".
"""

import streamlit as st
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────
# FORMATTERS
# ──────────────────────────────────────────────

def fmt_number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}"


def fmt_currency(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def fmt_metric(label: str, value: float, delta=None) -> None:
    if delta is not None:
        st.metric(label=label, value=fmt_number(value), delta=fmt_number(delta))
    else:
        st.metric(label=label, value=fmt_number(value))


# ──────────────────────────────────────────────
# STORE / FAMILY HELPERS
# ──────────────────────────────────────────────

def get_store_info(stores_df: pd.DataFrame, store_nbr: int) -> dict:
    row = stores_df[stores_df["store_nbr"] == store_nbr]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_family_encoder(master_df: pd.DataFrame) -> dict:
    fam = master_df[["family", "family_enc"]].drop_duplicates()
    return dict(zip(fam["family"], fam["family_enc"]))


def get_store_type_encoder(master_df: pd.DataFrame) -> dict:
    st_df = master_df[["type", "store_type_enc"]].drop_duplicates()
    return dict(zip(st_df["type"], st_df["store_type_enc"]))


def get_store_cluster(master_df: pd.DataFrame, store_nbr: int) -> int:
    row = master_df[master_df["store_nbr"] == store_nbr]
    if row.empty:
        return 1
    return int(row["cluster"].iloc[0])


def get_store_type_enc(master_df: pd.DataFrame, store_nbr: int) -> int:
    row = master_df[master_df["store_nbr"] == store_nbr]
    if row.empty:
        return 0
    return int(row["store_type_enc"].iloc[0])


# ──────────────────────────────────────────────
# LAG / ROLLING LOOKUPS
# ──────────────────────────────────────────────

def get_recent_stats(
    df: pd.DataFrame,
    store_nbr: int,
    family: str,
    ref_date,
) -> dict:
    mask = (
        (df["store_nbr"] == store_nbr) &
        (df["family"]    == family) &
        (df["date"]      < ref_date)
    )
    hist = df[mask].sort_values("date")

    def _lag(n):
        if len(hist) < n:
            return float(hist["sales"].mean()) if len(hist) else 0.0
        return float(hist["sales"].iloc[-n])

    def _roll_mean(n):
        return float(hist["sales"].tail(n).mean()) if len(hist) >= 1 else 0.0

    def _roll_std(n):
        tail = hist["sales"].tail(n)
        return float(tail.std()) if len(tail) >= 2 else 0.0

    def _promo_roll(n):
        return float(hist["onpromotion"].tail(n).sum()) if len(hist) >= 1 else 0.0

    return {
        "sales_lag_7":        _lag(7),
        "sales_lag_14":       _lag(14),
        "sales_lag_28":       _lag(28),
        "sales_roll_mean_7":  _roll_mean(7),
        "sales_roll_mean_14": _roll_mean(14),
        "sales_roll_std_7":   _roll_std(7),
        "sales_roll_std_14":  _roll_std(14),
        "promo_roll_7":       _promo_roll(7),
        "promo_roll_14":      _promo_roll(14),
    }


# ──────────────────────────────────────────────
# LEGACY SHIMS
# ──────────────────────────────────────────────

def card(content: str) -> None:
    st.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #E5E7EB;'
        f'border-radius:10px;padding:0.9rem 1rem;font-size:12.5px;'
        f'color:#374151;line-height:1.65;margin-bottom:0.75rem;'
        f'box-shadow:0 1px 3px rgba(0,0,0,0.04);">{content}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:#111827;'
        f'letter-spacing:-0.01em;margin:1.5rem 0 0.6rem 0;padding-bottom:7px;'
        f'border-bottom:2px solid #E5E7EB;">{title}</div>',
        unsafe_allow_html=True,
    )


def prediction_card(value: float) -> None:
    v = fmt_currency(value)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
                border:1.5px solid #93C5FD;border-radius:14px;
                padding:1.5rem;text-align:center;margin:1rem 0;">
        <div style="font-size:3rem;font-weight:800;color:#1D4ED8;
                    letter-spacing:-0.03em;">{v}</div>
        <div style="font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:0.07em;color:#3B82F6;margin-top:4px;">
            Predicted Sales Units
        </div>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, color: str = "blue") -> str:
    styles = {
        "green":  ("D1FAE5", "065F46"),
        "blue":   ("DBEAFE", "1E40AF"),
        "red":    ("FEE2E2", "991B1B"),
        "amber":  ("FEF3C7", "92400E"),
        "gray":   ("F3F4F6", "374151"),
    }
    bg, fg = styles.get(color, ("DBEAFE", "1E40AF"))
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:99px;font-size:11px;font-weight:700;'
        f'background:#{bg};color:#{fg};">{text}</span>'
    )


# ══════════════════════════════════════════════════════════════
# CSS — Complete override stylesheet
# ══════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* === 1. FORCE LIGHT MODE ON EVERY STREAMLIT WRAPPER === */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"],
.main {
    background-color: #F5F7FB !important;
    color: #111827 !important;
}

.block-container,
[data-testid="block-container"] {
    padding-top: 1.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1300px !important;
    background-color: #F5F7FB !important;
}

[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="column"] {
    background-color: transparent !important;
}

/* === 2. HIDE CHROME === */
#MainMenu { visibility: hidden !important; }
footer    { visibility: hidden !important; }
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background-color: transparent !important;
    display: none !important;
}

/* === 3. SIDEBAR === */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF !important;
    padding-top: 1.25rem !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #111827 !important;
}

/* === 4. GLOBAL TYPOGRAPHY === */
p, span, label, h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #111827 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* === 5. TABS === */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 2px solid #E5E7EB !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.25rem !important;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 2.5px solid transparent !important;
    color: #6B7280 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 0.7rem 1.1rem !important;
    margin-bottom: -2px !important;
    letter-spacing: -0.01em !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #2563EB !important;
    background-color: #F3F4F6 !important;
    border-radius: 6px 6px 0 0 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2.5px solid #2563EB !important;
    background-color: transparent !important;
}
[data-baseweb="tab-panel"] {
    background-color: transparent !important;
    padding: 0 !important;
}

/* === 6. METRIC CONTAINERS === */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 12px !important;
    padding: 1rem 1.15rem !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="metric-container"] label {
    font-size: 10.5px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #6B7280 !important;
}
[data-testid="stMetricValue"] div {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #111827 !important;
    letter-spacing: -0.025em !important;
}

/* === 7. BUTTONS === */
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.3) !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #374151 !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #2563EB !important;
    color: #2563EB !important;
    background: #EFF6FF !important;
}

/* === 8. SELECTBOX === */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    transition: border-color 0.15s ease !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #2563EB !important;
}
[data-baseweb="select"] span { color: #111827 !important; font-size: 13px !important; }
[data-baseweb="popover"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.12) !important;
}
[data-baseweb="menu"] li { color: #111827 !important; font-size: 13px !important; }
[data-baseweb="menu"] li:hover { background-color: #EFF6FF !important; }

/* === 9. DATE INPUT === */
[data-testid="stDateInput"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 13px !important;
}
[data-testid="stDateInput"] input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* === 10. EXPANDERS === */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 12.5px !important;
    font-weight: 600 !important;
    color: #374151 !important;
    background: #F8FAFC !important;
}
[data-testid="stExpander"] summary:hover { background: #F3F4F6 !important; }

/* === 11. DATAFRAME === */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* === 12. ALERTS === */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left-width: 4px !important;
    font-size: 13px !important;
}

/* === 13. SLIDER === */
[data-testid="stSlider"] label {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #374151 !important;
}

/* === 14. TOGGLES & RADIOS === */
[data-testid="stToggle"] label p,
[data-testid="stRadio"] label p {
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: #111827 !important;
}

/* === 15. DIVIDERS === */
hr {
    border: none !important;
    border-top: 1px solid #E5E7EB !important;
    margin: 1.25rem 0 !important;
}

/* === 16. SCROLLBARS === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F3F4F6; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #9CA3AF; }
</style>
"""


def inject_css() -> None:
    """Inject the complete design-system CSS into Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)