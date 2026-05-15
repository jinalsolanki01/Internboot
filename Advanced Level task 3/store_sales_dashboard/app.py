"""
app.py
------
Store Sales Forecasting Dashboard  —  Day 3  (Redesigned UI)
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time

# ── Page config MUST be the first Streamlit call ──────────────
st.set_page_config(
    page_title="Store Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Internal modules ──────────────────────────────────────────
from src.utils import (
    inject_css, badge, fmt_currency, fmt_number,
    get_family_encoder, get_store_type_enc, get_store_cluster,
    get_recent_stats, get_store_info,
)
from src.ui_components import (
    page_header, section_header, kpi_card, metric_row,
    chart_card_start, chart_card_end, info_card, render_badge,
    prediction_result_card, store_info_banner, stat_table,
    model_quality_banner, divider, empty_state, sidebar_label,
)
from src.preprocessing import (
    build_master_df, get_store_list, get_family_list,
    filter_store_family, load_stores,
)
from src.feature_engineering import (
    build_features, build_prediction_row, FEATURE_COLS,
)
from src.model_training import (
    train_model, save_model, load_model, model_exists,
    predict_store_family, get_feature_importance,
)
from src.visualization import (
    plot_actual_vs_predicted, plot_sales_trend, plot_monthly_heatmap,
    plot_feature_importance, plot_residuals, plot_dow_pattern,
    plot_scatter_actual_predicted, plot_top_stores,
)

# ── CSS injection ─────────────────────────────────────────────
inject_css()


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "model":        None,
        "metrics":      None,
        "feature_cols": None,
        "master_df":    None,
        "feat_df":      None,
        "trained":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ═══════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_master():
    return build_master_df()

@st.cache_data(show_spinner=False)
def get_features(_master_df):
    return build_features(_master_df)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

def render_sidebar(master_df: pd.DataFrame):
    with st.sidebar:

        # ── Logo / Brand ──────────────────────────────────────
        st.markdown("""
        <div style="padding:0.25rem 0 1rem 0;">
            <div style="display:flex;align-items:center;gap:9px;margin-bottom:4px;">
                <div style="width:34px;height:34px;background:#2563EB;
                            border-radius:9px;display:flex;align-items:center;
                            justify-content:center;font-size:18px;flex-shrink:0;">
                    📊
                </div>
                <div>
                    <div style="font-size:15px;font-weight:800;color:#111827;
                                letter-spacing:-0.02em;">Sales Forecast</div>
                    <div style="font-size:10.5px;color:#9CA3AF;font-weight:500;
                                margin-top:1px;">Analytics Dashboard</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Data Summary ──────────────────────────────────────
        sidebar_label("Dataset")
        n_rows   = len(master_df)
        n_stores = master_df["store_nbr"].nunique()
        n_family = master_df["family"].nunique()
        date_min = master_df["date"].min().strftime("%b %Y")
        date_max = master_df["date"].max().strftime("%b %Y")

        st.markdown(f"""
        <div style="background:#F8FAFC;border:1px solid #E5E7EB;border-radius:9px;
                    padding:0.75rem 0.9rem;font-size:12px;color:#374151;
                    line-height:1.85;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#6B7280;">Rows</span>
                <span style="font-weight:700;color:#111827;">{n_rows:,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#6B7280;">Stores</span>
                <span style="font-weight:700;color:#111827;">{n_stores}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#6B7280;">Families</span>
                <span style="font-weight:700;color:#111827;">{n_family}</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#6B7280;">Period</span>
                <span style="font-weight:700;color:#111827;">{date_min}–{date_max}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Model Status ──────────────────────────────────────
        sidebar_label("Model")
        if model_exists():
            st.markdown(
                badge("✓ Saved on Disk", "green"),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                badge("⚠ Not Trained", "red"),
                unsafe_allow_html=True
            )
        st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

        model_type = st.selectbox(
            "Algorithm",
            options=["gradient_boosting", "random_forest", "ridge"],
            format_func=lambda x: {
                "gradient_boosting": "Gradient Boosting (Best)",
                "random_forest":     "Random Forest (Fast)",
                "ridge":             "Ridge Regression (Baseline)",
            }[x],
        )

        train_btn = st.button(
            "🚀  Train Model",
            use_container_width=True,
            type="primary",
        )

        st.divider()

        # ── Filters ───────────────────────────────────────────
        sidebar_label("Explorer Filters")
        stores    = get_store_list(master_df)
        store_nbr = st.selectbox(
            "Store",
            options=stores,
            format_func=lambda x: f"Store {x}",
        )
        families = get_family_list(master_df)
        family   = st.selectbox("Product Family", options=families)

        st.divider()

        # ── Chart Controls ────────────────────────────────────
        sidebar_label("Chart Controls")
        chart_days = st.slider(
            "Last N days",
            min_value=30, max_value=365, value=90, step=30,
        )
        freq = st.radio(
            "Trend period",
            options=["W", "M"],
            format_func=lambda x: "Weekly" if x == "W" else "Monthly",
            horizontal=True,
        )

        st.divider()

        # ── Footer ────────────────────────────────────────────
        st.markdown("""
        <div style="font-size:10.5px;color:#9CA3AF;line-height:1.75;">
            Day 3 — Internship Project<br>
            Kaggle Store Sales Dataset<br>
            Streamlit · Scikit-learn · Python
        </div>
        """, unsafe_allow_html=True)

    return store_nbr, family, model_type, train_btn, chart_days, freq


# ═══════════════════════════════════════════════════════════════
# TRAINING HANDLER
# ═══════════════════════════════════════════════════════════════

def handle_training(feat_df, model_type: str):
    with st.spinner("Training model — this may take 1–3 min…"):
        t0 = time.time()
        model, metrics, feature_cols = train_model(
            feat_df, model_type=model_type, sample_size=150_000,
        )
        save_model(model, metrics, feature_cols)
        elapsed = time.time() - t0

    st.session_state.model        = model
    st.session_state.metrics      = metrics
    st.session_state.feature_cols = feature_cols
    st.session_state.trained      = True

    st.success(
        f"✅ Trained in {elapsed:.1f}s  ·  "
        f"MAE {metrics['MAE']:.2f}  ·  "
        f"RMSE {metrics['RMSE']:.2f}  ·  "
        f"R² {metrics['R2']:.4f}"
    )
    return model, metrics, feature_cols


def ensure_model_loaded():
    if st.session_state.model is None and model_exists():
        m, metrics, fc = load_model()
        st.session_state.model        = m
        st.session_state.metrics      = metrics
        st.session_state.feature_cols = fc
        st.session_state.trained      = True


# ═══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════

def render_overview(master_df, store_nbr, family, chart_days, freq):

    # ── Global KPIs ───────────────────────────────────────────
    section_header("Dataset Overview", "All stores · All families")

    total_sales = master_df["sales"].sum()
    avg_daily   = master_df.groupby("date")["sales"].sum().mean()
    promo_rate  = (master_df["onpromotion"] > 0).mean() * 100
    holiday_pct = master_df["is_holiday"].mean() * 100

    metric_row([
        {"title": "Total Sales",       "value": fmt_currency(total_sales), "icon": "💰", "accent_color": "#2563EB"},
        {"title": "Avg Daily Sales",   "value": fmt_currency(avg_daily),   "icon": "📅", "accent_color": "#10B981"},
        {"title": "Promotion Rate",    "value": f"{promo_rate:.1f}%",      "icon": "🏷️", "accent_color": "#F59E0B"},
        {"title": "Holiday Days",      "value": f"{holiday_pct:.1f}%",     "icon": "🎉", "accent_color": "#8B5CF6"},
    ])

    divider()

    # ── Store + Family Section ────────────────────────────────
    section_header(f"Store {store_nbr}  ·  {family}", "Filtered view")

    filtered = filter_store_family(master_df, store_nbr, family)

    if filtered.empty:
        empty_state("🔍", "No Data Found",
                    "No records match this store + family combination.")
        return

    # Store info banner
    stores_df = load_stores()
    info      = get_store_info(stores_df, store_nbr)
    if info:
        store_info_banner(
            city       = info.get("city", "—"),
            state      = info.get("state", "—"),
            store_type = info.get("type", "—"),
            cluster    = info.get("cluster", "—"),
        )

    # Store+Family KPIs
    metric_row([
        {"title": "Total Sales",     "value": fmt_currency(filtered["sales"].sum()),  "icon": "📦", "accent_color": "#2563EB"},
        {"title": "Avg Daily",       "value": fmt_currency(filtered["sales"].mean()), "icon": "📈", "accent_color": "#10B981"},
        {"title": "Peak Day",        "value": fmt_currency(filtered["sales"].max()),  "icon": "🏆", "accent_color": "#F59E0B"},
        {"title": "Days of History", "value": f"{len(filtered):,}",                  "icon": "📅", "accent_color": "#6366F1"},
    ])

    divider()

    # ── Charts row 1 ──────────────────────────────────────────
    col_l, col_r = st.columns([3, 2])

    with col_l:
        chart_card_start("Sales Trend",
                         "Weekly" if freq == "W" else "Monthly")
        fig = plot_sales_trend(filtered, store_nbr, family, freq=freq)
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    with col_r:
        chart_card_start("Day-of-Week Pattern")
        fig = plot_dow_pattern(filtered, store_nbr, family)
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    # ── Charts row 2 ──────────────────────────────────────────
    chart_card_start("Monthly Heatmap",
                     "Average daily sales by month × year")
    fig = plot_monthly_heatmap(filtered, store_nbr, family)
    st.pyplot(fig, use_container_width=True)
    chart_card_end()

    chart_card_start("Top 10 Stores by Total Sales")
    fig = plot_top_stores(master_df, top_n=10)
    st.pyplot(fig, use_container_width=True)
    chart_card_end()


# ═══════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════

def render_model_performance(master_df, feat_df, store_nbr, family, chart_days):

    if st.session_state.model is None:
        empty_state(
            "⚙️",
            "No Model Loaded",
            "Click 'Train Model' in the sidebar to train and save a model.",
        )
        return

    model        = st.session_state.model
    metrics      = st.session_state.metrics
    feature_cols = st.session_state.feature_cols

    # ── Quality Banner ────────────────────────────────────────
    section_header("Model Evaluation", "Hold-out test set performance")
    model_quality_banner(
        r2   = metrics["R2"],
        mae  = metrics["MAE"],
        rmse = metrics["RMSE"],
    )

    # ── Native metric cards ───────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "MAE  (Mean Abs Error)",
        fmt_number(metrics["MAE"]),
        help="Lower is better. Average absolute prediction error.",
    )
    c2.metric(
        "RMSE (Root Mean Sq Error)",
        fmt_number(metrics["RMSE"]),
        help="Lower is better. Penalises large errors more.",
    )
    c3.metric(
        "R²  (Coefficient of Det.)",
        fmt_number(metrics["R2"], 4),
        help="Closer to 1.0 is better. Variance explained by the model.",
    )

    divider()

    # ── Batch predictions for selected store + family ─────────
    section_header(
        f"Actual vs Predicted  ·  Store {store_nbr}  ·  {family}",
        f"Last {chart_days} days shown in the line chart",
    )

    slice_feat = feat_df[
        (feat_df["store_nbr"] == store_nbr) &
        (feat_df["family"]    == family)
    ].copy()

    if slice_feat.empty:
        empty_state("📭", "No Data",
                    "No feature data available for this store + family.")
        return

    result_df = predict_store_family(model, slice_feat, feature_cols)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        chart_card_start("Actual vs Predicted", "Line chart")
        fig = plot_actual_vs_predicted(result_df, store_nbr, family, chart_days)
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    with col_r:
        chart_card_start("Scatter Plot", "Actual vs predicted correlation")
        fig = plot_scatter_actual_predicted(result_df)
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    # ── Residuals + Feature Importance ───────────────────────
    col_l2, col_r2 = st.columns([2, 3])
    with col_l2:
        chart_card_start("Residuals Distribution")
        fig = plot_residuals(result_df)
        st.pyplot(fig, use_container_width=True)
        chart_card_end()

    with col_r2:
        fi_df = get_feature_importance(model, feature_cols)
        if not fi_df.empty:
            chart_card_start("Feature Importance", "Top 15 features")
            fig = plot_feature_importance(fi_df)
            st.pyplot(fig, use_container_width=True)
            chart_card_end()
        else:
            info_card("Feature importance is not available for linear models.")

    # ── Raw table ─────────────────────────────────────────────
    with st.expander("📋  View Prediction Data  (last 50 rows)"):
        display = result_df.tail(50).copy()
        display["error"]  = (display["actual"] - display["predicted"]).round(2)
        display["error%"] = (
            (display["error"].abs() / display["actual"].replace(0, np.nan)) * 100
        ).round(1)
        st.dataframe(
            display[["date", "actual", "predicted", "error", "error%"]],
            use_container_width=True,
            height=300,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — PREDICT SALES
# ═══════════════════════════════════════════════════════════════

def render_prediction(master_df, store_nbr, family):

    if st.session_state.model is None:
        empty_state(
            "🔮",
            "No Model Available",
            "Train a model first using the 'Train Model' button in the sidebar.",
        )
        return

    model        = st.session_state.model
    feature_cols = st.session_state.feature_cols

    section_header("Sales Prediction", f"Store {store_nbr}  ·  {family}")

    # ── Encoders ──────────────────────────────────────────────
    fam_encoder  = get_family_encoder(master_df)
    family_enc   = fam_encoder.get(family, 0)
    store_type_e = get_store_type_enc(master_df, store_nbr)
    cluster      = get_store_cluster(master_df, store_nbr)

    # ── Two-column form layout ────────────────────────────────
    col_form, col_context = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("""
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.07em;color:#6B7280;margin-bottom:10px;">
            Prediction Inputs
        </div>
        """, unsafe_allow_html=True)

        pred_date = st.date_input(
            "Prediction Date",
            value=datetime.date(2017, 8, 15),
            min_value=datetime.date(2013, 1, 1),
            max_value=datetime.date(2025, 12, 31),
            help="Select the date you want to forecast sales for.",
        )

        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)

        onpromotion = st.toggle(
            "Product on Promotion",
            value=False,
            help="Is this product running a promotion on the selected date?",
        )
        is_holiday = st.toggle(
            "National Holiday",
            value=False,
            help="Is the selected date a national holiday in Ecuador?",
        )

    with col_context:
        stats = get_recent_stats(
            master_df, store_nbr, family,
            ref_date=pd.Timestamp(pred_date),
        )
        stat_table(
            rows=[
                ("7-day lag sales",    f"{stats['sales_lag_7']:.2f}"),
                ("14-day lag sales",   f"{stats['sales_lag_14']:.2f}"),
                ("28-day lag sales",   f"{stats['sales_lag_28']:.2f}"),
                ("7-day rolling avg",  f"{stats['sales_roll_mean_7']:.2f}"),
                ("14-day rolling avg", f"{stats['sales_roll_mean_14']:.2f}"),
            ],
            title="Historical Context (auto-filled)",
        )

    st.markdown("<div style='margin:1rem 0 0.5rem 0;'></div>",
                unsafe_allow_html=True)

    # ── Predict button ────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 1, 1])
    with btn_col:
        predict_btn = st.button(
            "⚡  Predict Sales",
            use_container_width=True,
            type="primary",
        )

    # ── Result ────────────────────────────────────────────────
    if predict_btn:
        X_pred = build_prediction_row(
            store_nbr      = store_nbr,
            family_enc     = family_enc,
            store_type_enc = store_type_e,
            cluster        = cluster,
            date           = pd.Timestamp(pred_date),
            onpromotion    = int(onpromotion),
            is_holiday     = int(is_holiday),
            **stats,
        )
        pred_cols = [c for c in feature_cols if c in X_pred.columns]
        X_pred    = X_pred[pred_cols]
        predicted = float(np.clip(model.predict(X_pred)[0], 0, None))

        mae   = st.session_state.metrics["MAE"]
        lower = max(0.0, predicted - mae)
        upper = predicted + mae

        divider()
        section_header("Prediction Result")

        prediction_result_card(
            value    = fmt_currency(predicted),
            lower    = fmt_currency(lower),
            upper    = fmt_currency(upper),
            store    = store_nbr,
            family   = family,
            date_str = pred_date.strftime("%d %b %Y"),
            promo    = bool(onpromotion),
            holiday  = bool(is_holiday),
        )

        st.markdown("<div style='margin-top:1rem;'></div>",
                    unsafe_allow_html=True)
        info_card(
            f"Confidence range based on model MAE of <b>{mae:.2f}</b> units. "
            f"Expected range: <b>[{lower:.1f} – {upper:.1f}]</b>.",
            color="#10B981",
        )

    else:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0;color:#9CA3AF;">
            <div style="font-size:1.75rem;margin-bottom:8px;">🔮</div>
            <div style="font-size:13px;font-weight:600;color:#6B7280;">
                Configure inputs and click <b>Predict Sales</b>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Load data ─────────────────────────────────────────────
    try:
        master_df = get_master()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

    st.session_state.master_df = master_df

    # ── Sidebar ───────────────────────────────────────────────
    store_nbr, family, model_type, train_btn, chart_days, freq = \
        render_sidebar(master_df)

    # ── Train ─────────────────────────────────────────────────
    if train_btn:
        with st.spinner("Building features…"):
            feat_df = get_features(master_df)
            st.session_state.feat_df = feat_df
        handle_training(feat_df, model_type)

    # ── Auto-load saved model ─────────────────────────────────
    ensure_model_loaded()

    # ── Ensure feat_df is ready ───────────────────────────────
    if st.session_state.feat_df is None:
        with st.spinner("Preparing features…"):
            feat_df = get_features(master_df)
            st.session_state.feat_df = feat_df
    else:
        feat_df = st.session_state.feat_df

    # ── Page header ───────────────────────────────────────────
    page_header(
        title    = "Store Sales Forecasting Dashboard",
        subtitle = "Kaggle Time-Series Competition  ·  GradientBoosting Regression  ·  Interactive Predictions",
    )

    # ── Tabs ──────────────────────────────────────────────────
    tab_overview, tab_performance, tab_predict = st.tabs([
        "  📊  Overview  ",
        "  🎯  Model Performance  ",
        "  🔮  Predict Sales  ",
    ])

    with tab_overview:
        render_overview(master_df, store_nbr, family, chart_days, freq)

    with tab_performance:
        render_model_performance(master_df, feat_df, store_nbr, family, chart_days)

    with tab_predict:
        render_prediction(master_df, store_nbr, family)


if __name__ == "__main__":
    main()