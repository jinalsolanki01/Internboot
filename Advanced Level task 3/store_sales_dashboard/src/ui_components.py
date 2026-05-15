"""
ui_components.py
----------------
Reusable premium HTML component builders for the dashboard.
All components render via st.markdown(unsafe_allow_html=True).

Design system
-------------
  BG         #F5F7FB   main background
  CARD       #FFFFFF   card surfaces
  BORDER     #E5E7EB   subtle borders
  TEXT       #111827   primary text
  MUTED      #6B7280   secondary text
  FAINT      #9CA3AF   placeholder / hint text
  ACCENT     #2563EB   primary blue
  SUCCESS    #10B981   green
  DANGER     #EF4444   red
  WARNING    #F59E0B   amber
"""

import streamlit as st


# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str) -> None:
    """
    Render the main dashboard hero header.
    Clean two-line title + subtitle with a bottom separator line.
    """
    st.markdown(f"""
    <div style="
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    ">
        <div style="
            font-size: 1.65rem;
            font-weight: 800;
            color: #111827;
            letter-spacing: -0.03em;
            line-height: 1.15;
        ">{title}</div>
        <div style="
            font-size: 12.5px;
            color: #6B7280;
            margin-top: 5px;
            font-weight: 400;
            letter-spacing: 0.01em;
        ">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SECTION HEADER
# ─────────────────────────────────────────────────────────────

def section_header(title: str, subtitle: str = "") -> None:
    """
    Renders a labelled section divider.
    Optional subtitle shown below in muted text.
    """
    sub_html = (
        f'<div style="font-size:11.5px;color:#6B7280;margin-top:3px;">'
        f'{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(f"""
    <div style="margin: 1.75rem 0 0.9rem 0;">
        <div style="
            font-size: 13px;
            font-weight: 700;
            color: #111827;
            letter-spacing: -0.01em;
            display: flex;
            align-items: center;
            gap: 7px;
        ">{title}</div>
        {sub_html}
        <div style="
            height: 2px;
            background: linear-gradient(to right, #E5E7EB 60%, transparent);
            margin-top: 8px;
            border-radius: 2px;
        "></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KPI CARD  (custom HTML — richer than st.metric)
# ─────────────────────────────────────────────────────────────

def kpi_card(
    title: str,
    value: str,
    icon: str = "",
    delta: str = "",
    delta_good: bool = True,
    accent_color: str = "#2563EB",
) -> None:
    """
    Premium KPI card with icon, value, label and optional delta.

    Parameters
    ----------
    title        : Label above the value (e.g. "Total Sales")
    value        : Formatted value string (e.g. "1.07B")
    icon         : Emoji or short icon string
    delta        : Optional change string (e.g. "+12.3%")
    delta_good   : True = green delta, False = red delta
    accent_color : Top border accent colour
    """
    delta_color = "#10B981" if delta_good else "#EF4444"
    delta_html  = (
        f'<div style="margin-top:6px;font-size:11px;font-weight:600;'
        f'color:{delta_color};">{delta}</div>'
        if delta else ""
    )
    icon_html = (
        f'<div style="font-size:20px;margin-bottom:6px;">{icon}</div>'
        if icon else ""
    )
    st.markdown(f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-top: 3px solid {accent_color};
        border-radius: 10px;
        padding: 1rem 1.1rem 0.85rem 1.1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
        height: 100%;
    ">
        {icon_html}
        <div style="
            font-size: 10.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #6B7280;
            margin-bottom: 4px;
        ">{title}</div>
        <div style="
            font-size: 1.55rem;
            font-weight: 800;
            color: #111827;
            letter-spacing: -0.025em;
            line-height: 1.1;
        ">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# METRIC ROW (4-up grid using st.columns + kpi_card)
# ─────────────────────────────────────────────────────────────

def metric_row(items: list[dict]) -> None:
    """
    Render a responsive row of KPI cards.

    Parameters
    ----------
    items : list of dicts, each with keys:
              title, value, icon (opt), delta (opt),
              delta_good (opt, default True),
              accent_color (opt)

    Example
    -------
    metric_row([
        {"title": "Total Sales",  "value": "1.07B", "icon": "💰"},
        {"title": "Avg Daily",    "value": "637K",  "icon": "📅"},
    ])
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            kpi_card(
                title        = item.get("title", ""),
                value        = item.get("value", "—"),
                icon         = item.get("icon", ""),
                delta        = item.get("delta", ""),
                delta_good   = item.get("delta_good", True),
                accent_color = item.get("accent_color", "#2563EB"),
            )


# ─────────────────────────────────────────────────────────────
# CHART CARD WRAPPER
# ─────────────────────────────────────────────────────────────

def chart_card_start(title: str, subtitle: str = "") -> None:
    """
    Render the opening of a chart card container.
    Call chart_card_end() after st.pyplot().
    """
    sub_html = (
        f'<span style="font-size:11px;color:#9CA3AF;margin-left:8px;">'
        f'{subtitle}</span>'
        if subtitle else ""
    )
    st.markdown(f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.1rem 1.25rem 0.5rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    ">
        <div style="
            font-size: 12.5px;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.75rem;
        ">{title}{sub_html}</div>
    """, unsafe_allow_html=True)


def chart_card_end() -> None:
    """Close the chart card container div."""
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INFO CARD
# ─────────────────────────────────────────────────────────────

def info_card(content: str, color: str = "#2563EB") -> None:
    """
    Render a coloured left-border info card.
    content : raw HTML or text
    color   : left border colour
    """
    st.markdown(f"""
    <div style="
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 0.85rem 1rem;
        font-size: 12.5px;
        color: #374151;
        line-height: 1.65;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    ">{content}</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# BADGE
# ─────────────────────────────────────────────────────────────

_BADGE_STYLES = {
    "green":  ("D1FAE5", "065F46"),
    "blue":   ("DBEAFE", "1E40AF"),
    "red":    ("FEE2E2", "991B1B"),
    "amber":  ("FEF3C7", "92400E"),
    "purple": ("EDE9FE", "5B21B6"),
    "gray":   ("F3F4F6", "374151"),
}


def badge(text: str, color: str = "blue") -> str:
    """Return inline HTML for a coloured pill badge."""
    bg, fg = _BADGE_STYLES.get(color, ("DBEAFE", "1E40AF"))
    return (
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:99px;font-size:11px;font-weight:700;'
        f'background:#{bg};color:#{fg};">{text}</span>'
    )


def render_badge(text: str, color: str = "blue") -> None:
    """Render a badge directly via st.markdown."""
    st.markdown(badge(text, color), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STATUS ROW  (model trained / data loaded indicators)
# ─────────────────────────────────────────────────────────────

def status_row(items: list[tuple[str, str, str]]) -> None:
    """
    Render a horizontal row of status indicators.
    items : list of (label, badge_text, badge_color)
    """
    parts = "  ·  ".join(
        f'{label}: {badge(btext, bcolor)}'
        for label, btext, bcolor in items
    )
    st.markdown(
        f'<div style="font-size:11.5px;color:#6B7280;'
        f'margin-bottom:0.5rem;">{parts}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# PREDICTION RESULT CARD
# ─────────────────────────────────────────────────────────────

def prediction_result_card(
    value: str,
    lower: str,
    upper: str,
    store: int,
    family: str,
    date_str: str,
    promo: bool,
    holiday: bool,
) -> None:
    """
    Full premium prediction result display card.
    Shows main value + confidence range + context chips.
    """
    promo_chip   = _chip("🏷️ Promo ON",    "#DBEAFE", "#1E40AF") if promo   else _chip("No Promo",    "#F3F4F6", "#6B7280")
    holiday_chip = _chip("🎉 Holiday",     "#D1FAE5", "#065F46") if holiday else _chip("Regular Day", "#F3F4F6", "#6B7280")

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #EFF6FF 0%, #F0FDF4 100%);
        border: 1.5px solid #BFDBFE;
        border-radius: 16px;
        padding: 2rem 2rem 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(37,99,235,0.08);
    ">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.1em;color:#3B82F6;margin-bottom:8px;">
            Predicted Sales
        </div>
        <div style="font-size:3.2rem;font-weight:900;color:#1D4ED8;
                    letter-spacing:-0.04em;line-height:1;">
            {value}
        </div>
        <div style="
            display:flex;justify-content:center;gap:10px;
            margin-top:14px;flex-wrap:wrap;
        ">
            <div style="background:#FFFFFF;border:1px solid #BFDBFE;
                        border-radius:8px;padding:6px 14px;font-size:11.5px;">
                <span style="color:#6B7280;">Low</span>
                <span style="font-weight:700;color:#111827;margin-left:6px;">{lower}</span>
            </div>
            <div style="background:#FFFFFF;border:1px solid #BFDBFE;
                        border-radius:8px;padding:6px 14px;font-size:11.5px;">
                <span style="color:#6B7280;">High</span>
                <span style="font-weight:700;color:#111827;margin-left:6px;">{upper}</span>
            </div>
        </div>
        <div style="
            display:flex;justify-content:center;gap:8px;
            margin-top:14px;flex-wrap:wrap;align-items:center;
        ">
            {_chip(f"🏪 Store {store}", "#F3F4F6", "#374151")}
            {_chip(f"📦 {family}",     "#F3F4F6", "#374151")}
            {_chip(f"📅 {date_str}",   "#F3F4F6", "#374151")}
            {promo_chip}
            {holiday_chip}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _chip(text: str, bg: str, fg: str) -> str:
    """Return HTML for a small context chip."""
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 10px;'
        f'border-radius:99px;font-size:11px;font-weight:600;">{text}</span>'
    )


# ─────────────────────────────────────────────────────────────
# STORE INFO BANNER
# ─────────────────────────────────────────────────────────────

def store_info_banner(city: str, state: str, store_type: str, cluster: int) -> None:
    """Render a compact store metadata banner."""
    st.markdown(f"""
    <div style="
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        display: flex;
        gap: 20px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 1rem;
        font-size: 12.5px;
        color: #374151;
    ">
        <span>🏙️ <b>{city}</b>, {state}</span>
        <span style="color:#D1D5DB;">|</span>
        <span>Type: {badge(store_type, 'blue')}</span>
        <span style="color:#D1D5DB;">|</span>
        <span>Cluster <b>{cluster}</b></span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STAT TABLE  (key-value pairs, used in Historical Context)
# ─────────────────────────────────────────────────────────────

def stat_table(rows: list[tuple[str, str]], title: str = "") -> None:
    """
    Render a clean key-value table inside a card.
    rows : list of (label, value) tuples
    """
    rows_html = "".join(
        f"""<div style="display:flex;justify-content:space-between;
                        align-items:center;padding:7px 0;
                        border-bottom:1px solid #F3F4F6;">
                <span style="font-size:12px;color:#6B7280;">{k}</span>
                <span style="font-size:12px;font-weight:600;color:#111827;">{v}</span>
            </div>"""
        for k, v in rows
    )
    title_html = (
        f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:#9CA3AF;margin-bottom:8px;">{title}</div>'
        if title else ""
    )
    st.markdown(f"""
    <div style="
        background:#FFFFFF;
        border:1px solid #E5E7EB;
        border-radius:10px;
        padding:0.9rem 1rem;
        box-shadow:0 1px 3px rgba(0,0,0,0.04);
    ">
        {title_html}
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL QUALITY BANNER
# ─────────────────────────────────────────────────────────────

def model_quality_banner(r2: float, mae: float, rmse: float) -> None:
    """
    Render a horizontal quality summary banner with colour-coded R².
    """
    if r2 >= 0.90:
        label, bg, fg, border = "Excellent", "#D1FAE5", "#065F46", "#6EE7B7"
    elif r2 >= 0.75:
        label, bg, fg, border = "Good",      "#DBEAFE", "#1E40AF", "#93C5FD"
    elif r2 >= 0.50:
        label, bg, fg, border = "Fair",      "#FEF3C7", "#92400E", "#FCD34D"
    else:
        label, bg, fg, border = "Needs Work","#FEE2E2", "#991B1B", "#FCA5A5"

    st.markdown(f"""
    <div style="
        background:{bg};
        border:1.5px solid {border};
        border-radius:12px;
        padding:1rem 1.5rem;
        display:flex;
        align-items:center;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:12px;
        margin-bottom:1rem;
    ">
        <div>
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.07em;color:{fg};opacity:0.75;">
                Model Quality
            </div>
            <div style="font-size:1.4rem;font-weight:800;color:{fg};
                        letter-spacing:-0.02em;margin-top:2px;">
                {label}
            </div>
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                            color:{fg};opacity:0.7;letter-spacing:0.05em;">R²</div>
                <div style="font-size:1.2rem;font-weight:800;color:{fg};">{r2:.4f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                            color:{fg};opacity:0.7;letter-spacing:0.05em;">MAE</div>
                <div style="font-size:1.2rem;font-weight:800;color:{fg};">{mae:.2f}</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                            color:{fg};opacity:0.7;letter-spacing:0.05em;">RMSE</div>
                <div style="font-size:1.2rem;font-weight:800;color:{fg};">{rmse:.2f}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DIVIDER
# ─────────────────────────────────────────────────────────────

def divider(margin: str = "1.25rem") -> None:
    """Render a clean horizontal rule."""
    st.markdown(
        f'<hr style="border:none;border-top:1px solid #E5E7EB;'
        f'margin:{margin} 0;" />',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────

def empty_state(icon: str, title: str, subtitle: str) -> None:
    """Centered empty-state placeholder."""
    st.markdown(f"""
    <div style="
        text-align:center;
        padding:3rem 1rem;
        color:#9CA3AF;
    ">
        <div style="font-size:2.5rem;margin-bottom:12px;">{icon}</div>
        <div style="font-size:14px;font-weight:700;color:#374151;
                    margin-bottom:6px;">{title}</div>
        <div style="font-size:12.5px;color:#9CA3AF;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR SECTION LABEL
# ─────────────────────────────────────────────────────────────

def sidebar_label(text: str) -> None:
    """Render a sidebar section group label."""
    st.markdown(
        f'<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:#9CA3AF;margin:0.85rem 0 0.35rem 0;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )