import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config 
st.set_page_config(
    page_title="Supply Chain AI System",
    page_icon="⛓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@400;600;800&display=swap');

:root {
    --bg: #0A0E1A;
    --surface: #111827;
    --border: #1e2d45;
    --accent: #00D4FF;
    --green: #00E5A0;
    --orange: #FF6B35;
    --yellow: #FFD166;
    --red: #FF4757;
    --purple: #A855F7;
    --text-primary: #E8F4FD;
    --text-secondary: #7A8FA8;
    --text-muted: #3D5068;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0A0E1A !important;
    color: #E8F4FD !important;
}

.main { background-color: #0A0E1A !important; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1e2d45 !important;
}
[data-testid="stSidebar"] * { color: #E8F4FD !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #7A8FA8 !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: rgba(0, 212, 255, 0.1) !important;
    color: #00D4FF !important;
    border: 1px solid rgba(0, 212, 255, 0.3) !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] { color: #7A8FA8 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: #00D4FF !important; font-family: 'JetBrains Mono', monospace !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] > div { font-size: 11px !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #1e2d45; border-radius: 10px; overflow: hidden; }

/* Plotly charts BG */
.js-plotly-plot { border-radius: 12px; }

/* Custom badges */
.badge { display: inline-block; border-radius: 4px; font-size: 10px; font-weight: 700;
         padding: 2px 8px; letter-spacing: 0.06em; font-family: 'JetBrains Mono', monospace; }
.badge-high { background: #2D0D0D; color: #FF4757; border: 1px solid rgba(255,71,87,0.25); }
.badge-medium { background: #2D1F00; color: #FFD166; border: 1px solid rgba(255,209,102,0.25); }
.badge-low { background: #0D2D1A; color: #00E5A0; border: 1px solid rgba(0,229,160,0.25); }

/* Alert boxes */
.alert-critical { background: rgba(255,71,87,0.05); border: 1px solid rgba(255,71,87,0.2);
                  border-left: 3px solid #FF4757; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
.alert-warning { background: rgba(255,209,102,0.05); border: 1px solid rgba(255,209,102,0.2);
                 border-left: 3px solid #FFD166; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }
.alert-info { background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.2);
              border-left: 3px solid #00D4FF; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px; }

.alert-title { font-size: 12px; font-weight: 700; color: #E8F4FD; margin-bottom: 4px; }
.alert-detail { font-size: 11px; color: #7A8FA8; line-height: 1.5; }
.alert-time { font-size: 10px; color: #3D5068; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }

/* Pipeline steps */
.pipeline-step { border-radius: 10px; padding: 14px 16px; margin-bottom: 8px; }
.pipeline-done { background: #0D2D1A; border: 1px solid #00E5A0; }
.pipeline-active { background: #0D1F2D; border: 1px solid #00D4FF; box-shadow: 0 0 20px rgba(0,212,255,0.1); }
.pipeline-pending { background: #111827; border: 1px solid #1e2d45; }

.step-title { font-size: 13px; font-weight: 700; color: #E8F4FD; }
.step-file { font-size: 10px; color: #3D5068; font-family: 'JetBrains Mono', monospace; }
.step-detail { font-size: 11px; color: #7A8FA8; margin-top: 6px; }

/* Section header */
.section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.section-bar { width: 3px; height: 18px; background: #00D4FF; border-radius: 2px; display: inline-block; }
.section-title { font-size: 13px; font-weight: 700; color: #E8F4FD; letter-spacing: 0.04em; text-transform: uppercase; }

/* Formula card */
.formula-card { background: #111827; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; }
.formula-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px; }
.formula-code { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700;
                color: #E8F4FD; background: #0A0E1A; padding: 10px 14px; border-radius: 8px; margin-bottom: 10px; }
.formula-detail { font-size: 11px; color: #7A8FA8; line-height: 1.6; }

.pipeline-icon { width: 28px; height: 28px; border-radius: 6px; display: inline-flex;
                 align-items: center; justify-content: center; font-size: 12px; font-weight: 800;
                 font-family: 'JetBrains Mono', monospace; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }

/* Selectbox / filter */
[data-testid="stSelectbox"] > div { background: #111827 !important; border: 1px solid #1e2d45 !important; border-radius: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Data 
SEGMENTS = pd.DataFrame([
    {"Sub-Category": "Bookcases",   "Region": "Central", "Market": "US",    "ADD/day": 2.41, "Safety Stock": 38.2,  "Reorder Point": 52.6,  "Avg Inventory": 22,  "Risk": "HIGH",   "Demand Trend %": 12},
    {"Sub-Category": "Chairs",      "Region": "West",    "Market": "US",    "ADD/day": 3.90, "Safety Stock": 61.4,  "Reorder Point": 84.3,  "Avg Inventory": 49,  "Risk": "HIGH",   "Demand Trend %": -4},
    {"Sub-Category": "Tables",      "Region": "South",   "Market": "US",    "ADD/day": 1.85, "Safety Stock": 29.3,  "Reorder Point": 42.1,  "Avg Inventory": 38,  "Risk": "MEDIUM", "Demand Trend %": 7},
    {"Sub-Category": "Copiers",     "Region": "East",    "Market": "APAC",  "ADD/day": 0.92, "Safety Stock": 14.6,  "Reorder Point": 20.8,  "Avg Inventory": 18,  "Risk": "MEDIUM", "Demand Trend %": 18},
    {"Sub-Category": "Phones",      "Region": "Central", "Market": "EU",    "ADD/day": 5.62, "Safety Stock": 88.7,  "Reorder Point": 119.4, "Avg Inventory": 145, "Risk": "LOW",    "Demand Trend %": -2},
    {"Sub-Category": "Accessories", "Region": "West",    "Market": "US",    "ADD/day": 8.14, "Safety Stock": 128.4, "Reorder Point": 172.9, "Avg Inventory": 210, "Risk": "LOW",    "Demand Trend %": 9},
    {"Sub-Category": "Paper",       "Region": "East",    "Market": "LATAM", "ADD/day": 12.3, "Safety Stock": 193.9, "Reorder Point": 261.1, "Avg Inventory": 89,  "Risk": "HIGH",   "Demand Trend %": -8},
    {"Sub-Category": "Binders",     "Region": "South",   "Market": "APAC",  "ADD/day": 6.71, "Safety Stock": 105.8, "Reorder Point": 142.5, "Avg Inventory": 168, "Risk": "LOW",    "Demand Trend %": 3},
])

FORECAST_DATA = {
    "Month": ["Month 1", "Month 2", "Month 3"],
    "Technology": [4820, 5140, 6090],
    "Furniture": [1980, 2110, 2640],
    "Office Supplies": [3410, 3650, 4320],
}

COLORS = {
    "bg": "#0A0E1A", "surface": "#111827", "border": "#1e2d45",
    "accent": "#00D4FF", "green": "#00E5A0", "orange": "#FF6B35",
    "yellow": "#FFD166", "red": "#FF4757", "purple": "#A855F7",
    "text": "#E8F4FD", "muted": "#7A8FA8",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#111827", plot_bgcolor="#111827",
    font=dict(color="#E8F4FD", family="DM Sans"),
    margin=dict(t=20, b=20, l=10, r=10),
)

GRID_AXIS = dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45")

# ─── Helpers 
def section_header(title, sub=""):
    html = f"""
    <div style="margin-bottom:16px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <div style="width:3px;height:18px;background:#00D4FF;border-radius:2px;flex-shrink:0;"></div>
        <div style="font-size:13px;font-weight:700;color:#E8F4FD;letter-spacing:0.04em;text-transform:uppercase;">{title}</div>
      </div>
      {"<div style='font-size:12px;color:#7A8FA8;padding-left:13px;'>"+sub+"</div>" if sub else ""}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)

def pipeline_step(num, title, file, status, detail=""):
    cls = {"DONE": "pipeline-done", "ACTIVE": "pipeline-active", "PENDING": "pipeline-pending"}[status]
    icon_map = {"DONE": ("✓", "#00E5A0"), "ACTIVE": ("◉", "#00D4FF"), "PENDING": (str(num), "#3D5068")}
    icon, icon_color = icon_map[status]
    badge_colors = {"DONE": ("#0D2D1A", "#00E5A0", "rgba(0,229,160,0.3)"),
                    "ACTIVE": ("#0D1F2D", "#00D4FF", "rgba(0,212,255,0.3)"),
                    "PENDING": ("#111827", "#7A8FA8", "rgba(61,80,104,0.3)")}
    bg, bc, bb = badge_colors[status]
    st.markdown(f"""
    <div class="{cls} pipeline-step">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <div style="width:28px;height:28px;border-radius:6px;background:{icon_color}20;border:1px solid {icon_color};
                    display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;
                    color:{icon_color};font-family:'JetBrains Mono',monospace;flex-shrink:0;">{icon}</div>
        <div style="flex:1;">
          <div class="step-title">{title}</div>
          <div class="step-file">{file}</div>
        </div>
        <span style="background:{bg};color:{bc};border:1px solid {bb};border-radius:4px;
                     font-size:10px;font-weight:700;padding:2px 8px;font-family:'JetBrains Mono',monospace;">{status}</span>
      </div>
      {"<div class='step-detail'>"+detail+"</div>" if detail else ""}
    </div>""", unsafe_allow_html=True)

def alert_box(atype, title, detail, time):
    cls = {"CRITICAL": "alert-critical", "WARNING": "alert-warning", "INFO": "alert-info"}[atype]
    st.markdown(f"""
    <div class="{cls}">
      <div class="alert-title">{title}</div>
      <div class="alert-detail">{detail}</div>
      <div class="alert-time">{time}</div>
    </div>""", unsafe_allow_html=True)

def risk_color(risk):
    return {"HIGH": "#FF4757", "MEDIUM": "#FFD166", "LOW": "#00E5A0"}.get(risk, "#7A8FA8")

# ─── Sidebar 
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:8px 0 20px;">
      <div style="width:38px;height:38px;border-radius:10px;background:linear-gradient(135deg,#00D4FF,#0099CC);
                  display:flex;align-items:center;justify-content:center;font-size:20px;">⛓</div>
      <div>
        <div style="font-size:15px;font-weight:800;color:#E8F4FD;">Supply Chain AI</div>
        <div style="font-size:10px;color:#7A8FA8;font-family:'JetBrains Mono',monospace;">Superstore Dataset</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;background:#0D2D1A;
                border:1px solid rgba(0,229,160,0.3);border-radius:8px;margin-bottom:20px;">
      <div style="width:8px;height:8px;border-radius:50%;background:#00E5A0;"></div>
      <span style="font-size:11px;color:#00E5A0;font-weight:600;">Pipeline Active</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Dataset**", unsafe_allow_html=False)
    st.markdown("""
    <div style="font-size:11px;color:#7A8FA8;line-height:1.8;">
    📂 51,290 rows · 35 columns<br>
    🗓 2011 – 2015<br>
    🌍 7 markets<br>
    📦 6,284 segments
    </div><br>""", unsafe_allow_html=True)

    st.markdown("**Pipeline Status**", unsafe_allow_html=False)
    for s in [("Step 3", "Preprocessing", "DONE"), ("Step 4", "EDA Analysis", "DONE"),
              ("Step 5", "Forecasting", "DONE"), ("Step 6", "Inventory Opt.", "ACTIVE"),
              ("Step 7", "Export Final", "PENDING"), ("Step 8", "Power BI", "PENDING")]:
        col_map = {"DONE": "#00E5A0", "ACTIVE": "#00D4FF", "PENDING": "#3D5068"}
        c = col_map[s[2]]
        st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #1e2d45;">
          <span style="font-size:11px;color:#7A8FA8;">{s[0]} · {s[1]}</span>
          <span style="font-size:10px;font-weight:700;color:{c};font-family:'JetBrains Mono',monospace;">{s[2]}</span>
        </div>""", unsafe_allow_html=True)

# ─── Header 
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:16px 0;border-bottom:1px solid #1e2d45;margin-bottom:24px;">
  <div>
    <div style="font-size:22px;font-weight:800;color:#E8F4FD;letter-spacing:-0.02em;">
      Supply Chain AI System
    </div>
    <div style="font-size:11px;color:#7A8FA8;font-family:'JetBrains Mono',monospace;margin-top:3px;">
      Superstore Dataset · 51,290 rows · Steps 3–6 Complete
    </div>
  </div>
  <div style="font-size:11px;color:#3D5068;font-family:'JetBrains Mono',monospace;">
    step6 → inventory_optimization
  </div>
</div>""", unsafe_allow_html=True)

# ─── Main Tabs 
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "EDA Insights", "Forecasting", "Inventory", "Export"])

# 
# TAB 1 — OVERVIEW
# 
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows Cleaned", "51,290", "35 cols · 0 nulls after imputation")
    c2.metric("Best Model MAE", "3.41 units", "XGBoost / GBM — 200 estimators")
    c3.metric("High-Risk Segments", "847", "Of 6,284 total inventory segments")
    c4.metric("Forecast Horizon", "+3 Months", "3-month demand rollout ready")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Pipeline Progress", "Data science workflow — step3 through step8")

    cols = st.columns(3)
    steps = [
        (3, "Load & Clean Data",       "step3_preprocessing.py",         "DONE",    "Rows: 51,290 · 0 nulls after imputation"),
        (4, "EDA Analysis",            "step4_eda.py",                   "DONE",    "2 charts saved · Business insights complete"),
        (5, "Demand Forecasting",      "step5_forecasting.py",           "DONE",    "Best MAE: 3.41 units · model.pkl saved"),
        (6, "Inventory Optimization",  "step6_inventory_optimization.py","ACTIVE",  "6,284 segments · ADD + Safety Stock + ROP"),
        (7, "Export Final Dataset",    "step7_export_final.py",          "PENDING", "28 columns · final_supply_chain_data.csv"),
        (8, "Power BI Dashboard",      "step8_powerbi_guide.md",         "PENDING", "Guide + visuals ready after step7"),
    ]
    for i, step in enumerate(steps):
        with cols[i % 3]:
            pipeline_step(*step)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Model Comparison", "MAE (lower is better) — trained on 80/20 split")
        models = {"Linear Regression": 12.84, "Random Forest (200 est.)": 6.17, "XGBoost / GBM ← Best": 3.41}
        model_colors = {"Linear Regression": "#3D5068", "Random Forest (200 est.)": "#00D4FF", "XGBoost / GBM ← Best": "#00E5A0"}
        fig_model = go.Figure()
        for name, val in models.items():
            fig_model.add_trace(go.Bar(
                x=[val], y=[name], orientation="h",
                marker_color=model_colors[name],
                marker_line_width=0,
                text=f"  MAE: {val}", textposition="outside",
                textfont=dict(color="#7A8FA8", size=11, family="JetBrains Mono"),
            ))
        fig_model.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                                height=200, xaxis_title="MAE (units)",
                                xaxis=dict(range=[0, 16], **GRID_AXIS),
                                yaxis=dict(gridcolor="rgba(0,0,0,0)", zerolinecolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_model, use_container_width=True)
        st.markdown("""<div style="padding:10px 14px;background:rgba(0,229,160,0.08);border-radius:8px;border:1px solid rgba(0,229,160,0.2);">
          <div style="font-size:11px;color:#00E5A0;font-weight:700;">Best Model: XGBoost / GradientBoosting</div>
          <div style="font-size:11px;color:#7A8FA8;margin-top:3px;">MAE = 3.41 · RMSE = 5.89 · Saved to models/model.pkl</div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        section_header("Inventory Risk Distribution", "Across 6,284 category × sub-cat × region × market segments")
        fig_risk = go.Figure(go.Pie(
            labels=["LOW — Stock adequate", "MEDIUM — Below reorder point", "HIGH — Below safety stock"],
            values=[3299, 2138, 847],
            hole=0.65,
            marker_colors=["#00E5A0", "#FFD166", "#FF4757"],
            textinfo="none",
        ))
        fig_risk.add_annotation(text="6,284<br>segments", x=0.5, y=0.5, showarrow=False,
                                font=dict(size=14, color="#E8F4FD", family="JetBrains Mono"))
        fig_risk.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=True,
                               legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=10, color="#7A8FA8")))
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Active Alerts", "Generated from step6 inventory optimization output")
    ac1, ac2 = st.columns(2)
    with ac1:
        alert_box("CRITICAL", "847 segments below safety stock threshold",
                  "Immediate reorder required. Inventory level < Safety Stock (Z=1.65 × StdDev × √LeadTime). High stockout risk for Bookcases/Central, Chairs/West, Paper/LATAM.",
                  "step6 · risk_flag = HIGH")
        alert_box("CRITICAL", "Stock cover below lead time in 847 segments",
                  "Days-of-stock < avg_lead_time days. You'll run out before the next shipment arrives.",
                  "Formula: days_of_stock = avg_inventory / ADD")
    with ac2:
        alert_box("WARNING", "Festival months incoming — demand spike expected",
                  "Months Jan, Aug, Oct, Nov, Dec flagged as festival months. Historical lift = +14% avg across categories.",
                  "is_festival_month feature · step3 engineering")
        alert_box("INFO", "forecast_data.csv ready — 3-month rollout complete",
                  "XGBoost model predicted demand for all 6,284 segments across next 3 months.",
                  "Saved → data/forecast_data.csv · step5")

# 
# TAB 2 — EDA INSIGHTS
# 
with tab2:
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Total Revenue", "$12.6M", "Across all markets, 2011–2015")
    e2.metric("Total Units Sold", "178,312", "All categories combined")
    e3.metric("Avg Profit Margin", "11.4%", "Technology leads at 17.2%")
    e4.metric("Markets Covered", "7", "US, EU, APAC, LATAM, Africa, MEA, Canada")

    st.markdown("<br>", unsafe_allow_html=True)
    el, er = st.columns([1.4, 1])

    with el:
        section_header("Forecasted Demand — Next 3 Months", "By category: Technology · Furniture · Office Supplies")
        df_fc = pd.DataFrame(FORECAST_DATA)
        fig_bar = go.Figure()
        for cat, color in [("Technology", "#00D4FF"), ("Furniture", "#00E5A0"), ("Office Supplies", "#FF6B35")]:
            fig_bar.add_trace(go.Bar(name=cat, x=df_fc["Month"], y=df_fc[cat],
                                     marker_color=color, marker_line_width=0, opacity=0.85))
        fig_bar.update_layout(**PLOTLY_LAYOUT, barmode="group", height=260,
                              xaxis=GRID_AXIS, yaxis=GRID_AXIS,
                              legend=dict(orientation="h", y=-0.15, font=dict(size=10, color="#7A8FA8")))
        st.plotly_chart(fig_bar, use_container_width=True)

    with er:
        section_header("Revenue Share by Category", "step4_eda.py — analysis")
        for cat, pct, rev, color in [
            ("Technology", 37.4, "$4.7M", "#00D4FF"),
            ("Furniture", 32.1, "$4.0M", "#FF6B35"),
            ("Office Supplies", 30.5, "$3.9M", "#00E5A0"),
        ]:
            st.markdown(f"""
            <div style="margin-bottom:14px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:12px;color:#E8F4FD;font-weight:600;">{cat}</span>
                <span style="font-size:12px;color:{color};font-family:'JetBrains Mono',monospace;font-weight:700;">{rev} ({pct}%)</span>
              </div>
              <div style="height:8px;background:#1e2d45;border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='border-top:1px solid #1e2d45;padding-top:14px;'>", unsafe_allow_html=True)
        section_header("Profit Margin by Category", "avg_profit / revenue")
        for cat, margin, color in [("Technology", 17.2, "#00D4FF"), ("Office Supplies", 13.7, "#00E5A0"), ("Furniture", 3.4, "#FF6B35")]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
              <span style="font-size:12px;color:#7A8FA8;width:120px;">{cat}</span>
              <div style="flex:1;height:6px;background:#1e2d45;border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{margin*5}%;background:{color};border-radius:99px;"></div>
              </div>
              <span style="font-size:12px;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;width:40px;text-align:right;">{margin}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fl, fr = st.columns(2)

    with fl:
        section_header("Festival Month Demand Lift", "Festival vs Normal month avg units sold")
        for cat, lift in [("Technology", 18.4), ("Office Supplies", 14.1), ("Furniture", 9.7)]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
              <span style="background:#1A0D2D;color:#A855F7;border:1px solid rgba(168,85,247,0.3);
                           border-radius:4px;font-size:10px;font-weight:700;padding:2px 8px;
                           font-family:'JetBrains Mono',monospace;">FESTIVAL</span>
              <span style="font-size:12px;color:#E8F4FD;flex:1;">{cat}</span>
              <span style="font-size:14px;font-weight:800;color:#A855F7;font-family:'JetBrains Mono',monospace;">+{lift}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="margin-top:12px;padding:10px 12px;background:rgba(168,85,247,0.08);
                        border-radius:8px;border:1px solid rgba(168,85,247,0.2);">
          <div style="font-size:11px;color:#7A8FA8;">
            Festival months: <span style="color:#A855F7;font-weight:700;">Jan · Aug · Oct · Nov · Dec</span>
            — inventory buffer recommended
          </div>
        </div>""", unsafe_allow_html=True)

    with fr:
        section_header("Top 5 Profitable Sub-Categories", "Total profit · step4 business insights")
        for rank, sub, profit in [(1, "Copiers", "$258,568"), (2, "Phones", "$216,242"),
                                   (3, "Accessories", "$178,093"), (4, "Paper", "$134,166"), (5, "Binders", "$121,348")]:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;padding:8px 10px;
                        background:#0A0E1A;border-radius:8px;">
              <span style="font-size:11px;color:#3D5068;font-family:'JetBrains Mono',monospace;width:20px;">#{rank}</span>
              <span style="flex:1;font-size:12px;color:#E8F4FD;font-weight:600;">{sub}</span>
              <span style="font-size:13px;color:#00E5A0;font-weight:800;font-family:'JetBrains Mono',monospace;">{profit}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="margin-top:8px;padding:8px 10px;background:rgba(255,71,87,0.05);
                        border-radius:8px;border:1px solid rgba(255,71,87,0.15);">
          <div style="font-size:11px;color:#FF4757;font-weight:700;margin-bottom:4px;">Loss-making: Tables (−$17,725) · Bookcases (−$3,473)</div>
          <div style="font-size:10px;color:#7A8FA8;">High discount rates driving negative margins</div>
        </div>""", unsafe_allow_html=True)

# 
# TAB 3 — FORECASTING
# 
with tab3:
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Train Rows", "41,032", "80% of monthly aggregated data")
    f2.metric("Test Rows", "10,258", "20% held out for evaluation")
    f3.metric("Features Used", "13", "Encoded cats + time + price signals")
    f4.metric("Forecast Rows", "18,852", "6,284 segments × 3 months")

    st.markdown("<br>", unsafe_allow_html=True)
    fl2, fr2 = st.columns(2)

    with fl2:
        section_header("Feature Importance (Random Forest)", "Which features drive demand prediction most")
        features = [
            ("avg_price", 0.312, "#00D4FF"), ("sub_category_enc", 0.241, "#00E5A0"),
            ("month", 0.183, "#A855F7"), ("is_festival_month", 0.097, "#FF6B35"),
            ("category_enc", 0.071, "#FFD166"), ("avg_lead_time", 0.056, "#7A8FA8"),
            ("region_enc", 0.040, "#3D5068"),
        ]
        for name, val, color in features:
            st.markdown(f"""
            <div style="margin-bottom:11px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:11px;color:#E8F4FD;font-family:'JetBrains Mono',monospace;">{name}</span>
                <span style="font-size:11px;color:{color};font-weight:700;font-family:'JetBrains Mono',monospace;">{val*100:.1f}%</span>
              </div>
              <div style="height:5px;background:#1e2d45;border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{val*100*3.2}%;background:{color};border-radius:99px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

    with fr2:
        section_header("3-Month Forecast by Category", "Forecasted units across all regions & markets")
        df_fc2 = pd.DataFrame(FORECAST_DATA)
        fig_fc2 = go.Figure()
        for cat, color in [("Technology", "#00D4FF"), ("Furniture", "#00E5A0"), ("Office Supplies", "#FF6B35")]:
            fig_fc2.add_trace(go.Bar(name=cat, x=df_fc2["Month"], y=df_fc2[cat],
                                      marker_color=color, marker_line_width=0, opacity=0.85))
        fig_fc2.update_layout(**PLOTLY_LAYOUT, barmode="group", height=200,
                               xaxis=GRID_AXIS, yaxis=GRID_AXIS,
                               legend=dict(orientation="h", y=-0.2, font=dict(size=10, color="#7A8FA8")))
        st.plotly_chart(fig_fc2, use_container_width=True)

        months = [
            ("Month 1", 4820, 1980, 3410, False),
            ("Month 2", 5140, 2110, 3650, False),
            ("Month 3", 6090, 2640, 4320, True),
        ]
        mcols = st.columns(3)
        for i, (mo, tech, furn, off, fest) in enumerate(months):
            with mcols[i]:
                fest_badge = '<span style="background:#1A0D2D;color:#A855F7;border:1px solid rgba(168,85,247,0.3);border-radius:4px;font-size:9px;font-weight:700;padding:1px 6px;font-family:\'JetBrains Mono\',monospace;">FEST</span>' if fest else ""
                st.markdown(f"""
                <div style="background:#0A0E1A;border-radius:8px;padding:10px 12px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:11px;color:#7A8FA8;">{mo}</span>{fest_badge}
                  </div>
                  <div style="font-size:10px;color:#3D5068;margin-bottom:2px;">Technology</div>
                  <div style="font-size:13px;font-weight:700;color:#00D4FF;font-family:'JetBrains Mono',monospace;">{tech:,}</div>
                  <div style="font-size:10px;color:#3D5068;margin-top:4px;margin-bottom:2px;">Office Supplies</div>
                  <div style="font-size:13px;font-weight:700;color:#FF6B35;font-family:'JetBrains Mono',monospace;">{off:,}</div>
                  <div style="font-size:10px;color:#3D5068;margin-top:4px;margin-bottom:2px;">Furniture</div>
                  <div style="font-size:13px;font-weight:700;color:#00E5A0;font-family:'JetBrains Mono',monospace;">{furn:,}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Training Pipeline — step5_forecasting.py", "Full model selection workflow")
    train_steps = [
        ("5.1", "Monthly Aggregation", "Group by cat, sub-cat, region, market, year, month · 8 agg columns", "#00D4FF"),
        ("5.2", "Label Encoding", "4 categorical cols → _enc suffix · enc_map stored in model.pkl", "#A855F7"),
        ("5.3", "80/20 Train Split", "random_state=42 · 41,032 train / 10,258 test rows", "#FFD166"),
        ("5.4", "3 Models Trained", "LinearRegression · RandomForest(200) · XGBoost/GBM(200, lr=0.05)", "#FF6B35"),
        ("5.8", "Forecast Generated", "All segments × 3 months → 18,852 rows → forecast_data.csv", "#00E5A0"),
    ]
    tcols = st.columns(5)
    for i, (step, label, detail, color) in enumerate(train_steps):
        with tcols[i]:
            st.markdown(f"""
            <div style="background:#0A0E1A;border-radius:8px;padding:14px;">
              <div style="font-size:10px;color:{color};font-family:'JetBrains Mono',monospace;font-weight:700;margin-bottom:6px;">STEP {step}</div>
              <div style="font-size:12px;font-weight:700;color:#E8F4FD;margin-bottom:6px;">{label}</div>
              <div style="font-size:10px;color:#7A8FA8;line-height:1.5;">{detail}</div>
            </div>""", unsafe_allow_html=True)

# 
# TAB 4 — INVENTORY
# 
with tab4:
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Total Segments", "6,284", "cat × sub-cat × region × market")
    i2.metric("Avg Lead Time", "3.8 days", "Used in ROP & Safety Stock")
    i3.metric("Service Level Z", "1.65", "95% service level target")
    i4.metric("Critical Coverage", "847", "Segments below lead-time coverage")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Inventory Formulas", "Core calculations used in step6_inventory_optimization.py")
    fc1, fc2, fc3 = st.columns(3)
    formulas = [
        ("Average Daily Demand (ADD)", "ADD = Total Units Sold ÷ Total Days",
         "Baseline consumption rate per segment. Used in both ROP and days-of-stock calculations.", "#00D4FF"),
        ("Safety Stock", "SS = 1.65 × StdDev × √Lead Time",
         "Buffer stock for 95% service level. Z=1.65 protects against demand variability during replenishment.", "#FFD166"),
        ("Reorder Point (ROP)", "ROP = (ADD × Lead Time) + Safety Stock",
         "The inventory level at which you must place a new order to avoid stockout.", "#00E5A0"),
    ]
    for col, (title, formula, detail, color) in zip([fc1, fc2, fc3], formulas):
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid {color}30;border-radius:12px;padding:18px 20px;">
              <div style="font-size:11px;color:{color};font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;">{title}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#E8F4FD;
                          background:#0A0E1A;padding:10px 14px;border-radius:8px;margin-bottom:10px;border:1px solid {color}20;">{formula}</div>
              <div style="font-size:11px;color:#7A8FA8;line-height:1.6;">{detail}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Inventory Segments", "ADD · Safety Stock · ROP · Risk Flag · Demand Trend")

    col_filter = st.columns([1, 1, 1, 1, 3])
    risk_filter = col_filter[0].selectbox("Filter by Risk", ["ALL", "HIGH", "MEDIUM", "LOW"], label_visibility="collapsed")

    df_display = SEGMENTS if risk_filter == "ALL" else SEGMENTS[SEGMENTS["Risk"] == risk_filter]

    def style_risk(val):
        colors_map = {"HIGH": "color: #FF4757; font-weight: bold",
                      "MEDIUM": "color: #FFD166; font-weight: bold",
                      "LOW": "color: #00E5A0; font-weight: bold"}
        return colors_map.get(val, "")

    def style_inv(val):
        return "color: #FF4757; font-weight: bold" if val < 50 else "color: #E8F4FD"

    def style_trend(val):
        return f"color: {'#00E5A0' if val > 0 else '#FF4757'}; font-weight: bold"

    styled_df = df_display.style \
        .applymap(style_risk, subset=["Risk"]) \
        .applymap(style_inv, subset=["Avg Inventory"]) \
        .applymap(style_trend, subset=["Demand Trend %"]) \
        .format({"ADD/day": "{:.2f}", "Safety Stock": "{:.1f}", "Reorder Point": "{:.1f}", "Demand Trend %": "{:+d}%"}) \
        .set_properties(**{"background-color": "#111827", "color": "#E8F4FD",
                           "border": "1px solid #1e2d45", "font-size": "12px"}) \
        .set_table_styles([
            {"selector": "th", "props": [("background-color", "#0A0E1A"), ("color", "#7A8FA8"),
                                          ("font-size", "10px"), ("text-transform", "uppercase"),
                                          ("letter-spacing", "0.06em"), ("border", "1px solid #1e2d45")]},
        ])
    st.dataframe(styled_df, use_container_width=True, height=340)

    st.markdown(f"""<div style="font-size:11px;color:#3D5068;margin-top:8px;">
      Showing {len(df_display)} of 6,284 segments · Avg Inventory &lt; Safety Stock = stockout risk
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Risk Distribution by Region", "Heatmap of inventory risk across geographies")
    pivot = SEGMENTS.groupby(["Region", "Risk"]).size().unstack(fill_value=0)
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0, "#0D2D1A"], [0.5, "#2D1F00"], [1, "#2D0D0D"]],
        text=pivot.values, texttemplate="%{text}", textfont=dict(color="#E8F4FD", size=12),
        showscale=False,
    ))
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=200, xaxis=GRID_AXIS, yaxis=GRID_AXIS)
    st.plotly_chart(fig_heat, use_container_width=True)

# 
# TAB 5 — EXPORT
# 
with tab5:
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Final Columns", "28", "All key outputs merged into one file")
    x2.metric("Final Rows", "51,290", "final_supply_chain_data.csv")
    x3.metric("Low Stock Alerts", "1,241", "inventory_level < reorder_point")
    x4.metric("Overstock Alerts", "388", "inventory_level > ROP × 3")

    st.markdown("<br>", unsafe_allow_html=True)
    xl, xr = st.columns(2)

    with xl:
        section_header("Merge Logic — step7_export_final.py", "Three CSVs joined by category × sub_category × region × market")
        for file, role, cols, color in [
            ("cleaned_data.csv", "Base historical data", "35 columns · 51,290 rows", "#00D4FF"),
            ("inventory_stats.csv", "ADD, Safety Stock, ROP, Risk flags", "6,284 segment rows", "#FFD166"),
            ("forecast_data.csv", "3-month demand predictions", "18,852 rows → median per segment", "#A855F7"),
        ]:
            st.markdown(f"""
            <div style="margin-bottom:10px;padding:12px 14px;background:#0A0E1A;border-radius:8px;
                        border:1px solid {color}20;display:flex;gap:12px;">
              <div style="width:3px;border-radius:99px;background:{color};flex-shrink:0;"></div>
              <div>
                <div style="font-size:12px;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{file}</div>
                <div style="font-size:11px;color:#E8F4FD;margin-top:3px;">{role}</div>
                <div style="font-size:10px;color:#3D5068;margin-top:2px;">{cols}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div style="margin-top:12px;padding:12px 14px;background:rgba(0,229,160,0.08);
                        border-radius:8px;border:1px solid rgba(0,229,160,0.2);">
          <div style="font-size:11px;color:#00E5A0;font-weight:700;">OUTPUT → final_supply_chain_data.csv</div>
          <div style="font-size:10px;color:#7A8FA8;margin-top:3px;">28 columns · Power BI ready · low_stock_alert + overstock_alert + demand_trend fields included</div>
        </div>""", unsafe_allow_html=True)

    with xr:
        section_header("Final 28 Columns", "Complete schema for Power BI import")
        schema = [
            ("category", "Identifiers"), ("sub_category", "Identifiers"),
            ("region", "Identifiers"), ("market", "Identifiers"),
            ("year", "Identifiers"), ("month", "Identifiers"),
            ("units_sold", "Historical Demand"), ("revenue", "Historical Demand"),
            ("profit", "Historical Demand"), ("avg_unit_price", "Historical Demand"),
            ("avg_discount", "Historical Demand"), ("avg_lead_time", "Historical Demand"),
            ("n_orders", "Historical Demand"), ("inventory_level", "Inventory"),
            ("inventory_value", "Inventory"), ("avg_daily_demand", "Optimization"),
            ("safety_stock", "Optimization"), ("reorder_point", "Optimization"),
            ("days_of_stock", "Optimization"), ("risk_flag", "Optimization"),
            ("stock_cover_flag", "Optimization"), ("forecasted_demand_next3m", "Forecast"),
            ("demand_forecast_delta", "Forecast"), ("demand_trend", "Forecast"),
            ("low_stock_alert", "Alerts"), ("overstock_alert", "Alerts"),
            ("profit_margin_pct", "Profitability"),
        ]
        group_colors = {
            "Identifiers": "#3D5068", "Historical Demand": "#00D4FF",
            "Inventory": "#FF6B35", "Optimization": "#00E5A0",
            "Forecast": "#A855F7", "Alerts": "#FF4757", "Profitability": "#FFD166",
        }
        cols_per_row = 2
        rows = [schema[i:i+cols_per_row] for i in range(0, len(schema), cols_per_row)]
        table_html = "<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;'>"
        for col, group in schema:
            c = group_colors.get(group, "#3D5068")
            table_html += f"""<div style="display:flex;gap:6px;align-items:center;padding:3px 0;">
              <div style="width:6px;height:6px;border-radius:50%;background:{c};flex-shrink:0;"></div>
              <span style="font-size:10px;color:#7A8FA8;font-family:'JetBrains Mono',monospace;">{col}</span>
            </div>"""
        table_html += "</div>"
        st.markdown(table_html, unsafe_allow_html=True)

        legend_html = "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-top:14px;'>"
        for group, color in group_colors.items():
            legend_html += f"""<div style="display:flex;align-items:center;gap:5px;">
              <div style="width:8px;height:8px;border-radius:50%;background:{color};"></div>
              <span style="font-size:10px;color:#7A8FA8;">{group}</span>
            </div>"""
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Demand Trend Summary", "demand_trend column — based on forecasted_demand_next3m vs units_sold")
    tc1, tc2, tc3 = st.columns(3)
    trends = [
        ("INCREASING", "28,412", "55.4%", "forecasted_demand_next3m > units_sold · Buffer stock up recommended", "#00E5A0"),
        ("DECREASING", "16,819", "32.8%", "forecasted_demand_next3m < units_sold · Review overstock risk", "#FF4757"),
        ("STABLE", "6,059", "11.8%", "forecast delta ≈ 0 · Maintain current inventory policy", "#7A8FA8"),
    ]
    for col, (label, count, pct, detail, color) in zip([tc1, tc2, tc3], trends):
        with col:
            st.markdown(f"""
            <div style="background:#0A0E1A;border-radius:10px;padding:16px 18px;border:1px solid {color}20;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="background:{color}15;color:{color};border:1px solid {color}40;border-radius:4px;
                             font-size:10px;font-weight:700;padding:2px 8px;font-family:'JetBrains Mono',monospace;">{label}</span>
                <span style="font-size:11px;color:{color};font-family:'JetBrains Mono',monospace;">{pct}</span>
              </div>
              <div style="font-size:22px;font-weight:800;color:{color};font-family:'JetBrains Mono',monospace;margin-bottom:6px;">{count}</div>
              <div style="font-size:10px;color:#7A8FA8;line-height:1.5;">{detail}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="padding:16px 20px;background:#111827;border:1px solid #1e2d45;border-radius:12px;text-align:center;">
      <div style="font-size:12px;color:#7A8FA8;margin-bottom:8px;">Pipeline Complete — Ready for Power BI</div>
      <div style="font-size:11px;color:#3D5068;font-family:'JetBrains Mono',monospace;">
        python step7_export_final.py → final_supply_chain_data.csv → Import to Power BI → step8_powerbi_guide.md
      </div>
    </div>""", unsafe_allow_html=True)