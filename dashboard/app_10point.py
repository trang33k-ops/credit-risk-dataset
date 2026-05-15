# ============================================================
# CREDIT RISK INTELLIGENCE DASHBOARD — CLASSIC CINEMATIC EDITION
# FinTech Credit Scoring + Explainable AI + Scenario Simulation
# Author: Generated for Nguyen Thu Trang's Big Project
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence | 10-Point Edition",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
VISUAL_DIR = BASE_DIR / "visuals"
EXPLAIN_DIR = BASE_DIR / "explainability"
PIPELINE_DIR = BASE_DIR / "pipeline"

# ------------------------------------------------------------
# 1. CINEMATIC CLASSIC CSS
# ------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #18130b;
  --espresso: #2b2118;
  --brown: #6b4f2a;
  --gold: #c9a24e;
  --cream: #fff7e6;
  --paper: #fbf3df;
  --emerald: #0f7b55;
  --amber: #b7791f;
  --ruby: #a52a2a;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 12% 5%, rgba(201,162,78,.22), transparent 28%),
    radial-gradient(circle at 85% 12%, rgba(111,78,40,.18), transparent 32%),
    linear-gradient(135deg, #21170f 0%, #5a3d1e 26%, #f8edd3 27%, #fffaf0 100%);
}

.block-container { padding-top: 1.05rem; padding-bottom: 2.2rem; max-width: 1480px; }

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(31,22,13,.98), rgba(65,42,18,.97)),
    url("https://www.transparenttextures.com/patterns/black-paper.png");
  border-right: 1px solid rgba(201,162,78,.35);
}
[data-testid="stSidebar"] * { color: #fff4d2 !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  border: 1px solid rgba(201,162,78,.20); border-radius: 14px; padding: 7px 9px; margin: 4px 0;
  background: rgba(255,255,255,.04);
}

.hero {
  position: relative; overflow: hidden; padding: 30px 34px; border-radius: 30px;
  background:
    linear-gradient(135deg, rgba(28,19,10,.97), rgba(95,62,25,.92)),
    radial-gradient(circle at top right, rgba(255,215,128,.22), transparent 35%);
  color: #fff4d2; border: 1px solid rgba(255,221,128,.35);
  box-shadow: 0 24px 70px rgba(36,24,10,.35); margin-bottom: 18px;
}
.hero::after {
  content:""; position:absolute; inset:-20%; background: linear-gradient(110deg, transparent 30%, rgba(255,255,255,.08) 48%, transparent 62%);
  transform: rotate(4deg); animation: shine 7s infinite linear;
}
@keyframes shine { from { transform: translateX(-70%) rotate(4deg);} to { transform: translateX(70%) rotate(4deg);} }
.hero h1 { font-family:'Cinzel', serif; font-size: 42px; letter-spacing: .5px; margin: 0 0 8px 0; position:relative; z-index:1; }
.hero p { font-size: 15px; color: #f7e8be; line-height: 1.75; max-width: 980px; margin:0; position:relative; z-index:1; }
.ribbon { display:inline-flex; gap:8px; align-items:center; padding:8px 12px; border-radius:999px; background:rgba(255,244,210,.10); border:1px solid rgba(255,244,210,.25); font-weight:800; color:#ffe9a3; margin-bottom:12px; position:relative; z-index:1; }

.card {
  background: rgba(255, 250, 240, .88); border: 1px solid rgba(107,79,42,.18); border-radius: 24px; padding: 20px;
  box-shadow: 0 14px 40px rgba(60,38,14,.13); backdrop-filter: blur(12px); margin-bottom: 16px;
}
.card.dark {
  background: linear-gradient(135deg, rgba(35,24,12,.96), rgba(74,49,22,.94)); color:#fff4d2; border:1px solid rgba(201,162,78,.35);
}
.kpi {
  background: linear-gradient(180deg, rgba(255,250,240,.98), rgba(244,231,199,.95));
  border: 1px solid rgba(107,79,42,.20); border-radius: 24px; padding: 18px 18px 15px 18px;
  box-shadow: 0 12px 28px rgba(70,45,18,.12); min-height: 132px;
}
.kpi .label { color: #6b4f2a; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight:800; }
.kpi .value { color:#1b1309; font-family:'Cinzel', serif; font-size: 30px; font-weight:800; margin-top:6px; }
.kpi .caption { color:#7d6848; font-size: 13px; margin-top:5px; line-height:1.45; }
.section-title { font-family:'Cinzel', serif; font-size: 26px; color:#21170f; font-weight:800; margin: 16px 0 10px; }
.small-note { color:#6b4f2a; font-size:13px; line-height:1.65; }
.badge { display:inline-block; padding:7px 12px; border-radius:999px; font-weight:800; font-size:12px; margin:3px 5px 3px 0; border:1px solid rgba(0,0,0,.06); }
.green { background:#def7ec; color:#075f43; } .yellow { background:#fff2bd; color:#7a4d00; } .red { background:#ffe1de; color:#8f1d1d; } .blue { background:#dfeeff; color:#17457c; } .gold { background:#f7e6b7; color:#61420b; }
.story-step { padding: 13px 15px; border-left: 4px solid #c9a24e; background: rgba(255,244,210,.65); border-radius: 14px; margin: 8px 0; }
.stButton > button, .stDownloadButton > button {
  border-radius: 15px !important; border: 1px solid rgba(201,162,78,.45) !important;
  background: linear-gradient(135deg, #2b2118, #8a642b) !important; color: #fff4d2 !important; font-weight: 800 !important;
  box-shadow: 0 10px 24px rgba(75,48,17,.25) !important;
}
[data-testid="stMetric"] { background: rgba(255,250,240,.86); border:1px solid rgba(107,79,42,.18); padding: 15px; border-radius: 18px; }
hr { border: none; border-top: 1px solid rgba(107,79,42,.16); }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 2. LOADERS
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_pickle(path: Path):
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, object]:
    names = ["best_model", "best_model_name", "model_columns", "best_threshold"]
    out = {}
    for name in names:
        obj = load_pickle(MODEL_DIR / f"{name}.pkl")
        if obj is not None:
            out[name] = obj
    pipe = load_pickle(PIPELINE_DIR / "pipeline.pkl")
    if pipe is not None:
        out["pipeline"] = pipe
    return out

# Data outputs from Person 1/2/3
data_clean = read_csv(DATA_DIR / "data_clean.csv")
data_feature = read_csv(DATA_DIR / "data_feature.csv")
credit_scored = read_csv(DATA_DIR / "credit_scored_data.csv")
pca_features = read_csv(DATA_DIR / "pca_features.csv")

person1_summary = read_csv(REPORT_DIR / "person1_summary_report.csv")
person2_summary = read_csv(REPORT_DIR / "person2_summary_report.csv")
model_comparison = read_csv(REPORT_DIR / "model_comparison_table.csv")
threshold_opt = read_csv(REPORT_DIR / "threshold_optimization.csv")
feature_importance = read_csv(REPORT_DIR / "feature_importance.csv")
shap_importance = read_csv(REPORT_DIR / "shap_feature_importance.csv")
shap_waterfall = read_csv(REPORT_DIR / "shap_waterfall_sample_data.csv")
eda_grade = read_csv(REPORT_DIR / "eda_default_by_grade.csv")
eda_intent = read_csv(REPORT_DIR / "eda_default_by_intent.csv")
eda_country = read_csv(REPORT_DIR / "eda_default_by_country.csv")
pca_var = read_csv(REPORT_DIR / "pca_explained_variance.csv")
audit_before = read_csv(REPORT_DIR / "data_audit_before_cleaning.csv")
audit_after = read_csv(REPORT_DIR / "data_audit_after_cleaning.csv")
feature_dict = read_csv(REPORT_DIR / "feature_dictionary.csv")

artifacts = load_artifacts()
df = credit_scored.copy() if not credit_scored.empty else data_clean.copy()

# ------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------
def fmt_int(x) -> str:
    try: return f"{int(round(float(x))):,}"
    except Exception: return "—"

def fmt_money(x) -> str:
    try: return f"${float(x):,.0f}"
    except Exception: return "—"

def fmt_pct(x) -> str:
    try: return f"{float(x) * 100:.2f}%"
    except Exception: return "—"

def kpi(label: str, value: str, caption: str = ""):
    st.markdown(f"""
    <div class="kpi"><div class="label">{label}</div><div class="value">{value}</div><div class="caption">{caption}</div></div>
    """, unsafe_allow_html=True)

def section(title: str, note: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="small-note">{note}</div>', unsafe_allow_html=True)

def badge(text: str, cls: str = "gold") -> str:
    return f'<span class="badge {cls}">{text}</span>'

def plot_empty(message="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=18))
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def classic_layout(fig, title: Optional[str] = None, height: int = 430):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        height=height,
        font=dict(family="Inter", color="#21170f"),
        paper_bgcolor="rgba(255,250,240,0)",
        plot_bgcolor="rgba(255,250,240,.68)",
        margin=dict(l=20, r=20, t=55, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(107,79,42,.15)", zerolinecolor="rgba(107,79,42,.25)")
    fig.update_yaxes(gridcolor="rgba(107,79,42,.15)", zerolinecolor="rgba(107,79,42,.25)")
    return fig

def calculate_credit_score(prob_default: float) -> int:
    score = 850 - prob_default * 550
    return int(round(max(300, min(850, score))))

def classify_risk_level(score: int) -> str:
    if score >= 700: return "LOW"
    if score >= 600: return "MEDIUM"
    return "HIGH"

def make_decision(prob_default: float, score: int) -> str:
    if prob_default < 0.25 and score >= 720: return "APPROVE"
    if prob_default < 0.60 and score >= 500: return "REVIEW"
    return "REJECT"

def preprocess_input(raw: dict) -> pd.DataFrame:
    cols = artifacts.get("model_columns", [])
    row = {c: 0 for c in cols}
    for k, v in raw.items():
        if k in row:
            row[k] = v
        encoded = f"{k}_{v}"
        if encoded in row:
            row[encoded] = 1
    return pd.DataFrame([row]).reindex(columns=cols, fill_value=0)

def predict_one(raw: dict) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    model = artifacts.get("best_model")
    cols = artifacts.get("model_columns")
    if model is None or cols is None:
        return None, None
    X = preprocess_input(raw)
    prob = float(model.predict_proba(X)[0][1])
    threshold = float(artifacts.get("best_threshold", 0.5))
    score = calculate_credit_score(prob)
    return {
        "probability_of_default": round(prob, 4),
        "prediction": int(prob >= threshold),
        "credit_score": score,
        "risk_level": classify_risk_level(score),
        "decision": make_decision(prob, score),
        "threshold_used": threshold,
    }, X

def gauge(value: float, title: str, min_v=300, max_v=850):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": 700, "increasing": {"color": "#0f7b55"}, "decreasing": {"color": "#a52a2a"}},
        title={"text": title, "font": {"size": 20, "family": "Cinzel"}},
        gauge={
            "axis": {"range": [min_v, max_v]},
            "bar": {"color": "#c9a24e"},
            "bgcolor": "rgba(255,250,240,.6)",
            "borderwidth": 1,
            "bordercolor": "#6b4f2a",
            "steps": [
                {"range": [300, 500], "color": "#f7c7c0"},
                {"range": [500, 700], "color": "#f7e6b7"},
                {"range": [700, 850], "color": "#ccebdd"},
            ],
            "threshold": {"line": {"color": "#a52a2a", "width": 4}, "thickness": .75, "value": value},
        },
    ))
    return classic_layout(fig, height=320)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

# ------------------------------------------------------------
# 4. SIDEBAR
# ------------------------------------------------------------
st.sidebar.markdown("# 🏛️ Credit Risk")
st.sidebar.markdown("**Classic Intelligence Edition**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Điều hướng",
    [
        "🏛️ Executive Hall",
        "🧬 Data Chamber",
        "🤖 Model Arena",
        "🔎 Explainable AI",
        "🎯 Prediction Studio",
        "🧪 Scenario Lab",
        "🌪️ Stress Test",
        "📜 Story & Report",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Trạng thái hệ thống")
checks = {
    "Data clean": not data_clean.empty,
    "Scored portfolio": not credit_scored.empty,
    "Best model": "best_model" in artifacts,
    "Threshold": "best_threshold" in artifacts,
    "SHAP explainability": not shap_importance.empty or (EXPLAIN_DIR / "shap_bar.png").exists(),
}
for name, ok in checks.items():
    st.sidebar.markdown(f"{'✅' if ok else '⚠️'} {name}")
st.sidebar.caption("Mục tiêu: dashboard có logic, thẩm mỹ, insight và khả năng trình bày như sản phẩm thật.")

# ------------------------------------------------------------
# 5. HERO + DATA VALIDATION
# ------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="ribbon">🏛️ CLASSIC FINTECH • AI CREDIT SCORING • EXECUTIVE DECISION SYSTEM</div>
  <h1>Credit Risk Intelligence System</h1>
  <p>Dashboard không chỉ vẽ biểu đồ, mà kể một câu chuyện đầy đủ: dữ liệu được đưa từ SQL, làm sạch, xây dựng đặc trưng, huấn luyện nhiều mô hình, tối ưu threshold, giải thích bằng SHAP, chuyển xác suất vỡ nợ thành credit score và quyết định tín dụng.</p>
</div>
""",
    unsafe_allow_html=True,
)

if df.empty:
    st.error("Chưa tìm thấy dữ liệu đầu ra. Hãy chạy lần lượt: person1_data_lead.py → person2_ml_model.py → person3_ml_system.py rồi mở dashboard.")
    st.stop()

# Normalized filtered data
filtered = df.copy()
with st.expander("🎛️ Bộ lọc danh mục khách hàng toàn cục", expanded=False):
    c1, c2, c3, c4, c5 = st.columns(5)
    if "risk_level" in filtered.columns:
        opts = sorted(filtered["risk_level"].dropna().unique().tolist())
        choice = c1.multiselect("Risk level", opts, default=opts)
        filtered = filtered[filtered["risk_level"].isin(choice)]
    if "decision" in filtered.columns:
        opts = sorted(filtered["decision"].dropna().unique().tolist())
        choice = c2.multiselect("Decision", opts, default=opts)
        filtered = filtered[filtered["decision"].isin(choice)]
    if "loan_grade" in filtered.columns:
        opts = sorted(filtered["loan_grade"].dropna().unique().tolist())
        choice = c3.multiselect("Loan grade", opts, default=opts)
        filtered = filtered[filtered["loan_grade"].isin(choice)]
    if "loan_intent" in filtered.columns:
        opts = sorted(filtered["loan_intent"].dropna().unique().tolist())
        choice = c4.multiselect("Loan intent", opts, default=opts)
        filtered = filtered[filtered["loan_intent"].isin(choice)]
    if "person_income" in filtered.columns:
        min_i, max_i = float(filtered["person_income"].min()), float(filtered["person_income"].max())
        lo, hi = c5.slider("Income range", min_i, max_i, (min_i, max_i))
        filtered = filtered[(filtered["person_income"] >= lo) & (filtered["person_income"] <= hi)]

# ------------------------------------------------------------
# PAGE 1: EXECUTIVE HALL
# ------------------------------------------------------------
if page == "🏛️ Executive Hall":
    section("Executive Hall", "Trang tổng quan cho ban lãnh đạo: quy mô danh mục, chất lượng tín dụng, tỷ lệ rủi ro và quyết định cuối cùng.")
    total = len(filtered)
    default_rate = filtered["loan_status"].mean() if "loan_status" in filtered.columns else np.nan
    avg_score = filtered["credit_score"].mean() if "credit_score" in filtered.columns else np.nan
    avg_pd = filtered["probability_of_default"].mean() if "probability_of_default" in filtered.columns else np.nan
    exposure = filtered["loan_amnt"].sum() if "loan_amnt" in filtered.columns else np.nan
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Customers", fmt_int(total), "Số khách hàng sau bộ lọc")
    with c2: kpi("Observed Default", f"{default_rate*100:.2f}%" if not np.isnan(default_rate) else "—", "Tỷ lệ vỡ nợ thực tế")
    with c3: kpi("Average Score", f"{avg_score:.0f}" if not np.isnan(avg_score) else "—", "Điểm tín dụng 300–850")
    with c4: kpi("Average PD", f"{avg_pd*100:.2f}%" if not np.isnan(avg_pd) else "—", "Xác suất vỡ nợ dự đoán")
    with c5: kpi("Loan Exposure", fmt_money(exposure), "Tổng dư nợ mô phỏng")

    c1, c2 = st.columns([1.25, .9])
    with c1:
        if "decision" in filtered.columns:
            dec = filtered["decision"].value_counts().rename_axis("decision").reset_index(name="customers")
            fig = px.funnel(dec, x="customers", y="decision", color="decision", color_discrete_map={"APPROVE":"#0f7b55", "REVIEW":"#c9a24e", "REJECT":"#a52a2a"})
            st.plotly_chart(classic_layout(fig, "Decision Funnel: Approve → Review → Reject", 430), use_container_width=True)
        else:
            st.plotly_chart(plot_empty("No decision column"), use_container_width=True)
    with c2:
        if "credit_score" in filtered.columns:
            st.plotly_chart(gauge(float(filtered["credit_score"].mean()), "Portfolio Credit Score"), use_container_width=True)
        st.markdown(
            f"""
<div class="card dark">
<b>Góc nhìn sản phẩm</b><br><br>
{badge('APPROVE', 'green')} khách hàng tốt, rủi ro thấp.<br>
{badge('REVIEW', 'yellow')} cần kiểm tra thêm hồ sơ hoặc tài sản đảm bảo.<br>
{badge('REJECT', 'red')} rủi ro cao, nên từ chối hoặc điều chỉnh chính sách.
</div>
""",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if "credit_score" in filtered.columns:
            fig = px.histogram(filtered, x="credit_score", nbins=42, color="decision" if "decision" in filtered.columns else None, color_discrete_map={"APPROVE":"#0f7b55", "REVIEW":"#c9a24e", "REJECT":"#a52a2a"})
            fig.add_vline(x=720, line_dash="dash", line_color="#0f7b55", annotation_text="Auto Approve")
            fig.add_vline(x=500, line_dash="dash", line_color="#a52a2a", annotation_text="Reject Boundary")
            st.plotly_chart(classic_layout(fig, "Credit Score Distribution", 430), use_container_width=True)
    with c2:
        if "probability_of_default" in filtered.columns:
            th = float(artifacts.get("best_threshold", 0.5))
            fig = px.histogram(filtered, x="probability_of_default", nbins=42, color="loan_status" if "loan_status" in filtered.columns else None)
            fig.add_vline(x=th, line_dash="dash", line_color="#a52a2a", annotation_text=f"Best threshold = {th:.2f}")
            st.plotly_chart(classic_layout(fig, "Predicted Probability of Default", 430), use_container_width=True)

    if {"loan_amnt", "probability_of_default"}.issubset(filtered.columns):
        temp = filtered.copy()
        temp["expected_loss_proxy"] = temp["loan_amnt"] * temp["probability_of_default"]
        top = temp.sort_values("expected_loss_proxy", ascending=False).head(15)
        fig = px.bar(top, x="client_ID" if "client_ID" in top.columns else top.index.astype(str), y="expected_loss_proxy", color="decision" if "decision" in top.columns else None, title="Top 15 Expected Loss Customers")
        st.plotly_chart(classic_layout(fig, height=420), use_container_width=True)

# ------------------------------------------------------------
# PAGE 2: DATA CHAMBER
# ------------------------------------------------------------
elif page == "🧬 Data Chamber":
    section("Data Chamber", "Lớp dữ liệu: SQL → audit → cleaning → feature engineering → PCA. Đây là phần chứng minh bài không chỉ làm ML mà có quy trình dữ liệu thật.")
    c1, c2, c3, c4 = st.columns(4)
    before = audit_before.iloc[0] if not audit_before.empty else {}
    after = audit_after.iloc[0] if not audit_after.empty else {}
    with c1: kpi("Raw Rows", fmt_int(before.get("total_rows", len(df))), "Trước làm sạch")
    with c2: kpi("Clean Rows", fmt_int(after.get("total_rows", len(df))), "Sau làm sạch")
    with c3: kpi("Missing After", fmt_int(after.get("missing_values", filtered.isna().sum().sum())), "Mục tiêu: 0")
    with c4: kpi("Features", fmt_int(data_feature.shape[1] if not data_feature.empty else df.shape[1]), "Sau encoding")

    tab1, tab2, tab3, tab4 = st.tabs(["EDA Insight", "PCA Map", "Data Quality", "Data Dictionary"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            if not eda_grade.empty:
                fig = px.bar(eda_grade, x="loan_grade", y="default_rate_percent", color="default_rate_percent", text="default_rate_percent", color_continuous_scale="YlOrBr")
                st.plotly_chart(classic_layout(fig, "Default Rate by Loan Grade", 420), use_container_width=True)
            else: st.plotly_chart(plot_empty("Missing eda_default_by_grade.csv"), use_container_width=True)
        with c2:
            if not eda_intent.empty:
                fig = px.bar(eda_intent, x="loan_intent", y="default_rate_percent", color="default_rate_percent", text="default_rate_percent", color_continuous_scale="OrRd")
                fig.update_xaxes(tickangle=-25)
                st.plotly_chart(classic_layout(fig, "Default Rate by Loan Intent", 420), use_container_width=True)
            else: st.plotly_chart(plot_empty("Missing eda_default_by_intent.csv"), use_container_width=True)
        if not eda_country.empty and {"avg_dti", "default_rate_percent", "total_customers"}.issubset(eda_country.columns):
            fig = px.scatter(eda_country, x="avg_dti", y="default_rate_percent", size="total_customers", color="country", hover_name="country")
            st.plotly_chart(classic_layout(fig, "Country Risk Landscape: DTI vs Default Rate", 440), use_container_width=True)
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if not pca_var.empty:
                fig = px.bar(pca_var, x="principal_component", y="explained_variance_ratio", color="explained_variance_ratio", color_continuous_scale="YlOrBr")
                st.plotly_chart(classic_layout(fig, "PCA Explained Variance", 400), use_container_width=True)
            else: st.plotly_chart(plot_empty("Missing PCA report"), use_container_width=True)
        with c2:
            if not pca_var.empty:
                fig = px.line(pca_var, x="principal_component", y="cumulative_variance", markers=True)
                st.plotly_chart(classic_layout(fig, "PCA Cumulative Variance", 400), use_container_width=True)
        if not pca_features.empty and {"PC1", "PC2", "loan_status"}.issubset(pca_features.columns):
            sample = pca_features.sample(min(3500, len(pca_features)), random_state=42)
            fig = px.scatter(sample, x="PC1", y="PC2", color="loan_status", opacity=.72)
            st.plotly_chart(classic_layout(fig, "2D Risk Space after PCA", 480), use_container_width=True)
    with tab3:
        miss = filtered.isna().sum().sort_values(ascending=False).head(20).reset_index()
        miss.columns = ["feature", "missing_values"]
        fig = px.bar(miss, x="missing_values", y="feature", orientation="h", color="missing_values", color_continuous_scale="YlOrBr")
        st.plotly_chart(classic_layout(fig, "Missing Values Control Panel", 430), use_container_width=True)
        st.dataframe(filtered.head(300), use_container_width=True, height=360)
    with tab4:
        if not feature_dict.empty:
            st.dataframe(feature_dict, use_container_width=True, height=520)
        else:
            st.info("Chưa có reports/feature_dictionary.csv")

# ------------------------------------------------------------
# PAGE 3: MODEL ARENA
# ------------------------------------------------------------
elif page == "🤖 Model Arena":
    section("Model Arena", "So sánh nhiều thuật toán, tuning model mạnh nhất và tối ưu threshold để phù hợp bài toán tín dụng.")
    if not person2_summary.empty:
        row = person2_summary.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi("Best Model", str(row.get("best_model", artifacts.get("best_model_name", "—"))), "Sau tuning")
        with c2: kpi("Accuracy", fmt_pct(row.get("accuracy", np.nan)), "Không dùng một mình")
        with c3: kpi("Precision", fmt_pct(row.get("precision", np.nan)), "Ít false alarm")
        with c4: kpi("Recall", fmt_pct(row.get("recall", np.nan)), "Bắt default tốt")
        with c5: kpi("ROC-AUC", f"{float(row.get('roc_auc', 0)):.4f}", "Khả năng phân biệt")

    tab1, tab2, tab3 = st.tabs(["Benchmark", "Threshold Strategy", "Evaluation Board"])
    with tab1:
        if not model_comparison.empty:
            metrics = [m for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"] if m in model_comparison.columns]
            fig = px.bar(model_comparison, x="model", y=metrics, barmode="group")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(classic_layout(fig, "Model Performance Comparison", 460), use_container_width=True)
            st.dataframe(model_comparison, use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Run person2_ml_model.py to generate model comparison"), use_container_width=True)
    with tab2:
        if not threshold_opt.empty:
            fig = go.Figure()
            for col, color in [("precision", "#8a642b"), ("recall", "#a52a2a"), ("f1_score", "#0f7b55"), ("accuracy", "#17457c")]:
                if col in threshold_opt.columns:
                    fig.add_trace(go.Scatter(x=threshold_opt["threshold"], y=threshold_opt[col], mode="lines", name=col))
            th = float(artifacts.get("best_threshold", threshold_opt.sort_values("f1_score", ascending=False).iloc[0]["threshold"] if "f1_score" in threshold_opt.columns else .5))
            fig.add_vline(x=th, line_dash="dash", line_color="#a52a2a", annotation_text=f"Chosen threshold {th:.2f}")
            st.plotly_chart(classic_layout(fig, "Threshold Optimization: Credit Risk is not a 0.5 game", 460), use_container_width=True)
            st.markdown("""
<div class="card"><b>Ý nghĩa:</b> Bài toán tín dụng không nên chỉ dùng threshold mặc định 0.5. Nếu bỏ sót khách hàng vỡ nợ, tổn thất tài chính có thể lớn hơn việc cảnh báo nhầm. Vì vậy dashboard đưa threshold thành một quyết định kinh doanh.</div>
""", unsafe_allow_html=True)
        else:
            st.info("Chưa có reports/threshold_optimization.csv")
    with tab3:
        images = [
            "confusion_matrix_professional.png",
            "roc_curve_professional.png",
            "precision_recall_curve_professional.png",
            "probability_distribution_professional.png",
        ]
        cols = st.columns(2)
        for i, img in enumerate(images):
            with cols[i % 2]:
                path = VISUAL_DIR / img
                if path.exists(): st.image(str(path), use_container_width=True)
                else: st.warning(f"Thiếu {img}")

# ------------------------------------------------------------
# PAGE 4: EXPLAINABLE AI
# ------------------------------------------------------------
elif page == "🔎 Explainable AI":
    section("Explainable AI", "Phần giúp bài có chiều sâu: mô hình không phải hộp đen, mà giải thích được biến nào làm tăng/giảm rủi ro.")
    c1, c2 = st.columns([1, 1])
    with c1:
        if not shap_importance.empty:
            top = shap_importance.head(20)
            y = "mean_abs_shap" if "mean_abs_shap" in top.columns else top.columns[-1]
            fig = px.bar(top.sort_values(y), x=y, y="feature", orientation="h", color=y, color_continuous_scale="YlOrBr")
            st.plotly_chart(classic_layout(fig, "Top SHAP Drivers", 560), use_container_width=True)
        elif (EXPLAIN_DIR / "shap_bar.png").exists():
            st.image(str(EXPLAIN_DIR / "shap_bar.png"), use_container_width=True)
        else:
            st.plotly_chart(plot_empty("No SHAP output yet"), use_container_width=True)
    with c2:
        if not shap_waterfall.empty and {"feature", "shap_value"}.issubset(shap_waterfall.columns):
            temp = shap_waterfall.sort_values("shap_value")
            fig = px.bar(temp, x="shap_value", y="feature", orientation="h", color="shap_value", color_continuous_scale="RdYlGn_r")
            fig.add_vline(x=0, line_color="#21170f")
            st.plotly_chart(classic_layout(fig, "High-Risk Customer Waterfall", 560), use_container_width=True)
        elif (EXPLAIN_DIR / "shap_waterfall_sample.png").exists():
            st.image(str(EXPLAIN_DIR / "shap_waterfall_sample.png"), use_container_width=True)
        else:
            st.info("Chưa có waterfall sample.")
    if (EXPLAIN_DIR / "shap_summary.png").exists():
        st.markdown("### SHAP Summary Image")
        st.image(str(EXPLAIN_DIR / "shap_summary.png"), use_container_width=True)

# ------------------------------------------------------------
# PAGE 5: PREDICTION STUDIO
# ------------------------------------------------------------
elif page == "🎯 Prediction Studio":
    section("Prediction Studio", "Nhập hồ sơ khách hàng và xem hệ thống đưa ra PD, credit score, risk level và quyết định tín dụng.")
    c1, c2 = st.columns([.95, 1.05])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Customer Application Form")
        age = st.slider("Age", 18, 80, 32)
        income = st.number_input("Annual income", min_value=0, value=65000, step=1000)
        loan_amnt = st.number_input("Loan amount", min_value=100, value=12000, step=500)
        loan_int_rate = st.slider("Interest rate (%)", 1.0, 35.0, 12.5, .1)
        loan_percent_income = st.slider("Loan percent income", 0.01, 1.00, min(loan_amnt / max(income, 1), .75), .01)
        dti = st.slider("Debt-to-income ratio", 0.00, 2.00, .35, .01)
        util = st.slider("Credit utilization ratio", 0.00, 1.50, .45, .01)
        delinq = st.slider("Past delinquencies", 0, 20, 1)
        hist = st.slider("Credit history length", 1, 35, 6)
        home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        intent = st.selectbox("Loan intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        grade = st.selectbox("Loan grade", list("ABCDEFG"))
        default_file = st.selectbox("Default on file", ["N", "Y"])
        raw = {
            "person_age": age, "person_income": income, "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate, "loan_percent_income": loan_percent_income,
            "debt_to_income_ratio": dti, "credit_utilization_ratio": util,
            "past_delinquencies": delinq, "cb_person_cred_hist_length": hist,
            "person_home_ownership": home, "loan_intent": intent, "loan_grade": grade,
            "cb_person_default_on_file": default_file,
            "loan_to_income_ratio": loan_amnt / max(income, 1),
        }
        run = st.button("🏛️ Run Credit Decision")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        result, X_input = predict_one(raw)
        if result is None:
            st.error("Chưa tìm thấy model_columns.pkl hoặc best_model.pkl. Hãy chạy Person 2 và Person 3 trước.")
        else:
            pd_value = result["probability_of_default"]
            score = result["credit_score"]
            decision = result["decision"]
            cls = "green" if decision == "APPROVE" else "yellow" if decision == "REVIEW" else "red"
            st.markdown(f"""
<div class="card dark">
<h2 style="font-family:Cinzel;margin-top:0;">Decision Result</h2>
{badge(decision, cls)} {badge(result['risk_level'], cls)} {badge('PD ' + str(round(pd_value*100,2)) + '%', 'gold')}
<p style="line-height:1.7;">Hệ thống chuyển xác suất vỡ nợ thành điểm tín dụng và ra quyết định theo rule được thiết kế ở Person 3.</p>
</div>
""", unsafe_allow_html=True)
            c21, c22 = st.columns(2)
            with c21: st.plotly_chart(gauge(score, "Customer Score"), use_container_width=True)
            with c22:
                fig = go.Figure(go.Indicator(mode="number+gauge", value=pd_value*100, number={"suffix":"%"}, title={"text":"Probability of Default"}, gauge={"shape":"bullet", "axis":{"range":[0,100]}, "bar":{"color":"#a52a2a"}, "threshold":{"line":{"color":"#21170f","width":3}, "value":float(result['threshold_used'])*100}}))
                st.plotly_chart(classic_layout(fig, height=280), use_container_width=True)
            explanation = ""
            if decision == "APPROVE": explanation = "Hồ sơ có rủi ro thấp, điểm tín dụng tốt và xác suất vỡ nợ dưới vùng nguy hiểm."
            elif decision == "REVIEW": explanation = "Hồ sơ chưa đủ xấu để từ chối, nhưng cần kiểm tra thêm thu nhập, lịch sử nợ hoặc tài sản đảm bảo."
            else: explanation = "Hồ sơ có tín hiệu rủi ro cao; nên từ chối tự động hoặc yêu cầu điều kiện bảo vệ rủi ro mạnh hơn."
            st.markdown(f'<div class="card"><b>Business explanation:</b> {explanation}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# PAGE 6: SCENARIO LAB
# ------------------------------------------------------------
elif page == "🧪 Scenario Lab":
    section("Scenario Lab", "Tính năng sáng tạo: mô phỏng khách hàng thay đổi thu nhập, khoản vay, lãi suất để thấy quyết định tín dụng biến đổi như thế nào.")
    if "best_model" not in artifacts:
        st.error("Cần chạy Person 2/3 để có model trước.")
    else:
        base_income = st.slider("Base income", 10000, 200000, 65000, 5000)
        base_loan = st.slider("Base loan amount", 1000, 80000, 15000, 1000)
        base_rate = st.slider("Interest rate", 3.0, 35.0, 13.0, .5)
        rates = np.linspace(max(3, base_rate - 6), min(35, base_rate + 8), 18)
        loans = np.linspace(max(1000, base_loan * .4), base_loan * 2.2, 18)
        records = []
        for r in rates:
            for l in loans:
                raw = {
                    "person_age": 32, "person_income": base_income, "loan_amnt": l, "loan_int_rate": r,
                    "loan_percent_income": l / max(base_income, 1), "loan_to_income_ratio": l / max(base_income, 1),
                    "debt_to_income_ratio": .35, "credit_utilization_ratio": .45, "past_delinquencies": 1,
                    "cb_person_cred_hist_length": 6, "person_home_ownership": "RENT", "loan_intent": "PERSONAL",
                    "loan_grade": "C", "cb_person_default_on_file": "N",
                }
                result, _ = predict_one(raw)
                if result:
                    records.append({"interest_rate": r, "loan_amount": l, "pd": result["probability_of_default"], "score": result["credit_score"], "decision": result["decision"]})
        sim = pd.DataFrame(records)
        fig = px.density_heatmap(sim, x="interest_rate", y="loan_amount", z="pd", histfunc="avg", color_continuous_scale="YlOrRd")
        st.plotly_chart(classic_layout(fig, "Risk Heatmap: PD changes by loan amount and interest rate", 560), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(sim, x="loan_amount", y="score", color="decision", size="pd", color_discrete_map={"APPROVE":"#0f7b55", "REVIEW":"#c9a24e", "REJECT":"#a52a2a"})
            st.plotly_chart(classic_layout(fig, "Score Sensitivity by Loan Amount", 430), use_container_width=True)
        with c2:
            st.dataframe(sim.sort_values("pd", ascending=False).head(30), use_container_width=True, height=430)

# ------------------------------------------------------------
# PAGE 7: STRESS TEST
# ------------------------------------------------------------
elif page == "🌪️ Stress Test":
    section("Portfolio Stress Test", "Mô phỏng cú sốc kinh tế: thu nhập giảm, lãi suất tăng, credit utilization tăng. Đây là góc nhìn nâng cấp như ngân hàng thật.")
    if filtered.empty:
        st.warning("Không có dữ liệu sau bộ lọc.")
    else:
        c1, c2, c3 = st.columns(3)
        income_shock = c1.slider("Income shock", -50, 10, -15, 5) / 100
        rate_shock = c2.slider("Interest rate shock", 0, 10, 3, 1)
        util_shock = c3.slider("Credit utilization shock", 0, 50, 15, 5) / 100
        base_default = filtered["loan_status"].mean() if "loan_status" in filtered.columns else np.nan
        stress = filtered.copy()
        if "person_income" in stress.columns:
            stress["stress_income"] = stress["person_income"] * (1 + income_shock)
        if "loan_int_rate" in stress.columns:
            stress["stress_interest_rate"] = stress["loan_int_rate"] + rate_shock
        if "credit_utilization_ratio" in stress.columns:
            stress["stress_credit_utilization"] = np.minimum(stress["credit_utilization_ratio"] + util_shock, 2)
        if "probability_of_default" in stress.columns:
            stress["stress_pd_proxy"] = np.clip(
                stress["probability_of_default"] * (1 + abs(income_shock)*1.2 + rate_shock/20 + util_shock*.9),
                0, 1
            )
            stress["stress_score"] = stress["stress_pd_proxy"].apply(calculate_credit_score)
            stress["stress_decision"] = stress.apply(lambda r: make_decision(r["stress_pd_proxy"], r["stress_score"]), axis=1)
            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Base PD", fmt_pct(stress["probability_of_default"].mean()), "Hiện tại")
            with c2: kpi("Stress PD", fmt_pct(stress["stress_pd_proxy"].mean()), "Sau cú sốc")
            with c3: kpi("Score Drop", f"{stress['credit_score'].mean() - stress['stress_score'].mean():.0f}" if "credit_score" in stress.columns else "—", "Điểm giảm trung bình")
            with c4: kpi("Reject After Stress", fmt_pct((stress["stress_decision"] == "REJECT").mean()), "Tỷ lệ từ chối")
            comp = pd.DataFrame({
                "state": ["Before", "After Stress"],
                "avg_pd": [stress["probability_of_default"].mean(), stress["stress_pd_proxy"].mean()],
                "reject_rate": [(stress["decision"] == "REJECT").mean() if "decision" in stress.columns else np.nan, (stress["stress_decision"] == "REJECT").mean()],
                "avg_score": [stress["credit_score"].mean() if "credit_score" in stress.columns else np.nan, stress["stress_score"].mean()]
            })
            fig = px.bar(comp, x="state", y=["avg_pd", "reject_rate"], barmode="group")
            st.plotly_chart(classic_layout(fig, "Before vs After Stress", 430), use_container_width=True)
            fig = px.histogram(stress, x="stress_score", color="stress_decision", nbins=40, color_discrete_map={"APPROVE":"#0f7b55", "REVIEW":"#c9a24e", "REJECT":"#a52a2a"})
            st.plotly_chart(classic_layout(fig, "Stressed Score Distribution", 430), use_container_width=True)
        else:
            st.info("Cần data/credit_scored_data.csv để chạy stress test đầy đủ.")

# ------------------------------------------------------------
# PAGE 8: STORY & REPORT
# ------------------------------------------------------------
elif page == "📜 Story & Report":
    section("Story & Report", "Trang này dùng để thuyết trình: nói rõ logic từng người và tại sao dashboard xứng đáng điểm cao.")
    st.markdown(
        """
<div class="card">
<div class="story-step"><b>1. Person 1 — Data Intelligence:</b> đưa dữ liệu từ MySQL, audit, làm sạch, xử lý missing/duplicate, EDA, feature engineering và PCA.</div>
<div class="story-step"><b>2. Person 2 — Model Arena:</b> so sánh Logistic Regression, Decision Tree, Random Forest, XGBoost; tuning và tối ưu threshold theo bản chất Credit Risk.</div>
<div class="story-step"><b>3. Person 3 — Deployment System:</b> đóng gói pipeline, chấm điểm tín dụng, rule APPROVE/REVIEW/REJECT và giải thích bằng SHAP.</div>
<div class="story-step"><b>4. Dashboard — Business Product:</b> biến toàn bộ kết quả thành sản phẩm ra quyết định, có filter, prediction studio, scenario lab và stress test.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("### Checklist điểm cao")
    st.markdown(
        """
- Có quy trình dữ liệu từ SQL đến dashboard.
- Có nhiều thuật toán và so sánh công bằng bằng Accuracy, Precision, Recall, F1, ROC-AUC.
- Có threshold optimization, không dùng máy móc ngưỡng 0.5.
- Có Explainable AI/SHAP để giải thích mô hình.
- Có credit score và decision rule phù hợp nghiệp vụ ngân hàng.
- Có mô phỏng kịch bản và stress test, thể hiện tư duy ứng dụng thực tế.
- Giao diện có concept rõ: classic fintech, hiện đại, trực quan, có tính trình bày.
"""
    )
    with st.expander("📄 Xem báo cáo Person 2 nếu có"):
        txt = read_text(REPORT_DIR / "evaluation_report.txt")
        st.text(txt if txt else "Chưa có reports/evaluation_report.txt")
    st.download_button("⬇️ Download filtered portfolio CSV", filtered.to_csv(index=False).encode("utf-8-sig"), "filtered_credit_portfolio.csv", "text/csv")
