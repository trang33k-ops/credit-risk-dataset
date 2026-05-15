import streamlit as st
import plotly.graph_objects as go


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 45%, #ECFEFF 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .sidebar-title {
            font-size: 26px;
            font-weight: 900;
            color: #0F172A;
            margin-bottom: 2px;
        }
        .sidebar-subtitle {
            font-size: 13px;
            color: #64748B;
            margin-bottom: 20px;
        }
        .hero {
            padding: 28px 30px;
            border-radius: 28px;
            background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 55%, #0891B2 100%);
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            color: white;
            margin-bottom: 24px;
        }
        .hero-badge {
            display: inline-block;
            padding: 6px 12px;
            background: rgba(255,255,255,0.18);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            font-size: 13px;
            margin-bottom: 12px;
        }
        .hero-title {
            font-size: 42px;
            font-weight: 900;
            line-height: 1.05;
            margin: 0;
        }
        .hero-subtitle {
            font-size: 17px;
            color: #E0F2FE;
            margin-top: 12px;
            max-width: 980px;
        }
        .kpi-card {
            background: rgba(255,255,255,0.88);
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 22px;
            padding: 20px 20px;
            box-shadow: 0 10px 30px rgba(15,23,42,0.08);
            min-height: 122px;
        }
        .kpi-label {
            color: #64748B;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .kpi-value {
            color: #0F172A;
            font-size: 30px;
            font-weight: 900;
            margin-top: 8px;
        }
        .kpi-help {
            color: #64748B;
            font-size: 13px;
            margin-top: 6px;
        }
        .section-title {
            margin-top: 16px;
            margin-bottom: 10px;
        }
        .section-title h2 {
            color: #0F172A;
            font-size: 25px;
            font-weight: 900;
            margin-bottom: 0px;
        }
        .section-title p {
            color: #64748B;
            margin-top: 4px;
        }
        .glass-card {
            background: rgba(255,255,255,0.82);
            border: 1px solid rgba(148, 163, 184, 0.30);
            border-radius: 26px;
            padding: 25px;
            box-shadow: 0 15px 40px rgba(15,23,42,0.08);
        }
        .badge-approve, .badge-review, .badge-reject, .badge-low, .badge-medium, .badge-high {
            padding: 18px;
            border-radius: 20px;
            color: white;
            text-align: center;
            font-size: 25px;
            font-weight: 900;
            box-shadow: 0 10px 24px rgba(15,23,42,0.12);
        }
        .badge-approve, .badge-low { background: linear-gradient(135deg, #16A34A, #22C55E); }
        .badge-review, .badge-medium { background: linear-gradient(135deg, #F59E0B, #FACC15); color: #111827; }
        .badge-reject, .badge-high { background: linear-gradient(135deg, #DC2626, #EF4444); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section(title: str, subtitle: str, badge: str):
    st.markdown(
        f"""
        <div class='hero'>
            <div class='hero-badge'>{badge}</div>
            <h1 class='hero-title'>{title}</h1>
            <div class='hero-subtitle'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{value}</div>
            <div class='kpi-help'>{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class='section-title'>
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_badge(decision: str):
    css = {
        "APPROVE": "badge-approve",
        "REVIEW": "badge-review",
        "REJECT": "badge-reject",
    }.get(decision, "badge-review")
    st.markdown(f"<div class='{css}'>{decision}</div>", unsafe_allow_html=True)


def risk_badge(risk: str):
    css = {
        "LOW": "badge-low",
        "MEDIUM": "badge-medium",
        "HIGH": "badge-high",
    }.get(risk, "badge-medium")
    st.markdown(f"<div class='{css}'>{risk} RISK</div>", unsafe_allow_html=True)


def credit_score_gauge(score: int):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credit Score"},
        gauge={
            "axis": {"range": [300, 850]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [300, 500], "color": "#FEE2E2"},
                {"range": [500, 720], "color": "#FEF3C7"},
                {"range": [720, 850], "color": "#DCFCE7"},
            ],
            "threshold": {"line": {"color": "#DC2626", "width": 4}, "thickness": 0.75, "value": score},
        }
    ))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_footer():
    st.markdown("---")
    st.caption("PTIT FinTech Project — AI Credit Risk Scoring System | Person 1 + Person 2 + Person 3")
