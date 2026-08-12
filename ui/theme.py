"""Colorful Analytics Forge theme (teal / coral / navy — not purple defaults)."""
from __future__ import annotations

import streamlit as st

DOMAIN_COLORS = {
    "predictive_maintenance": "#F97316",
    "sales_forecasting": "#0D9488",
    "hospital_medical": "#EC4899",
    "warehouse_efficiency": "#2563EB",
    "customer_segmentation": "#CA8A04",
    "marketing_campaign": "#DC2626",
    "executive_dashboard": "#0F766E",
    "generic": "#64748B",
}

KPI_PALETTE = [
    ("#0D9488", "#CCFBF1"),
    ("#EA580C", "#FFEDD5"),
    ("#2563EB", "#DBEAFE"),
    ("#DB2777", "#FCE7F3"),
    ("#CA8A04", "#FEF9C3"),
    ("#7C3AED", "#EDE9FE"),
    ("#0891B2", "#CFFAFE"),
    ("#B91C1C", "#FEE2E2"),
]


def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"]  {
  font-family: 'DM Sans', sans-serif;
}

.stApp {
  background: linear-gradient(160deg, #F0FDFA 0%, #ECFEFF 40%, #FFF7ED 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #134E4A 0%, #0F766E 55%, #115E59 100%);
}
[data-testid="stSidebar"] * {
  color: #F0FDFA !important;
}
[data-testid="stSidebar"] .stRadio label {
  background: rgba(255,255,255,0.08);
  border-radius: 10px;
  padding: 0.35rem 0.6rem;
  margin-bottom: 0.25rem;
}
[data-testid="stSidebar"] [data-baseweb="radio"] > div:hover {
  background: rgba(251, 146, 60, 0.25);
  border-radius: 10px;
}

.af-hero {
  background: linear-gradient(120deg, #0F766E 0%, #0D9488 45%, #FB923C 100%);
  color: white;
  padding: 1.25rem 1.5rem;
  border-radius: 18px;
  margin-bottom: 1.2rem;
  box-shadow: 0 10px 30px rgba(13, 148, 136, 0.25);
}
.af-hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.8rem;
  margin: 0 0 0.35rem 0;
  color: white !important;
}
.af-hero p {
  margin: 0;
  opacity: 0.95;
  color: #ECFDF5 !important;
}

.af-section {
  background: rgba(255,255,255,0.78);
  border: 1px solid rgba(13, 148, 136, 0.18);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
  box-shadow: 0 4px 18px rgba(15, 118, 110, 0.08);
}

.af-badge {
  display: inline-block;
  padding: 0.25rem 0.7rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 700;
  color: white;
  margin-bottom: 0.6rem;
}

.af-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 0.85rem;
  margin: 0.5rem 0 1rem 0;
}
.af-kpi {
  border-radius: 16px;
  padding: 1.05rem 1.1rem;
  min-height: 110px;
  aspect-ratio: 1.35 / 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  border: 1px solid rgba(15, 23, 42, 0.06);
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
  background: #fff;
}
.af-kpi .label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  opacity: 0.8;
  margin-bottom: 0.45rem;
}
.af-kpi .value {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  line-height: 1.15;
}
.af-slicer-bar {
  background: rgba(255,255,255,0.9);
  border: 1px solid rgba(13, 148, 136, 0.2);
  border-radius: 14px;
  padding: 0.85rem 1rem;
  margin-bottom: 1rem;
}

div.stButton > button {
  border-radius: 12px;
  font-weight: 650;
  border: none;
  background: linear-gradient(90deg, #0D9488, #14B8A6);
  color: white;
}
div.stButton > button:hover {
  background: linear-gradient(90deg, #EA580C, #FB923C);
  color: white;
  border: none;
}
div.stButton > button[kind="primary"] {
  background: linear-gradient(90deg, #EA580C, #F97316);
}

[data-testid="stMetric"] {
  background: white;
  border-radius: 14px;
  padding: 0.75rem;
  border-left: 5px solid #0D9488;
  box-shadow: 0 4px 12px rgba(13,148,136,0.12);
}

.stDownloadButton > button {
  background: linear-gradient(90deg, #2563EB, #38BDF8) !important;
  color: white !important;
  border-radius: 12px !important;
}

.stAlert {
  border-radius: 12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def page_hero(title: str, subtitle: str, domain: str | None = None) -> None:
    color = DOMAIN_COLORS.get(domain or "generic", DOMAIN_COLORS["generic"])
    badge = ""
    if domain:
        badge = f'<div class="af-badge" style="background:{color}">{domain.replace("_", " ").title()}</div>'
    st.markdown(
        f"""
<div class="af-hero">
  {badge}
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def section_start() -> None:
    st.markdown('<div class="af-section">', unsafe_allow_html=True)


def section_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
