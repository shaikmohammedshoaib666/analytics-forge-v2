"""Template briefing text for each domain."""
from __future__ import annotations

from typing import Any, Optional

from core.classify import load_domains
from core.kpis import format_kpi_value


BRIEF_TEMPLATES = {
    "predictive_maintenance": (
        "Predictive Maintenance briefing: monitoring {n_rows:,} sensor records across "
        "equipment. Focus on failure risk, RUL, and sensor anomalies (temperature, "
        "vibration, pressure). Recommended next step: train a regressor on RUL or a "
        "classifier on failure, then surface high-risk machines on the dashboard."
    ),
    "sales_forecasting": (
        "Sales Forecasting briefing: {n_rows:,} order rows ingested. Track revenue, "
        "units, and regional mix. Recommended next step: time-based charts and a "
        "regression/forecast model on revenue."
    ),
    "hospital_medical": (
        "Hospital / Medical briefing: {n_rows:,} clinical/operational rows. "
        "Prioritize length of stay, admissions, and outcome indicators. Validate PHI "
        "handling before sharing packs externally."
    ),
    "warehouse_efficiency": (
        "Warehouse Efficiency briefing: {n_rows:,} logistics rows. Focus on throughput, "
        "lead time, and SKU coverage. Recommended charts: bar by aisle/dock and lead-time histograms."
    ),
    "customer_segmentation": (
        "Customer Segmentation briefing: {n_rows:,} customer rows. Use RFM-style KPIs "
        "and clustering (K-Means) to define personas, then validate churn drivers."
    ),
    "marketing_campaign": (
        "Marketing Campaign briefing: {n_rows:,} campaign rows. Optimize spend vs "
        "conversions (CTR/CPC/ROAS). Compare channels before scaling budget."
    ),
    "executive_dashboard": (
        "Executive Dashboard briefing: {n_rows:,} rows ready for board-level KPIs. "
        "Highlight revenue, cost, and margin; keep charts sparse and decision-oriented."
    ),
    "generic": (
        "Generic Analytics briefing: {n_rows:,} rows × {n_cols} columns cleaned and "
        "profiled. Explore Auto KPIs, add charts to the dashboard, and try ML Studio "
        "with an automatically selected target."
    ),
}


def build_briefing(
    domain: str,
    df_shape: tuple[int, int],
    kpis: Optional[dict[str, Any]] = None,
    classification: Optional[dict] = None,
) -> str:
    n_rows, n_cols = df_shape
    tmpl = BRIEF_TEMPLATES.get(domain, BRIEF_TEMPLATES["generic"])
    text = tmpl.format(n_rows=n_rows, n_cols=n_cols)

    domains = load_domains()
    label = domains.get(domain, {}).get("label", domain)
    header = f"## {label}\n\n{text}\n"

    if classification and classification.get("scores"):
        top = sorted(classification["scores"].items(), key=lambda x: -x[1])[:3]
        score_lines = ", ".join(f"{d}={s:.2f}" for d, s in top)
        header += f"\nDomain confidence (top): {score_lines}\n"

    if kpis:
        header += "\n### Key KPIs\n"
        for kid, item in list(kpis.items())[:8]:
            if isinstance(item, dict):
                name = item.get("name", kid)
                val = format_kpi_value(item.get("value"))
            else:
                name, val = kid, format_kpi_value(item)
            header += f"- **{name}**: {val}\n"

    return header
