"""Auto SCADA-style dashboard for live mode — auto-generated KPIs and charts."""
from __future__ import annotations

from typing import Any

import pandas as pd


def build_auto_dashboard(df: pd.DataFrame, domain: str = "generic") -> dict[str, Any]:
    """Generate auto dashboard config from live data slice."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    kpi_cards = []
    for col in num_cols[:6]:
        kpi_cards.append({
            "name": col,
            "value": round(float(df[col].mean()), 2),
            "label": f"Avg {col}",
        })

    charts = []
    if num_cols:
        charts.append({
            "chart_type": "line",
            "x": df.columns[0],
            "y": num_cols[0],
            "title": f"{num_cols[0]} trend",
            "lib": "plotly",
        })
    if len(num_cols) >= 2:
        charts.append({
            "chart_type": "scatter",
            "x": num_cols[0],
            "y": num_cols[1],
            "title": f"{num_cols[0]} vs {num_cols[1]}",
            "lib": "plotly",
        })
    if cat_cols and num_cols:
        charts.append({
            "chart_type": "bar",
            "x": cat_cols[0],
            "y": num_cols[0],
            "title": f"{num_cols[0]} by {cat_cols[0]}",
            "lib": "plotly",
        })

    alerts = []
    for col in num_cols[:3]:
        mean = df[col].mean()
        std = df[col].std()
        high = df[col].max()
        if high > mean + 2 * std:
            alerts.append(f"⚠️ {col} has values {high:.1f} exceeding 2σ ({mean + 2*std:.1f})")

    return {
        "kpi_cards": kpi_cards,
        "charts": charts,
        "alerts": alerts,
        "domain": domain,
        "rows": len(df),
    }
