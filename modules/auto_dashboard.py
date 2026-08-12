"""Auto dashboard builder — KPIs + pie/bar/line for live or one-click manual generate."""
from __future__ import annotations

from typing import Any

import pandas as pd

from modules.dashboard_slicers import auto_chart_specs


def build_auto_dashboard(df: pd.DataFrame, domain: str = "generic") -> dict[str, Any]:
    """Generate auto dashboard config from a data slice."""
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    kpi_list = []
    for col in num_cols[:6]:
        kpi_list.append({
            "name": col,
            "value": round(float(df[col].mean()), 2),
            "label": f"Avg {col}",
        })
        kpi_list.append({
            "name": f"{col}_sum",
            "value": round(float(df[col].sum()), 2),
            "label": f"Total {col}",
        })
    # de-dupe to 8
    kpi_list = kpi_list[:8]

    charts = auto_chart_specs(df)

    alerts = []
    for col in num_cols[:3]:
        mean = df[col].mean()
        std = df[col].std()
        high = df[col].max()
        if std and high > mean + 2 * std:
            alerts.append(f"⚠️ {col} has values {high:.1f} exceeding 2σ ({mean + 2*std:.1f})")

    return {
        "kpi_cards": kpi_list,
        "charts": charts,
        "alerts": alerts,
        "domain": domain,
        "rows": len(df),
    }
