"""Business alerts — domain-aware threshold checks on live or manual data."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_alerts(df: pd.DataFrame, domain: str = "generic", ml_result: dict | None = None) -> list[dict[str, Any]]:
    """Generate business alerts based on data and optional ML results."""
    alerts: list[dict[str, Any]] = []

    # Domain-specific checks
    if domain == "predictive_maintenance":
        _pdm_alerts(df, alerts)
    elif domain == "sales_forecasting":
        _sales_alerts(df, alerts, ml_result)
    else:
        _generic_alerts(df, alerts)

    # ML-based alerts
    if ml_result and ml_result.get("ok"):
        _ml_alerts(ml_result, alerts)

    return alerts


def _pdm_alerts(df: pd.DataFrame, alerts: list) -> None:
    col_map = {c.lower(): c for c in df.columns}
    if "failure" in col_map:
        fail_rate = df[col_map["failure"]].mean()
        if fail_rate > 0.1:
            alerts.append({"severity": "high", "message": f"Failure rate {fail_rate*100:.1f}% exceeds 10% threshold"})
    if "rul" in col_map:
        min_rul = df[col_map["rul"]].min()
        if min_rul < 50:
            alerts.append({"severity": "critical", "message": f"Machine with RUL={min_rul} — schedule maintenance NOW"})
    if "temperature" in col_map:
        max_temp = df[col_map["temperature"]].max()
        if max_temp > 95:
            alerts.append({"severity": "warning", "message": f"Temperature spike: {max_temp:.1f}°C exceeds 95°C limit"})


def _sales_alerts(df: pd.DataFrame, alerts: list, ml_result: dict | None) -> None:
    col_map = {c.lower(): c for c in df.columns}
    if "revenue" in col_map:
        rev = df[col_map["revenue"]]
        if rev.iloc[-1] < rev.mean() * 0.7:
            alerts.append({"severity": "warning", "message": "Latest revenue 30%+ below average — investigate."})
    if ml_result and ml_result.get("metrics", {}).get("pct_change"):
        pct = ml_result["metrics"]["pct_change"]
        if pct < -0.15:
            alerts.append({"severity": "high", "message": f"Forecast shows {pct*100:.0f}% decline ahead"})
        elif pct > 0.2:
            alerts.append({"severity": "info", "message": f"Forecast shows +{pct*100:.0f}% growth ahead"})


def _generic_alerts(df: pd.DataFrame, alerts: list) -> None:
    null_pct = df.isna().sum().sum() / max(1, df.size) * 100
    if null_pct > 10:
        alerts.append({"severity": "warning", "message": f"Data has {null_pct:.1f}% missing values"})
    dup_pct = df.duplicated().sum() / max(1, len(df)) * 100
    if dup_pct > 5:
        alerts.append({"severity": "info", "message": f"{dup_pct:.1f}% duplicate rows detected"})


def _ml_alerts(ml_result: dict, alerts: list) -> None:
    metrics = ml_result.get("metrics", {})
    if "r2" in metrics and metrics["r2"] < 0.3:
        alerts.append({"severity": "info", "message": f"Model R²={metrics['r2']:.2f} is low — consider more features or data."})
    if "accuracy" in metrics and metrics["accuracy"] < 0.7:
        alerts.append({"severity": "info", "message": f"Classification accuracy {metrics['accuracy']*100:.0f}% — may need more training data."})
