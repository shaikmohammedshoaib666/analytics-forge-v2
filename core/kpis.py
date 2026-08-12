"""Field-aware KPI computation."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core.classify import load_domains


def _find_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # partial match
    for cand in candidates:
        for lc, orig in lower_map.items():
            if cand.lower() in lc:
                return orig
    return None


def _safe_mean(s: pd.Series) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().any():
        return float(s.mean())
    return None


def _safe_sum(s: pd.Series) -> Optional[float]:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().any():
        return float(s.sum())
    return None


def _eval_formula(df: pd.DataFrame, formula: str) -> Any:
    f = formula.strip().lower()
    if f == "count(*)":
        return int(len(df))
    if f == "ncol":
        return int(df.shape[1])
    if f.startswith("mean(") and f.endswith(")"):
        inner = formula.strip()[5:-1].strip()
        if inner.lower() == "numeric":
            nums = df.select_dtypes(include=[np.number])
            if nums.empty:
                return None
            return {c: float(nums[c].mean()) for c in nums.columns[:8]}
        col = _find_col(df, inner)
        return _safe_mean(df[col]) if col else None
    if f.startswith("sum(") and f.endswith(")"):
        inner = formula.strip()[4:-1].strip()
        col = _find_col(df, inner)
        return _safe_sum(df[col]) if col else None
    if f.startswith("nunique(") and f.endswith(")"):
        inner = formula.strip()[8:-1].strip()
        col = _find_col(df, inner)
        return int(df[col].nunique()) if col else None
    if "/" in f and "sum(" in f:
        # simple (sum(a)-sum(b))/sum(a)
        try:
            # naive margin: (sum(revenue)-sum(cost))/sum(revenue)
            rev = _find_col(df, "revenue", "sales", "income")
            cost = _find_col(df, "cost", "expense", "spend")
            if rev and cost:
                r = _safe_sum(df[rev]) or 0.0
                c = _safe_sum(df[cost]) or 0.0
                return (r - c) / r if r else None
        except Exception:
            return None
    return None


def compute_generic_kpis(df: pd.DataFrame) -> dict[str, Any]:
    kpis: dict[str, Any] = {
        "row_count": {"name": "Rows", "value": int(len(df))},
        "col_count": {"name": "Columns", "value": int(df.shape[1])},
        "missing_pct": {
            "name": "Missing %",
            "value": float(df.isna().mean().mean() * 100) if len(df) else 0.0,
        },
    }
    nums = df.select_dtypes(include=[np.number])
    for c in nums.columns[:5]:
        kpis[f"mean_{c}"] = {"name": f"Mean {c}", "value": float(nums[c].mean())}
    return kpis


def compute_kpis(
    df: pd.DataFrame,
    domain: str = "generic",
    ml_metrics: Optional[dict] = None,
    domains: Optional[dict] = None,
) -> dict[str, Any]:
    """Compute domain KPI defs + optional ML metrics."""
    domains = domains or load_domains()
    meta = domains.get(domain, domains.get("generic", {}))
    kpi_defs = meta.get("kpi_defs", [])
    kpis: dict[str, Any] = {}

    for kd in kpi_defs:
        kid = kd.get("id", kd.get("name", "kpi"))
        name = kd.get("name", kid)
        formula = kd.get("formula", "count(*)")
        val = _eval_formula(df, formula)
        kpis[kid] = {"name": name, "value": val, "formula": formula}

    # Always include basics
    base = compute_generic_kpis(df)
    for k, v in base.items():
        if k not in kpis:
            kpis[k] = v

    # Domain-specific extras
    if domain == "predictive_maintenance":
        fail_col = _find_col(df, "failure", "failed", "fault")
        if fail_col is not None:
            s = pd.to_numeric(df[fail_col], errors="coerce")
            kpis["failure_count"] = {
                "name": "Failure Count",
                "value": int(s.fillna(0).sum()),
            }
        rul_col = _find_col(df, "rul", "remaining_useful_life")
        if rul_col is not None:
            kpis["min_rul"] = {
                "name": "Min RUL",
                "value": _safe_mean(df[rul_col].nsmallest(1))
                if len(df)
                else None,
            }
            try:
                kpis["min_rul"] = {
                    "name": "Min RUL",
                    "value": float(pd.to_numeric(df[rul_col], errors="coerce").min()),
                }
            except Exception:
                pass

    if domain == "sales_forecasting":
        rev = _find_col(df, "revenue", "sales", "amount")
        date_col = _find_col(df, "order_date", "date", "timestamp")
        if rev and date_col:
            tmp = df[[date_col, rev]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[rev] = pd.to_numeric(tmp[rev], errors="coerce")
            today = tmp[date_col].max()
            if pd.notna(today):
                day = tmp[tmp[date_col].dt.date == today.date()]
                kpis["sales_latest_day"] = {
                    "name": "Sales (latest day)",
                    "value": float(day[rev].sum()) if len(day) else 0.0,
                }

    if ml_metrics:
        for mk, mv in ml_metrics.items():
            if isinstance(mv, (int, float, np.floating)):
                kpis[f"ml_{mk}"] = {"name": f"ML {mk.upper()}", "value": float(mv)}
            elif mk == "metrics" and isinstance(mv, dict):
                for mk2, mv2 in mv.items():
                    if isinstance(mv2, (int, float, np.floating)):
                        kpis[f"ml_{mk2}"] = {
                            "name": f"ML {mk2.upper()}",
                            "value": float(mv2),
                        }

    return kpis


def format_kpi_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {format_kpi_value(v)}" for k, v in list(value.items())[:4])
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
