"""Power BI-style dashboard slicers — filter the view without re-running the pipeline."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def suggest_slicer_columns(df: pd.DataFrame, max_cats: int = 8) -> dict[str, list[str]]:
    """Pick useful categorical + numeric columns for dashboard filters."""
    cats: list[str] = []
    nums: list[str] = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if pd.api.types.is_numeric_dtype(df[c]):
            if nun >= 2:
                nums.append(c)
        else:
            if 2 <= nun <= 40:
                cats.append(c)
    return {
        "categorical": cats[:max_cats],
        "numeric": nums[:max_cats],
    }


def apply_slicers(
    df: pd.DataFrame,
    cat_filters: dict[str, list[Any]],
    num_ranges: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Apply Power BI-like slicers to a dataframe copy."""
    out = df.copy()
    for col, selected in (cat_filters or {}).items():
        if not selected or col not in out.columns:
            continue
        out = out[out[col].astype(str).isin([str(s) for s in selected])]
    for col, (lo, hi) in (num_ranges or {}).items():
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        out = out[(series >= lo) & (series <= hi)]
    return out.reset_index(drop=True)


def auto_chart_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Generate a professional chart pack including pie when possible."""
    cats = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    nums = df.select_dtypes(include=["number"]).columns.tolist()
    specs: list[dict[str, Any]] = []

    if cats and nums:
        specs.append({
            "chart_type": "pie",
            "lib": "plotly",
            "names": cats[0],
            "values": nums[0],
            "title": f"{nums[0]} by {cats[0]}",
        })
        specs.append({
            "chart_type": "bar",
            "lib": "plotly",
            "x": cats[0],
            "y": nums[0],
            "title": f"{nums[0]} by {cats[0]}",
        })
    if len(nums) >= 2:
        specs.append({
            "chart_type": "scatter",
            "lib": "plotly",
            "x": nums[0],
            "y": nums[1],
            "title": f"{nums[0]} vs {nums[1]}",
        })
    date_cols = [c for c in df.columns if any(h in str(c).lower() for h in ("date", "time", "timestamp"))]
    if date_cols and nums:
        specs.append({
            "chart_type": "line",
            "lib": "plotly",
            "x": date_cols[0],
            "y": nums[0],
            "title": f"{nums[0]} over time",
        })
    elif nums:
        specs.append({
            "chart_type": "line",
            "lib": "plotly",
            "x": df.columns[0],
            "y": nums[0],
            "title": f"{nums[0]} trend",
        })
    return specs[:5]
