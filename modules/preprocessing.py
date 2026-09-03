"""Preprocessing add-ons for the Clean page — scaling (Standard/Robust/MinMax) + IQR outlier removal.

Pure pandas/sklearn helpers: no Streamlit import so they stay unit-testable and can be
called from `clean_data()` without changing the existing DWDM clean behaviour.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

SCALER_CHOICES = ["None", "Standard", "Robust", "MinMax"]
SCALER_HELP = {
    "None": "Keep raw units (default) — KPIs and currency stay readable.",
    "Standard": "Zero mean / unit variance. Best when features are roughly gaussian.",
    "Robust": "Median / IQR scaling. Best when outliers or heavy tails are present.",
    "MinMax": "Squash to 0–1. Best for bounded sensors and neural-style models.",
}

_SKIP_SUFFIXES = ("_id", "_bin", "_flag")
_SKIP_NAMES = {"id", "index", "uuid", "row", "rownum"}


def scalable_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns worth scaling — skips ids, binned/engineered labels and constants."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out: list[str] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        name = str(col)
        low = name.lower()
        if low in _SKIP_NAMES or low.endswith(_SKIP_SUFFIXES) or "unnamed" in low:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.nunique() <= 2:  # constants and 0/1 flags (often the target) stay untouched
            continue
        out.append(name)
    return out


def build_scaler(kind: str):
    """Return a fresh sklearn scaler for `kind` ("standard" / "robust" / "minmax")."""
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    key = str(kind or "").strip().lower()
    if key in ("standard", "standardscaler", "zscore", "z-score"):
        return StandardScaler()
    if key in ("robust", "robustscaler", "iqr"):
        return RobustScaler()
    if key in ("minmax", "min-max", "min_max", "minmaxscaler"):
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {kind!r} (use Standard / Robust / MinMax)")


def apply_scaling(
    df: pd.DataFrame,
    kind: str,
    columns: Optional[Iterable[str]] = None,
    suffix: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Scale numeric columns with the chosen scaler.

    `suffix=None` rescales in place so downstream pages consume scaled values;
    passing e.g. `"_scaled"` keeps the raw columns and appends new ones.
    """
    meta: dict[str, Any] = {"ok": False, "scaler": kind, "columns": [], "log": [], "suffix": suffix}
    if not isinstance(df, pd.DataFrame) or df.empty:
        meta["error"] = "empty frame"
        return df, meta
    if str(kind or "None").strip().lower() in ("", "none", "raw", "off"):
        meta.update({"ok": True, "scaler": "None", "log": ["scaling: off (raw units kept)"]})
        return df.copy(), meta

    cols = [str(c) for c in (columns if columns is not None else scalable_columns(df))]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        meta["error"] = "no numeric columns available to scale"
        return df.copy(), meta

    out = df.copy()
    values = out[cols].apply(pd.to_numeric, errors="coerce")
    filled = values.fillna(values.median(numeric_only=True))
    filled = filled.fillna(0.0)
    try:
        scaler = build_scaler(kind)
        scaled = scaler.fit_transform(filled.to_numpy(dtype=float))
    except Exception as exc:  # keep the clean pipeline alive on any sklearn hiccup
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return df.copy(), meta

    scaled_df = pd.DataFrame(scaled, columns=cols, index=out.index)
    if suffix:
        for col in cols:
            out[f"{col}{suffix}"] = scaled_df[col]
        meta["log"].append(f"{kind} scaler → {len(cols)} new `{suffix}` columns")
    else:
        for col in cols:
            out[col] = scaled_df[col]
        meta["log"].append(f"{kind} scaler applied in place to {len(cols)} numeric columns")
    meta.update({"ok": True, "columns": cols, "scaler": kind})
    return out, meta


def iqr_bounds(series: pd.Series, multiplier: float = 1.5) -> tuple[float, float]:
    """Tukey fence for one numeric series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def iqr_outlier_mask(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    multiplier: float = 1.5,
) -> tuple[pd.Series, pd.DataFrame]:
    """Boolean mask (True = outlier row) plus a per-column breakdown table."""
    empty_report = pd.DataFrame(columns=["column", "lower", "upper", "outliers"])
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=bool), empty_report

    cols = [str(c) for c in (columns if columns is not None else scalable_columns(df))]
    cols = [c for c in cols if c in df.columns]
    mask = pd.Series(False, index=df.index)
    rows: list[dict[str, Any]] = []
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 5:
            continue
        low, high = iqr_bounds(series, multiplier)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue
        hits = (series < low) | (series > high)
        hits = hits.fillna(False)
        if hits.any():
            mask = mask | hits
        rows.append(
            {
                "column": col,
                "lower": round(float(low), 4),
                "upper": round(float(high), 4),
                "outliers": int(hits.sum()),
            }
        )
    report = pd.DataFrame(rows).sort_values("outliers", ascending=False) if rows else empty_report
    return mask, report.reset_index(drop=True)


def remove_iqr_outliers(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    multiplier: float = 1.5,
    max_removed_frac: float = 0.6,
    min_rows_kept: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop rows outside the Tukey fence, refusing to gut the dataset.

    A too-aggressive multiplier on skewed industrial data can wipe out most rows, so the
    filter is skipped (and reported) when it would remove more than `max_removed_frac`.
    """
    meta: dict[str, Any] = {
        "ok": False,
        "multiplier": float(multiplier),
        "rows_before": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "rows_after": int(len(df)) if isinstance(df, pd.DataFrame) else 0,
        "removed": 0,
        "columns": [],
        "log": [],
    }
    if not isinstance(df, pd.DataFrame) or df.empty:
        meta["error"] = "empty frame"
        return df, meta

    mask, report = iqr_outlier_mask(df, columns=columns, multiplier=multiplier)
    meta["report"] = report
    meta["columns"] = report["column"].tolist() if not report.empty else []
    if mask.empty or not bool(mask.any()):
        meta.update({"ok": True, "log": [f"IQR filter (k={multiplier}): no outlier rows found"]})
        return df.copy(), meta

    removed = int(mask.sum())
    kept = len(df) - removed
    if kept < min_rows_kept or removed / max(1, len(df)) > max_removed_frac:
        meta["error"] = (
            f"skipped — would drop {removed}/{len(df)} rows "
            f"({removed / max(1, len(df)) * 100:.0f}%); raise the multiplier"
        )
        meta["log"].append(meta["error"])
        return df.copy(), meta

    out = df.loc[~mask].reset_index(drop=True)
    meta.update(
        {
            "ok": True,
            "removed": removed,
            "rows_after": int(len(out)),
            "log": [f"IQR filter (k={multiplier}) removed {removed} outlier rows"],
        }
    )
    return out, meta
