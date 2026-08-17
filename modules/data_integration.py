"""Multi-file data integration with SQL-style joins (ported from analytics-forge v1)."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

JOIN_TYPES = {
    "inner": "INNER JOIN — only matching keys in both tables",
    "left": "LEFT JOIN — all rows from left + matches from right",
    "right": "RIGHT JOIN — all rows from right + matches from left",
    "outer": "FULL OUTER JOIN — all rows from both tables",
}

_PREFERRED_KEYS = {"machine_id", "asset_id", "id", "timestamp", "date", "customer_id", "sku"}


def _is_frame(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame)


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load csv/tsv/xlsx/json/parquet into a DataFrame from a Streamlit UploadedFile or path-like."""
    name = getattr(uploaded_file, "name", str(uploaded_file)).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(uploaded_file)
    if name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    if name.endswith(".tsv"):
        return pd.read_csv(uploaded_file, sep="\t")
    df = pd.read_csv(uploaded_file)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def suggest_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Intersect column names as candidate join keys. Never uses DataFrame truthiness."""
    if not _is_frame(left) or not _is_frame(right):
        return []
    common = sorted(set(left.columns) & set(right.columns))
    preferred = [c for c in common if str(c).lower() in _PREFERRED_KEYS]
    rest = [c for c in common if c not in preferred]
    return preferred + rest


def join_two(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on: Optional[list[str]] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffixes: tuple[str, str] = ("_l", "_r"),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not _is_frame(left) or not _is_frame(right):
        raise TypeError("join_two requires pandas DataFrames (empty frames are allowed).")

    how = (how or "inner").lower()
    if how not in JOIN_TYPES:
        raise ValueError(f"Unsupported join type: {how}. Use one of {list(JOIN_TYPES)}")

    meta: dict[str, Any] = {
        "how": how,
        "left_rows": len(left),
        "right_rows": len(right),
    }
    keys = [k for k in (on or []) if k]
    if keys:
        merged = pd.merge(left, right, how=how, on=keys, suffixes=suffixes)
        meta["keys"] = keys
    elif left_on and right_on:
        merged = pd.merge(left, right, how=how, left_on=left_on, right_on=right_on, suffixes=suffixes)
        meta["keys"] = [left_on, right_on]
    else:
        auto = suggest_join_keys(left, right)
        if not auto:
            raise ValueError("No common columns to join on. Pick join keys explicitly.")
        merged = pd.merge(left, right, how=how, on=auto[:1], suffixes=suffixes)
        meta["keys"] = auto[:1]
        meta["auto_key"] = True

    meta["result_rows"] = len(merged)
    meta["result_cols"] = list(merged.columns)
    return merged, meta


def join_many(
    tables: dict[str, pd.DataFrame],
    steps: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Chain joins across 3+ named tables.

    steps example:
      [
        {"left": "sensors", "right": "maintenance", "how": "left", "on": ["machine_id"]},
        {"left": "_result", "right": "costs", "how": "inner", "on": ["machine_id"]},
      ]
    After step 1, the working frame is registered as "_result".
    """
    if not tables:
        raise ValueError("No tables provided")
    if not steps:
        raise ValueError("Provide at least one join step")

    registry: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        if _is_frame(df):
            registry[name] = df

    first_left = steps[0].get("left")
    if first_left not in registry:
        raise KeyError(f"Unknown left table: {first_left}")

    working = registry[first_left].copy()
    logs: list[dict[str, Any]] = []

    for i, step in enumerate(steps):
        right_name = step["right"]
        if right_name not in registry:
            raise KeyError(f"Unknown right table: {right_name}")
        left_name = step.get("left", "_result")
        if i == 0 and left_name != "_result":
            left_df = registry[left_name]
        else:
            left_df = working
        right_df = registry[right_name]
        working, meta = join_two(
            left_df,
            right_df,
            how=step.get("how", "inner"),
            on=step.get("on"),
            left_on=step.get("left_on"),
            right_on=step.get("right_on"),
        )
        meta["step"] = i + 1
        meta["left_name"] = left_name
        meta["right_name"] = right_name
        logs.append(meta)
        registry["_result"] = working

    return working, logs
