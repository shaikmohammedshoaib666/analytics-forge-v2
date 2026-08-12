"""Generic data cleaning with step logging."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply a standard cleaning pipeline.
    Returns (clean_df, log) where log entries document pandas operations used.
    """
    log: list[dict[str, Any]] = []
    out = df.copy()
    rows_before = len(out)

    # 1. Strip column names
    new_cols = [str(c).strip() for c in out.columns]
    if list(out.columns) != new_cols:
        out.columns = new_cols
        log.append(
            {
                "operation": "strip_column_names",
                "detail": "df.columns = [str(c).strip() for c in df.columns]",
                "rows_before": rows_before,
                "rows_after": len(out),
            }
        )

    # 2. Drop fully empty columns
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        log.append(
            {
                "operation": "drop_empty_columns",
                "detail": f"df.drop(columns={empty_cols})",
                "rows_before": rows_before,
                "rows_after": len(out),
            }
        )

    # 3. Drop duplicate column names (keep first)
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
        log.append(
            {
                "operation": "drop_duplicate_columns",
                "detail": "df.loc[:, ~df.columns.duplicated()]",
                "rows_before": rows_before,
                "rows_after": len(out),
            }
        )

    # 4. Strip string cells
    obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    if obj_cols:
        for c in obj_cols:
            out[c] = out[c].apply(lambda x: x.strip() if isinstance(x, str) else x)
        log.append(
            {
                "operation": "strip_strings",
                "detail": f"strip() on columns {obj_cols}",
                "rows_before": len(out),
                "rows_after": len(out),
            }
        )

    # 5. Replace common null sentinels
    sentinels = ["", "NA", "N/A", "n/a", "null", "Null", "NULL", "None", "-", "--"]
    before_na = int(out.isna().sum().sum())
    out = out.replace(sentinels, np.nan)
    after_na = int(out.isna().sum().sum())
    if after_na != before_na:
        log.append(
            {
                "operation": "replace_null_sentinels",
                "detail": f"df.replace({sentinels}, np.nan)",
                "rows_before": len(out),
                "rows_after": len(out),
            }
        )

    # 6. Coerce numeric-looking object columns
    for c in list(out.columns):
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            # If majority converts, keep numeric
            non_null_orig = out[c].notna().sum()
            if non_null_orig > 0 and converted.notna().sum() / non_null_orig >= 0.8:
                out[c] = converted
                log.append(
                    {
                        "operation": "to_numeric",
                        "detail": f"pd.to_numeric(df['{c}'], errors='coerce')",
                        "rows_before": len(out),
                        "rows_after": len(out),
                    }
                )

    # 7. Parse datetimes for columns with date-like names or values
    date_hints = ("date", "time", "timestamp", "datetime", "day", "month", "year")
    for c in list(out.columns):
        cl = str(c).lower()
        if any(h in cl for h in date_hints) and out[c].dtype == object:
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().sum() > 0:
                out[c] = parsed
                log.append(
                    {
                        "operation": "to_datetime",
                        "detail": f"pd.to_datetime(df['{c}'], errors='coerce')",
                        "rows_before": len(out),
                        "rows_after": len(out),
                    }
                )

    # 8. Drop fully empty rows
    before = len(out)
    out = out.dropna(how="all")
    if len(out) != before:
        log.append(
            {
                "operation": "drop_empty_rows",
                "detail": "df.dropna(how='all')",
                "rows_before": before,
                "rows_after": len(out),
            }
        )

    # 9. Drop duplicate rows
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(
            {
                "operation": "drop_duplicates",
                "detail": "df.drop_duplicates()",
                "rows_before": before,
                "rows_after": len(out),
            }
        )

    # 10. Fill numeric NaN with median; categorical with mode / 'Unknown'
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if out[c].isna().any():
            med = out[c].median()
            out[c] = out[c].fillna(med)
            log.append(
                {
                    "operation": "fillna_median",
                    "detail": f"df['{c}'].fillna(df['{c}'].median()) -> {med}",
                    "rows_before": len(out),
                    "rows_after": len(out),
                }
            )

    cat_cols = out.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill)
            log.append(
                {
                    "operation": "fillna_mode",
                    "detail": f"df['{c}'].fillna('{fill}')",
                    "rows_before": len(out),
                    "rows_after": len(out),
                }
            )

    # 11. Reset index
    out = out.reset_index(drop=True)
    log.append(
        {
            "operation": "reset_index",
            "detail": "df.reset_index(drop=True)",
            "rows_before": len(out),
            "rows_after": len(out),
        }
    )

    if not log:
        log.append(
            {
                "operation": "noop",
                "detail": "No cleaning changes required",
                "rows_before": rows_before,
                "rows_after": len(out),
            }
        )

    return out, log
