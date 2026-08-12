"""Top filters for live mode — gate fetches so full plant data is never loaded."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import pandas as pd

MAX_ROWS_HARD_CAP = 50_000
DEFAULT_MAX_ROWS = 5_000


@dataclass
class TopFilters:
    site: Optional[str] = None
    line: Optional[str] = None
    machine: Optional[str] = None
    product: Optional[str] = None
    region: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    max_rows: int = DEFAULT_MAX_ROWS

    def effective_max_rows(self) -> int:
        return min(self.max_rows, MAX_ROWS_HARD_CAP)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def apply_buffer_filters(df: pd.DataFrame, filters: TopFilters) -> pd.DataFrame:
    """Apply filters to an already-loaded in-memory buffer."""
    out = df.copy()
    col_map = {c.lower(): c for c in out.columns}

    for fkey, col_hint in [
        ("site", "site"), ("line", "line"), ("machine", "machine"),
        ("product", "product"), ("region", "region"),
    ]:
        val = getattr(filters, fkey, None)
        if val and col_hint in col_map:
            real_col = col_map[col_hint]
            out = out[out[real_col].astype(str).str.lower() == val.lower()]

    if filters.date_from or filters.date_to:
        date_col = next((col_map[k] for k in ("timestamp", "date", "order_date", "datetime") if k in col_map), None)
        if date_col:
            dt = pd.to_datetime(out[date_col], errors="coerce")
            if filters.date_from:
                out = out[dt >= pd.Timestamp(filters.date_from)]
            if filters.date_to:
                out = out[dt <= pd.Timestamp(filters.date_to)]

    return out.head(filters.effective_max_rows()).reset_index(drop=True)
