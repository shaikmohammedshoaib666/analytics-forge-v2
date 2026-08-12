"""Industry-adaptive top filters — fields change by domain (factory ≠ healthcare ≠ sales)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import pandas as pd

from core.templates import get_template

MAX_ROWS_HARD_CAP = 50_000
DEFAULT_MAX_ROWS = 5_000

# Shared aliases so buffer filtering works across industry column names
COLUMN_ALIASES: dict[str, list[str]] = {
    "site": ["site", "plant", "facility", "location"],
    "line": ["line", "production_line", "assembly_line"],
    "machine": ["machine", "machine_id", "asset", "equipment"],
    "product": ["product", "sku", "category", "item"],
    "region": ["region", "market", "geo"],
    "hospital": ["hospital", "facility", "site"],
    "department": ["department", "dept", "specialty"],
    "ward": ["ward", "unit", "clinic"],
    "doctor": ["doctor", "physician", "provider"],
    "diagnosis": ["diagnosis", "dx", "condition"],
    "warehouse": ["warehouse", "dc", "depot", "site"],
    "aisle": ["aisle", "zone", "bin"],
    "sku": ["sku", "item", "product"],
    "carrier": ["carrier", "shipper"],
    "channel": ["channel", "sales_channel"],
    "store": ["store", "boutique", "outlet"],
    "campaign": ["campaign", "promo"],
    "tenant": ["tenant", "org", "organization", "account"],
    "app": ["app", "application", "module"],
    "environment": ["environment", "env", "stage"],
    "resource": ["resource", "service", "endpoint"],
    "source": ["source", "system"],
}


@dataclass
class TopFilters:
    """Dynamic filter bag — keys depend on industry template."""
    values: dict[str, Optional[str]] = field(default_factory=dict)
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    max_rows: int = DEFAULT_MAX_ROWS
    domain: str = "generic"

    def effective_max_rows(self) -> int:
        return min(int(self.max_rows), MAX_ROWS_HARD_CAP)

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.values.items() if v}
        if self.date_from:
            out["date_from"] = self.date_from
        if self.date_to:
            out["date_to"] = self.date_to
        out["max_rows"] = self.effective_max_rows()
        out["domain"] = self.domain
        return out

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)


def filter_schema_for_domain(domain: str) -> list[dict[str, str]]:
    """Return industry-specific filter field definitions."""
    tpl = get_template(domain) or get_template("generic")
    fields = tpl.get("filter_fields") or []
    if fields:
        return fields
    # Fallback generic
    return [
        {"key": "region", "label": "Region / Segment", "hint": "optional"},
        {"key": "product", "label": "Product / Category", "hint": "optional"},
    ]


def apply_buffer_filters(df: pd.DataFrame, filters: TopFilters) -> pd.DataFrame:
    """Apply filters to an already-loaded in-memory buffer using column aliases."""
    out = df.copy()
    col_map = {c.lower(): c for c in out.columns}

    for key, val in (filters.values or {}).items():
        if not val:
            continue
        aliases = COLUMN_ALIASES.get(key, [key])
        matched_col = next((col_map[a] for a in aliases if a in col_map), None)
        # also try exact key.lower()
        if matched_col is None and key.lower() in col_map:
            matched_col = col_map[key.lower()]
        if matched_col is None:
            continue
        out = out[out[matched_col].astype(str).str.lower() == str(val).lower()]

    if filters.date_from or filters.date_to:
        date_col = next(
            (
                col_map[k]
                for k in (
                    "timestamp", "date", "order_date", "datetime",
                    "admit_date", "event_time", "created_at",
                )
                if k in col_map
            ),
            None,
        )
        if date_col:
            dt = pd.to_datetime(out[date_col], errors="coerce")
            if filters.date_from:
                out = out[dt >= pd.Timestamp(filters.date_from)]
            if filters.date_to:
                out = out[dt <= pd.Timestamp(filters.date_to)]

    return out.head(filters.effective_max_rows()).reset_index(drop=True)
