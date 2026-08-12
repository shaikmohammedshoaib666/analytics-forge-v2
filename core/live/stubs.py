"""Live Demo Simulator — generates realistic plant-like data filtered by top filters.

Virtual universe: ~1.25M rows (5 sites × 10 lines × 25 machines × 1000 days).
Never materialized — only the filtered slice is returned.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.filters import TopFilters

VIRTUAL_SITES = [f"Site-{i}" for i in range(1, 6)]
VIRTUAL_LINES = [f"Line-{i}" for i in range(1, 11)]
VIRTUAL_MACHINES = [f"M-{i:03d}" for i in range(1, 26)]
VIRTUAL_PRODUCTS = ["ProductA", "ProductB", "ProductC", "ProductD"]
VIRTUAL_REGIONS = ["North", "South", "East", "West"]

CONNECTORS = {
    "demo_simulator": {"label": "Live Demo Simulator", "protocol": "sim"},
    "pymodbus": {"label": "Modbus TCP (sim)", "protocol": "modbus"},
    "opcua": {"label": "OPC-UA (sim)", "protocol": "opcua"},
    "api": {"label": "REST API (sim)", "protocol": "http"},
    "smps": {"label": "SMPS Serial (sim)", "protocol": "serial"},
}

VIRTUAL_UNIVERSE_SIZE = len(VIRTUAL_SITES) * len(VIRTUAL_LINES) * len(VIRTUAL_MACHINES) * 1000


def list_connectors() -> dict[str, dict[str, str]]:
    return CONNECTORS


def fetch_live_data(connector_id: str, filters: TopFilters) -> pd.DataFrame:
    """Simulate a filtered live fetch. All connectors route to the same sim backend."""
    rng = np.random.default_rng(42)
    max_rows = filters.effective_max_rows()

    sites = [filters.site] if filters.site else VIRTUAL_SITES[:2]
    lines = [filters.line] if filters.line else VIRTUAL_LINES[:3]
    machines = [filters.machine] if filters.machine else VIRTUAL_MACHINES[:5]
    products = [filters.product] if filters.product else VIRTUAL_PRODUCTS
    regions = [filters.region] if filters.region else VIRTUAL_REGIONS

    rows: list[dict[str, Any]] = []
    base_date = pd.Timestamp("2024-01-01")

    for site in sites:
        for line in lines:
            for machine in machines:
                n_points = min(20, max(1, max_rows // (len(sites) * len(lines) * len(machines))))
                for t in range(n_points):
                    rows.append({
                        "timestamp": base_date + pd.Timedelta(hours=t * 6),
                        "site": site,
                        "line": line,
                        "machine_id": machine,
                        "product": rng.choice(products),
                        "region": rng.choice(regions),
                        "temperature": round(float(rng.normal(75, 8)), 1),
                        "vibration": round(float(rng.exponential(0.5)), 3),
                        "pressure": round(float(rng.normal(101, 3)), 1),
                        "failure": int(rng.random() < 0.05),
                        "rul": int(rng.integers(10, 500)),
                    })
                    if len(rows) >= max_rows:
                        break
                if len(rows) >= max_rows:
                    break
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    df = pd.DataFrame(rows[:max_rows])
    return df
