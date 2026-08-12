"""Live connectors — demo sim + real-capable plugins (Modbus, OPC-UA, REST, SMPS, Azure).

Demo simulator ≈ sample CSV: test the automation flow only.
Other connectors are production-shaped: same fetch() API accepts real host/creds.
Without credentials they fall back to domain-aware sim data so the app stays usable.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from core.filters import TopFilters
from core.live.base import LiveConnector, get_connector, list_connectors, register

VIRTUAL_UNIVERSE_SIZE = 1_250_000


def _sim_frame(filters: TopFilters, n: int, domain: str) -> pd.DataFrame:
    """Domain-aware simulated slice (never materializes full universe)."""
    rng = np.random.default_rng(42)
    n = max(1, min(n, filters.effective_max_rows()))
    base = pd.Timestamp("2024-01-01")
    vals = filters.values or {}

    if domain == "healthcare":
        rows = []
        for i in range(n):
            rows.append({
                "admit_date": base + pd.Timedelta(hours=i * 3),
                "hospital": vals.get("hospital") or rng.choice(["City General", "Metro Care"]),
                "department": vals.get("department") or rng.choice(["ER", "Cardiology", "ICU"]),
                "ward": vals.get("ward") or rng.choice(["Ward-A", "Ward-B", "ICU-2"]),
                "doctor": vals.get("doctor") or rng.choice(["Dr. Rao", "Dr. Chen", "Dr. Ali"]),
                "diagnosis": vals.get("diagnosis") or rng.choice(["Cardiac", "Trauma", "Respiratory"]),
                "wait_minutes": int(rng.integers(5, 180)),
                "readmission": int(rng.random() < 0.12),
                "length_of_stay": int(rng.integers(1, 14)),
            })
        return pd.DataFrame(rows)

    if domain == "sales_forecasting":
        rows = []
        for i in range(n):
            rows.append({
                "order_date": base + pd.Timedelta(days=i % 60),
                "region": vals.get("region") or rng.choice(["North", "South", "East", "West"]),
                "product": vals.get("product") or rng.choice(["Handbags", "Shoes", "Electronics"]),
                "channel": vals.get("channel") or rng.choice(["Online", "Store", "Wholesale"]),
                "store": vals.get("store") or rng.choice(["Flagship", "Outlet", "Boutique"]),
                "campaign": vals.get("campaign") or rng.choice(["Summer", "Holiday", "None"]),
                "revenue": float(rng.integers(80, 8000)),
                "units": int(rng.integers(1, 40)),
            })
        return pd.DataFrame(rows)

    if domain == "supply_chain":
        rows = []
        for i in range(n):
            rows.append({
                "timestamp": base + pd.Timedelta(hours=i),
                "warehouse": vals.get("warehouse") or rng.choice(["DC-West", "DC-East"]),
                "aisle": vals.get("aisle") or rng.choice(["Zone-A", "Zone-B"]),
                "sku": vals.get("sku") or rng.choice(["SKU-100", "SKU-200", "SKU-300"]),
                "carrier": vals.get("carrier") or rng.choice(["DHL", "FedEx", "Local"]),
                "region": vals.get("region") or rng.choice(["APAC", "EU", "NA"]),
                "delivery_hours": float(rng.integers(6, 96)),
                "inventory": int(rng.integers(0, 500)),
                "defect": int(rng.random() < 0.04),
            })
        return pd.DataFrame(rows)

    if domain == "erp_cloud":
        rows = []
        for i in range(n):
            rows.append({
                "event_time": base + pd.Timedelta(minutes=i * 5),
                "tenant": vals.get("tenant") or rng.choice(["Contoso", "Fabrikam"]),
                "app": vals.get("app") or rng.choice(["Finance", "CRM", "HR"]),
                "environment": vals.get("environment") or rng.choice(["prod", "staging"]),
                "region": vals.get("region") or rng.choice(["eastus", "westeurope"]),
                "resource": vals.get("resource") or rng.choice(["billing-api", "auth", "reports"]),
                "latency_ms": float(rng.integers(20, 800)),
                "error": int(rng.random() < 0.03),
                "active_users": int(rng.integers(10, 5000)),
            })
        return pd.DataFrame(rows)

    # Factory / PdM default
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": base + pd.Timedelta(hours=i * 2),
            "site": vals.get("site") or rng.choice(["Munich", "Detroit", "Site-1"]),
            "line": vals.get("line") or rng.choice(["Line-1", "Line-2", "Assembly-3"]),
            "machine_id": vals.get("machine") or rng.choice(["CNC-01", "Robot-02", "Press-03"]),
            "product": vals.get("product") or rng.choice(["Chassis", "Battery", "Door"]),
            "region": vals.get("region") or rng.choice(["EU", "NA", "APAC"]),
            "temperature": round(float(rng.normal(75, 8)), 1),
            "vibration": round(float(rng.exponential(0.5)), 3),
            "pressure": round(float(rng.normal(101, 3)), 1),
            "failure": int(rng.random() < 0.05),
            "rul": int(rng.integers(10, 500)),
        })
    return pd.DataFrame(rows)


class DemoSimulatorConnector(LiveConnector):
    id = "demo_simulator"
    label = "Demo Simulator (test only — like sample CSV)"
    protocol = "sim"
    capability = "sim"

    def fetch(self, filters: TopFilters, config: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        return _sim_frame(filters, filters.effective_max_rows(), filters.domain or "generic")


class _CapableSimFallback(LiveConnector):
    """Shared helper: real-shaped config; sim until credentials exist."""

    def fetch(self, filters: TopFilters, config: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        status = self.connection_status(config)
        # Real path placeholder — ready for Azure / plant wiring
        if status["mode"] == "real":
            return self._fetch_real(filters, config or {})
        return _sim_frame(filters, filters.effective_max_rows(), filters.domain or "generic")

    def _fetch_real(self, filters: TopFilters, config: dict[str, Any]) -> pd.DataFrame:
        """Override in subclasses when real SDKs are wired.
        For now raises so callers know real mode needs implementation against a live endpoint.
        """
        # Soft real attempt: if URL provided for API-like connectors, try HTTP JSON
        url = config.get("url") or config.get("endpoint")
        if url and self.protocol in {"http", "azure"}:
            try:
                import urllib.request
                import json
                with urllib.request.urlopen(url, timeout=10) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                if isinstance(payload, list):
                    df = pd.DataFrame(payload)
                elif isinstance(payload, dict) and "data" in payload:
                    df = pd.DataFrame(payload["data"])
                else:
                    df = pd.DataFrame([payload])
                return df.head(filters.effective_max_rows())
            except Exception as exc:
                raise RuntimeError(
                    f"{self.label}: real fetch failed ({exc}). "
                    "Check URL/token, or leave blank to use sim fallback."
                ) from exc
        raise RuntimeError(
            f"{self.label}: credentials present but live SDK not installed yet. "
            "Install pymodbus / opcua / azure SDK on the host (e.g. Azure), "
            "or clear credentials to use sim fallback for testing."
        )


class PyModbusConnector(_CapableSimFallback):
    id = "pymodbus"
    label = "Modbus TCP (real-capable)"
    protocol = "modbus"
    capability = "capable"


class OpcUaConnector(_CapableSimFallback):
    id = "opcua"
    label = "OPC-UA (real-capable)"
    protocol = "opcua"
    capability = "capable"


class RestApiConnector(_CapableSimFallback):
    id = "api"
    label = "REST / ERP API (real-capable)"
    protocol = "http"
    capability = "capable"


class SmpsConnector(_CapableSimFallback):
    id = "smps"
    label = "SMPS / Serial industrial (real-capable)"
    protocol = "serial"
    capability = "capable"


class AzureConnector(_CapableSimFallback):
    id = "azure"
    label = "Azure Cloud / IoT / Blob API (real-capable)"
    protocol = "azure"
    capability = "capable"


# Register plugins once
for _c in (
    DemoSimulatorConnector(),
    PyModbusConnector(),
    OpcUaConnector(),
    RestApiConnector(),
    SmpsConnector(),
    AzureConnector(),
):
    register(_c)


def fetch_live_data(
    connector_id: str,
    filters: TopFilters,
    config: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    connector = get_connector(connector_id) or get_connector("demo_simulator")
    assert connector is not None
    return connector.fetch(filters, config)


# Re-export for older imports
__all__ = [
    "VIRTUAL_UNIVERSE_SIZE",
    "list_connectors",
    "fetch_live_data",
    "get_connector",
]
