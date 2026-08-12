"""Live connector plugin interface.

Design:
- Demo simulator = like sample CSV: for testing the pipeline only.
- pymodbus / OPC-UA / REST / SMPS / Azure = real-capable plugins.
  Right now they use sim backends when credentials are missing,
  but the same class methods accept real host/url/creds for industry later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from core.filters import TopFilters


class LiveConnector(ABC):
    """Base plugin for live / industry data sources."""

    id: str = "base"
    label: str = "Base Connector"
    protocol: str = "none"
    # "sim" = demo only; "capable" = real connection path implemented (creds optional)
    capability: str = "sim"

    @abstractmethod
    def fetch(self, filters: TopFilters, config: Optional[dict[str, Any]] = None) -> pd.DataFrame:
        """Pull ONLY the filtered slice — never the full source."""

    def connection_status(self, config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Report whether real credentials are present."""
        cfg = config or {}
        has_host = bool(cfg.get("host") or cfg.get("url") or cfg.get("endpoint"))
        has_auth = bool(cfg.get("token") or cfg.get("username") or cfg.get("connection_string"))
        if self.capability == "sim":
            return {
                "mode": "demo",
                "ready_for_real": False,
                "message": "Demo simulator only — for pipeline testing (like sample CSV).",
            }
        if has_host and has_auth:
            return {
                "mode": "real",
                "ready_for_real": True,
                "message": f"Credentials present — {self.label} will attempt a real fetch.",
            }
        return {
            "mode": "sim_fallback",
            "ready_for_real": False,
            "message": (
                f"{self.label} is REAL-CAPABLE. No host/credentials yet → using sim data. "
                "Add host/url + token (or Azure connection string) to connect to industry later."
            ),
        }


# Registry filled by stubs / real plugins
_REGISTRY: dict[str, LiveConnector] = {}


def register(connector: LiveConnector) -> None:
    _REGISTRY[connector.id] = connector


def get_connector(connector_id: str) -> Optional[LiveConnector]:
    return _REGISTRY.get(connector_id)


def list_connectors() -> dict[str, dict[str, str]]:
    return {
        cid: {
            "label": c.label,
            "protocol": c.protocol,
            "capability": c.capability,
        }
        for cid, c in _REGISTRY.items()
    }
