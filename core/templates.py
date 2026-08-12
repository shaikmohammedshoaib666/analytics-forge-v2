"""Industry templates loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "config" / "industry_templates.yaml"


def load_templates() -> dict[str, Any]:
    if not TEMPLATES_PATH.exists():
        return {}
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("templates", {})


def get_template(domain: str) -> dict[str, Any]:
    templates = load_templates()
    if domain in templates:
        return templates[domain]
    # Map common aliases
    aliases = {
        "pdm": "predictive_maintenance",
        "manufacturing": "predictive_maintenance",
        "factory": "predictive_maintenance",
        "sales": "sales_forecasting",
        "retail": "sales_forecasting",
        "hospital": "healthcare",
        "warehouse": "supply_chain",
        "logistics": "supply_chain",
        "azure": "erp_cloud",
        "cloud": "erp_cloud",
        "erp": "erp_cloud",
    }
    mapped = aliases.get((domain or "").lower())
    if mapped and mapped in templates:
        return templates[mapped]
    return templates.get("generic", {})


def insight_text(domain: str, kind: str) -> str:
    tpl = get_template(domain)
    return (tpl.get("insight_templates") or {}).get(kind, "")
