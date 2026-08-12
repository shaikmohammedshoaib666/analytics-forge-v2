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
    return templates.get(domain, templates.get("generic", {}))
