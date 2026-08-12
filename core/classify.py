"""Auto-detect one of the 8 solution fields."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import yaml

from config.settings import CONFIG_DIR


def load_domains() -> dict[str, Any]:
    path = CONFIG_DIR / "domains.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    domains = data.get("domains", {})
    # Normalize: ensure label alias for name
    for domain_id, meta in domains.items():
        if isinstance(meta, dict) and "label" not in meta:
            meta["label"] = meta.get("name", domain_id)
    return domains


def classify_dataframe(
    df: pd.DataFrame,
    override: Optional[str] = None,
) -> dict[str, Any]:
    domains = load_domains()

    if override and override in domains:
        meta = domains[override]
        return {
            "domain": override,
            "confidence": 1.0,
            "label": meta.get("label", meta.get("name", override)),
            "scores": {override: 1.0},
            "meta": meta,
            "all_domains": domains,
            "recommended_models": meta.get("recommended_models", []),
            "override": True,
        }

    blob = " ".join([str(c).lower() for c in df.columns])
    sample = " ".join(
        df.astype(str).head(20).fillna("").astype(str).values.ravel()[:200]
    ).lower()
    text = blob + " " + sample

    scores: dict[str, float] = {}
    for domain_id, meta in domains.items():
        if domain_id == "generic":
            continue
        keywords = [str(k).lower() for k in meta.get("keywords", [])]
        hit = sum(1 for k in keywords if k in text)
        scores[domain_id] = float(hit)

    if not scores or max(scores.values()) <= 0:
        best = "generic"
        confidence = 0.35
    else:
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        total = sum(scores.values()) or 1.0
        confidence = min(0.95, 0.45 + (scores[best] / total) * 0.5)

    meta = domains.get(best, domains.get("generic", {}))
    return {
        "domain": best,
        "confidence": round(confidence, 3),
        "label": meta.get("label", meta.get("name", best)),
        "scores": scores,
        "meta": meta,
        "all_domains": domains,
        "recommended_models": meta.get("recommended_models", []),
        "override": False,
    }


def classify(df: pd.DataFrame, override: Optional[str] = None) -> dict[str, Any]:
    """Public alias used by app/pipeline."""
    return classify_dataframe(df, override=override)
