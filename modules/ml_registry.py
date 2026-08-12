"""ML model catalog loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from config.settings import MODELS_CATALOG_YAML

# Aliases so smoke_test / callers can use snake_case ids
_ALIASES = {
    "random_forest_regressor": "RandomForestRegressor",
    "random_forest_classifier": "RandomForestClassifier",
    "linear_regression": "LinearRegression",
    "logistic_regression": "LogisticRegression",
    "gradient_boosting_regressor": "GradientBoostingRegressor",
    "isolation_forest": "IsolationForest",
    "xgb_regressor": "XGBRegressor",
    "xgb_classifier": "XGBClassifier",
    "xgboost": "XGBRegressor",
    "lightgbm": "LGBMRegressor",
    "lgbm_regressor": "LGBMRegressor",
    "prophet": "Prophet",
    "kmeans": "KMeans",
    "k_means": "KMeans",
    "pca": "PCA",
    "statsmodels_ols": "StatsmodelsOLS",
    "statsmodels": "StatsmodelsOLS",
    "ols": "StatsmodelsOLS",
    "dbscan": "DBSCAN",
    "pulp": "PuLP",
    "optimization_pulp": "PuLP",
    "arima": "ARIMA",
    "data_insights": "DataInsights",
    "datainsights": "DataInsights",
    "insights": "DataInsights",
}


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize catalog entry to a consistent shape."""
    mid = entry.get("id") or entry.get("model_id") or ""
    est = entry.get("estimator") or entry.get("class") or ""
    soft = bool(entry.get("optional") or entry.get("requires_license") or entry.get("soft_fail"))
    return {
        "id": mid,
        "label": entry.get("label") or entry.get("name") or mid,
        "group": entry.get("group", "other"),
        "library": entry.get("library", "sklearn"),
        "task": entry.get("task", "regression"),
        "class": est,
        "estimator": est,
        "default_params": entry.get("default_params") or entry.get("params") or {},
        "soft_fail": soft,
        "optional": soft,
        "note": entry.get("note", ""),
        "requires_license": bool(entry.get("requires_license")),
    }


def load_models_catalog(path: Optional[Path] = None) -> dict:
    p = path or MODELS_CATALOG_YAML
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("models", {})
    models: dict[str, dict] = {}

    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            norm = _normalize_entry(entry)
            if norm["id"]:
                models[norm["id"]] = norm
    elif isinstance(raw, dict):
        for mid, meta in raw.items():
            if not isinstance(meta, dict):
                continue
            entry = {"id": mid, **meta}
            models[mid] = _normalize_entry(entry)

    viz = data.get("viz_libraries") or data.get("viz_libs") or ["plotly", "matplotlib", "seaborn"]
    return {"models": models, "viz_libs": viz, "viz_libraries": viz}


def list_models(task: Optional[str] = None, include_soft_fail: bool = True) -> dict:
    catalog = load_models_catalog()
    models = catalog.get("models", {})
    out = {}
    for mid, meta in models.items():
        if not include_soft_fail and meta.get("soft_fail"):
            continue
        if task and meta.get("task") != task:
            continue
        out[mid] = meta
    return out


def get_model(model_id: str) -> Optional[dict]:
    models = load_models_catalog().get("models", {})
    if model_id in models:
        return models[model_id]
    alias = _ALIASES.get(model_id.lower().replace("-", "_"))
    if alias and alias in models:
        return models[alias]
    # case-insensitive fallback
    lower = {k.lower(): k for k in models}
    key = lower.get(model_id.lower())
    if key:
        return models[key]
    return None


def list_viz_libs() -> list:
    return load_models_catalog().get("viz_libs", ["plotly", "matplotlib", "seaborn"])
