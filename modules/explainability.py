"""Model explainability for Analytics Forge v2 — SHAP plots + model feature importance.

`shap` is an optional dependency: every helper degrades to a clear message (and the
built-in importance / permutation fallback) when it is not installed, so the Dashboard
keeps rendering on slim Render / Streamlit Cloud images.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

MAX_SHAP_ROWS = 150
MAX_BACKGROUND = 100


def shap_available() -> tuple[bool, str]:
    """(installed?, message) — never raises, used to gate the SHAP UI."""
    try:
        import shap  # noqa: F401
    except Exception as exc:  # ImportError, but broken builds can raise others
        return False, f"shap not installed ({type(exc).__name__}). Run `pip install shap`."
    try:
        version = getattr(__import__("shap"), "__version__", "?")
    except Exception:
        version = "?"
    return True, f"shap {version} ready"


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def model_feature_importance(model: Any, feature_names: Sequence[str]) -> dict[str, Any]:
    """Native importance from a fitted estimator (`feature_importances_` or `|coef_|`)."""
    names = [str(f) for f in feature_names]
    if model is None:
        return {"ok": False, "error": "No trained model in this session."}

    values: Optional[np.ndarray] = None
    source = ""
    raw = getattr(model, "feature_importances_", None)
    if raw is not None:
        values = np.asarray(raw, dtype=float)
        source = "feature_importances_"
    else:
        coef = getattr(model, "coef_", None)
        if coef is not None:
            arr = np.asarray(coef, dtype=float)
            values = np.abs(arr).mean(axis=0) if arr.ndim > 1 else np.abs(arr)
            source = "|coef_|"

    if values is None:
        return {"ok": False, "error": f"{type(model).__name__} exposes no feature importances."}
    if len(values) != len(names):
        names = [f"f{i}" for i in range(len(values))]

    table = pd.DataFrame({"feature": names, "importance": np.nan_to_num(values)}).sort_values(
        "importance", ascending=False, ignore_index=True
    )
    return {"ok": True, "table": table, "source": source, "model_name": type(model).__name__}


def permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Model-agnostic fallback when the estimator has no native importances."""
    if model is None or not isinstance(X, pd.DataFrame) or X.empty:
        return {"ok": False, "error": "Need a trained model and feature matrix."}
    from sklearn.inspection import permutation_importance

    try:
        result = permutation_importance(
            model, X, y, n_repeats=int(n_repeats), random_state=random_state, n_jobs=1
        )
    except Exception as exc:
        return {"ok": False, "error": f"Permutation importance failed: {type(exc).__name__}: {exc}"}
    table = pd.DataFrame(
        {"feature": list(X.columns), "importance": np.asarray(result.importances_mean, dtype=float)}
    ).sort_values("importance", ascending=False, ignore_index=True)
    return {"ok": True, "table": table, "source": "permutation_importance", "model_name": type(model).__name__}


def top_prediction_rows(model: Any, X: pd.DataFrame, k: int = 5) -> list[int]:
    """Positional indices of the most extreme predictions (highest risk / value)."""
    if not isinstance(X, pd.DataFrame) or X.empty:
        return []
    k = int(max(1, min(k, len(X))))
    try:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(X), dtype=float)
            scores = proba[:, -1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
        else:
            scores = np.asarray(model.predict(X), dtype=float).ravel()
    except Exception:
        return list(range(k))
    order = np.argsort(-np.nan_to_num(scores))
    return [int(i) for i in order[:k]]


def compute_shap(
    model: Any,
    X: pd.DataFrame,
    max_rows: int = MAX_SHAP_ROWS,
    class_index: Optional[int] = None,
) -> dict[str, Any]:
    """Fit a SHAP explainer and return an `Explanation` reduced to a single output.

    Tries the fast tree path first, then falls back to the model-agnostic explainer.
    """
    ok, msg = shap_available()
    if not ok:
        return {"ok": False, "error": msg, "missing_shap": True}
    if model is None or not isinstance(X, pd.DataFrame) or X.empty:
        return {"ok": False, "error": "Need a trained model and its feature matrix."}

    import shap

    sample = X.iloc[: int(max(1, min(max_rows, len(X))))].copy()
    explanation = None
    mode = ""
    try:
        explainer = shap.TreeExplainer(model)
        explanation = explainer(sample, check_additivity=False)
        mode = "TreeExplainer"
    except Exception:
        try:
            background = shap.utils.sample(X, min(MAX_BACKGROUND, len(X)), random_state=0)
            explainer = shap.Explainer(model.predict, background)
            explanation = explainer(sample)
            mode = "Explainer(predict)"
        except Exception as exc:
            return {"ok": False, "error": f"SHAP failed: {type(exc).__name__}: {exc}"}

    values = np.asarray(explanation.values)
    n_classes = int(values.shape[2]) if values.ndim == 3 else 1
    if values.ndim == 3:
        idx = int(class_index if class_index is not None else n_classes - 1)
        idx = max(0, min(idx, n_classes - 1))
        explanation = explanation[:, :, idx]
    else:
        idx = None

    return {
        "ok": True,
        "explanation": explanation,
        "X": sample,
        "mode": mode,
        "n_classes": n_classes,
        "class_index": idx,
        "features": list(sample.columns),
    }


def shap_summary_figure(payload: dict[str, Any], kind: str = "bar", max_display: int = 12):
    """Global SHAP view — `bar` (mean |SHAP|) or `beeswarm` (per-row distribution)."""
    if not payload or not payload.get("ok"):
        return None
    import shap

    plt = _matplotlib()
    explanation = payload["explanation"]
    fig = plt.figure(figsize=(7, max(3.0, 0.35 * min(max_display, len(payload.get("features") or [])) + 1.5)))
    try:
        if str(kind).lower().startswith("bee"):
            shap.plots.beeswarm(explanation, max_display=int(max_display), show=False)
        else:
            shap.plots.bar(explanation, max_display=int(max_display), show=False)
    except Exception:
        plt.close(fig)
        return None
    out = plt.gcf()
    out.tight_layout()
    return out


def shap_waterfall_figure(payload: dict[str, Any], row: int = 0, max_display: int = 12):
    """Per-prediction SHAP waterfall for one row of the explained sample."""
    if not payload or not payload.get("ok"):
        return None
    import shap

    plt = _matplotlib()
    explanation = payload["explanation"]
    idx = int(max(0, min(int(row), len(explanation) - 1)))
    fig = plt.figure(figsize=(7, max(3.0, 0.35 * int(max_display) + 1.5)))
    try:
        shap.plots.waterfall(explanation[idx], max_display=int(max_display), show=False)
    except Exception:
        plt.close(fig)
        return None
    out = plt.gcf()
    out.tight_layout()
    return out


def shap_importance_table(payload: dict[str, Any]) -> pd.DataFrame:
    """Mean |SHAP| per feature — usable when matplotlib rendering is unavailable."""
    if not payload or not payload.get("ok"):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    values = np.abs(np.asarray(payload["explanation"].values, dtype=float))
    if values.ndim == 3:
        values = values.mean(axis=2)
    return pd.DataFrame(
        {"feature": payload["features"], "mean_abs_shap": values.mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
