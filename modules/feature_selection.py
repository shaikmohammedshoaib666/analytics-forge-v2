"""Advanced feature selection for Analytics Forge v2.

Domain-agnostic scoring so the Field page can rank columns for any pack
(PdM, sales, healthcare, churn, ...):

* ANOVA F-test        — linear class separation / linear signal
* Mutual information  — non-linear dependency
* Correlation pairs   — redundancy detection with an auto-drop suggestion
* Tree importance     — RandomForest `feature_importances_`
* RFE                 — recursive elimination curve + ranking
* PCA                 — explained-variance curve

Every entry point returns a plain dict with `ok` / `error` so the UI can degrade
gracefully instead of raising inside a Streamlit rerun.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

MAX_ROWS = 20_000
CORR_DEFAULT_THRESHOLD = 0.95

_SKIP_NAMES = {"id", "uuid", "index", "row", "rownum"}
_DATE_HINTS = ("timestamp", "datetime", "date", "time")


def candidate_features(df: pd.DataFrame, target: Optional[str] = None) -> list[str]:
    """Columns usable as model inputs — mirrors the ML Studio feature picker rules."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    feats: list[str] = []
    n = len(df)
    for col in df.columns:
        name = str(col)
        low = name.lower()
        if target is not None and name == str(target):
            continue
        if low in _SKIP_NAMES or low.endswith("_id") or "unnamed" in low:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if any(h in low for h in _DATE_HINTS) and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > min(40, max(15, int(n * 0.4))):
                continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        feats.append(name)
    return feats


def infer_task(y: pd.Series) -> str:
    """`classification` for labels / low-cardinality integers, else `regression`."""
    series = pd.Series(y).dropna()
    if series.empty:
        return "regression"
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    nunique = int(series.nunique())
    if nunique <= 2:
        return "classification"
    if nunique <= 20 and float(pd.to_numeric(series, errors="coerce").dropna().mod(1).abs().max() or 0) == 0.0:
        return "classification"
    return "regression"


def encode_matrix(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Numeric-only matrix: categoricals factorized, NaNs filled with the column median."""
    cols = [c for c in features if c in df.columns]
    out = df[cols].copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.factorize(out[col].astype(str))[0].astype(float)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    medians = out.median(numeric_only=True)
    return out.fillna(medians).fillna(0.0)


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    """Shared X/y builder used by every scorer below."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "No data loaded."}
    if not target or target not in df.columns:
        return {"ok": False, "error": f"Target `{target}` not found in the working table."}

    feats = [str(c) for c in (features if features is not None else candidate_features(df, target))]
    feats = [c for c in feats if c in df.columns and c != target]
    if not feats:
        return {"ok": False, "error": "No usable feature columns for this target."}

    work = df[feats + [target]].dropna(subset=[target])
    if len(work) > max_rows:
        work = work.sample(max_rows, random_state=42)
    if len(work) < 10:
        return {"ok": False, "error": f"Need ≥10 labelled rows, found {len(work)}."}

    task = infer_task(work[target])
    y_raw = work[target]
    if task == "classification":
        y = pd.Series(pd.factorize(y_raw.astype(str))[0], index=work.index, name=target)
        if y.nunique() < 2:
            return {"ok": False, "error": "Target has a single class — pick another target."}
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        keep = y.notna()
        work, y = work.loc[keep], y.loc[keep]
        if len(work) < 10:
            return {"ok": False, "error": "Numeric target has too few valid rows."}

    X = encode_matrix(work, feats)
    X = X.loc[:, X.std(numeric_only=True).fillna(0) > 0]
    if X.empty:
        return {"ok": False, "error": "All candidate features are constant."}
    return {"ok": True, "X": X, "y": y, "task": task, "features": list(X.columns), "target": target}


def top_k_features(table: pd.DataFrame, k: int, score_col: str = "score") -> list[str]:
    """Top-k feature names from any scoring table produced here."""
    if not isinstance(table, pd.DataFrame) or table.empty or "feature" not in table.columns:
        return []
    if score_col not in table.columns:
        score_col = table.columns[-1]
    ranked = table.sort_values(score_col, ascending=False)
    return [str(f) for f in ranked["feature"].head(max(1, int(k))).tolist()]


def anova_scores(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """ANOVA F-test: `f_classif` for labels, `f_regression` for numeric targets."""
    prep = prepare_xy(df, target, features)
    if not prep.get("ok"):
        return prep
    from sklearn.feature_selection import f_classif, f_regression

    X, y, task = prep["X"], prep["y"], prep["task"]
    try:
        scorer = f_classif if task == "classification" else f_regression
        scores, pvalues = scorer(X.to_numpy(dtype=float), y.to_numpy())
    except Exception as exc:
        return {"ok": False, "error": f"ANOVA failed: {type(exc).__name__}: {exc}"}

    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "score": np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0),
            "p_value": np.nan_to_num(np.asarray(pvalues, dtype=float), nan=1.0),
        }
    ).sort_values("score", ascending=False, ignore_index=True)
    table["significant"] = table["p_value"] < 0.05
    return {"ok": True, "table": table, "task": task, "method": "anova", "target": target}


def mutual_info_scores(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mutual information — catches non-linear dependencies ANOVA misses."""
    prep = prepare_xy(df, target, features, max_rows=min(MAX_ROWS, 5000))
    if not prep.get("ok"):
        return prep
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    X, y, task = prep["X"], prep["y"], prep["task"]
    try:
        scorer = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = scorer(X.to_numpy(dtype=float), y.to_numpy(), random_state=random_state)
    except Exception as exc:
        return {"ok": False, "error": f"Mutual information failed: {type(exc).__name__}: {exc}"}

    table = pd.DataFrame(
        {"feature": list(X.columns), "score": np.nan_to_num(np.asarray(scores, dtype=float))}
    ).sort_values("score", ascending=False, ignore_index=True)
    return {"ok": True, "table": table, "task": task, "method": "mutual_info", "target": target}


def correlation_pairs(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    threshold: float = CORR_DEFAULT_THRESHOLD,
    target: Optional[str] = None,
) -> dict[str, Any]:
    """Highly correlated pairs (|r| > threshold) plus the redundant column to drop."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "No data loaded."}
    feats = [str(c) for c in (features if features is not None else candidate_features(df, target))]
    feats = [c for c in feats if c in df.columns]
    if len(feats) < 2:
        return {"ok": False, "error": "Need ≥2 candidate features to check correlation."}

    X = encode_matrix(df, feats)
    X = X.loc[:, X.std(numeric_only=True).fillna(0) > 0]
    if X.shape[1] < 2:
        return {"ok": False, "error": "Need ≥2 non-constant features to check correlation."}

    corr = X.corr(numeric_only=True).abs()
    rows: list[dict[str, Any]] = []
    redundant: list[str] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            value = float(corr.loc[a, b])
            if not np.isfinite(value) or value <= threshold:
                continue
            # Keep the column that appears first in the table, drop the later duplicate.
            keep, drop = (a, b) if a not in redundant else (b, a)
            rows.append(
                {"feature_a": a, "feature_b": b, "abs_corr": round(value, 4), "keep": keep, "drop": drop}
            )
            if drop not in redundant:
                redundant.append(drop)

    table = (
        pd.DataFrame(rows).sort_values("abs_corr", ascending=False, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr", "keep", "drop"])
    )
    return {
        "ok": True,
        "table": table,
        "redundant": redundant,
        "threshold": float(threshold),
        "method": "correlation",
    }


def tree_importance(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Quick RandomForest fit → `feature_importances_` (impurity based)."""
    prep = prepare_xy(df, target, features, max_rows=min(MAX_ROWS, 10_000))
    if not prep.get("ok"):
        return prep
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X, y, task = prep["X"], prep["y"], prep["task"]
    model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    try:
        model = model_cls(n_estimators=int(n_estimators), random_state=random_state, n_jobs=-1)
        model.fit(X, y)
    except Exception as exc:
        return {"ok": False, "error": f"RandomForest fit failed: {type(exc).__name__}: {exc}"}

    table = pd.DataFrame(
        {"feature": list(X.columns), "score": np.asarray(model.feature_importances_, dtype=float)}
    ).sort_values("score", ascending=False, ignore_index=True)
    return {
        "ok": True,
        "table": table,
        "task": task,
        "method": "tree_importance",
        "model": model,
        "X": X,
        "y": y,
        "target": target,
        "model_name": model_cls.__name__,
    }


def rfe_ranking(
    df: pd.DataFrame,
    target: str,
    features: Optional[Iterable[str]] = None,
    min_features: int = 1,
    max_points: int = 8,
    n_estimators: int = 60,
    random_state: int = 42,
) -> dict[str, Any]:
    """Recursive feature elimination: cross-validated curve + final ranking.

    The curve is sampled at up to `max_points` sizes so the fit budget stays bounded
    on wide industrial tables.
    """
    prep = prepare_xy(df, target, features, max_rows=min(MAX_ROWS, 4000))
    if not prep.get("ok"):
        return prep
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import cross_val_score

    X, y, task = prep["X"], prep["y"], prep["task"]
    total = X.shape[1]
    if total < 2:
        return {"ok": False, "error": "RFE needs ≥2 features."}

    model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    estimator = model_cls(n_estimators=int(n_estimators), random_state=random_state, n_jobs=-1)
    scoring = "accuracy" if task == "classification" else "r2"
    cv = int(min(5, max(2, len(X) // 10)))

    sizes = sorted({int(round(v)) for v in np.linspace(max(1, min_features), total, min(max_points, total))})
    curve: list[dict[str, Any]] = []
    try:
        for n in sizes:
            selector = RFE(estimator, n_features_to_select=n, step=1)
            selector.fit(X, y)
            chosen = list(X.columns[selector.support_])
            score = float(cross_val_score(estimator, X[chosen], y, cv=cv, scoring=scoring).mean())
            curve.append({"n_features": n, "cv_score": round(score, 4), "features": ", ".join(chosen[:8])})
        final = RFE(estimator, n_features_to_select=max(1, min_features), step=1).fit(X, y)
    except Exception as exc:
        return {"ok": False, "error": f"RFE failed: {type(exc).__name__}: {exc}"}

    curve_df = pd.DataFrame(curve)
    best_row = curve_df.loc[curve_df["cv_score"].idxmax()]
    best_n = int(best_row["n_features"])
    best_selector = RFE(estimator, n_features_to_select=best_n, step=1).fit(X, y)
    ranking = pd.DataFrame(
        {
            "feature": list(X.columns),
            "rank": np.asarray(final.ranking_, dtype=int),
            "selected_at_best_n": np.asarray(best_selector.support_, dtype=bool),
        }
    ).sort_values("rank", ignore_index=True)
    return {
        "ok": True,
        "curve": curve_df,
        "ranking": ranking,
        "best_n": best_n,
        "best_score": float(best_row["cv_score"]),
        "best_features": list(X.columns[best_selector.support_]),
        "scoring": scoring,
        "task": task,
        "method": "rfe",
    }


def pca_variance(
    df: pd.DataFrame,
    features: Optional[Iterable[str]] = None,
    n_components: int = 3,
    target: Optional[str] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Standard-scaled PCA: explained / cumulative variance table plus component loadings."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "No data loaded."}
    feats = [str(c) for c in (features if features is not None else candidate_features(df, target))]
    feats = [c for c in feats if c in df.columns]
    X = encode_matrix(df, feats) if feats else pd.DataFrame()
    if X.empty:
        return {"ok": False, "error": "No numeric features available for PCA."}
    X = X.loc[:, X.std(numeric_only=True).fillna(0) > 0]
    if X.shape[1] < 2 or len(X) < 5:
        return {"ok": False, "error": "PCA needs ≥2 non-constant features and ≥5 rows."}

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_comp = int(max(1, min(int(n_components), X.shape[1], len(X))))
    try:
        scaled = StandardScaler().fit_transform(X.to_numpy(dtype=float))
        pca = PCA(n_components=n_comp, random_state=random_state)
        scores = pca.fit_transform(scaled)
    except Exception as exc:
        return {"ok": False, "error": f"PCA failed: {type(exc).__name__}: {exc}"}

    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    table = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(n_comp)],
            "explained_variance": np.round(ratios, 4),
            "cumulative_variance": np.round(np.cumsum(ratios), 4),
        }
    )
    loadings = pd.DataFrame(
        np.round(pca.components_.T, 4),
        index=list(X.columns),
        columns=[f"PC{i + 1}" for i in range(n_comp)],
    )
    return {
        "ok": True,
        "table": table,
        "loadings": loadings,
        "total_explained": round(float(ratios.sum()), 4),
        "n_components": n_comp,
        "scores": pd.DataFrame(scores[:200], columns=[f"PC{i + 1}" for i in range(n_comp)]),
        "method": "pca",
    }


def selection_summary(
    selected: Iterable[str],
    dropped: Iterable[str],
    method: str,
    target: Optional[str] = None,
) -> dict[str, Any]:
    """Session-state payload describing an applied selection."""
    sel = [str(c) for c in selected]
    return {
        "features": sel,
        "dropped": [str(c) for c in dropped],
        "method": str(method),
        "target": target,
        "n_selected": len(sel),
    }
