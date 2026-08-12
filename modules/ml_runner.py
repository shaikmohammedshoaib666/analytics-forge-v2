"""Run sklearn + catalog models with leakage-safe feature/target prep.

XGBoost / LightGBM / statsmodels / PuLP ship via main requirements.txt.
Prophet is optional (requirements-optional.txt) — soft-fails if missing.
"""
from __future__ import annotations

import importlib
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from modules.ml_registry import get_model


TARGET_PRIORITY_REG = [
    "rul",
    "remaining_useful_life",
    "revenue",
    "sales",
    "amount",
    "units",
    "price",
    "score",
    "value",
    "target",
    "y",
]
TARGET_PRIORITY_CLF = [
    "failure",
    "failed",
    "churn",
    "converted",
    "conversion",
    "label",
    "target",
    "class",
    "status",
]

_ID_EXACT = {"id", "uuid", "guid", "index", "row_id", "rowid", "pk"}
_ID_SUFFIXES = ("_id", "_uuid", "_guid", "_key", "_pk")

# Soft cap for huge CSVs — typical college demo files are far smaller.
_MAX_FIT_ROWS = 25_000


def _resolve_class(dotted: str):
    module_path, _, cls_name = dotted.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def _is_leak_id_column(col: str, series: pd.Series, n_rows: int) -> bool:
    """Drop obvious identifier columns that leak or inflate accuracy."""
    cl = str(col).lower().strip()
    if cl in _ID_EXACT or "unnamed" in cl:
        return True
    if any(cl.endswith(suf) for suf in _ID_SUFFIXES):
        return True
    # near-unique object/int identifiers
    nun = series.nunique(dropna=True)
    if nun >= max(50, int(n_rows * 0.9)) and not pd.api.types.is_float_dtype(series):
        return True
    return False


def pick_target(df: pd.DataFrame, task: str) -> Optional[str]:
    cols_lower = {str(c).lower(): c for c in df.columns}
    priority = TARGET_PRIORITY_CLF if task == "classification" else TARGET_PRIORITY_REG
    for name in priority:
        if name in cols_lower:
            return cols_lower[name]
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if task == "classification":
        for c in df.columns:
            if _is_leak_id_column(c, df[c], len(df)):
                continue
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 10:
                return c
        return None
    usable = [c for c in nums if not _is_leak_id_column(c, df[c], len(df))]
    if usable:
        return usable[-1]
    return nums[-1] if nums else None


def pick_features(df: pd.DataFrame, target: str) -> list[str]:
    feats = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        cl = str(c).lower().strip()
        # Never use date/time columns as supervised features (one-hot explosion)
        if any(h in cl for h in ("date", "time", "timestamp", "datetime", "ds")):
            continue
        if _is_leak_id_column(c, df[c], n):
            continue
        # Drop high-cardinality categoricals (near-unique strings)
        if not pd.api.types.is_numeric_dtype(df[c]):
            nun = df[c].nunique(dropna=True)
            if nun > min(40, max(15, int(n * 0.4))):
                continue
        feats.append(c)
    return feats


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("oh", _one_hot_encoder()),
                    ]
                ),
                cat_cols,
            )
        )
    if not transformers:
        raise RuntimeError("No numeric or categorical feature columns after filtering.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _maybe_subsample(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, bool]:
    """Light guard for huge uploads; demo CSVs are unaffected."""
    if len(df) <= _MAX_FIT_ROWS:
        return df, False
    return df.sample(n=_MAX_FIT_ROWS, random_state=random_state).copy(), True


def _build_estimator(model_id: str, meta: dict):
    dotted = meta.get("class") or meta.get("estimator") or ""
    library = (meta.get("library") or "").lower()
    task = meta.get("task", "")

    if library == "prophet" or model_id.lower() == "prophet":
        raise RuntimeError("PROPHET_ROUTE")
    if library == "pulp" or model_id.lower() == "pulp" or task == "optimization":
        raise RuntimeError("PULP_ROUTE")
    if library == "statsmodels" or model_id == "StatsmodelsOLS" or "statsmodels" in dotted:
        raise RuntimeError("STATSMODELS_ROUTE")
    if task in {"dimensionality", "pca"} or model_id == "PCA":
        raise RuntimeError("PCA_ROUTE")
    if library in {"gurobipy", "ortools", "torch", "pyspark"} or meta.get("requires_license"):
        raise RuntimeError(
            f"{model_id} is not available in this Streamlit / college install. "
            "Use PuLP for optimization, or sklearn / XGBoost / LightGBM / Prophet."
        )
    if not dotted or "." not in dotted or dotted in {"stub", "gurobipy", "ortools"}:
        raise RuntimeError(f"Model {model_id} has no importable estimator class.")

    params = dict(meta.get("default_params") or {})
    if model_id == "KMeans":
        params.setdefault("n_clusters", 3)
        params.setdefault("random_state", 42)
        params.setdefault("n_init", 10)
    if model_id == "LogisticRegression":
        params.setdefault("max_iter", 1000)
    if model_id == "IsolationForest":
        params.setdefault("contamination", 0.05)
        params.setdefault("random_state", 42)
    if model_id.startswith(("RandomForest", "ExtraTrees", "GradientBoosting")):
        params.setdefault("n_estimators", 80)
        params.setdefault("random_state", 42)
    if model_id.startswith(("DecisionTree",)):
        params.setdefault("random_state", 42)
    if model_id in {"SVC", "SVR"}:
        params.setdefault("C", 1.0)

    if library == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost is listed in requirements.txt but failed to import. "
                "Re-run: pip install -r requirements.txt  (pip install xgboost)"
            ) from exc
        params.setdefault("random_state", 42)
        if model_id == "XGBClassifier":
            params.setdefault("eval_metric", "logloss")
    if library == "lightgbm":
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is listed in requirements.txt but failed to import. "
                "Re-run: pip install -r requirements.txt  (pip install lightgbm)"
            ) from exc
        params.setdefault("random_state", 42)
        params.setdefault("verbosity", -1)

    cls = _resolve_class(dotted)
    try:
        return cls(**params)
    except TypeError:
        return cls()


def run_model(
    df: pd.DataFrame,
    model_id: str,
    target: Optional[str] = None,
    features: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train/evaluate a catalog model with leakage-safe prep:
    split first → fit encoders/scaler on train only → transform → fit → evaluate on test.
    """
    meta = get_model(model_id)
    if not meta:
        return {"ok": False, "error": f"Unknown model: {model_id}"}

    task = meta.get("task", "regression")
    library = (meta.get("library") or "").lower()

    if task == "clustering":
        return _run_clustering(df, model_id, meta, features, random_state=random_state)
    if task == "anomaly" or model_id == "IsolationForest":
        return _run_anomaly(df, model_id, meta, features, random_state=random_state)
    if task in {"dimensionality", "pca"} or model_id == "PCA":
        return _run_pca(df, model_id, meta, features, random_state=random_state)
    if library == "statsmodels" or model_id == "StatsmodelsOLS":
        return _run_statsmodels_ols(
            df,
            model_id,
            meta,
            target=target,
            features=features,
            test_size=test_size,
            random_state=random_state,
        )
    if task == "forecast" or model_id == "Prophet":
        return _run_prophet(df, target=target)
    if task == "optimization" or model_id.lower() == "pulp":
        return _run_pulp(df, target=target, features=features)
    if task in {"deep_learning", "big_data"}:
        return {
            "ok": False,
            "model_id": model_id,
            "task": task,
            "error": meta.get("note")
            or f"{model_id} requires a stronger host (not Streamlit free tier).",
        }

    try:
        estimator = _build_estimator(model_id, meta)
    except Exception as exc:
        msg = str(exc)
        if msg == "PROPHET_ROUTE":
            return _run_prophet(df, target=target)
        if msg == "PULP_ROUTE":
            return _run_pulp(df, target=target, features=features)
        if msg == "STATSMODELS_ROUTE":
            return _run_statsmodels_ols(
                df,
                model_id,
                meta,
                target=target,
                features=features,
                test_size=test_size,
                random_state=random_state,
            )
        if msg == "PCA_ROUTE":
            return _run_pca(df, model_id, meta, features, random_state=random_state)
        return {"ok": False, "error": msg, "model_id": model_id, "task": task}

    tgt = target or pick_target(df, task)
    if not tgt or tgt not in df.columns:
        return {"ok": False, "error": "Could not determine target column.", "task": task}

    feats = features or pick_features(df, tgt)
    feats = [f for f in feats if f != tgt and f in df.columns]
    if not feats:
        return {"ok": False, "error": "No usable feature columns (IDs/target excluded).", "target": tgt}

    work = df[feats + [tgt]].copy()
    work = work.dropna(subset=[tgt])
    work, subsampled = _maybe_subsample(work, random_state=random_state)
    if len(work) < 10:
        return {"ok": False, "error": "Need at least 10 rows after dropping null targets."}

    X = work[feats]
    y = work[tgt]
    if task == "classification":
        if pd.api.types.is_float_dtype(y) and y.nunique() > 20:
            return {
                "ok": False,
                "error": "Target looks continuous; pick a classifier target or use a regressor.",
                "target": tgt,
            }
    else:
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

    if len(X) < 10:
        return {"ok": False, "error": "Need at least 10 clean rows for train/test split."}

    # --- Leakage-safe order: SPLIT FIRST ---
    stratify = (
        y
        if task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2
        else None
    )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # Encode non-numeric class labels (fit on train only)
    if task == "classification" and (
        not pd.api.types.is_numeric_dtype(y_train) or y_train.dtype == object
    ):
        le = LabelEncoder()
        y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index)
        try:
            y_test = pd.Series(le.transform(y_test.astype(str)), index=y_test.index)
        except ValueError:
            return {
                "ok": False,
                "error": "Test set has class labels not seen in train; try a different split or larger data.",
                "target": tgt,
            }

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    pre = _build_preprocessor(num_cols, cat_cols)

    # Fit encoders/scaler ONLY on train, then transform both
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    try:
        estimator.fit(X_train_t, y_train)
        preds = estimator.predict(X_test_t)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": task,
            "error": f"{model_id} failed while fitting/predicting: {exc}",
            "target": tgt,
        }

    metrics: dict[str, Any] = {}
    if task == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        try:
            metrics["f1"] = round(
                float(f1_score(y_test, preds, average="weighted", zero_division=0)),
                4,
            )
        except Exception:
            pass
    else:
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)

    if subsampled:
        metrics["subsampled_rows"] = int(_MAX_FIT_ROWS)

    preview = pd.DataFrame({"y_true": np.asarray(y_test), "y_pred": np.asarray(preds)})
    preview = preview.head(25).reset_index(drop=True)

    return {
        "ok": True,
        "model_id": model_id,
        "task": task,
        "target": tgt,
        "features": feats,
        "metrics": metrics,
        "predictions_preview": preview,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "leakage_safe": True,
        "subsampled": subsampled,
    }


def _numeric_feature_frame(
    df: pd.DataFrame,
    features: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    exclude = set(exclude or [])
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = features or nums
    feats = [
        f
        for f in feats
        if f in df.columns
        and f not in exclude
        and not _is_leak_id_column(f, df[f], len(df))
    ]
    if not feats:
        return pd.DataFrame(), []
    X = df[feats].apply(pd.to_numeric, errors="coerce").dropna()
    return X, feats


def _run_clustering(
    df: pd.DataFrame,
    model_id: str,
    meta: dict,
    features: Optional[list[str]] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    try:
        estimator = _build_estimator(model_id, meta)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model_id": model_id, "task": "clustering"}

    X, feats = _numeric_feature_frame(df, features)
    feats = feats[:8]
    X = X[feats] if feats else X
    if len(feats) < 2:
        return {"ok": False, "error": "Need at least 2 numeric features for clustering."}
    X, subsampled = _maybe_subsample(X, random_state=random_state)
    if len(X) < 5:
        return {"ok": False, "error": "Not enough rows for clustering."}

    # Fit scaler on all rows for unsupervised demo (no held-out labels)
    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    labels = estimator.fit_predict(Xt)
    unique = set(int(x) for x in np.unique(labels))
    n_clusters = len(unique - {-1}) if -1 in unique else len(unique)
    preview = X.head(25).copy()
    preview["cluster"] = labels[: len(preview)]

    metrics: dict[str, Any] = {
        "n_clusters": int(n_clusters),
        "n_rows": int(len(X)),
    }
    if hasattr(estimator, "inertia_"):
        metrics["inertia"] = round(float(estimator.inertia_), 4)
    if -1 in unique:
        metrics["noise_points"] = int((labels == -1).sum())
    if subsampled:
        metrics["subsampled_rows"] = int(_MAX_FIT_ROWS)

    return {
        "ok": True,
        "model_id": model_id,
        "task": "clustering",
        "target": None,
        "features": feats,
        "metrics": metrics,
        "predictions_preview": preview.reset_index(drop=True),
        "n_train": int(len(X)),
        "n_test": 0,
        "subsampled": subsampled,
    }


def _run_anomaly(
    df: pd.DataFrame,
    model_id: str,
    meta: dict,
    features: Optional[list[str]] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    try:
        estimator = _build_estimator(model_id, meta)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "model_id": model_id, "task": "anomaly"}

    prefer = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not str(c).endswith("_outlier_flag")
    ]
    X, feats = _numeric_feature_frame(df, features or prefer[:10])
    if not feats:
        return {"ok": False, "error": "Need numeric columns for IsolationForest / anomaly detection."}
    X, subsampled = _maybe_subsample(X, random_state=random_state)
    if len(X) < 10:
        return {"ok": False, "error": "Need at least 10 clean numeric rows for anomaly detection."}

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    labels = estimator.fit_predict(Xt)
    scores = None
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(Xt)
    anomaly_count = int((labels == -1).sum())
    preview = X.head(25).copy()
    preview["anomaly_label"] = labels[: len(preview)]
    if scores is not None:
        preview["anomaly_score"] = scores[: len(preview)]

    metrics: dict[str, Any] = {
        "anomaly_count": anomaly_count,
        "anomaly_rate_pct": round(float(anomaly_count / len(X) * 100), 3),
        "n_rows": int(len(X)),
    }
    if subsampled:
        metrics["subsampled_rows"] = int(_MAX_FIT_ROWS)

    return {
        "ok": True,
        "model_id": model_id,
        "task": "anomaly",
        "target": None,
        "features": feats,
        "metrics": metrics,
        "predictions_preview": preview.reset_index(drop=True),
        "n_train": int(len(X)),
        "n_test": 0,
        "subsampled": subsampled,
    }


def _run_pca(
    df: pd.DataFrame,
    model_id: str,
    meta: dict,
    features: Optional[list[str]] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Dedicated unsupervised PCA path — explained variance, no fake R²."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "dimensionality",
            "error": f"PCA unavailable ({exc}). scikit-learn must be installed.",
        }

    X, feats = _numeric_feature_frame(df, features)
    feats = feats[:12]
    X = X[feats] if feats else X
    if len(feats) < 2:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "dimensionality",
            "error": "PCA needs at least 2 numeric features (IDs excluded).",
        }
    X, subsampled = _maybe_subsample(X, random_state=random_state)
    if len(X) < 5:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "dimensionality",
            "error": "Need at least 5 clean rows for PCA.",
        }

    params = dict(meta.get("default_params") or {})
    max_comp = min(X.shape[0], X.shape[1])
    n_comp = params.get("n_components", min(5, max_comp))
    if isinstance(n_comp, float) and 0 < n_comp < 1:
        # variance fraction retained — sklearn accepts this directly
        pca_kw: dict[str, Any] = {"n_components": n_comp}
    else:
        n_comp_i = int(n_comp) if n_comp is not None else min(5, max_comp)
        n_comp_i = max(1, min(n_comp_i, max_comp))
        pca_kw = {"n_components": n_comp_i}

    scaler = StandardScaler()
    Xt = scaler.fit_transform(X)
    pca = PCA(**pca_kw)
    try:
        transformed = pca.fit_transform(Xt)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "dimensionality",
            "error": f"PCA failed: {exc}",
        }

    evr = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cum = round(float(np.cumsum(pca.explained_variance_ratio_)[-1]), 4)
    n_out = int(getattr(pca, "n_components_", transformed.shape[1]))
    pc_cols = [f"PC{i + 1}" for i in range(n_out)]
    preview = pd.DataFrame(transformed[:25], columns=pc_cols)

    metrics: dict[str, Any] = {
        "n_components": n_out,
        "cumulative_variance": cum,
        "n_features": int(len(feats)),
        "n_rows": int(len(X)),
        # Scalar-friendly string for KPI strip (not fake R²)
        "explained_variance_ratio": ", ".join(str(v) for v in evr),
    }
    for i, v in enumerate(evr[:5]):
        metrics[f"pc{i + 1}_var"] = v
    if subsampled:
        metrics["subsampled_rows"] = int(_MAX_FIT_ROWS)

    return {
        "ok": True,
        "model_id": model_id,
        "task": "dimensionality",
        "target": None,
        "features": feats,
        "metrics": metrics,
        "predictions_preview": preview.reset_index(drop=True),
        "n_train": int(len(X)),
        "n_test": 0,
        "subsampled": subsampled,
    }


def _run_statsmodels_ols(
    df: pd.DataFrame,
    model_id: str,
    meta: dict,
    target: Optional[str] = None,
    features: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Statsmodels OLS with train-fit / test metrics; soft-fail only if import fails."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": (
                f"statsmodels is listed in requirements.txt but failed to import ({exc}). "
                "Re-run: pip install -r requirements.txt — or use LinearRegression (sklearn)."
            ),
        }

    tgt = target or pick_target(df, "regression")
    if not tgt or tgt not in df.columns:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": "Could not determine numeric target for StatsmodelsOLS.",
        }

    feats = features or pick_features(df, tgt)
    feats = [f for f in feats if f != tgt and f in df.columns]
    if not feats:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": "No usable feature columns (IDs/target excluded).",
            "target": tgt,
        }

    work = df[feats + [tgt]].copy()
    work[tgt] = pd.to_numeric(work[tgt], errors="coerce")
    work = work.dropna(subset=[tgt])
    work, subsampled = _maybe_subsample(work, random_state=random_state)
    if len(work) < 10:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": "Need at least 10 rows for StatsmodelsOLS.",
            "target": tgt,
        }

    X = work[feats]
    y = work[tgt]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    try:
        pre = _build_preprocessor(num_cols, cat_cols)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": str(exc),
            "target": tgt,
        }

    X_train_t = np.asarray(pre.fit_transform(X_train), dtype=float)
    X_test_t = np.asarray(pre.transform(X_test), dtype=float)
    y_train_a = np.asarray(y_train, dtype=float)
    y_test_a = np.asarray(y_test, dtype=float)

    X_train_c = sm.add_constant(X_train_t, has_constant="add")
    X_test_c = sm.add_constant(X_test_t, has_constant="add")
    # Align constant column count if transform shape differs
    if X_test_c.shape[1] != X_train_c.shape[1]:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": "Train/test feature shapes diverged after preprocessing.",
            "target": tgt,
        }

    try:
        model = sm.OLS(y_train_a, X_train_c).fit()
        preds = np.asarray(model.predict(X_test_c), dtype=float)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "task": "regression",
            "error": (
                f"StatsmodelsOLS failed: {exc}. "
                "Try LinearRegression if the design matrix is singular."
            ),
            "target": tgt,
        }

    metrics: dict[str, Any] = {
        "r2": round(float(r2_score(y_test_a, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test_a, preds))), 4),
        "mae": round(float(mean_absolute_error(y_test_a, preds)), 4),
        "train_rsquared": round(float(model.rsquared), 4),
        "n_params": int(len(model.params)),
    }
    if subsampled:
        metrics["subsampled_rows"] = int(_MAX_FIT_ROWS)

    preview = pd.DataFrame({"y_true": y_test_a, "y_pred": preds}).head(25).reset_index(drop=True)
    return {
        "ok": True,
        "model_id": model_id,
        "task": "regression",
        "target": tgt,
        "features": feats,
        "metrics": metrics,
        "predictions_preview": preview,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "leakage_safe": True,
        "subsampled": subsampled,
    }


def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
        except (TypeError, ValueError):
            pass
    for c in df.columns:
        cl = str(c).lower()
        if any(h in cl for h in ("date", "time", "timestamp", "ds")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(10, int(len(df) * 0.5)):
                return c
    return None


def _run_prophet(df: pd.DataFrame, target: Optional[str] = None) -> dict[str, Any]:
    try:
        from prophet import Prophet
    except Exception as exc:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": (
                f"Prophet is not installed ({exc}). "
                "Local: pip install -r requirements-optional.txt. "
                "On Streamlit Cloud, Prophet is omitted from requirements.txt to avoid "
                "long/hanging builds (cmdstan); use holdout regression models instead."
            ),
        }

    date_col = _find_date_column(df)
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if not _is_leak_id_column(c, df[c], len(df))
    ]
    if not date_col or not num_cols:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": (
                "Prophet needs a date/time column AND a numeric target. "
                "Pick a datetime column in your data and a numeric metric (revenue, RUL, sales)."
            ),
        }

    if target and target in num_cols:
        ycol = target
    elif target and target == date_col:
        ycol = next(
            (
                c
                for c in num_cols
                if any(k in str(c).lower() for k in ("revenue", "sales", "rul", "units", "y", "amount"))
            ),
            num_cols[0],
        )
    else:
        ycol = next(
            (
                c
                for c in num_cols
                if any(k in str(c).lower() for k in ("revenue", "sales", "rul", "units", "y", "amount"))
            ),
            num_cols[0],
        )

    tmp = (
        pd.DataFrame(
            {
                "ds": pd.to_datetime(df[date_col], errors="coerce"),
                "y": pd.to_numeric(df[ycol], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="last")
    )
    if len(tmp) < 10:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": "Need >= 10 dated rows for Prophet.",
        }

    # Time-based holdout (not random split) — fit on past, score future rows only
    n_test = max(3, int(len(tmp) * 0.2))
    train = tmp.iloc[:-n_test].copy()
    test = tmp.iloc[-n_test:].copy()
    if len(train) < 8:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": "Not enough history after holdout split for Prophet.",
        }

    try:
        import logging

        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
        logging.getLogger("prophet").setLevel(logging.WARNING)
        # Fewer uncertainty samples keeps college/Cloud demos responsive
        m = Prophet(uncertainty_samples=200)
        m.fit(train)
        # Score on actual holdout timestamps (precise alignment, no daily-freq mismatch)
        fc_holdout = m.predict(test[["ds"]].copy())
        freq = pd.infer_freq(train["ds"]) or "D"
        horizon = min(14, max(7, len(train) // 5))
        future = m.make_future_dataframe(periods=horizon, freq=freq)
        fc = m.predict(future)
    except Exception as exc:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": f"Prophet failed while fitting: {exc}",
        }

    merged = test.merge(fc_holdout[["ds", "yhat"]], on="ds", how="inner")
    if len(merged) < 2:
        return {
            "ok": False,
            "model_id": "Prophet",
            "task": "forecast",
            "error": "Could not align holdout dates with Prophet forecast.",
        }

    metrics = {
        "r2": round(float(r2_score(merged["y"], merged["yhat"])), 4),
        "rmse": round(float(mean_squared_error(merged["y"], merged["yhat"]) ** 0.5), 4),
        "mae": round(float(mean_absolute_error(merged["y"], merged["yhat"])), 4),
        "target": ycol,
        "date_col": date_col,
        "horizon": int(horizon),
        "holdout_rows": int(len(merged)),
    }
    preview = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(20).reset_index(drop=True)

    # Manager-friendly forecast numbers (future rows only)
    try:
        last_actual = float(train["y"].iloc[-1])
        future_only = fc[fc["ds"] > train["ds"].max()]
        if len(future_only) == 0:
            future_only = preview
        end_row = future_only.iloc[-1]
        mean_yhat = float(future_only["yhat"].mean())
        end_yhat = float(end_row["yhat"])
        metrics["last_actual"] = round(last_actual, 4)
        metrics["forecast_mean"] = round(mean_yhat, 4)
        metrics["forecast_end"] = round(end_yhat, 4)
        metrics["forecast_lower"] = round(float(end_row["yhat_lower"]), 4)
        metrics["forecast_upper"] = round(float(end_row["yhat_upper"]), 4)
        if last_actual != 0:
            metrics["pct_change"] = round((end_yhat - last_actual) / abs(last_actual), 4)
    except Exception:
        pass

    return {
        "ok": True,
        "model_id": "Prophet",
        "task": "forecast",
        "target": ycol,
        "features": [date_col, ycol],
        "metrics": metrics,
        "predictions_preview": preview,
        "n_train": int(len(train)),
        "n_test": int(len(merged)),
        "leakage_safe": True,
    }


def _run_pulp(
    df: pd.DataFrame,
    target: Optional[str] = None,
    features: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Simple open-source LP demo (PuLP) as a student-friendly Gurobi stand-in."""
    try:
        import pulp
    except ImportError as exc:
        return {
            "ok": False,
            "model_id": "PuLP",
            "task": "optimization",
            "error": (
                f"PuLP is listed in requirements.txt but failed to import ({exc}). "
                "Re-run: pip install -r requirements.txt  (pip install pulp)"
            ),
        }

    X, feats = _numeric_feature_frame(df, features, exclude=[target] if target else None)
    if len(feats) < 2:
        return {
            "ok": False,
            "model_id": "PuLP",
            "task": "optimization",
            "error": "PuLP demo needs at least 2 numeric columns (IDs excluded).",
        }

    feats = feats[:8]
    X = X[feats]
    # Coefficients = cleaned column means; maximize weighted allocation under budget
    coeffs = {}
    for c in feats:
        mu = float(X[c].mean())
        coeffs[c] = mu if np.isfinite(mu) else 0.0

    # If all non-positive, use absolute means so the demo still solves
    if all(v <= 0 for v in coeffs.values()):
        coeffs = {c: abs(v) + 1e-3 for c, v in coeffs.items()}

    prob = pulp.LpProblem("analytics_forge_allocation", pulp.LpMaximize)
    xs = pulp.LpVariable.dicts("x", feats, lowBound=0, upBound=1, cat="Continuous")
    prob += pulp.lpSum(coeffs[c] * xs[c] for c in feats), "weighted_value"
    # Budget: total allocation <= 1 (simple mix constraint)
    prob += pulp.lpSum(xs[c] for c in feats) <= 1.0, "budget"
    status_code = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus.get(status_code, str(status_code))

    allocation = {c: float(pulp.value(xs[c]) or 0.0) for c in feats}
    objective = float(pulp.value(prob.objective) or 0.0)
    preview = pd.DataFrame(
        {
            "feature": list(allocation.keys()),
            "mean_coeff": [coeffs[c] for c in allocation],
            "allocation": list(allocation.values()),
        }
    ).sort_values("allocation", ascending=False)

    return {
        "ok": status == "Optimal",
        "model_id": "PuLP",
        "task": "optimization",
        "target": target,
        "features": feats,
        "metrics": {
            "status": status,
            "objective": round(objective, 6),
            "budget": 1.0,
            "n_vars": len(feats),
        },
        "predictions_preview": preview.reset_index(drop=True),
        "n_train": int(len(X)),
        "n_test": 0,
        "error": None if status == "Optimal" else f"PuLP solver status: {status}",
    }


# Manager-friendly UX copy for ML Studio (What / Why / What it does + target hint)
MODEL_GUIDANCE: dict[str, dict[str, str]] = {
    "RandomForestRegressor": {
        "what": "Predicts a number from your other columns (like remaining life or revenue).",
        "why": "A safe first model when you have a spreadsheet with mixed columns.",
        "what_it_does": "Learns patterns from many small decision trees and averages them.",
        "target": "Pick the number to predict — e.g. RUL, revenue, or sales.",
        "good_for": "Predicts a number from your other columns (like remaining life or revenue).",
    },
    "RandomForestClassifier": {
        "what": "Predicts a yes/no or category (failure, churn, converted).",
        "why": "Use when the answer is a label, not a continuous number.",
        "what_it_does": "Votes across many trees to pick the most likely class.",
        "target": "Pick the label column — e.g. failure (0/1) or churn.",
        "good_for": "Predicts a yes/no or category (failure, churn, converted).",
    },
    "LinearRegression": {
        "what": "Draws a straight-line relationship to predict a number.",
        "why": "Fast sanity check before trying heavier models.",
        "what_it_does": "Fits a simple line (or plane) from inputs to your target number.",
        "target": "Pick a number column — revenue, sales, or RUL.",
        "good_for": "Draws a straight-line relationship to predict a number.",
    },
    "LogisticRegression": {
        "what": "Predicts a category with simple, readable odds.",
        "why": "Good baseline when you care about yes/no outcomes.",
        "what_it_does": "Estimates the chance of each class from your columns.",
        "target": "Pick a label — failure, churn, or similar.",
        "good_for": "Predicts a category with simple, readable odds.",
    },
    "XGBRegressor": {
        "what": "Predicts a number using a strong boosting model (often more accurate).",
        "why": "When Random Forest is not accurate enough on tabular data.",
        "what_it_does": "Builds trees one after another, each fixing the last one's mistakes.",
        "target": "Pick the number to predict — revenue, RUL, or sales.",
        "good_for": "Predicts a number using a strong boosting model (often more accurate).",
    },
    "XGBClassifier": {
        "what": "Predicts a category with a strong boosting model.",
        "why": "When you need better classification accuracy on tables.",
        "what_it_does": "Builds trees sequentially to improve class predictions.",
        "target": "Pick the label column — failure, churn, etc.",
        "good_for": "Predicts a category with a strong boosting model.",
    },
    "LGBMRegressor": {
        "what": "Predicts a number quickly on larger spreadsheets.",
        "why": "Faster boosting alternative when files get bigger.",
        "what_it_does": "Uses LightGBM boosting to learn numeric patterns.",
        "target": "Pick the number to predict — revenue, RUL, or sales.",
        "good_for": "Predicts a number quickly on larger spreadsheets.",
    },
    "LGBMClassifier": {
        "what": "Predicts a category quickly on larger spreadsheets.",
        "why": "Fast boosting for yes/no or multi-class labels.",
        "what_it_does": "Uses LightGBM boosting to learn class patterns.",
        "target": "Pick the label column — failure, churn, etc.",
        "good_for": "Predicts a category quickly on larger spreadsheets.",
    },
    "IsolationForest": {
        "what": "Flags unusual rows (outliers) — no target needed.",
        "why": "Spot odd sensor readings or suspicious transactions.",
        "what_it_does": "Marks rows that look different from the rest.",
        "target": "Leave as (auto) — this model does not need a target.",
        "good_for": "Flags unusual rows (outliers) — no target needed.",
    },
    "KMeans": {
        "what": "Groups similar rows into clusters — no target needed.",
        "why": "Segment customers, machines, or products by similarity.",
        "what_it_does": "Puts each row into the nearest group of look-alikes.",
        "target": "Leave as (auto) — clustering does not need a target.",
        "good_for": "Groups similar rows into clusters — no target needed.",
    },
    "DBSCAN": {
        "what": "Finds dense groups and marks noise — no target needed.",
        "why": "When clusters are irregular shapes, not neat balls.",
        "what_it_does": "Groups dense areas and labels sparse points as noise.",
        "target": "Leave as (auto) — no target column needed.",
        "good_for": "Finds dense groups and marks noise — no target needed.",
    },
    "PCA": {
        "what": "Summarizes many numeric columns into a few main patterns.",
        "why": "Explore structure or compress sensors before other models.",
        "what_it_does": "Reports how much variance each component explains (not R²).",
        "target": "Leave as (auto) — no target column needed.",
        "good_for": "Summarizes many numeric columns into a few main patterns.",
    },
    "StatsmodelsOLS": {
        "what": "Classic linear regression with clear fit statistics.",
        "why": "When you want a textbook-style regression check.",
        "what_it_does": "Fits ordinary least squares and scores holdout R² / RMSE / MAE.",
        "target": "Pick a number column — revenue, sales, or RUL.",
        "good_for": "Classic linear regression with clear fit statistics.",
    },
    "Prophet": {
        "what": (
            "Forecasts a number over time — like next month's sales, demand, or RUL. "
            "Pick Target = revenue / sales / RUL (the number). Date column is found automatically."
        ),
        "why": "Needs a timeline plus a number to predict (dates + metric).",
        "what_it_does": (
            "Looks at past values over time, learns trend and seasonality, "
            "then projects that number into the future."
        ),
        "target": "Pick the number to forecast — revenue, sales, or RUL. (Date/time is auto-detected.)",
        "good_for": (
            "Forecasts a number over time — like next month's sales, demand, or RUL. "
            "Pick Target = revenue / sales / RUL (the number). Date column is found automatically."
        ),
    },
    "PuLP": {
        "what": "Suggests how to allocate a limited budget across options.",
        "why": "Planning / resource decisions — not prediction.",
        "what_it_does": "Solves a small linear program (open-source; no commercial license).",
        "target": "Optional — leave (auto); uses numeric columns as options.",
        "good_for": "Suggests how to allocate a limited budget across options.",
    },
    "GradientBoostingRegressor": {
        "what": "Predicts a number with sklearn boosting (no extra packages).",
        "why": "Middle ground between Random Forest and XGBoost.",
        "what_it_does": "Builds trees sequentially to improve numeric predictions.",
        "target": "Pick the number to predict — revenue, RUL, or sales.",
        "good_for": "Predicts a number with sklearn boosting (no extra packages).",
    },
    "ExtraTreesRegressor": {
        "what": "Predicts a number with a fast random-forest-style ensemble.",
        "why": "Quick alternative when you want another tree ensemble.",
        "what_it_does": "Averages many randomized trees for a numeric prediction.",
        "target": "Pick the number to predict — revenue, RUL, or sales.",
        "good_for": "Predicts a number with a fast random-forest-style ensemble.",
    },
}


def _normalize_guidance(guide: dict[str, str]) -> dict[str, str]:
    """Ensure What / Why / What it does / target keys exist for the ML UI."""
    what = guide.get("what") or guide.get("good_for") or ""
    why = guide.get("why") or ""
    what_it_does = guide.get("what_it_does") or why
    target = guide.get("target") or ""
    return {
        "what": what,
        "why": why,
        "what_it_does": what_it_does,
        "target": target,
        "good_for": what,  # backward-compatible alias
    }


def model_guidance(model_id: str) -> dict[str, str]:
    if model_id in MODEL_GUIDANCE:
        return _normalize_guidance(MODEL_GUIDANCE[model_id])
    meta = get_model(model_id) or {}
    task = meta.get("task", "regression")
    defaults = {
        "regression": {
            "what": "Predicts a continuous number from your other columns.",
            "why": "Use when the answer is a metric (revenue, RUL, cost).",
            "what_it_does": f"{model_id} learns a numeric relationship from the training rows.",
            "target": "Pick a number column (not an ID).",
        },
        "classification": {
            "what": "Predicts a class or label (yes/no, failure, churn).",
            "why": "Use when the answer is a category, not a continuous number.",
            "what_it_does": f"{model_id} learns which label fits each row.",
            "target": "Pick a label column with a few distinct values.",
        },
        "clustering": {
            "what": "Groups similar rows together — no target needed.",
            "why": "Segmentation and discovery without labeled outcomes.",
            "what_it_does": "Puts rows into groups based on numeric similarity.",
            "target": "Leave as (auto).",
        },
        "anomaly": {
            "what": "Finds unusual / outlier rows — no target needed.",
            "why": "Spot odd sensor or transaction patterns.",
            "what_it_does": "Flags rows that look different from the rest.",
            "target": "Leave as (auto).",
        },
        "dimensionality": {
            "what": "Compresses many numbers into a few main patterns.",
            "why": "Explore structure; not a prediction score like R².",
            "what_it_does": "Reports explained variance per component.",
            "target": "Leave as (auto).",
        },
        "forecast": {
            "what": "Forecasts a number over time (like next month's sales).",
            "why": "Needs time plus a number to predict.",
            "what_it_does": "Uses past dates and values to project the metric forward.",
            "target": "Pick the number to forecast (revenue / sales / RUL). Date is found automatically when possible.",
        },
        "optimization": {
            "what": "Allocates limited resources under simple constraints.",
            "why": "Planning decisions — not forecasting or classification.",
            "what_it_does": "Solves a small linear program on numeric columns.",
            "target": "Optional — leave (auto).",
        },
    }
    return _normalize_guidance(defaults.get(task, defaults["regression"]))
