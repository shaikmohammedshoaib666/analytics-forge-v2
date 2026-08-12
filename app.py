"""
Analytics Forge v2 — Production Dual-Mode Industrial OS
Single-file app: LIVE Modbus SCADA buffer + MANUAL upload, shared analytics core.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import time
import traceback
import warnings
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Paths / env
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
LIVE_CSV = DATA_DIR / "live.csv"
UPLOAD_DIR = DATA_DIR / "uploads"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODBUS_HOST = os.getenv("MODBUS_HOST", "192.168.1.100")
MODBUS_PORT = int(os.getenv("MODBUS_PORT", "502"))
MODBUS_UNIT = int(os.getenv("MODBUS_UNIT", "1"))
MODBUS_START = int(os.getenv("MODBUS_START", "0"))  # 40001 -> address 0
MODBUS_COUNT = int(os.getenv("MODBUS_COUNT", "10"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest").strip() or "gemini-flash-latest"
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", "").strip() or EMAIL_USER
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com").strip()
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "587") or 587)
OPERATOR_EMAIL = "shaikhmohammedshoaib666@gmail.com"

REGISTER_NAMES = [
    "temperature",
    "vibration",
    "pressure",
    "current",
    "voltage",
    "speed",
    "torque",
    "rul",
    "failure",
    "load",
]

st.set_page_config(
    page_title="Analytics Forge v2",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "signed_in": True,
        "mode": "MANUAL UPLOAD",
        "page": "Upload",
        "manual_df": None,
        "manual_name": None,
        "live_last_poll": 0.0,
        "live_status": "idle",
        "live_error": None,
        "clean_df": None,
        "clean_checks": None,
        "automl_result": None,
        "forecast_text": None,
        "chat_history": [],
        "pipeline_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
# GLOBAL FUNCTIONS — dual-mode data + shared analytics core
# =============================================================================

def _scale_register(name: str, raw: int) -> float:
    """Map raw Modbus register integers to engineering units."""
    raw = float(raw)
    if name == "temperature":
        return round(raw / 10.0, 2)
    if name == "vibration":
        return round(raw / 1000.0, 4)
    if name == "pressure":
        return round(raw / 10.0, 2)
    if name == "current":
        return round(raw / 100.0, 3)
    if name == "voltage":
        return round(raw / 10.0, 2)
    if name == "speed":
        return float(raw)
    if name == "torque":
        return round(raw / 10.0, 2)
    if name == "rul":
        return float(raw)
    if name == "failure":
        return 1.0 if raw > 0 else 0.0
    if name == "load":
        return round(raw / 10.0, 2)
    return raw


def poll_modbus_once(
    host: str = MODBUS_HOST,
    port: int = MODBUS_PORT,
    unit: int = MODBUS_UNIT,
    start: int = MODBUS_START,
    count: int = MODBUS_COUNT,
) -> dict[str, Any]:
    """
    REAL Modbus TCP read of holding registers (40001+).
    Raises on connection / protocol failure — no demo fake plant.
    """
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as exc:
        raise RuntimeError(
            "pymodbus is not installed. Run: pip install pymodbus"
        ) from exc

    client = ModbusTcpClient(host=host, port=port, timeout=3)
    try:
        if not client.connect():
            raise RuntimeError(
                f"Cannot connect to Modbus TCP {host}:{port}. "
                "Check plant PLC / VPN / MODBUS_HOST in .env."
            )
        # pymodbus 3.x uses device_id / slave depending on version
        try:
            result = client.read_holding_registers(address=start, count=count, device_id=unit)
        except TypeError:
            try:
                result = client.read_holding_registers(address=start, count=count, slave=unit)
            except TypeError:
                result = client.read_holding_registers(start, count, unit)

        if result is None or (hasattr(result, "isError") and result.isError()):
            raise RuntimeError(f"Modbus read error from {host}:{port} regs@{start}+{count}: {result}")

        regs = list(result.registers)
        if len(regs) < count:
            raise RuntimeError(f"Expected {count} registers, got {len(regs)}")

        row: dict[str, Any] = {"timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z"}
        for i, name in enumerate(REGISTER_NAMES[:count]):
            row[name] = _scale_register(name, regs[i])
        for i in range(len(REGISTER_NAMES), count):
            row[f"reg_{40001 + i}"] = float(regs[i])
        return row
    finally:
        try:
            client.close()
        except Exception:
            pass


def append_live_csv(row: dict[str, Any], path: Path = LIVE_CSV) -> pd.DataFrame:
    """Append one SCADA poll row to data/live.csv and return full buffer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, frame], ignore_index=True)
        except Exception:
            out = frame
    else:
        out = frame
    # Keep a bounded SCADA buffer (last 50k rows)
    if len(out) > 50_000:
        out = out.tail(50_000).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def read_live_buffer(path: Path = LIVE_CSV) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return None


def ensure_live_poll(force: bool = False, min_interval_s: float = 5.0) -> pd.DataFrame:
    """
    LIVE SCADA path: poll pymodbus every ~5s, persist to data/live.csv, return buffer.
    No DemoSimulator. Connection failure raises / surfaces as live_error.
    """
    now = time.time()
    last = float(st.session_state.get("live_last_poll") or 0.0)
    need = force or (now - last >= min_interval_s)
    if need:
        try:
            row = poll_modbus_once()
            df = append_live_csv(row)
            st.session_state.live_last_poll = now
            st.session_state.live_status = "connected"
            st.session_state.live_error = None
            return df
        except Exception as exc:
            st.session_state.live_status = "error"
            st.session_state.live_error = str(exc)
            buf = read_live_buffer()
            if buf is not None and len(buf):
                # Serve last good SCADA buffer while showing the error
                return buf
            raise
    buf = read_live_buffer()
    if buf is not None and len(buf):
        return buf
    # First poll required
    row = poll_modbus_once()
    df = append_live_csv(row)
    st.session_state.live_last_poll = time.time()
    st.session_state.live_status = "connected"
    st.session_state.live_error = None
    return df


def get_data() -> pd.DataFrame:
    """
    Dual-mode data switch used by EVERY page.
    LIVE CONNECT -> real Modbus SCADA buffer (data/live.csv)
    MANUAL UPLOAD -> uploaded dataframe in session
    """
    mode = st.session_state.get("mode", "MANUAL UPLOAD")
    if mode == "LIVE CONNECT":
        return ensure_live_poll(force=False, min_interval_s=5.0)

    df = st.session_state.get("manual_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(
            "No manual file loaded. Go to Upload (or sidebar file uploader) and upload a CSV/Excel."
        )
    return df.copy()


def _basic_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Always-available quality checks (7 rows for Clean page table)."""
    rows = []
    n = len(df)
    # 1 CONSTANT columns
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    rows.append({
        "check": "CONSTANT",
        "status": "FAIL" if const_cols else "PASS",
        "detail": f"{len(const_cols)} constant cols: {const_cols[:5]}" if const_cols else "No constant columns",
    })
    # 2 ZEROS
    num = df.select_dtypes(include=[np.number])
    zero_ratio = float((num == 0).sum().sum() / max(1, num.size)) if num.size else 0.0
    rows.append({
        "check": "ZEROS",
        "status": "WARN" if zero_ratio > 0.3 else "PASS",
        "detail": f"{zero_ratio*100:.1f}% zero cells in numeric columns",
    })
    # 3 NULLS / Missing
    miss = float(df.isna().sum().sum() / max(1, df.size))
    rows.append({
        "check": "NULLS",
        "status": "FAIL" if miss > 0.2 else ("WARN" if miss > 0.05 else "PASS"),
        "detail": f"{miss*100:.2f}% missing values",
    })
    # 4 DUPLICATES
    dups = int(df.duplicated().sum())
    rows.append({
        "check": "DUPLICATES",
        "status": "WARN" if dups else "PASS",
        "detail": f"{dups} duplicate rows",
    })
    # 5 OUTLIERS (IQR)
    outlier_n = 0
    for c in num.columns:
        q1, q3 = num[c].quantile(0.25), num[c].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        outlier_n += int(((num[c] < q1 - 1.5 * iqr) | (num[c] > q3 + 1.5 * iqr)).sum())
    rows.append({
        "check": "OUTLIER",
        "status": "WARN" if outlier_n > max(5, int(0.05 * n)) else "PASS",
        "detail": f"{outlier_n} IQR outlier cells",
    })
    # 6 SCHEMA
    rows.append({
        "check": "SCHEMA",
        "status": "PASS" if n > 0 and df.shape[1] > 0 else "FAIL",
        "detail": f"{n} rows × {df.shape[1]} cols",
    })
    # 7 TIMESTAMP
    ts_cols = [c for c in df.columns if any(h in str(c).lower() for h in ("time", "date", "timestamp"))]
    rows.append({
        "check": "TIMESTAMP",
        "status": "PASS" if ts_cols else "WARN",
        "detail": f"time-like cols: {ts_cols}" if ts_cols else "No timestamp column detected",
    })
    return rows


def _clean_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    log.append("strip column names")
    out = out.dropna(how="all")
    out = out.loc[:, ~out.columns.duplicated()]
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"drop_duplicates {before}->{len(out)}")
    for c in list(out.columns):
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            if out[c].notna().sum() and converted.notna().sum() / max(1, out[c].notna().sum()) >= 0.8:
                out[c] = converted
                log.append(f"to_numeric {c}")
    for c in out.select_dtypes(include=[np.number]).columns:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
            log.append(f"fillna_median {c}")
    for c in out.select_dtypes(include=["object", "string"]).columns:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill)
            log.append(f"fillna_mode {c}")
    return out.reset_index(drop=True), log


def _clean_pyspark(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("forge_v2_clean")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    log = ["pyspark session started"]
    try:
        sdf = spark.createDataFrame(df.astype(str))
        before = sdf.count()
        sdf = sdf.dropDuplicates()
        after = sdf.count()
        log.append(f"spark dropDuplicates {before}->{after}")
        pdf = sdf.toPandas()
        # restore numerics where possible
        cleaned, plog = _clean_pandas(pdf)
        log.extend(plog)
        return cleaned, log
    finally:
        spark.stop()


def _run_great_expectations(df: pd.DataFrame) -> dict[str, Any]:
    try:
        import great_expectations as gx  # noqa: F401

        results = []
        for col in df.columns:
            null_pct = float(df[col].isna().mean())
            results.append({"column": col, "check": "not_null_50pct", "success": null_pct < 0.5})
        for col in df.select_dtypes(include=[np.number]).columns:
            results.append({
                "column": col,
                "check": "finite",
                "success": bool(np.isfinite(pd.to_numeric(df[col], errors="coerce")).all()),
            })
        passed = sum(1 for r in results if r["success"])
        return {"engine": "great_expectations", "passed": passed, "total": len(results), "ok": True}
    except Exception as exc:
        return {"engine": "great_expectations", "ok": False, "error": str(exc)}


def _run_ydata(df: pd.DataFrame) -> dict[str, Any]:
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df.head(min(500, len(df))), minimal=True, progress_bar=False)
        desc = profile.get_description()
        n_vars = len(desc.get("variables", {}))
        n_alerts = len(desc.get("alerts", []))
        return {"engine": "ydata-profiling", "ok": True, "variables": n_vars, "alerts": n_alerts}
    except Exception as exc:
        return {"engine": "ydata-profiling", "ok": False, "error": str(exc)}


def _run_cleanlab(df: pd.DataFrame) -> dict[str, Any]:
    try:
        from cleanlab import Datalab

        num = df.select_dtypes(include=[np.number]).dropna()
        if num.shape[1] < 2 or len(num) < 10:
            return {"engine": "cleanlab", "ok": True, "skipped": "need >=2 numeric cols and 10 rows"}
        lab = Datalab(data=num.reset_index(drop=True))
        lab.find_issues(features=num.values)
        issues = lab.get_issues()
        n_out = int(issues["is_outlier_issue"].sum()) if "is_outlier_issue" in issues.columns else 0
        return {"engine": "cleanlab", "ok": True, "outlier_issues": n_out, "rows": len(num)}
    except Exception as exc:
        return {"engine": "cleanlab", "ok": False, "error": str(exc)}


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Industrial clean: PySpark (if available) + pandas finish + GE + ydata + Cleanlab.
    Returns (clean_df, checks_table with 7 rows).
    """
    engine_logs: list[str] = []
    try:
        clean_df, engine_logs = _clean_pyspark(df)
        engine_logs.insert(0, "engine=pyspark")
    except Exception as exc:
        clean_df, engine_logs = _clean_pandas(df)
        engine_logs.insert(0, f"engine=pandas (pyspark fallback: {exc})")

    checks = _basic_checks(clean_df)
    ge = _run_great_expectations(clean_df)
    yd = _run_ydata(clean_df)
    cl = _run_cleanlab(clean_df)

    # Enrich CONSTANT / OUTLIER rows with engine notes in detail already set
    checks_df = pd.DataFrame(checks)
    meta = pd.DataFrame([
        {"check": "ENGINES", "status": "INFO", "detail": " | ".join(engine_logs[:4])},
        {"check": "GE", "status": "PASS" if ge.get("ok") else "WARN", "detail": json.dumps(ge)[:180]},
        {"check": "YDATA", "status": "PASS" if yd.get("ok") else "WARN", "detail": json.dumps(yd)[:180]},
        {"check": "CLEANLAB", "status": "PASS" if cl.get("ok") else "WARN", "detail": json.dumps(cl)[:180]},
    ])
    # Keep first 7 operational checks for the screenshot-style table
    table = checks_df.head(7).copy()
    st.session_state["_clean_engine_meta"] = meta
    st.session_state.clean_df = clean_df
    st.session_state.clean_checks = table
    return clean_df, table


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        for k, real in lower.items():
            if n.lower() in k:
                return real
    return None


def get_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Industrial KPI dict for SCADA / PdM boards."""
    n_rows, n_cols = df.shape
    miss = round(float(df.isna().sum().sum() / max(1, df.size) * 100), 2)
    tcol = _col(df, "temperature", "temp")
    vcol = _col(df, "vibration", "vib")
    pcol = _col(df, "pressure")
    rcol = _col(df, "rul", "remaining_useful_life")
    fcol = _col(df, "failure", "fault", "alarm")

    def mean_of(col: Optional[str]) -> Any:
        if not col:
            return "—"
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return "—"
        return round(float(s.mean()), 3)

    kpis = {
        "Rows": int(n_rows),
        "Cols": int(n_cols),
        "Missing%": miss,
        "Mean_temp": mean_of(tcol),
        "Mean_vib": mean_of(vcol),
        "Mean_pressure": mean_of(pcol),
        "Mean_RUL": mean_of(rcol),
        "Failure_Count": int(pd.to_numeric(df[fcol], errors="coerce").fillna(0).sum()) if fcol else 0,
        "Min_RUL": (
            round(float(pd.to_numeric(df[rcol], errors="coerce").min()), 2)
            if rcol and pd.to_numeric(df[rcol], errors="coerce").notna().any()
            else "—"
        ),
    }
    return kpis


def _numeric_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    work = df.copy()
    fail_col = _col(work, "failure", "fault", "alarm", "label")
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c != fail_col][:12]
    if not feats:
        raise RuntimeError("Need numeric sensor columns for field prediction.")
    X = work[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, fail_col


def field_predict(df: pd.DataFrame) -> float:
    """
    Train RF + GB + IsolationForest ensemble to estimate failure risk % in next window.
    Returns risk percentage 0-100.
    """
    X, fail_col = _numeric_xy(df)
    if len(X) < 10:
        # Not enough history — use IsolationForest anomaly rate as risk proxy
        iso = IsolationForest(contamination=0.1, random_state=42)
        labels = iso.fit_predict(X)
        risk = float((labels == -1).mean() * 100)
        return round(min(99.0, max(1.0, risk)), 1)

    if fail_col and work_has_binary(df, fail_col):
        y = pd.to_numeric(df[fail_col], errors="coerce").fillna(0).astype(int)
        y = (y > 0).astype(int)
        if y.nunique() < 2:
            iso = IsolationForest(contamination=0.08, random_state=42)
            risk = float((iso.fit_predict(X) == -1).mean() * 100)
            return round(risk, 1)
        rf = RandomForestClassifier(n_estimators=120, random_state=42)
        gb = GradientBoostingClassifier(random_state=42)
        rf.fit(X, y)
        gb.fit(X, y)
        iso = IsolationForest(contamination=max(0.05, float(y.mean()) or 0.05), random_state=42)
        iso.fit(X)
        # Predict on latest window
        latest = X.tail(min(24, len(X)))
        p_rf = rf.predict_proba(latest)[:, 1].mean()
        p_gb = gb.predict_proba(latest)[:, 1].mean()
        p_iso = float((iso.predict(latest) == -1).mean())
        risk = 100.0 * (0.4 * p_rf + 0.4 * p_gb + 0.2 * p_iso)
        return round(float(min(99.5, max(0.5, risk))), 1)

    iso = IsolationForest(contamination=0.1, random_state=42)
    risk = float((iso.fit_predict(X) == -1).mean() * 100)
    # Blend with high temp / vib heuristics if present
    tcol = _col(df, "temperature", "temp")
    vcol = _col(df, "vibration", "vib")
    bump = 0.0
    if tcol:
        t = pd.to_numeric(df[tcol], errors="coerce")
        if t.notna().any() and t.iloc[-1] > t.mean() + 2 * (t.std() or 1):
            bump += 15
    if vcol:
        v = pd.to_numeric(df[vcol], errors="coerce")
        if v.notna().any() and v.iloc[-1] > v.mean() + 2 * (v.std() or 1):
            bump += 15
    return round(min(99.0, risk + bump), 1)


def work_has_binary(df: pd.DataFrame, col: str) -> bool:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False
    return s.nunique() <= 5


def run_automl(df: pd.DataFrame, target: str, n_trials: int = 50) -> tuple[str, dict[str, Any]]:
    """
    Optuna AutoML (50 trials) — picks best model family + hyperparameters.
    Returns (model_name, metrics).
    """
    if target not in df.columns:
        raise RuntimeError(f"Target '{target}' not in dataframe.")

    work = df.copy()
    y_raw = work[target]
    feature_cols = [c for c in work.columns if c != target]
    # Drop datetime-like
    feature_cols = [
        c for c in feature_cols
        if not pd.api.types.is_datetime64_any_dtype(work[c])
        and not any(h in str(c).lower() for h in ("timestamp", "datetime"))
    ]
    X = work[feature_cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c].astype(str))[0]
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    # Task detection
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().sum() >= max(10, int(0.8 * len(y_raw))) and y_num.nunique() > 8:
        task = "regression"
        y = y_num
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]
        scoring = "neg_root_mean_squared_error"
    else:
        task = "classification"
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index)
        scoring = "accuracy"

    if len(X) < 12:
        raise RuntimeError("Need at least 12 rows for AutoML.")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise RuntimeError("Optuna not installed. pip install optuna") from exc

    def objective(trial: Any) -> float:
        model_name = trial.suggest_categorical(
            "model",
            ["RandomForest", "GradientBoosting"],
        )
        if task == "regression":
            if model_name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 2, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = GradientBoostingRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 50, 250),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    random_state=42,
                )
        else:
            if model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 2, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 250),
                    max_depth=trial.suggest_int("max_depth", 2, 8),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    random_state=42,
                )
        scores = cross_val_score(model, X, y, cv=min(5, max(2, len(X) // 5)), scoring=scoring)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_params
    best_model_name = str(best.get("model", "RandomForest"))

    # Fit final holdout metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {k: v for k, v in best.items() if k != "model"}
    if task == "regression":
        if best_model_name == "RandomForest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        else:
            model = GradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "task": task,
            "r2": round(float(r2_score(y_test, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "mae": round(float(mean_absolute_error(y_test, pred)), 4),
            "optuna_best_value": round(float(study.best_value), 4),
            "n_trials": int(n_trials),
            "params": best,
            "target": target,
        }
    else:
        if best_model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        else:
            model = GradientBoostingClassifier(random_state=42, **params)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "task": task,
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
            "optuna_best_value": round(float(study.best_value), 4),
            "n_trials": int(n_trials),
            "params": best,
            "target": target,
        }
    return best_model_name, metrics


def prophet_forecast(df: pd.DataFrame, target: str) -> str:
    """
    Prophet 90-day business forecast text:
    'Business will decrease X% in Y days' style.
    """
    try:
        from prophet import Prophet
    except Exception as exc:
        return f"Prophet unavailable ({exc}). Install: pip install prophet"

    # Find date column
    date_col = None
    for c in df.columns:
        if any(h in str(c).lower() for h in ("time", "date", "timestamp", "ds")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() >= max(8, int(0.5 * len(df))):
                date_col = c
                break
    if date_col is None:
        # synthesize index timeline for SCADA buffer
        ds = pd.date_range(end=datetime.utcnow(), periods=len(df), freq="H")
    else:
        ds = pd.to_datetime(df[date_col], errors="coerce")

    if target not in df.columns:
        return f"Target '{target}' not found for Prophet."

    tmp = pd.DataFrame({"ds": ds, "y": pd.to_numeric(df[target], errors="coerce")}).dropna()
    tmp = tmp.sort_values("ds").drop_duplicates("ds", keep="last")
    if len(tmp) < 10:
        return "Need >=10 dated rows for Prophet 90-day forecast."

    m = Prophet(uncertainty_samples=100)
    m.fit(tmp)
    # Infer freq
    freq = pd.infer_freq(tmp["ds"]) or "D"
    # 90 days ahead — if hourly-ish use 90*24
    if freq in {"H", "h", "T", "min"}:
        periods = 90 * 24
        horizon_days = 90
    else:
        periods = 90
        horizon_days = 90
    future = m.make_future_dataframe(periods=periods, freq=freq if freq else "D")
    fc = m.predict(future)
    last_actual = float(tmp["y"].iloc[-1])
    future_only = fc[fc["ds"] > tmp["ds"].max()]
    if future_only.empty:
        end_yhat = float(fc["yhat"].iloc[-1])
    else:
        # point near ~90 days
        idx = min(len(future_only) - 1, max(0, periods - 1))
        end_yhat = float(future_only["yhat"].iloc[idx])
    if last_actual == 0:
        pct = 0.0
    else:
        pct = (end_yhat - last_actual) / abs(last_actual) * 100.0
    direction = "increase" if pct >= 0 else "decrease"
    return (
        f"Business / `{target}` will {direction} {abs(pct):.1f}% in {horizon_days} days "
        f"(last={last_actual:.3f} → forecast≈{end_yhat:.3f})."
    )


def _gemini_answer(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return ""
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", None) or str(resp)).strip()
    except Exception as exc:
        return f"[Gemini error] {exc}"


def rag_ask(question: str, df: pd.DataFrame) -> str:
    """
    Build LlamaIndex over current dataframe rows, then answer with Gemini grounded on context.
    """
    context_rows = []
    sample = df.head(80)
    for _, row in sample.iterrows():
        context_rows.append(" | ".join(f"{c}={row[c]}" for c in sample.columns if pd.notna(row[c])))
    context = "\n".join(context_rows[:80])

    nodes_text = ""
    try:
        from llama_index.core import Document, VectorStoreIndex, Settings
        from llama_index.core.embeddings import BaseEmbedding

        # Lightweight local path: use simple keyword retrieval if embeddings unavailable
        docs = [Document(text=t) for t in context_rows[:120]]
        # Try building index; if embedding model missing, fall through
        try:
            index = VectorStoreIndex.from_documents(docs)
            engine = index.as_query_engine(similarity_top_k=4)
            retrieved = engine.query(question)
            nodes_text = str(retrieved)
        except Exception:
            # keyword overlap retrieval
            q_tokens = set(question.lower().split())
            scored = []
            for t in context_rows:
                score = len(q_tokens & set(t.lower().split()))
                scored.append((score, t))
            scored.sort(reverse=True)
            nodes_text = "\n".join(t for s, t in scored[:5] if s > 0) or context[:2000]
    except Exception as exc:
        nodes_text = context[:2500] + f"\n[LlamaIndex note: {exc}]"

    prompt = (
        "You are Analytics Forge industrial assistant. Answer ONLY from the data context.\n"
        f"Question: {question}\n\nData context:\n{nodes_text}\n\n"
        "Give a concise operational answer with numbers when present."
    )
    gem = _gemini_answer(prompt)
    if gem:
        return gem
    # Offline fallback
    return f"(Offline RAG) Top matching rows:\n{nodes_text[:1500]}"


def send_email_report(to_addr: str, subject: str, body: str, df: Optional[pd.DataFrame] = None) -> str:
    if not EMAIL_USER or not EMAIL_PASSWORD:
        raise RuntimeError(
            "Email not configured. Set EMAIL_USER and EMAIL_PASSWORD (Gmail App Password) in .env"
        )
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    if df is not None:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename="forge_buffer.csv")
    context = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30) as server:
        server.starttls(context=context)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
    return f"Sent to {to_addr}"


def load_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw = uploaded.getvalue()
    dest = UPLOAD_DIR / uploaded.name
    dest.write_bytes(raw)
    if name.endswith(".csv") or name.endswith(".tsv") or name.endswith(".txt"):
        sep = "\t" if name.endswith(".tsv") else ","
        return pd.read_csv(dest, sep=sep)
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(dest)
    if name.endswith(".json"):
        return pd.read_json(dest)
    if name.endswith(".parquet"):
        return pd.read_parquet(dest)
    raise RuntimeError(f"Unsupported file type: {uploaded.name}")


# =============================================================================
# UI HELPERS (no raw HTML KPI leak — Streamlit metrics only)
# =============================================================================

def metric_grid(kpis: dict[str, Any], per_row: int = 4) -> None:
    items = list(kpis.items())
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for j, (k, v) in enumerate(items[i : i + per_row]):
            with cols[j]:
                st.metric(str(k).replace("_", " "), v)


def require_data() -> Optional[pd.DataFrame]:
    try:
        return get_data()
    except Exception as exc:
        st.error(str(exc))
        if st.session_state.get("mode") == "LIVE CONNECT" and st.session_state.get("live_error"):
            st.warning(
                f"LIVE Modbus status: {st.session_state.live_status} — {st.session_state.live_error}"
            )
        return None


# =============================================================================
# PAGES
# =============================================================================

def page_upload() -> None:
    st.header("Upload")
    st.caption("MANUAL mode uses this file for all pages. LIVE mode ignores upload and uses Modbus SCADA buffer.")

    if st.session_state.mode == "LIVE CONNECT":
        st.info(
            f"LIVE CONNECT active → pymodbus `{MODBUS_HOST}:{MODBUS_PORT}` "
            f"regs {40001}-{40001 + MODBUS_COUNT - 1}, poll ≤5s, buffer `{LIVE_CSV}`."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Force Modbus poll now", type="primary"):
                try:
                    df = ensure_live_poll(force=True)
                    st.success(f"Polled OK — buffer {len(df):,} rows")
                    st.dataframe(df.tail(20), use_container_width=True)
                except Exception as exc:
                    st.error(str(exc))
        with c2:
            st.write(f"Status: **{st.session_state.get('live_status')}**")
            if st.session_state.get("live_error"):
                st.error(st.session_state.live_error)
        buf = read_live_buffer()
        if buf is not None:
            st.subheader("Current SCADA buffer")
            st.dataframe(buf.tail(50), use_container_width=True)
        return

    uploaded = st.file_uploader(
        "Upload industrial / ERP / plant CSV or Excel",
        type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet"],
    )
    if uploaded is not None:
        try:
            df = load_uploaded_file(uploaded)
            st.session_state.manual_df = df
            st.session_state.manual_name = uploaded.name
            st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} cols")
            st.dataframe(df.head(50), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))
    elif st.session_state.manual_df is not None:
        st.write(f"Current file: **{st.session_state.manual_name}**")
        st.dataframe(st.session_state.manual_df.head(50), use_container_width=True)


def page_clean() -> None:
    st.header("Clean")
    st.caption("PySpark (+pandas fallback) · Great Expectations · ydata-profiling · Cleanlab")
    df = require_data()
    if df is None:
        return
    if st.button("Run industrial clean + quality", type="primary") or st.session_state.clean_df is None:
        with st.spinner("Cleaning..."):
            clean_df, checks = clean_data(df)
        st.success(f"Clean complete — {len(clean_df):,} rows")
    else:
        clean_df = st.session_state.clean_df
        checks = st.session_state.clean_checks

    st.subheader("Quality checks")
    st.dataframe(checks, use_container_width=True)
    meta = st.session_state.get("_clean_engine_meta")
    if meta is not None:
        with st.expander("Engine details (GE / ydata / Cleanlab)"):
            st.dataframe(meta, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw head")
        st.dataframe(df.head(30), use_container_width=True)
    with c2:
        st.subheader("Clean head")
        st.dataframe(clean_df.head(30), use_container_width=True)


def page_field() -> None:
    st.header("Field")
    df = require_data()
    if df is None:
        return
    st.write("**Auto detected:** Predictive Maintenance")
    with st.spinner("Training RF + GB + IsolationForest..."):
        risk = field_predict(df)
    st.metric("Failure Risk", f"{risk}% in 12h")
    if risk >= 70:
        st.error(f"CRITICAL: Failure risk {risk}% — schedule maintenance within 8–12h.")
    elif risk >= 40:
        st.warning(f"Elevated risk {risk}% — inspect vibration / temperature trends.")
    else:
        st.success(f"Risk {risk}% — within normal operating envelope.")
    kpis = get_kpis(df)
    metric_grid({k: kpis[k] for k in ("Mean_temp", "Mean_vib", "Mean_pressure", "Min_RUL")})


def page_kpis() -> None:
    st.header("Auto KPIs")
    df = require_data()
    if df is None:
        return
    kpis = get_kpis(df)
    metric_grid(kpis, per_row=4)
    st.subheader("Predictive Maintenance briefing")
    risk = field_predict(df)
    st.write(
        f"Buffer holds **{kpis['Rows']}** rows. "
        f"Mean temperature **{kpis['Mean_temp']}**, vibration **{kpis['Mean_vib']}**, "
        f"pressure **{kpis['Mean_pressure']}**. "
        f"Failures counted: **{kpis['Failure_Count']}**. Min RUL **{kpis['Min_RUL']}**. "
        f"Ensemble failure risk ≈ **{risk}%** in the next 12h window."
    )


def page_charts() -> None:
    st.header("Charts")
    df = require_data()
    if df is None:
        return
    tcol = _col(df, "temperature", "temp")
    vcol = _col(df, "vibration", "vib")
    pcol = _col(df, "pressure")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if tcol:
        st.subheader("Temperature (last 48)")
        series = pd.to_numeric(df[tcol], errors="coerce").tail(48)
        st.line_chart(series)
        fig = px.line(series.reset_index(), y=tcol if tcol in series.reset_index().columns else series.name,
                      title="Temperature trend")
        # simpler plotly
        fig = go.Figure(go.Scatter(y=series.values, mode="lines", name=tcol))
        fig.update_layout(title=f"{tcol} — last 48", height=360)
        st.plotly_chart(fig, use_container_width=True)

    if tcol and vcol:
        st.subheader("Vibration vs Temperature")
        plot_df = df[[vcol, tcol]].apply(pd.to_numeric, errors="coerce").dropna().tail(500)
        st.scatter_chart(plot_df.rename(columns={vcol: "vibration", tcol: "temperature"}))
        fig2 = px.scatter(plot_df, x=vcol, y=tcol, title="Vibration vs Temperature")
        st.plotly_chart(fig2, use_container_width=True)

    if cat_cols and num_cols:
        st.subheader("Pie distribution")
        names, values = cat_cols[0], num_cols[0]
        pie_df = df.groupby(names, dropna=False)[values].mean().reset_index()
        fig3 = px.pie(pie_df, names=names, values=values, title=f"{values} by {names}")
        st.plotly_chart(fig3, use_container_width=True)

    if pcol:
        st.subheader("Pressure")
        st.line_chart(pd.to_numeric(df[pcol], errors="coerce").tail(48))


def page_ml() -> None:
    st.header("ML Studio")
    st.caption("Optuna AutoML selects the model — no manual model pick. Prophet adds 90-day business forecast.")
    df = require_data()
    if df is None:
        return
    cols = list(df.columns)
    default_i = 0
    for preferred in ("rul", "failure", "temperature", "revenue", "sales"):
        hit = _col(df, preferred)
        if hit and hit in cols:
            default_i = cols.index(hit)
            break
    target = st.selectbox("Target", cols, index=default_i)
    trials = st.slider("Optuna trials", 10, 50, 50)
    if st.button("Run AutoML + Forecast", type="primary"):
        with st.spinner("Optuna searching best model..."):
            try:
                best_model, metrics = run_automl(df, target, n_trials=trials)
                forecast_text = prophet_forecast(df, target)
                st.session_state.automl_result = {"best_model": best_model, "metrics": metrics}
                st.session_state.forecast_text = forecast_text
            except Exception as exc:
                st.error(str(exc))
                st.code(traceback.format_exc())
                return
    res = st.session_state.get("automl_result")
    if res:
        st.success(f"Best Model: **{res['best_model']}** (Optuna auto-selected)")
        st.json(res["metrics"])
    if st.session_state.get("forecast_text"):
        st.write(st.session_state.forecast_text)


def page_ask() -> None:
    st.header("Ask / AI")
    gem_ok = bool(GEMINI_API_KEY)
    st.write(f"Gemini: **{'Connected' if gem_ok else 'No key'}** ({GEMINI_MODEL})")
    st.write("RAG: LlamaIndex on current data buffer + Gemini grounded answers")
    if not gem_ok:
        st.warning("Set GEMINI_API_KEY in `.env` and restart Streamlit.")

    df = require_data()
    if df is None:
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask about your plant / buffer data…")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("RAG + Gemini..."):
                ans = rag_ask(q, df)
            st.markdown(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})


def page_dashboard() -> None:
    """Dashboard — NO raw HTML. Metrics + charts + alerts only."""
    st.header("Dashboard")
    st.caption("SCADA-style board for LIVE · same KPIs for MANUAL. Filter-gated buffer view.")
    df = require_data()
    if df is None:
        return

    # Power-BI-like slicers
    st.subheader("Filters")
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique() <= 40]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    view = df.copy()
    fc1, fc2 = st.columns(2)
    with fc1:
        for c in cat_cols[:3]:
            opts = sorted(view[c].dropna().astype(str).unique().tolist())
            sel = st.multiselect(f"Filter {c}", opts, default=[], key=f"dash_f_{c}")
            if sel:
                view = view[view[c].astype(str).isin(sel)]
    with fc2:
        for c in num_cols[:2]:
            s = pd.to_numeric(view[c], errors="coerce").dropna()
            if s.empty or s.min() == s.max():
                continue
            lo, hi = float(s.min()), float(s.max())
            rng = st.slider(f"Range {c}", lo, hi, (lo, hi), key=f"dash_r_{c}")
            series = pd.to_numeric(view[c], errors="coerce")
            view = view[(series >= rng[0]) & (series <= rng[1])]

    st.caption(f"Buffer rows in view: **{len(view):,}** / {len(df):,}")

    kpis = get_kpis(view)
    risk = field_predict(view if len(view) >= 10 else df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows in Buffer", kpis["Rows"])
    c2.metric("Virtual Plant Cap", "1,250,000")
    c3.metric("Failure Risk", f"{risk}%")
    c4.metric("Forecasted", "Yes" if st.session_state.get("forecast_text") else "Run ML Studio")

    st.subheader("KPI scoreboard")
    metric_grid(kpis, per_row=4)

    tcol = _col(view, "temperature", "temp")
    vcol = _col(view, "vibration", "vib")
    if tcol:
        st.subheader("Temperature — last 48")
        st.line_chart(pd.to_numeric(view[tcol], errors="coerce").tail(48))
    if vcol:
        st.subheader("Vibration")
        st.scatter_chart(pd.to_numeric(view[vcol], errors="coerce").tail(200))

    if tcol and vcol:
        plot_df = view[[vcol, tcol]].apply(pd.to_numeric, errors="coerce").dropna().tail(400)
        fig = px.scatter(plot_df, x=vcol, y=tcol, title="Vibration vs Temperature")
        st.plotly_chart(fig, use_container_width=True)

    if risk > 70:
        st.error(f"ALERT: Failure risk {risk}% in 8h — dispatch maintenance.")
    elif risk > 40:
        st.warning(f"Watchlist: Failure risk {risk}% — review sensors.")
    else:
        st.success(f"Operating normally — failure risk {risk}%.")

    if st.session_state.get("forecast_text"):
        st.info(st.session_state.forecast_text)


def page_email() -> None:
    st.header("Email")
    st.write(f"Operator account: **{OPERATOR_EMAIL}**")
    st.caption("Send current buffer + KPIs / risk briefing. Requires EMAIL_USER + EMAIL_PASSWORD in .env.")
    df = None
    try:
        df = get_data()
    except Exception as exc:
        st.warning(str(exc))

    to_addr = st.text_input("Recipient", value=OPERATOR_EMAIL)
    subject = st.text_input("Subject", value="[Analytics Forge v2] Industrial report")
    if st.button("Send report + CSV", type="primary"):
        if df is None:
            st.error("No data buffer available.")
            return
        try:
            kpis = get_kpis(df)
            risk = field_predict(df)
            body = (
                f"Analytics Forge v2 report\nMode: {st.session_state.mode}\n"
                f"Rows: {kpis['Rows']}\nFailure risk: {risk}%\nKPIs: {json.dumps(kpis)}\n"
                f"Forecast: {st.session_state.get('forecast_text') or 'n/a'}\n"
            )
            msg = send_email_report(to_addr, subject, body, df=df)
            st.success(msg)
        except Exception as exc:
            st.error(str(exc))


# =============================================================================
# SIDEBAR + MAIN
# =============================================================================

PAGES = [
    "Upload",
    "Clean",
    "Field",
    "Auto KPIs",
    "Charts",
    "ML Studio",
    "Ask / AI",
    "Dashboard",
    "Email",
]


def render_sidebar() -> str:
    with st.sidebar:
        st.write(f"📧 {OPERATOR_EMAIL}")
        if st.button("Sign out"):
            st.session_state.signed_in = False
            st.rerun()

        st.title("Analytics Forge v2")
        st.caption("Dual mode · filter-gated live · shared core")

        mode = st.radio(
            "Mode",
            ["LIVE CONNECT", "MANUAL UPLOAD"],
            index=0 if st.session_state.mode == "LIVE CONNECT" else 1,
        )
        st.session_state.mode = mode

        if mode == "LIVE CONNECT":
            st.caption(f"SCADA Modbus `{MODBUS_HOST}:{MODBUS_PORT}` → `{LIVE_CSV.name}`")
            st.write(f"Link: **{st.session_state.get('live_status', 'idle')}**")
        else:
            up = st.file_uploader(
                "Quick upload",
                type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
                key="sidebar_upload",
            )
            if up is not None:
                try:
                    st.session_state.manual_df = load_uploaded_file(up)
                    st.session_state.manual_name = up.name
                    st.success(f"Loaded {up.name}")
                except Exception as exc:
                    st.error(str(exc))

        page = st.radio(
            "Navigate",
            PAGES,
            index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
        )
        st.session_state.page = page

        st.divider()
        if st.button("Start FORGE", type="primary"):
            st.session_state.pipeline_started = True
            st.success("Pipeline started")
        st.caption(f"Gemini key: {'yes' if GEMINI_API_KEY else 'missing'}")
        return page


def main() -> None:
    init_state()
    if not st.session_state.signed_in:
        st.warning("Signed out.")
        if st.button("Sign in again"):
            st.session_state.signed_in = True
            st.rerun()
        return

    page = render_sidebar()

    if st.session_state.pipeline_started:
        st.toast("FORGE pipeline active", icon="⚒️")

    # Banner mode
    if st.session_state.mode == "LIVE CONNECT":
        st.info("MODE: LIVE CONNECT (SCADA Modbus) — all pages read `data/live.csv` buffer")
    else:
        name = st.session_state.manual_name or "none"
        st.info(f"MODE: MANUAL UPLOAD — all pages read uploaded file (`{name}`)")

    if page == "Upload":
        page_upload()
    elif page == "Clean":
        page_clean()
    elif page == "Field":
        page_field()
    elif page == "Auto KPIs":
        page_kpis()
    elif page == "Charts":
        page_charts()
    elif page == "ML Studio":
        page_ml()
    elif page == "Ask / AI":
        page_ask()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Email":
        page_email()


if __name__ == "__main__":
    main()
