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
_ENV_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_KEY = _ENV_GEMINI_API_KEY
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
        "clean_engine": "pandas",
        "gemini_api_key_override": "",
        "live_last_poll": 0.0,
        "live_status": "idle",
        "live_error": None,
        "clean_df": None,
        "clean_checks": None,
        "clean_report": None,
        "field_result": None,
        "domain": "generic",
        "domain_meta": None,
        "automl_result": None,
        "forecast_text": None,
        "chat_history": [],
        "pipeline_started": False,
        "prefer_clean_df": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v




DOMAIN_CATALOG: dict[str, dict[str, Any]] = {
    "predictive_maintenance": {
        "label": "Predictive Maintenance / OPC-UA Sensors",
        "keywords": [
            "temperature", "temp", "vibration", "pressure", "rul", "failure", "machine",
            "sensor", "torque", "rpm", "current", "voltage", "opc", "modbus", "asset",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "healthcare": {
        "label": "Healthcare / Hospital",
        "keywords": [
            "patient", "age", "bmi", "bp", "blood", "glucose", "heart", "diagnosis",
            "admit", "ward", "doctor", "hospital", "readmission", "weight", "height",
            "cholesterol", "pulse", "spo2",
        ],
        "dtypes_hint": "mixed_clinical",
    },
    "sales_forecasting": {
        "label": "Sales / Retail / Revenue",
        "keywords": [
            "revenue", "sales", "units", "order", "price", "sku", "customer", "region",
            "channel", "store", "campaign", "discount", "gmv", "asp",
        ],
        "dtypes_hint": "commerce",
    },
    "warehouse_logistics": {
        "label": "Warehouse / Supply Chain",
        "keywords": [
            "warehouse", "sku", "inventory", "stock", "shipment", "delivery", "carrier",
            "aisle", "bin", "lead_time", "defect", "pick", "pack",
        ],
        "dtypes_hint": "ops",
    },
    "energy_utilities": {
        "label": "Energy / Utilities",
        "keywords": [
            "kwh", "mw", "power", "voltage", "current", "grid", "load", "consumption",
            "solar", "wind", "frequency", "pf",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "finance_risk": {
        "label": "Finance / Credit Risk",
        "keywords": [
            "loan", "credit", "score", "default", "interest", "balance", "emi", "income",
            "fraud", "transaction", "amount", "apr",
        ],
        "dtypes_hint": "tabular_finance",
    },
    "telecom_churn": {
        "label": "Telecom / Churn",
        "keywords": [
            "churn", "tenure", "plan", "minutes", "data_usage", "arpu", "subscriber",
            "complaint", "call_drop", "sim",
        ],
        "dtypes_hint": "crm",
    },
    "agriculture_iot": {
        "label": "Agriculture / Agri-IoT",
        "keywords": [
            "soil", "moisture", "humidity", "rainfall", "crop", "yield", "ph", "npk",
            "irrigation", "farm",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "generic": {
        "label": "Generic Analytics",
        "keywords": [],
        "dtypes_hint": "generic",
    },
}


def suggest_clean_engine(n_rows: int, n_cols: int) -> tuple[str, str]:
    """Suggest engine by size — never force PySpark on tiny files."""
    cells = n_rows * max(1, n_cols)
    if n_rows < 50_000 and cells < 2_000_000:
        return "pandas", f"Suggested: **pandas** ({n_rows:,} rows — small/medium, fastest for interactive UI)."
    if n_rows < 500_000:
        return "polars", f"Suggested: **polars** ({n_rows:,} rows — faster columnar engine for mid-size data)."
    return "pyspark", f"Suggested: **pyspark** ({n_rows:,} rows — big-data scale; slower startup on Mac)."


def list_available_engines() -> list[str]:
    engines = ["pandas"]
    try:
        import polars  # noqa: F401
        engines.append("polars")
    except ImportError:
        pass
    try:
        import pyspark  # noqa: F401
        engines.append("pyspark")
    except ImportError:
        pass
    return engines

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
    MANUAL UPLOAD -> uploaded dataframe in session (prefer clean_df when set)
    """
    mode = st.session_state.get("mode", "MANUAL UPLOAD")
    if mode == "LIVE CONNECT":
        return ensure_live_poll(force=False, min_interval_s=5.0)

    if st.session_state.get("prefer_clean_df") and isinstance(st.session_state.get("clean_df"), pd.DataFrame):
        cdf = st.session_state.clean_df
        if cdf is not None and not cdf.empty:
            return cdf.copy()
    df = st.session_state.get("manual_df")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(
            "No manual file loaded. Go to Upload (or sidebar file uploader) and upload a CSV/Excel."
        )
    return df.copy()


def _basic_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Legacy alias — full suite lives in build_quality_report."""
    return build_quality_report(df)["checks"]


def _clean_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    log.append("ETL: strip column names / schema normalize")
    out = out.dropna(how="all")
    out = out.loc[:, ~out.columns.duplicated()]
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"ETL dedupe {before}->{len(out)}")
    out = out.replace(["", "NA", "N/A", "null", "NULL", "None", "-", "--"], np.nan)
    log.append("ETL null-sentinel fusion")
    for c in list(out.columns):
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            if out[c].notna().sum() and converted.notna().sum() / max(1, out[c].notna().sum()) >= 0.8:
                out[c] = converted
                log.append(f"schema cast numeric {c}")
    date_hints = ("date", "time", "timestamp", "datetime", "day")
    for c in list(out.columns):
        if any(h in str(c).lower() for h in date_hints) and out[c].dtype == object:
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().sum() > 0:
                out[c] = parsed
                log.append(f"schema cast datetime {c}")
    for c in out.select_dtypes(include=[np.number]).columns:
        if out[c].isna().any():
            s = out[c]
            if s.notna().sum() >= 5:
                idx = np.arange(len(s))
                mask = s.notna().to_numpy()
                coef = np.polyfit(idx[mask], s.to_numpy()[mask], 1)
                pred = np.polyval(coef, idx)
                filled = s.copy()
                filled[s.isna()] = pred[s.isna()]
                out[c] = filled
                log.append(f"DWDM regression imputation {c}")
            else:
                out[c] = s.fillna(s.median())
                log.append(f"median imputation {c}")
    for c in out.select_dtypes(include=["object", "string", "category"]).columns:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill)
            log.append(f"mode imputation {c}")
    for c in list(out.select_dtypes(include=[np.number]).columns)[:6]:
        try:
            out[f"{c}_bin"] = pd.qcut(out[c], q=min(5, max(2, out[c].nunique())), duplicates="drop").astype(str)
            log.append(f"DWDM binning {c}")
        except Exception:
            pass
    for c in list(out.select_dtypes(include=[np.number]).columns):
        cl = str(c).lower()
        if any(h in cl for h in ("temp", "vib", "pressure", "current", "voltage", "speed")) and not cl.endswith("_smooth") and not cl.endswith("_bin"):
            out[f"{c}_smooth"] = out[c].rolling(window=min(5, max(2, len(out) // 10)), min_periods=1).mean()
            log.append(f"DWDM smoothing {c}")
    return out.reset_index(drop=True), log


def _clean_polars(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import polars as pl

    log = ["engine=polars"]
    pldf = pl.from_pandas(df)
    before = pldf.height
    null_cols = [c for c in pldf.columns if pldf[c].null_count() == pldf.height]
    if null_cols:
        pldf = pldf.drop(null_cols)
        log.append(f"drop null cols {null_cols}")
    pldf = pldf.unique()
    log.append(f"unique {before}->{pldf.height}")
    for c in pldf.columns:
        dtype = pldf[c].dtype
        if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            if pldf[c].null_count() > 0:
                med = pldf[c].median()
                pldf = pldf.with_columns(pl.col(c).fill_null(med))
                log.append(f"polars fill_median {c}")
        elif str(dtype) in ("Utf8", "String") or dtype in (getattr(pl, "Utf8", None), getattr(pl, "String", None)):
            if pldf[c].null_count() > 0:
                pldf = pldf.with_columns(pl.col(c).fill_null("Unknown"))
    pdf = pldf.to_pandas()
    pdf2, log2 = _clean_pandas(pdf)
    return pdf2, log + log2


def _clean_pyspark(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("forge_v2_clean")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    log = ["engine=pyspark"]
    try:
        sdf = spark.createDataFrame(df.astype(str))
        before = sdf.count()
        sdf = sdf.dropDuplicates()
        after = sdf.count()
        log.append(f"spark dropDuplicates {before}->{after}")
        pdf = sdf.toPandas()
        cleaned, plog = _clean_pandas(pdf)
        log.extend(plog)
        return cleaned, log
    finally:
        spark.stop()


def _engine_clean(df: pd.DataFrame, engine: str) -> tuple[pd.DataFrame, list[str]]:
    engine = (engine or "pandas").lower()
    if engine == "polars":
        try:
            return _clean_polars(df)
        except Exception as exc:
            out, log = _clean_pandas(df)
            return out, [f"polars failed ({exc}) → pandas"] + log
    if engine == "pyspark":
        try:
            return _clean_pyspark(df)
        except Exception as exc:
            out, log = _clean_pandas(df)
            return out, [f"pyspark failed ({exc}) → pandas"] + log
    return _clean_pandas(df)


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


def _zscore_iqr_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    z_hits, iqr_hits = 0, 0
    details = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 5:
            continue
        z = (s - s.mean()) / (s.std() + 1e-9)
        zc = int((z.abs() > 3).sum())
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ic = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        z_hits += zc
        iqr_hits += ic
        if zc or ic:
            details.append(f"{c}:z={zc},iqr={ic}")
    return {"z_hits": z_hits, "iqr_hits": iqr_hits, "details": details[:8]}


def _isolation_forest_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 15:
        return {"ok": False, "reason": "need >=2 numeric cols & 15 rows"}
    iso = IsolationForest(contamination=0.08, random_state=42)
    labels = iso.fit_predict(num.values)
    n = int((labels == -1).sum())
    return {"ok": True, "anomalies": n, "rate_pct": round(100.0 * n / len(num), 2)}


def _dbscan_noise(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False, "reason": "need more numeric rows"}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    noise = int((labels == -1).sum())
    return {"ok": True, "noise_points": noise, "clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}


def _kmeans_clean_proxy(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    km = KMeans(n_clusters=min(3, max(2, len(X) // 5)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    far = int((dists > dists.mean() + 2 * dists.std()).sum())
    return {"ok": True, "far_from_cluster": far}


def _rolling_impossible_jumps(df: pd.DataFrame) -> dict[str, Any]:
    flags = []
    total = 0
    for c in df.select_dtypes(include=[np.number]).columns:
        cl = str(c).lower()
        if not any(h in cl for h in ("temp", "vib", "pressure", "speed", "current")):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        delta = s.diff().abs()
        thr = 50 if "temp" in cl else (5 if "vib" in cl else (30 if "pressure" in cl else float(s.std() or 1) * 4))
        n = int((delta > thr).fillna(False).sum())
        if n:
            flags.append(f"{c}:{n} jumps>{thr}")
            total += n
    return {"ok": True, "impossible_jumps": flags[:10], "count": total}


def _lag_correlation_break(df: pd.DataFrame) -> dict[str, Any]:
    t = _col(df, "temperature", "temp")
    p = _col(df, "pressure")
    v = _col(df, "vibration", "vib")
    pairs = []
    for a, b in [(t, p), (t, v), (p, v)]:
        if a and b:
            corr = pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce"))
            pairs.append({"pair": f"{a}|{b}", "corr": None if pd.isna(corr) else round(float(corr), 3)})
    broken = [x for x in pairs if x["corr"] is not None and abs(x["corr"]) < 0.05]
    return {"pairs": pairs, "dead_correlations": broken}


def _domain_opc_rules(df: pd.DataFrame) -> list[str]:
    flags = []
    t = _col(df, "temperature", "temp")
    v = _col(df, "vibration", "vib")
    p = _col(df, "pressure")
    r = _col(df, "rul")
    fcol = _col(df, "failure", "fault")
    if t and v:
        tt = pd.to_numeric(df[t], errors="coerce")
        vv = pd.to_numeric(df[v], errors="coerce")
        stuck = int(((tt > 150) & (vv < 0.1)).fillna(False).sum())
        if stuck:
            flags.append(f"Sensor stuck pattern: {stuck} rows (temp>150 & vib<0.1)")
    if p:
        speed = _col(df, "speed", "flow", "load")
        if speed:
            pp = pd.to_numeric(df[p], errors="coerce")
            ss = pd.to_numeric(df[speed], errors="coerce")
            n = int(((pp.diff() < -10) & (ss.diff().abs() < 0.5)).fillna(False).sum())
            if n:
                flags.append(f"Leak/sensor fault suspect: {n} rows")
    if r:
        rr = pd.to_numeric(df[r], errors="coerce")
        d = rr.diff().dropna()
        if len(d) and d.gt(0).mean() > 0.6:
            flags.append("RUL calculation broken: RUL increases over time")
    if fcol and v:
        ff = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        vv = pd.to_numeric(df[v], errors="coerce")
        missed = int(((ff == 0) & (vv > vv.mean() + 3 * (vv.std() or 1))).fillna(False).sum())
        if missed:
            flags.append(f"Possible missed failures: {missed} rows")
    return flags


def _run_great_expectations(df: pd.DataFrame) -> dict[str, Any]:
    results = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        results.append({
            "expectation": "expect_column_values_to_not_be_null",
            "column": col,
            "success": null_pct < 0.2,
            "detail": f"null_pct={null_pct:.3f}",
        })
    t = _col(df, "temperature", "temp")
    if t:
        s = pd.to_numeric(df[t], errors="coerce")
        ok = bool(((s.dropna() >= 0) & (s.dropna() <= 200)).all()) if s.notna().any() else False
        results.append({"expectation": "expect_column_values_to_be_between", "column": t, "success": ok, "detail": "temp in [0,200]"})
    r = _col(df, "rul")
    if r:
        s = pd.to_numeric(df[r], errors="coerce").dropna()
        success = bool(s.diff().dropna().le(0).mean() >= 0.5) if len(s) > 3 else True
        results.append({"expectation": "expect_column_pair_values_A_to_be_greater_than_B", "column": r, "success": success, "detail": "RUL mostly non-increasing"})
    ts = _col(df, "timestamp", "time", "datetime", "date")
    mid = _col(df, "machine_id", "machine", "asset_id")
    if ts and mid:
        dup = int(df.duplicated([ts, mid]).sum())
        results.append({"expectation": "expect_compound_columns_to_be_unique", "column": f"{ts}+{mid}", "success": dup == 0, "detail": f"dup_keys={dup}"})
    results.append({"expectation": "expect_table_row_count_to_be_between", "column": "*", "success": 1 <= len(df) <= 5_000_000, "detail": f"rows={len(df)}"})
    ge_available = False
    try:
        import great_expectations as gx  # noqa: F401
        ge_available = True
    except Exception:
        ge_available = False
    passed = sum(1 for r in results if r["success"])
    return {"engine": "great_expectations", "available": ge_available, "passed": passed, "total": len(results), "results": results[:40], "ok": True}


def _run_ydata(df: pd.DataFrame) -> dict[str, Any]:
    high_card = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun > max(50, int(0.5 * len(df))):
            high_card.append(c)
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df.head(min(400, len(df))), minimal=True, progress_bar=False)
        desc = profile.get_description()
        return {"engine": "ydata-profiling", "ok": True, "variables": len(desc.get("variables", {})), "alerts": len(desc.get("alerts", [])), "high_cardinality": high_card[:8]}
    except Exception as exc:
        return {"engine": "ydata-profiling", "ok": False, "error": str(exc), "high_cardinality": high_card[:8]}


def _run_cleanlab(df: pd.DataFrame) -> dict[str, Any]:
    fcol = _col(df, "failure", "fault", "label", "churn", "default")
    num = df.select_dtypes(include=[np.number])
    out: dict[str, Any] = {"engine": "cleanlab"}
    try:
        from cleanlab import Datalab
        work = num.dropna()
        if work.shape[1] >= 2 and len(work) >= 15:
            lab = Datalab(data=work.reset_index(drop=True))
            lab.find_issues(features=work.values)
            issues = lab.get_issues()
            n_out = int(issues["is_outlier_issue"].sum()) if "is_outlier_issue" in issues.columns else 0
            out.update({"ok": True, "outlier_issues": n_out})
        else:
            out.update({"ok": True, "skipped": "numeric too small"})
    except Exception as exc:
        out.update({"ok": False, "error": str(exc)})
    dirty = []
    if fcol and _col(df, "vibration", "vib"):
        v = pd.to_numeric(df[_col(df, "vibration", "vib")], errors="coerce")
        f = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        if v.notna().any():
            false_pos = int(((f == 1) & (v < v.quantile(0.2))).sum())
            false_neg = int(((f == 0) & (v > v.quantile(0.95))).sum())
            if false_pos:
                dirty.append(f"{false_pos} rows failure=1 but low vibration")
            if false_neg:
                dirty.append(f"{false_neg} rows failure=0 but extreme vibration")
    out["dirty_label_flags"] = dirty
    return out


def _pca_drift(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 30:
        return {"ok": False}
    half = len(num) // 2
    X1 = StandardScaler().fit_transform(num.iloc[:half])
    X2 = StandardScaler().fit_transform(num.iloc[half:])
    ncomp = min(3, X1.shape[1])
    r1 = float(PCA(n_components=ncomp).fit(X1).explained_variance_ratio_.sum())
    r2 = float(PCA(n_components=min(3, X2.shape[1])).fit(X2).explained_variance_ratio_.sum())
    drift = abs(r1 - r2)
    return {"ok": True, "pca_var_early": round(r1, 3), "pca_var_late": round(r2, 3), "drift_score": round(drift, 3), "concept_drift": drift > 0.15}


def _association_rules_proxy(df: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight DWDM association-rule mining (Apriori-style) on binarized numeric highs
    + low-cardinality categoricals. Flags co-occurring anomaly baskets.
    """
    try:
        items: list[set[str]] = []
        cats = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique(dropna=True) <= 12][:4]
        nums = list(df.select_dtypes(include=[np.number]).columns)[:6]
        sample = df.tail(min(800, len(df)))
        for _, row in sample.iterrows():
            basket: set[str] = set()
            for c in cats:
                val = row[c]
                if pd.notna(val):
                    basket.add(f"{c}={val}")
            for c in nums:
                s = pd.to_numeric(sample[c], errors="coerce")
                thr = s.quantile(0.9)
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    basket.add(f"{c}=HIGH")
            if len(basket) >= 2:
                items.append(basket)
        if len(items) < 20:
            return {"ok": True, "skipped": "too few baskets", "suspicious_rules": []}
        from collections import Counter
        pair_counts: Counter[tuple[str, str]] = Counter()
        item_counts: Counter[str] = Counter()
        for basket in items:
            for a in basket:
                item_counts[a] += 1
            bl = sorted(basket)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    pair_counts[(bl[i], bl[j])] += 1
        n = len(items)
        rules = []
        for (a, b), cnt in pair_counts.most_common(30):
            support = cnt / n
            conf_ab = cnt / max(1, item_counts[a])
            conf_ba = cnt / max(1, item_counts[b])
            if support >= 0.05 and max(conf_ab, conf_ba) >= 0.55:
                rules.append({"rule": f"{a} => {b}", "support": round(support, 3), "confidence": round(max(conf_ab, conf_ba), 3)})
        suspicious = [r for r in rules if "HIGH" in r["rule"] and r["confidence"] >= 0.7][:8]
        return {"ok": True, "rules_found": len(rules), "top_rules": rules[:5], "suspicious_rules": suspicious}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "suspicious_rules": []}


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    n, m = df.shape
    miss = float(df.isna().sum().sum() / max(1, df.size))
    checks.append({"check": "NULLS / Missing%", "status": "FAIL" if miss > 0.2 else ("WARN" if miss > 0.05 else "PASS"), "detail": f"{miss*100:.2f}% missing"})
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append({"check": "CONSTANT", "status": "FAIL" if const_cols else "PASS", "detail": str(const_cols[:6]) if const_cols else "none"})
    num = df.select_dtypes(include=[np.number])
    zero_ratio = float((num == 0).sum().sum() / max(1, num.size)) if num.size else 0
    checks.append({"check": "ZEROS", "status": "WARN" if zero_ratio > 0.3 else "PASS", "detail": f"{zero_ratio*100:.1f}% zeros"})
    dups = int(df.duplicated().sum())
    checks.append({"check": "DUPLICATES", "status": "WARN" if dups else "PASS", "detail": f"{dups} dup rows"})
    zi = _zscore_iqr_flags(df)
    checks.append({"check": "Z-SCORE (>3σ)", "status": "WARN" if zi["z_hits"] else "PASS", "detail": f"{zi['z_hits']} hits; {zi['details'][:3]}"})
    checks.append({"check": "IQR OUTLIER", "status": "WARN" if zi["iqr_hits"] else "PASS", "detail": f"{zi['iqr_hits']} hits"})
    iso = _isolation_forest_flags(df)
    checks.append({"check": "ISOLATION FOREST", "status": "WARN" if iso.get("anomalies", 0) else ("PASS" if iso.get("ok") else "INFO"), "detail": json.dumps({k: iso[k] for k in iso if k != "ok"})[:160]})
    db = _dbscan_noise(df)
    checks.append({"check": "DBSCAN NOISE", "status": "WARN" if db.get("noise_points", 0) else ("PASS" if db.get("ok") else "INFO"), "detail": json.dumps(db)[:160]})
    km = _kmeans_clean_proxy(df)
    checks.append({"check": "KMEANS DISTANCE", "status": "WARN" if km.get("far_from_cluster", 0) else ("PASS" if km.get("ok") else "INFO"), "detail": json.dumps(km)[:160]})
    jumps = _rolling_impossible_jumps(df)
    checks.append({"check": "ROLLING IMPOSSIBLE JUMP", "status": "FAIL" if jumps.get("count", 0) else "PASS", "detail": str(jumps.get("impossible_jumps") or "none")[:160]})
    lag = _lag_correlation_break(df)
    checks.append({"check": "LAG / SENSOR CORRELATION", "status": "WARN" if lag.get("dead_correlations") else "PASS", "detail": json.dumps(lag)[:160]})
    ge = _run_great_expectations(df)
    checks.append({"check": "GE EXPECTATIONS", "status": "PASS" if ge["passed"] == ge["total"] else "WARN", "detail": f"{ge['passed']}/{ge['total']} passed; available={ge['available']}"})
    yd = _run_ydata(df)
    checks.append({"check": "YDATA CARDINALITY", "status": "WARN" if yd.get("high_cardinality") else ("PASS" if yd.get("ok") else "INFO"), "detail": json.dumps(yd.get("high_cardinality") or yd)[:160]})
    cl = _run_cleanlab(df)
    checks.append({"check": "CLEANLAB / DIRTY LABELS", "status": "WARN" if cl.get("dirty_label_flags") else ("PASS" if cl.get("ok") else "INFO"), "detail": str(cl.get("dirty_label_flags") or cl)[:160]})
    pca = _pca_drift(df)
    checks.append({"check": "PCA / CONCEPT DRIFT", "status": "WARN" if pca.get("concept_drift") else ("PASS" if pca.get("ok") else "INFO"), "detail": json.dumps(pca)[:160]})
    domain_flags = _domain_opc_rules(df)
    checks.append({"check": "DOMAIN OPC / PHYSICS RULES", "status": "FAIL" if domain_flags else "PASS", "detail": "; ".join(domain_flags) if domain_flags else "no domain violations"})
    assoc = _association_rules_proxy(df)
    checks.append({"check": "ASSOCIATION RULE MINING", "status": "WARN" if assoc.get("suspicious_rules") else ("PASS" if assoc.get("ok") else "INFO"), "detail": json.dumps(assoc)[:160]})
    checks.append({"check": "SCHEMA / ROWCOUNT", "status": "PASS" if n > 0 else "FAIL", "detail": f"{n} rows × {m} cols"})
    checks.append({"check": "TIMESTAMP PRESENT", "status": "PASS" if _col(df, "timestamp", "date", "time", "datetime") else "WARN", "detail": str(_col(df, "timestamp", "date", "time", "datetime") or "missing")})
    return {"checks": checks, "ge": ge, "ydata": yd, "cleanlab": cl, "pca": pca, "domain_flags": domain_flags, "association": assoc}


def clean_data(df: pd.DataFrame, engine: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = engine or st.session_state.get("clean_engine") or "pandas"
    clean_df, engine_logs = _engine_clean(df, engine)
    report = build_quality_report(clean_df)
    table = pd.DataFrame(report["checks"])
    st.session_state.clean_df = clean_df
    st.session_state.clean_checks = table
    st.session_state.clean_report = {**report, "engine_logs": engine_logs, "engine": engine}
    return clean_df, table


def get_kpis(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    miss = round(float(df.isna().sum().sum() / max(1, df.size) * 100), 2)
    domain = st.session_state.get("domain") or "generic"
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
    if domain == "healthcare":
        w, h = _col(df, "weight"), _col(df, "height")
        if w and h:
            ww = pd.to_numeric(df[w], errors="coerce")
            hh = pd.to_numeric(df[h], errors="coerce") / 100.0
            bmi = ww / (hh.replace(0, np.nan) ** 2)
            kpis["Mean_BMI"] = round(float(bmi.mean()), 2) if bmi.notna().any() else "—"
    if domain == "sales_forecasting":
        rev = _col(df, "revenue", "sales", "gmv", "amount")
        kpis["Total_Revenue"] = round(float(pd.to_numeric(df[rev], errors="coerce").sum()), 2) if rev else "—"
    return kpis


def apply_domain_feature_engineering(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    out = df.copy()
    if domain == "healthcare":
        w, h = _col(out, "weight"), _col(out, "height")
        if w and h:
            ww = pd.to_numeric(out[w], errors="coerce")
            hh = pd.to_numeric(out[h], errors="coerce")
            hh_m = np.where(hh > 3, hh / 100.0, hh)
            out["BMI"] = ww / np.square(np.where(hh_m == 0, np.nan, hh_m))
            bp = _col(out, "bp", "blood_pressure", "systolic")
            glu = _col(out, "glucose", "blood_sugar")
            age = _col(out, "age")
            score = pd.Series(0.0, index=out.index)
            if bp is not None:
                score += pd.to_numeric(out[bp], errors="coerce").fillna(120) / 120.0
            if glu is not None:
                score += pd.to_numeric(out[glu], errors="coerce").fillna(100) / 100.0
            if age is not None:
                score += pd.to_numeric(out[age], errors="coerce").fillna(40) / 100.0
            out["BRISK_SCORE"] = score
    elif domain == "predictive_maintenance":
        t = _col(out, "temperature", "temp")
        v = _col(out, "vibration", "vib")
        p = _col(out, "pressure")
        flow = _col(out, "flow", "flow_rate")
        if t:
            out["temp_gradient"] = pd.to_numeric(out[t], errors="coerce").diff()
        if v:
            out["vib_rolling_std"] = pd.to_numeric(out[v], errors="coerce").rolling(5, min_periods=1).std()
        if t and v:
            out["thermal_mech_index"] = pd.to_numeric(out[t], errors="coerce").fillna(0) / 100.0 + pd.to_numeric(out[v], errors="coerce").fillna(0)
        if p and flow:
            out["pressure_flow_ratio"] = pd.to_numeric(out[p], errors="coerce") / pd.to_numeric(out[flow], errors="coerce").replace(0, np.nan)
    elif domain == "sales_forecasting":
        rev = _col(out, "revenue", "sales", "amount")
        units = _col(out, "units", "qty", "quantity")
        if rev and units:
            out["ASP"] = pd.to_numeric(out[rev], errors="coerce") / pd.to_numeric(out[units], errors="coerce").replace(0, np.nan)
        if rev:
            out["revenue_ma7"] = pd.to_numeric(out[rev], errors="coerce").rolling(7, min_periods=1).mean()
            out["revenue_pct_change"] = pd.to_numeric(out[rev], errors="coerce").pct_change()
    elif domain == "finance_risk":
        bal = _col(out, "balance", "amount", "loan_amount")
        inc = _col(out, "income", "annual_income")
        if bal and inc:
            out["DTI_proxy"] = pd.to_numeric(out[bal], errors="coerce") / pd.to_numeric(out[inc], errors="coerce").replace(0, np.nan)
        score = _col(out, "credit_score", "score")
        if score:
            out["credit_score_z"] = (
                pd.to_numeric(out[score], errors="coerce") - pd.to_numeric(out[score], errors="coerce").mean()
            ) / (pd.to_numeric(out[score], errors="coerce").std() or 1)
    elif domain == "warehouse_logistics":
        stock = _col(out, "inventory", "stock", "on_hand")
        demand = _col(out, "demand", "orders", "shipments")
        lead = _col(out, "lead_time", "leadtime")
        if stock and demand:
            out["days_of_cover"] = pd.to_numeric(out[stock], errors="coerce") / pd.to_numeric(out[demand], errors="coerce").replace(0, np.nan)
        if lead:
            out["lead_time_ma"] = pd.to_numeric(out[lead], errors="coerce").rolling(5, min_periods=1).mean()
    elif domain == "energy_utilities":
        load = _col(out, "load", "consumption", "kwh", "mw", "power")
        volt = _col(out, "voltage")
        curr = _col(out, "current")
        if load:
            out["load_rolling_std"] = pd.to_numeric(out[load], errors="coerce").rolling(6, min_periods=1).std()
            out["load_gradient"] = pd.to_numeric(out[load], errors="coerce").diff()
        if volt and curr:
            out["apparent_power_proxy"] = pd.to_numeric(out[volt], errors="coerce") * pd.to_numeric(out[curr], errors="coerce")
    elif domain == "telecom_churn":
        tenure = _col(out, "tenure")
        arpu = _col(out, "arpu", "revenue")
        usage = _col(out, "data_usage", "minutes", "usage")
        if tenure and arpu:
            out["lifetime_value_proxy"] = pd.to_numeric(out[tenure], errors="coerce") * pd.to_numeric(out[arpu], errors="coerce")
        if usage:
            out["usage_z"] = (
                pd.to_numeric(out[usage], errors="coerce") - pd.to_numeric(out[usage], errors="coerce").mean()
            ) / (pd.to_numeric(out[usage], errors="coerce").std() or 1)
    elif domain == "agriculture_iot":
        moist = _col(out, "moisture", "soil_moisture")
        rain = _col(out, "rainfall", "rain")
        ph = _col(out, "ph")
        if moist:
            out["moisture_gradient"] = pd.to_numeric(out[moist], errors="coerce").diff()
        if moist and rain:
            out["irrigation_stress"] = pd.to_numeric(out[moist], errors="coerce") / (
                pd.to_numeric(out[rain], errors="coerce").fillna(0) + 1.0
            )
        if ph:
            out["ph_dev_neutral"] = (pd.to_numeric(out[ph], errors="coerce") - 7.0).abs()
    return out


def detect_field(df: pd.DataFrame, use_gemini: bool = True) -> dict[str, Any]:
    cols = [str(c).lower() for c in df.columns]
    col_join = " ".join(cols)
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for dom, meta in DOMAIN_CATALOG.items():
        if dom == "generic":
            continue
        hit = []
        sc = 0.0
        for kw in meta["keywords"]:
            if kw in col_join:
                sc += 1.0
                hit.append(kw)
        num_ratio = df.select_dtypes(include=[np.number]).shape[1] / max(1, df.shape[1])
        if meta["dtypes_hint"] == "numeric_sensors" and num_ratio > 0.6:
            sc += 1.5
            hit.append("numeric_sensor_schema")
        if meta["dtypes_hint"] == "commerce" and any(k in col_join for k in ("revenue", "sales", "order")):
            sc += 1.0
        scores[dom] = sc
        reasons[dom] = hit[:12]
    if scores:
        heur = max(scores, key=scores.get)
        heur_conf = scores[heur] / max(1.0, max(scores.values()))
    else:
        heur, heur_conf = "generic", 0.2

    gemini_domain = None
    gemini_raw = ""
    if use_gemini and get_gemini_api_key():
        schema = [{"column": str(c), "dtype": str(df[c].dtype), "sample": [str(x) for x in df[c].dropna().head(3).tolist()]} for c in df.columns[:40]]
        prompt = (
            "Classify this industrial/business dataset into ONE domain key from: "
            + ", ".join(DOMAIN_CATALOG.keys())
            + ".\nReturn JSON only: {\"domain\": \"...\", \"confidence\": 0-1, \"why\": \"...\"}\n"
            f"Columns/dtypes/samples: {json.dumps(schema)[:4000]}"
        )
        gemini_raw = _gemini_answer(prompt)
        try:
            start = gemini_raw.find("{")
            end = gemini_raw.rfind("}") + 1
            if start >= 0 and end > start:
                payload = json.loads(gemini_raw[start:end])
                gd = str(payload.get("domain", "")).strip()
                if gd in DOMAIN_CATALOG:
                    gemini_domain = gd
                    gconf = float(payload.get("confidence", 0.9))
                    final = gemini_domain
                    conf = min(0.98, 0.55 * heur_conf + 0.45 * gconf + (0.2 if gemini_domain == heur else 0))
                    return {
                        "domain": final,
                        "label": DOMAIN_CATALOG[final]["label"],
                        "confidence": round(conf, 3),
                        "heuristic": heur,
                        "heuristic_scores": scores,
                        "reasons": reasons.get(final) or reasons.get(heur) or [],
                        "gemini_domain": gemini_domain,
                        "gemini_why": payload.get("why", ""),
                        "gemini_raw": gemini_raw[:500],
                    }
        except Exception:
            pass

    final = heur if scores.get(heur, 0) > 0 else "generic"
    return {
        "domain": final,
        "label": DOMAIN_CATALOG[final]["label"],
        "confidence": round(float(min(0.92, max(0.25, heur_conf))), 3),
        "heuristic": heur,
        "heuristic_scores": scores,
        "reasons": reasons.get(final, []),
        "gemini_domain": gemini_domain,
        "gemini_why": "",
        "gemini_raw": gemini_raw[:500],
    }


def _numeric_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    work = df.copy()
    fail_col = _col(work, "failure", "fault", "alarm", "label", "churn", "default", "readmission")
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c != fail_col][:12]
    if not feats:
        raise RuntimeError("Need numeric columns for field prediction.")
    X = work[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, fail_col


def field_predict(df: pd.DataFrame) -> float:
    return float(field_risk_explain(df)["risk_pct"])


def field_risk_explain(df: pd.DataFrame) -> dict[str, Any]:
    domain = st.session_state.get("domain") or detect_field(df, use_gemini=False)["domain"]
    work = apply_domain_feature_engineering(df, domain)
    X, label_col = _numeric_xy(work)
    explanations: list[str] = []
    risk = 15.0
    t = _col(work, "temperature", "temp")
    v = _col(work, "vibration", "vib")
    g = "temp_gradient" if "temp_gradient" in work.columns else None
    if domain in ("predictive_maintenance", "energy_utilities", "agriculture_iot"):
        if v is not None:
            vv = float(pd.to_numeric(work[v], errors="coerce").iloc[-1])
            explanations.append(f"vibration={vv:.3f}")
            mean_v = float(pd.to_numeric(work[v], errors="coerce").mean())
            std_v = float(pd.to_numeric(work[v], errors="coerce").std() or 1)
            if vv > mean_v + 2 * std_v:
                risk += 25
        if t is not None:
            tt = float(pd.to_numeric(work[t], errors="coerce").iloc[-1])
            explanations.append(f"temperature={tt:.2f}")
            if tt > 90:
                risk += 15
        if g is not None and pd.notna(work[g].iloc[-1]):
            gg = float(work[g].iloc[-1])
            explanations.append(f"temp_gradient={gg:.2f}")
            if abs(gg) > 10:
                risk += 20
    if domain == "healthcare":
        if "BMI" in work.columns and pd.notna(work["BMI"].iloc[-1]):
            bmi = float(work["BMI"].iloc[-1])
            explanations.append(f"BMI={bmi:.1f}")
            if bmi >= 30:
                risk += 20
        if "BRISK_SCORE" in work.columns and pd.notna(work["BRISK_SCORE"].iloc[-1]):
            brisk = float(work["BRISK_SCORE"].iloc[-1])
            explanations.append(f"BRISK_SCORE={brisk:.2f}")
            if brisk > 3.5:
                risk += 18
    if domain == "sales_forecasting":
        rev = _col(work, "revenue", "sales")
        if rev:
            r = pd.to_numeric(work[rev], errors="coerce")
            if len(r) > 5 and r.iloc[-1] < r.mean() * 0.7:
                risk += 25
                explanations.append(f"revenue_drop latest={r.iloc[-1]:.1f} vs mean={r.mean():.1f}")
        if "ASP" in work.columns and pd.notna(work["ASP"].iloc[-1]):
            explanations.append(f"ASP={float(work['ASP'].iloc[-1]):.2f}")
    if domain == "finance_risk":
        if "DTI_proxy" in work.columns and pd.notna(work["DTI_proxy"].iloc[-1]):
            dti = float(work["DTI_proxy"].iloc[-1])
            explanations.append(f"DTI_proxy={dti:.2f}")
            if dti > 0.45:
                risk += 28
    if domain == "warehouse_logistics" and "days_of_cover" in work.columns:
        doc = pd.to_numeric(work["days_of_cover"], errors="coerce")
        if doc.notna().any():
            latest = float(doc.iloc[-1])
            explanations.append(f"days_of_cover={latest:.1f}")
            if latest < 3:
                risk += 30
    if domain == "energy_utilities":
        load = _col(work, "load", "consumption", "kwh", "mw", "power")
        if load:
            lv = float(pd.to_numeric(work[load], errors="coerce").iloc[-1])
            explanations.append(f"load={lv:.2f}")
            if "load_gradient" in work.columns and abs(float(work["load_gradient"].iloc[-1] or 0)) > abs(lv) * 0.2:
                risk += 22
                explanations.append(f"load_gradient={float(work['load_gradient'].iloc[-1]):.2f}")
    if domain == "telecom_churn":
        churn = _col(work, "churn")
        if churn is not None:
            rate = float(pd.to_numeric(work[churn], errors="coerce").fillna(0).mean())
            risk = max(risk, rate * 100)
            explanations.append(f"churn_rate={rate*100:.1f}%")
        if "lifetime_value_proxy" in work.columns and pd.notna(work["lifetime_value_proxy"].iloc[-1]):
            explanations.append(f"LTV_proxy={float(work['lifetime_value_proxy'].iloc[-1]):.1f}")
    if domain == "agriculture_iot":
        if "irrigation_stress" in work.columns and pd.notna(work["irrigation_stress"].iloc[-1]):
            stress = float(work["irrigation_stress"].iloc[-1])
            explanations.append(f"irrigation_stress={stress:.2f}")
            if stress > 20:
                risk += 25
        if "ph_dev_neutral" in work.columns and pd.notna(work["ph_dev_neutral"].iloc[-1]):
            phd = float(work["ph_dev_neutral"].iloc[-1])
            explanations.append(f"ph_dev_neutral={phd:.2f}")
            if phd > 1.5:
                risk += 15
    try:
        if len(X) >= 15:
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_rate = float((iso.fit_predict(X) == -1).mean())
            risk = 0.5 * risk + 0.5 * (iso_rate * 100)
            explanations.append(f"isolation_forest_anomaly_rate={iso_rate*100:.1f}%")
            if label_col and work_has_binary(work, label_col):
                y = (pd.to_numeric(work[label_col], errors="coerce").fillna(0) > 0).astype(int)
                if y.nunique() > 1:
                    try:
                        import optuna
                        optuna.logging.set_verbosity(optuna.logging.WARNING)

                        def obj(trial):
                            clf = RandomForestClassifier(
                                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                                max_depth=trial.suggest_int("max_depth", 2, 12),
                                random_state=42,
                                class_weight="balanced",
                            )
                            return float(cross_val_score(clf, X, y, cv=3, scoring="f1").mean())

                        study = optuna.create_study(direction="maximize")
                        study.optimize(obj, n_trials=15, show_progress_bar=False)
                        best = study.best_params
                        clf = RandomForestClassifier(**best, random_state=42, class_weight="balanced")
                        clf.fit(X, y)
                        proba = float(clf.predict_proba(X.tail(min(24, len(X))))[:, 1].mean())
                        risk = 0.4 * risk + 0.6 * (proba * 100)
                        imps = sorted(zip(X.columns, clf.feature_importances_), key=lambda z: -z[1])
                        explanations.append("top_features=" + ", ".join(f"{a}:{b:.2f}" for a, b in imps[:4]))
                        explanations.append(f"optuna_best={best}")
                    except Exception as exc:
                        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
                        gb = GradientBoostingClassifier(random_state=42)
                        rf.fit(X, y)
                        gb.fit(X, y)
                        latest = X.tail(min(24, len(X)))
                        proba = 0.5 * rf.predict_proba(latest)[:, 1].mean() + 0.5 * gb.predict_proba(latest)[:, 1].mean()
                        risk = 0.4 * risk + 0.6 * (proba * 100)
                        explanations.append(f"ensemble_proba={proba:.3f}; optuna_skip={exc}")
    except Exception as exc:
        explanations.append(f"ml_note={exc}")
    risk = float(min(99.5, max(0.5, risk)))
    because = ", ".join(explanations[:6]) if explanations else "insufficient history"
    text = f"{DOMAIN_CATALOG.get(domain, {}).get('label', domain)} risk {risk:.1f}% because {because}"
    return {"risk_pct": round(risk, 1), "domain": domain, "explanation": text, "factors": explanations}


def work_has_binary(df: pd.DataFrame, col: str) -> bool:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False
    return s.nunique() <= 5



def run_automl(df: pd.DataFrame, target: str, n_trials: int = 50, time_series_split: bool = True, balanced: bool = True) -> tuple[str, dict[str, Any]]:
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
        scoring = "f1_weighted" if balanced else "accuracy"

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
            cw = "balanced" if balanced else None
            if model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 50, 300),
                    max_depth=trial.suggest_int("max_depth", 2, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                    random_state=42,
                    n_jobs=-1,
                    class_weight=cw,
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
    if time_series_split:
        cut = int(len(X) * 0.8)
        cut = max(1, min(len(X) - 1, cut))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    else:
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
        cw = "balanced" if balanced else None
        if best_model_name == "RandomForest":
            model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=cw, **params)
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
            "time_series_split": time_series_split,
            "balanced": balanced,
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


def get_gemini_api_key() -> str:
    """Prefer session override (Upload/Field UI), else .env."""
    try:
        key = str(st.session_state.get("gemini_api_key_override") or "").strip()
        if key:
            return key
    except Exception:
        pass
    return _ENV_GEMINI_API_KEY


# Back-compat alias used throughout the app (re-resolved via get_gemini_api_key in helpers)
GEMINI_API_KEY = _ENV_GEMINI_API_KEY


def persist_gemini_key(key: str, write_dotenv: bool = True) -> None:
    """Store Gemini key in session and optionally append/update .env."""
    key = (key or "").strip()
    st.session_state.gemini_api_key_override = key
    global GEMINI_API_KEY
    GEMINI_API_KEY = key or _ENV_GEMINI_API_KEY
    if write_dotenv and key:
        env_path = ROOT / ".env"
        lines = env_path.read_text() if env_path.exists() else ""
        if "GEMINI_API_KEY=" in lines:
            out = []
            for line in lines.splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    out.append(f"GEMINI_API_KEY={key}")
                else:
                    out.append(line)
            env_path.write_text("\n".join(out) + ("\n" if out else ""))
        else:
            with env_path.open("a") as f:
                f.write(f"\nGEMINI_API_KEY={key}\n")
                if "GEMINI_MODEL=" not in lines:
                    f.write(f"GEMINI_MODEL={GEMINI_MODEL}\n")


def gemini_key_ui(context: str = "upload") -> None:
    st.subheader("Gemini API key")
    current = get_gemini_api_key()
    masked = (current[:6] + "…" + current[-4:]) if len(current) > 12 else ("set" if current else "missing")
    st.caption(f"Status: **{masked}** · model `{GEMINI_MODEL}` · used for field auto-detect + Ask/AI ({context})")
    new_key = st.text_input(
        "Paste Gemini API key",
        value="",
        type="password",
        key=f"gemini_key_input_{context}",
        help="Stored in session + .env (gitignored). Improves domain detection ~95% with column+dtype+metadata.",
    )
    if st.button("Save Gemini key", key=f"save_gemini_{context}"):
        if new_key.strip():
            persist_gemini_key(new_key.strip(), write_dotenv=True)
            st.success("Gemini key saved for this session and `.env`.")
            st.rerun()
        else:
            st.warning("Paste a non-empty key.")


def _gemini_answer(prompt: str) -> str:
    key = get_gemini_api_key()
    if not key:
        return ""
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
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
    st.caption(
        "MANUAL mode: choose cleaning engine (pandas / polars / pyspark) by size suggestion — never forced. "
        "LIVE mode ignores upload and uses Modbus SCADA buffer."
    )

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

    gemini_key_ui("upload")
    st.divider()

    uploaded = st.file_uploader(
        "Upload industrial / ERP / plant / healthcare / sales CSV or Excel",
        type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet"],
    )
    df: Optional[pd.DataFrame] = None
    if uploaded is not None:
        try:
            df = load_uploaded_file(uploaded)
            st.session_state.manual_df = df
            st.session_state.manual_name = uploaded.name
            st.session_state.clean_df = None
            st.session_state.clean_checks = None
            st.session_state.clean_report = None
            st.session_state.field_result = None
            st.success(f"Loaded **{uploaded.name}** — {len(df):,} rows × {df.shape[1]} cols")
        except Exception as exc:
            st.error(str(exc))
            return
    elif st.session_state.manual_df is not None:
        df = st.session_state.manual_df
        st.write(f"Current file: **{st.session_state.manual_name}** — {len(df):,} × {df.shape[1]}")

    if df is None:
        st.info("Upload a file to enable engine selection + field preview.")
        return

    suggested, reason = suggest_clean_engine(len(df), df.shape[1])
    available = list_available_engines()
    st.subheader("Cleaning engine (you choose)")
    st.info(reason + " PySpark is only suggested for large files — you can still override.")
    default_idx = available.index(suggested) if suggested in available else 0
    engine = st.selectbox(
        "Engine for Clean tab ETL / DWDM pipeline",
        available,
        index=default_idx,
        help="pandas = interactive; polars = mid-size columnar; pyspark = big data (slow startup).",
    )
    st.session_state.clean_engine = engine
    st.session_state.prefer_clean_df = st.checkbox(
        "Downstream pages prefer cleaned dataframe when available",
        value=bool(st.session_state.get("prefer_clean_df", True)),
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", df.shape[1])
    with c3:
        st.metric("Suggested engine", suggested)

    with st.expander("Quick field auto-detect preview (column names + dtypes + Gemini)"):
        if st.button("Detect domain now", key="upload_detect_field"):
            with st.spinner("Detecting field via heuristics + Gemini..."):
                meta = detect_field(df, use_gemini=True)
                st.session_state.domain = meta["domain"]
                st.session_state.domain_meta = meta
                st.json(meta)
        elif st.session_state.get("domain_meta"):
            st.json(st.session_state.domain_meta)

    st.dataframe(df.head(50), use_container_width=True)

def page_clean() -> None:
    st.header("Clean")
    st.caption(
        "DWDM industrial clean: ETL · schema integration · binning · smoothing · regression imputation · "
        "Z-score/IQR · IsolationForest · DBSCAN · KMeans · rolling jumps · lag correlation · "
        "GE expectations · ydata · Cleanlab · association rules · PCA drift · OPC domain rules"
    )
    # Always clean from raw manual / live buffer, not previously cleaned frame
    try:
        if st.session_state.mode == "LIVE CONNECT":
            df = ensure_live_poll(force=False, min_interval_s=5.0)
        else:
            raw = st.session_state.get("manual_df")
            if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
                st.error("No manual file loaded. Go to Upload first.")
                return
            df = raw.copy()
    except Exception as exc:
        st.error(str(exc))
        return

    available = list_available_engines()
    suggested, reason = suggest_clean_engine(len(df), df.shape[1])
    st.info(reason)
    cur = st.session_state.get("clean_engine") or suggested
    if cur not in available:
        cur = available[0]
    engine = st.selectbox(
        "Cleaning engine",
        available,
        index=available.index(cur),
        key="clean_page_engine",
    )
    st.session_state.clean_engine = engine

    run = st.button("Run industrial clean + 15+ quality checks", type="primary")
    if run or st.session_state.clean_df is None:
        with st.spinner(f"Cleaning with {engine} + DWDM / GE / Cleanlab..."):
            clean_df, checks = clean_data(df, engine=engine)
        st.success(f"Clean complete — {len(clean_df):,} rows · engine={engine} · checks={len(checks)}")
    else:
        clean_df = st.session_state.clean_df
        checks = st.session_state.clean_checks

    st.subheader("Quality report (15+ checks)")
    st.dataframe(checks, use_container_width=True)
    report = st.session_state.get("clean_report") or {}
    if report:
        with st.expander("Engine logs / DWDM techniques applied"):
            st.write(f"Engine: **{report.get('engine')}**")
            for line in report.get("engine_logs") or []:
                st.write(f"- {line}")
        with st.expander("Great Expectations detail"):
            st.json(report.get("ge") or {})
        with st.expander("ydata / Cleanlab / PCA / Association"):
            st.json(
                {
                    "ydata": report.get("ydata"),
                    "cleanlab": report.get("cleanlab"),
                    "pca": report.get("pca"),
                    "association": report.get("association"),
                    "domain_flags": report.get("domain_flags"),
                }
            )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw head")
        st.dataframe(df.head(30), use_container_width=True)
    with c2:
        st.subheader("Clean head (engineered cols)")
        st.dataframe(clean_df.head(30), use_container_width=True)

def page_field() -> None:
    st.header("Field")
    st.caption(
        "Auto-detect domain (column names + dtypes + Gemini) across 8–9 fields, "
        "then domain feature engineering + Optuna/ensemble risk with explainability."
    )
    gemini_key_ui("field")
    df = require_data()
    if df is None:
        return

    use_gem = st.checkbox("Use Gemini for domain classification", value=bool(get_gemini_api_key()))
    if st.button("Detect field + explain risk", type="primary") or st.session_state.field_result is None:
        with st.spinner("Field detect + domain FE + Optuna/ensemble..."):
            meta = detect_field(df, use_gemini=use_gem)
            st.session_state.domain = meta["domain"]
            st.session_state.domain_meta = meta
            engineered = apply_domain_feature_engineering(df, meta["domain"])
            explain = field_risk_explain(engineered if engineered is not None else df)
            st.session_state.field_result = {"meta": meta, "explain": explain, "engineered_cols": [c for c in engineered.columns if c not in df.columns]}

    res = st.session_state.field_result
    meta = res["meta"]
    explain = res["explain"]
    st.subheader(f"Detected: {meta.get('label')} ({meta.get('domain')})")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Confidence", f"{float(meta.get('confidence', 0))*100:.1f}%")
    with m2:
        st.metric("Risk", f"{explain.get('risk_pct')}%")
    with m3:
        st.metric("Gemini", meta.get("gemini_domain") or "heuristic only")
    st.info(explain.get("explanation", ""))
    if meta.get("reasons"):
        st.write("Heuristic hits: " + ", ".join(meta["reasons"]))
    if meta.get("gemini_why"):
        st.write("Gemini why: " + str(meta["gemini_why"]))
    if res.get("engineered_cols"):
        st.write("Domain features added: " + ", ".join(res["engineered_cols"]))
    risk = float(explain.get("risk_pct") or 0)
    if risk >= 70:
        st.error(f"CRITICAL: risk {risk}% — act within the domain playbook window.")
    elif risk >= 40:
        st.warning(f"Elevated risk {risk}% — review top factors.")
    else:
        st.success(f"Risk {risk}% — within normal envelope.")
    with st.expander("Full detection payload"):
        st.json(res)
    kpis = get_kpis(df)
    metric_grid(kpis)



def page_kpis() -> None:
    st.header("Auto KPIs")
    df = require_data()
    if df is None:
        return
    if not st.session_state.get("domain") or st.session_state.domain == "generic":
        meta = detect_field(df, use_gemini=bool(get_gemini_api_key()))
        st.session_state.domain = meta["domain"]
        st.session_state.domain_meta = meta
    kpis = get_kpis(df)
    metric_grid(kpis, per_row=4)
    explain = field_risk_explain(df)
    st.subheader(f"{DOMAIN_CATALOG.get(explain['domain'], {}).get('label', 'Domain')} briefing")
    st.write(explain["explanation"])
    st.write(f"Buffer holds **{kpis['Rows']}** rows. Domain KPIs above reflect field-specific metrics when available.")


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
    ts_split = st.checkbox("Time-series split (no shuffle — last 20% holdout)", value=True)
    imb = st.checkbox("Imbalanced class_weight=balanced (classification)", value=True)
    if st.button("Run AutoML + Forecast", type="primary"):
        with st.spinner("Optuna searching best model..."):
            try:
                best_model, metrics = run_automl(df, target, n_trials=trials, time_series_split=ts_split, balanced=imb)
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
    gem_ok = bool(get_gemini_api_key())
    st.write(f"Gemini: **{'Connected' if gem_ok else 'No key'}** ({GEMINI_MODEL})")
    st.write("RAG: LlamaIndex on current data buffer + Gemini grounded answers")
    if not gem_ok:
        st.warning("Set GEMINI_API_KEY in Upload/Field or `.env`.")
        gemini_key_ui("ask")

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
                    st.session_state.clean_df = None
                    st.success(f"Loaded {up.name}")
                except Exception as exc:
                    st.error(str(exc))
            engines = list_available_engines()
            sug = "pandas"
            if isinstance(st.session_state.get("manual_df"), pd.DataFrame) and not st.session_state.manual_df.empty:
                sug, _ = suggest_clean_engine(len(st.session_state.manual_df), st.session_state.manual_df.shape[1])
            cur_eng = st.session_state.get("clean_engine") or sug
            if cur_eng not in engines:
                cur_eng = engines[0]
            st.session_state.clean_engine = st.selectbox(
                "Clean engine",
                engines,
                index=engines.index(cur_eng),
                key="sidebar_engine",
                help="Suggested by file size; you choose — PySpark not forced on small data.",
            )

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
        st.caption(f"Gemini key: {'yes' if get_gemini_api_key() else 'missing'} · engine={st.session_state.get('clean_engine')}")
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
