"""
Analytics Forge v2 — Production Dual-Mode Industrial OS
Single-file app: LIVE Modbus SCADA buffer + MANUAL upload, shared analytics core.
"""
from __future__ import annotations

import json
import re
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
from modules.data_integration import (
    JOIN_TYPES,
    join_many,
    join_two,
    load_tabular_file,
    suggest_join_keys,
)
from modules.dwdm_sql import (
    DWDM_CONCEPTS,
    apply_dwdm_transforms,
    default_sql_examples,
    run_sql,
)
from modules.dashboard_charts import (
    assemble_dashboard_export,
    make_readable_bar,
    render_core_charts,
    render_export_controls,
    render_extended_charts,
)
from modules.dwdm_labs import (
    apriori_need_txn_hint,
    assign_kmeans,
    baskets_from_txn,
    baskets_row_bins,
    build_star_schema,
    mice_impute,
    mine_apriori,
    numeric_columns as lab_numeric_columns,
)
from modules.domain_detect import APP_TO_OS_DOMAIN, OS_TO_APP_DOMAIN
from modules.forge_os import (
    autosave_after_pipeline,
    gemini_issue_from_raw,
    get_gemini_api_key,
    get_gemini_model,
    persist_gemini_key,
    render_detection_ui,
    render_domain_hints,
    render_domain_selector,
    render_dollar_impact,
    render_gemini_key_ui,
    render_industry_banner,
    render_manager_brief,
    render_mapping_ui,
    render_session_sidebar,
    reset_domain_pick_for_new_frame,
    show_gemini_issue,
)
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
_ENV_GEMINI_API_KEY = get_gemini_api_key()
GEMINI_API_KEY = _ENV_GEMINI_API_KEY
GEMINI_MODEL = get_gemini_model()
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
        "ml_result": None,
        "dashboard_charts": [],
        "dashboard_insights": [],
        "llama_docs": None,
        "llama_index_obj": None,
        "llama_index_meta": None,
        "selected_dash_kpis": None,
        "uploaded_tables": {},
        "join_log": None,
        "sql_lab_result": None,
        "sql_lab_engine": None,
        "sql_lab_query": None,
        "usd_per_hour": 0.0,
        "usd_per_unit": 0.0,
        "column_roles": {},
        "forge_domain": "generic",
        "column_types": {},
        "forge_detect": None,
        "forge_session_id": None,
        "forge_session_title": "",
        "last_gemini_error": "",
        "domain_user_override": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v




DOMAIN_CATALOG: dict[str, dict[str, Any]] = {
    "predictive_maintenance": {
        "label": "Predictive Maintenance / OPC-UA Sensors",
        "keywords": [
            "temperature", "temp", "vibration", "pressure", "rul", "failure", "machine",
            "sensor", "torque", "rpm", "opc", "modbus", "asset", "bearing", "fault",
        ],
        "exclusive": ["vibration", "rul", "torque", "bearing", "modbus", "opc"],
        "negative": ["patient", "bmi", "churn", "revenue", "sku", "loan", "crop", "arpu"],
        "dtypes_hint": "numeric_sensors",
        "value_hints": {"vibration": (0, 50), "temperature": (-40, 500), "rul": (0, 100000)},
    },
    "healthcare": {
        "label": "Healthcare / Hospital",
        "keywords": [
            "patient", "bmi", "bp", "blood", "glucose", "heart", "diagnosis",
            "admit", "ward", "doctor", "hospital", "readmission",
            "cholesterol", "pulse", "spo2", "systolic", "diastolic", "icd",
        ],
        "exclusive": ["patient", "bmi", "glucose", "readmission", "spo2", "cholesterol", "ward", "hospital", "icd"],
        "negative": [
            "vibration", "rul", "sku", "churn", "kwh", "modbus",
            "student", "gpa", "marks", "exam", "attendance", "assignment", "course",
        ],
        "dtypes_hint": "mixed_clinical",
        "value_hints": {"bmi": (10, 60), "glucose": (40, 600)},
    },
    "sales_forecasting": {
        "label": "Sales / Retail / Revenue",
        "keywords": [
            "revenue", "sales", "units", "order", "price", "sku", "customer", "region",
            "channel", "store", "campaign", "discount", "gmv", "asp", "product", "qty",
        ],
        "exclusive": ["revenue", "sales", "gmv", "sku", "discount", "campaign", "asp"],
        "negative": ["vibration", "patient", "churn", "soil", "modbus", "loan"],
        "dtypes_hint": "commerce",
        "value_hints": {},
    },
    "warehouse_logistics": {
        "label": "Warehouse / Supply Chain",
        "keywords": [
            "warehouse", "sku", "inventory", "stock", "shipment", "delivery", "carrier",
            "aisle", "bin", "lead_time", "defect", "pick", "pack", "supplier", "po",
        ],
        "exclusive": ["warehouse", "inventory", "shipment", "aisle", "lead_time", "pick"],
        "negative": ["patient", "vibration", "churn", "bmi", "glucose"],
        "dtypes_hint": "ops",
        "value_hints": {},
    },
    "energy_utilities": {
        "label": "Energy / Utilities",
        "keywords": [
            "kwh", "mw", "power", "grid", "load", "consumption", "solar", "wind",
            "frequency", "pf", "transformer", "feeder", "demand_mw",
        ],
        "exclusive": ["kwh", "mw", "solar", "wind", "grid", "feeder", "transformer"],
        "negative": ["patient", "churn", "bmi", "sku", "vibration", "rul"],
        "dtypes_hint": "numeric_sensors",
        "value_hints": {},
    },
    "finance_risk": {
        "label": "Finance / Credit Risk",
        "keywords": [
            "loan", "credit", "credit_score", "default", "interest", "balance", "emi", "income",
            "fraud", "transaction", "amount", "apr", "collateral", "delinquent",
        ],
        "exclusive": ["loan", "credit", "default", "emi", "apr", "fraud", "delinquent"],
        "negative": ["vibration", "patient", "soil", "modbus", "warehouse", "student", "gpa", "marks", "exam"],
        "dtypes_hint": "tabular_finance",
        "value_hints": {},
    },
    "telecom_churn": {
        "label": "Telecom / Churn",
        "keywords": [
            "churn", "tenure", "plan", "minutes", "data_usage", "arpu", "subscriber",
            "complaint", "call_drop", "sim", "contract", "monthly_charges",
        ],
        "exclusive": ["churn", "arpu", "tenure", "call_drop", "sim", "subscriber"],
        "negative": ["vibration", "patient", "bmi", "soil", "rul", "warehouse"],
        "dtypes_hint": "crm",
        "value_hints": {"tenure": (0, 120), "arpu": (0, 5000)},
    },
    "agriculture_iot": {
        "label": "Agriculture / Agri-IoT",
        "keywords": [
            "soil", "moisture", "humidity", "rainfall", "crop", "yield", "ph", "npk",
            "irrigation", "farm", "nitrogen", "pesticide",
        ],
        "exclusive": ["soil", "crop", "yield", "npk", "irrigation", "farm"],
        "negative": ["patient", "churn", "loan", "sku", "modbus", "vibration"],
        "dtypes_hint": "numeric_sensors",
        "value_hints": {"ph": (3, 10), "moisture": (0, 100)},
    },
    "hr_people": {
        "label": "HR / People Analytics",
        "keywords": [
            "employee", "salary", "department", "attrition", "hire", "performance",
            "manager", "job", "satisfaction", "overtime", "hr", "headcount",
        ],
        "exclusive": ["employee", "attrition", "salary", "department", "headcount"],
        "negative": ["vibration", "patient", "soil", "modbus", "kwh", "student", "gpa"],
        "dtypes_hint": "hrm",
        "value_hints": {},
    },
    "education": {
        "label": "Education / Student",
        "keywords": [
            "student", "grade", "gpa", "marks", "exam", "course", "assignment",
            "attendance", "school", "university", "subject", "credits", "cgpa",
            "math_score", "reading_score", "writing_score",
        ],
        "exclusive": ["student", "gpa", "marks", "exam", "attendance", "assignment", "math_score", "reading_score", "writing_score"],
        "negative": ["patient", "hospital", "bmi", "glucose", "vibration", "rul", "oee", "modbus"],
        "dtypes_hint": "education",
        "value_hints": {"gpa": (0, 10), "attendance": (0, 100)},
    },
    "generic": {
        "label": "Generic Analytics",
        "keywords": [],
        "exclusive": [],
        "negative": [],
        "dtypes_hint": "generic",
        "value_hints": {},
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


def load_live_config() -> dict[str, Any]:
    """Load LIVE_MODE from config.yaml with .env overrides (+ session override from LIVE UI)."""
    if st.session_state.get("live_cfg_override"):
        return dict(st.session_state.live_cfg_override)
    cfg: dict[str, Any] = {
        "connection_type": "direct",
        "ocp_u_ip": MODBUS_HOST,
        "ocp_u_port": MODBUS_PORT,
        "unit_id": MODBUS_UNIT,
        "poll_interval_s": 5,
        "modbus_timeout_s": 2.5,
        "fastapi_url": "http://127.0.0.1:8088/live",
        "fastapi_health_url": "http://127.0.0.1:8088/health",
        "buffer_path": str(LIVE_CSV.relative_to(ROOT)) if LIVE_CSV.is_relative_to(ROOT) else "data/live.csv",
        "max_buffer_rows": 50000,
        "default_insight_engine": "prophet",
        "auto_dashboard": True,
        "registers": {
            name: {"address": i, "scale": _default_scale(name)}
            for i, name in enumerate(REGISTER_NAMES)
        },
    }
    conf_path = ROOT / "config.yaml"
    if conf_path.exists():
        try:
            import yaml
            raw = yaml.safe_load(conf_path.read_text(encoding="utf-8")) or {}
            live = dict(raw.get("LIVE_MODE") or {})
            cfg.update({k: v for k, v in live.items() if v is not None})
        except Exception as exc:
            st.session_state.live_config_warning = f"config.yaml parse warning: {exc}"
    # env wins for plant deploy
    if os.getenv("LIVE_CONNECTION_TYPE"):
        cfg["connection_type"] = os.getenv("LIVE_CONNECTION_TYPE", "direct").strip().lower()
    if os.getenv("OCP_U_IP") or os.getenv("MODBUS_HOST"):
        cfg["ocp_u_ip"] = os.getenv("OCP_U_IP") or os.getenv("MODBUS_HOST") or cfg["ocp_u_ip"]
    if os.getenv("OCP_U_PORT") or os.getenv("MODBUS_PORT"):
        cfg["ocp_u_port"] = int(os.getenv("OCP_U_PORT") or os.getenv("MODBUS_PORT") or cfg["ocp_u_port"])
    if os.getenv("FASTAPI_LIVE_URL"):
        cfg["fastapi_url"] = os.getenv("FASTAPI_LIVE_URL")
    cfg["connection_type"] = str(cfg.get("connection_type") or "direct").strip().lower()
    return cfg


def _default_scale(name: str) -> float:
    return {
        "temperature": 0.1,
        "vibration": 0.001,
        "pressure": 0.1,
        "current": 0.01,
        "smps_current": 0.01,
        "voltage": 0.1,
        "smps_voltage": 0.1,
        "torque": 0.1,
        "load": 0.1,
    }.get(name, 1.0)


def live_buffer_path(cfg: Optional[dict[str, Any]] = None) -> Path:
    cfg = cfg or load_live_config()
    p = Path(str(cfg.get("buffer_path") or "data/live.csv"))
    return p if p.is_absolute() else (ROOT / p)


def read_via_pymodbus(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """OPTION A — Direct: Streamlit/gateway → pymodbus → OCP-U."""
    cfg = cfg or load_live_config()
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as exc:
        raise RuntimeError("pymodbus not installed. pip install pymodbus") from exc

    host = str(cfg.get("ocp_u_ip") or MODBUS_HOST)
    port = int(cfg.get("ocp_u_port") or MODBUS_PORT)
    unit = int(cfg.get("unit_id") or MODBUS_UNIT)
    timeout = float(cfg.get("modbus_timeout_s") or 2.5)
    regs_map: dict[str, Any] = cfg.get("registers") or {}
    if not regs_map:
        raise RuntimeError("No register map in config.yaml LIVE_MODE.registers")

    addresses = [int(m["address"]) for m in regs_map.values()]
    start = min(addresses)
    count = max(addresses) - start + 1

    client = ModbusTcpClient(host=host, port=port, timeout=timeout)
    try:
        if not client.connect():
            raise RuntimeError(
                f"Cannot connect to OCP-U Modbus TCP {host}:{port}. "
                "Check Ethernet/VPN, IP, and that the PLC is online."
            )
        try:
            result = client.read_holding_registers(address=start, count=count, device_id=unit)
        except TypeError:
            try:
                result = client.read_holding_registers(address=start, count=count, slave=unit)
            except TypeError:
                result = client.read_holding_registers(start, count, unit)
        if result is None or (hasattr(result, "isError") and result.isError()):
            raise RuntimeError(f"Modbus read error from {host}:{port}: {result}")
        raw = list(result.registers)
        if len(raw) < count:
            raise RuntimeError(f"Expected {count} registers, got {len(raw)}")

        row: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source": "direct_pymodbus",
            "ocp_u": f"{host}:{port}",
        }
        for name, meta in regs_map.items():
            addr = int(meta["address"])
            scale = float(meta.get("scale", 1.0))
            val = float(raw[addr - start]) * scale
            if name == "failure":
                val = 1.0 if val > 0 else 0.0
            row[name] = round(val, 6) if name != "failure" else val
        return row
    finally:
        try:
            client.close()
        except Exception:
            pass


def read_via_fastapi(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """OPTION B — Production: Streamlit → FastAPI gateway → pymodbus → OCP-U."""
    cfg = cfg or load_live_config()
    url = str(cfg.get("fastapi_url") or "").strip()
    if not url:
        raise RuntimeError("fastapi_url missing in config.yaml")
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests not installed. pip install requests") from exc
    try:
        resp = requests.get(url, timeout=float(cfg.get("modbus_timeout_s") or 3) + 2)
    except Exception as exc:
        raise RuntimeError(
            f"FastAPI gateway unreachable at {url}: {exc}. "
            "Start gateway: uvicorn gateway:app --port 8088"
        ) from exc
    if resp.status_code >= 400:
        raise RuntimeError(f"Gateway error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Gateway did not return a JSON object")
    data.setdefault("source", "fastapi_gateway")
    data.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    return data


def fetch_live_row(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Single pipe entry: routes by connection_type."""
    cfg = cfg or load_live_config()
    ctype = str(cfg.get("connection_type") or "direct").lower()
    if ctype == "buffer_only":
        raise RuntimeError(
            "connection_type=buffer_only — Streamlit will not poll plant. "
            "Run gateway.py on the Pi to fill data/live.csv."
        )
    if ctype == "fastapi":
        return read_via_fastapi(cfg)
    return read_via_pymodbus(cfg)


def append_live_csv(row: dict[str, Any], path: Optional[Path] = None) -> pd.DataFrame:
    """Append one SCADA poll row to buffer and return full frame."""
    cfg = load_live_config()
    path = path or live_buffer_path(cfg)
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
    max_rows = int(cfg.get("max_buffer_rows") or 50000)
    if len(out) > max_rows:
        out = out.tail(max_rows).reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def read_live_buffer(path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    path = path or live_buffer_path()
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as exc:
        st.session_state.live_error = f"Buffer read failed: {exc}"
        return None


def ensure_live_poll(force: bool = False, min_interval_s: Optional[float] = None) -> pd.DataFrame:
    """
    LIVE SCADA path — buffer-first industrial pattern:
    1) Optionally poll plant (direct/fastapi) every N seconds
    2) Persist to data/live.csv
    3) On plant failure: serve last good buffer + surface error (never silent fake data)
    """
    cfg = load_live_config()
    interval = float(min_interval_s if min_interval_s is not None else cfg.get("poll_interval_s") or 5)
    ctype = str(cfg.get("connection_type") or "direct").lower()
    now = time.time()
    last = float(st.session_state.get("live_last_poll") or 0.0)
    need = force or (now - last >= interval)

    if ctype == "buffer_only":
        buf = read_live_buffer()
        if buf is not None and len(buf):
            st.session_state.live_status = "buffer_only"
            st.session_state.live_error = None
            return buf
        raise RuntimeError(
            "buffer_only mode: data/live.csv is empty. "
            "Start gateway.py near OCP-U so it writes the buffer."
        )

    if need:
        try:
            row = fetch_live_row(cfg)
            # fastapi gateway may already persist — still mirror locally for UI
            df = append_live_csv(row)
            st.session_state.live_last_poll = now
            st.session_state.live_status = f"connected:{ctype}"
            st.session_state.live_error = None
            st.session_state.live_last_row = row
            return df
        except Exception as exc:
            st.session_state.live_status = "error"
            st.session_state.live_error = str(exc)
            buf = read_live_buffer()
            if buf is not None and len(buf):
                return buf
            raise

    buf = read_live_buffer()
    if buf is not None and len(buf):
        return buf
    # first poll required
    row = fetch_live_row(cfg)
    df = append_live_csv(row)
    st.session_state.live_last_poll = time.time()
    st.session_state.live_status = f"connected:{ctype}"
    st.session_state.live_error = None
    st.session_state.live_last_row = row
    return df


def live_latest_metrics(df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    df = df if df is not None else read_live_buffer()
    if df is None or df.empty:
        return {}
    row = df.iloc[-1]
    keys = [
        "temperature", "vibration", "pressure", "smps_voltage", "smps_current",
        "voltage", "current", "speed", "torque", "rul", "failure", "load",
    ]
    out = {}
    for k in keys:
        if k in df.columns and pd.notna(row[k]):
            try:
                out[k] = float(row[k])
            except Exception:
                out[k] = row[k]
    if "timestamp" in df.columns:
        out["timestamp"] = str(row["timestamp"])
    return out


def live_auto_insights(df: pd.DataFrame, engine: str = "prophet") -> list[str]:
    """Right-rail insights for LIVE SCADA (domain-aware, buffer slice)."""
    lines: list[str] = []
    if df is None or df.empty:
        return ["No live buffer yet — poll OCP-U or start gateway."]
    metrics = live_latest_metrics(df)
    lines.append(f"Buffer **{len(df):,}** rows · latest `{metrics.get('timestamp', '—')}`")
    if "vibration" in metrics:
        vib = float(metrics["vibration"])
        mean_v = float(pd.to_numeric(df["vibration"], errors="coerce").mean()) if "vibration" in df.columns else vib
        if vib > mean_v * 1.5 + 0.1:
            lines.append(f"🚨 Vibration spike **{vib:.3f}** (mean {mean_v:.3f}) — inspect bearings / SMPS coupling.")
        else:
            lines.append(f"Vibration **{vib:.3f}** within recent envelope.")
    if "temperature" in metrics:
        lines.append(f"Temperature **{metrics['temperature']:.2f}°C**")
    if "smps_voltage" in metrics or "voltage" in metrics:
        v = metrics.get("smps_voltage", metrics.get("voltage"))
        lines.append(f"SMPS / supply voltage **{v}**")
    if "failure" in metrics and float(metrics["failure"]) >= 1:
        lines.append("🚨 Failure flag **1** on latest packet — open maintenance ticket.")

    engine = (engine or "prophet").lower()
    target = _col(df, "rul", "temperature", "vibration", "load")
    try:
        if engine == "prophet" and target:
            lines.append(prophet_forecast(df.tail(min(500, len(df))), target))
        elif engine == "pyspark" and target:
            # lightweight spark summary — fallback pandas if spark heavy
            try:
                from pyspark.sql import SparkSession
                spark = (
                    SparkSession.builder.master("local[1]")
                    .appName("forge_live_insight")
                    .config("spark.ui.enabled", "false")
                    .getOrCreate()
                )
                pdf = df[[target]].dropna().tail(200)
                sdf = spark.createDataFrame(pdf.astype(float))
                stats = sdf.describe(target).toPandas()
                lines.append(f"PySpark describe `{target}`:\n{stats.to_string(index=False)}")
                spark.stop()
            except Exception as exc:
                s = pd.to_numeric(df[target], errors="coerce")
                lines.append(f"PySpark unavailable ({exc}); pandas mean={s.mean():.3f} std={s.std():.3f}")
        else:
            if target:
                s = pd.to_numeric(df[target], errors="coerce").dropna()
                if len(s) >= 5:
                    slope = float(np.polyfit(np.arange(len(s.tail(50))), s.tail(50).to_numpy(), 1)[0])
                    lines.append(f"Pandas trend `{target}` slope={slope:.4f} (last 50).")
    except Exception as exc:
        lines.append(f"Insight engine note: {exc}")
    return lines


# backward-compat aliases used elsewhere
def poll_modbus_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
    cfg = load_live_config()
    if args or kwargs:
        if args:
            cfg["ocp_u_ip"] = args[0]
        if len(args) > 1:
            cfg["ocp_u_port"] = args[1]
        cfg["ocp_u_ip"] = kwargs.get("host", cfg["ocp_u_ip"])
        cfg["ocp_u_port"] = kwargs.get("port", cfg["ocp_u_port"])
        cfg["unit_id"] = kwargs.get("unit", cfg.get("unit_id"))
    return read_via_pymodbus(cfg)


def _scale_register(name: str, raw: int) -> float:
    """Legacy helper — prefer config.yaml scales."""
    return round(float(raw) * _default_scale(name), 6)



def get_data() -> pd.DataFrame:
    """
    Dual-mode data switch used by EVERY page.
    LIVE CONNECT -> real Modbus SCADA buffer (data/live.csv)
    MANUAL UPLOAD -> uploaded dataframe in session (prefer clean_df when set)
    """
    mode = st.session_state.get("mode", "MANUAL UPLOAD")
    if mode == "LIVE CONNECT":
        try:
            cfg = load_live_config()
            interval = float(cfg.get("poll_interval_s") or 5)
            return ensure_live_poll(force=False, min_interval_s=interval)
        except Exception as exc:
            st.session_state.live_status = "error"
            st.session_state.live_error = str(exc)
            buf = read_live_buffer()
            if buf is not None and len(buf):
                return buf
            raise RuntimeError(
                f"LIVE unavailable: {exc}. Fix OCP-U/FastAPI connection or switch to MANUAL UPLOAD."
            ) from exc

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


def _is_nonempty_frame(obj: Any) -> bool:
    """True only for a real DataFrame with rows — never `if df` / `df or ...`."""
    return isinstance(obj, pd.DataFrame) and not obj.empty


def join_table_registry() -> dict[str, pd.DataFrame]:
    """Named tables for SQL joins. LIVE buffer is listed but not swapped into get_data()."""
    tables: dict[str, pd.DataFrame] = {}
    stored = st.session_state.get("uploaded_tables") or {}
    if isinstance(stored, dict):
        for name, df in stored.items():
            if _is_nonempty_frame(df) and name not in ("joined", "_result"):
                tables[str(name)] = df
    manual = st.session_state.get("manual_df")
    if _is_nonempty_frame(manual):
        tables.setdefault("manual_df", manual)
    clean = st.session_state.get("clean_df")
    if _is_nonempty_frame(clean):
        tables.setdefault("clean_df", clean)
    if st.session_state.get("mode") == "LIVE CONNECT":
        buf = read_live_buffer()
        if _is_nonempty_frame(buf):
            tables.setdefault("live_buffer", buf)
    return tables


def dashboard_source_frame(fallback: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prefer cleaned / joined / warehouse session table for Dashboard charts."""
    tables = st.session_state.get("uploaded_tables") or {}
    joined = tables.get("joined") if isinstance(tables, dict) else None
    clean = st.session_state.get("clean_df")
    sql = st.session_state.get("sql_lab_result")
    if _is_nonempty_frame(clean):
        return clean.copy(), "cleaned / DWDM"
    if _is_nonempty_frame(joined):
        return joined.copy(), "joined / warehouse"
    if _is_nonempty_frame(sql):
        return sql.copy(), "SQL result"
    return fallback, "working"


def apply_joined_as_working(merged: pd.DataFrame, tables: dict[str, pd.DataFrame], logs: Any) -> None:
    """
    Join result becomes MANUAL working data via get_data() (clean_df + prefer_clean_df).
    LIVE CONNECT still reads the SCADA buffer — modes stay isolated.
    """
    st.session_state.clean_df = merged
    st.session_state.prefer_clean_df = True
    st.session_state.join_log = logs
    registry = dict(tables)
    registry["joined"] = merged
    st.session_state.uploaded_tables = registry


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
    try:
        autosave_after_pipeline(title=f"Clean · {st.session_state.get('manual_name') or 'session'}")
    except Exception:
        pass
    return clean_df, table



def _token_set(cols: list[str]) -> set[str]:
    toks: set[str] = set()
    for c in cols:
        for part in re.split(r"[^a-z0-9]+", str(c).lower()):
            if part:
                toks.add(part)
                toks.add(str(c).lower())
    return toks


def _heuristic_field_scores(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, list[str]], pd.DataFrame]:
    cols = [str(c).lower() for c in df.columns]
    col_join = " ".join(cols)
    toks = _token_set(cols)
    num_ratio = df.select_dtypes(include=[np.number]).shape[1] / max(1, df.shape[1])
    rows = []
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for dom, meta in DOMAIN_CATALOG.items():
        if dom == "generic":
            continue
        hit: list[str] = []
        sc = 0.0
        kw = meta.get("keywords") or []
        for k in kw:
            if k in toks or k in col_join:
                sc += 1.2
                hit.append(k)
        for k in meta.get("exclusive") or []:
            if k in toks or k in col_join:
                sc += 3.5
                hit.append(f"EXCL:{k}")
        for k in meta.get("negative") or []:
            if k in toks or k in col_join:
                sc -= 2.5
                hit.append(f"NEG:{k}")
        # dtype / schema hints
        if meta.get("dtypes_hint") == "numeric_sensors" and num_ratio > 0.55:
            sc += 0.8
        if meta.get("dtypes_hint") == "commerce" and any(k in toks for k in ("revenue", "sales", "order", "gmv")):
            sc += 1.5
        if meta.get("dtypes_hint") == "mixed_clinical" and any(k in toks for k in ("patient", "bmi", "glucose", "hospital")):
            sc += 1.5
        if meta.get("dtypes_hint") == "crm" and any(k in toks for k in ("churn", "arpu", "tenure")):
            sc += 1.5
        if meta.get("dtypes_hint") == "education" and any(k in toks for k in ("student", "gpa", "marks", "exam", "attendance")):
            sc += 1.5
        if dom == "healthcare" and not any(str(h).startswith("EXCL:") for h in hit):
            sc = min(sc, 0.8)
            hit.append("weak-only")
        # value-range fingerprints
        for hint_col, (lo, hi) in (meta.get("value_hints") or {}).items():
            real = _col(df, hint_col)
            if real:
                s = pd.to_numeric(df[real], errors="coerce").dropna()
                if len(s) >= 5:
                    m = float(s.median())
                    if lo <= m <= hi:
                        sc += 1.8
                        hit.append(f"RANGE:{hint_col}~{m:.2f}")
                    else:
                        sc -= 0.8
        scores[dom] = max(0.0, sc)
        reasons[dom] = hit[:14]
        rows.append({"domain": dom, "label": meta["label"], "heuristic_score": round(scores[dom], 3), "hits": ", ".join(hit[:8])})
    scoreboard = pd.DataFrame(rows).sort_values("heuristic_score", ascending=False).reset_index(drop=True)
    return scores, reasons, scoreboard


def _schema_feature_vector(df: pd.DataFrame) -> np.ndarray:
    """Numeric feature vector used by Optuna-tuned domain classifier."""
    cols = [str(c).lower() for c in df.columns]
    col_join = " ".join(cols)
    toks = _token_set(cols)
    feats: list[float] = []
    domain_keys = [d for d in DOMAIN_CATALOG if d != "generic"]
    for dom in domain_keys:
        meta = DOMAIN_CATALOG[dom]
        kw_hits = sum(1 for k in meta["keywords"] if k in toks or k in col_join)
        ex_hits = sum(1 for k in meta.get("exclusive", []) if k in toks or k in col_join)
        neg_hits = sum(1 for k in meta.get("negative", []) if k in toks or k in col_join)
        feats.extend([kw_hits, ex_hits, neg_hits, kw_hits / max(1, len(meta["keywords"]))])
    num_ratio = df.select_dtypes(include=[np.number]).shape[1] / max(1, df.shape[1])
    cat_ratio = df.select_dtypes(include=["object", "category"]).shape[1] / max(1, df.shape[1])
    has_dt = 1.0 if _col(df, "timestamp", "date", "datetime", "time") else 0.0
    feats.extend([num_ratio, cat_ratio, has_dt, float(df.shape[0]), float(df.shape[1])])
    return np.asarray(feats, dtype=float)


def _synthetic_domain_training_set(n_per: int = 12) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Tiny fingerprint matrix (column-name features), not the full user df."""
    """Build synthetic schema vectors so Optuna can tune a domain classifier without labels."""
    rng = np.random.default_rng(42)
    domain_keys = [d for d in DOMAIN_CATALOG if d != "generic"]
    X_rows: list[np.ndarray] = []
    y: list[str] = []
    for dom in domain_keys:
        meta = DOMAIN_CATALOG[dom]
        for _ in range(n_per):
            # fake dataframe with domain-ish columns
            chosen = list(rng.choice(meta["keywords"], size=min(8, len(meta["keywords"])), replace=False))
            for ex in meta.get("exclusive", [])[:3]:
                if ex not in chosen:
                    chosen.append(ex)
            # occasional noise columns from other domains
            other = [d for d in domain_keys if d != dom]
            noise_dom = rng.choice(other)
            noise = list(rng.choice(DOMAIN_CATALOG[noise_dom]["keywords"], size=2, replace=False))
            cols = chosen + noise + ["id", "notes"]
            data = {}
            for c in cols:
                if c in ("id", "notes"):
                    data[c] = [f"{c}{i}" for i in range(30)]
                else:
                    data[c] = rng.normal(50, 10, 30)
            fake = pd.DataFrame(data)
            X_rows.append(_schema_feature_vector(fake))
            y.append(dom)
    return np.vstack(X_rows), np.asarray(y), domain_keys


_OPTUNA_FIELD_CACHE: dict[int, dict[str, Any]] = {}
FIELD_DETECT_DEFAULT_TRIALS = 3
FIELD_DETECT_MAX_TRIALS = 40
FIELD_DETECT_HIGH_CONF = 0.72


def _fit_optuna_field_model(n_trials: int = 3) -> dict[str, Any]:
    """Optuna-tunes a small RF/GBM on schema fingerprints. Cached per n_trials."""
    n_trials = max(1, min(FIELD_DETECT_MAX_TRIALS, int(n_trials or FIELD_DETECT_DEFAULT_TRIALS)))
    hit = _OPTUNA_FIELD_CACHE.get(n_trials)
    if hit and hit.get("ok"):
        return hit
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    X, y, domains = _synthetic_domain_training_set(12)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    def objective(trial: Any) -> float:
        model_name = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting"])
        if model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 20, 50),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 4),
                random_state=42,
                n_jobs=-1,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=trial.suggest_int("n_estimators", 20, 40),
                max_depth=trial.suggest_int("max_depth", 2, 4),
                learning_rate=trial.suggest_float("learning_rate", 0.05, 0.25, log=True),
                random_state=42,
            )
        scores = cross_val_score(model, X, y_enc, cv=2, scoring="accuracy")
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_params
    model_name = best.pop("model")
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **best)
    else:
        model = GradientBoostingClassifier(random_state=42, **best)
    model.fit(X, y_enc)
    pack = {
        "ok": True,
        "model": model,
        "label_encoder": le,
        "domains": domains,
        "best_params": {"model": model_name, **best},
        "cv_accuracy": round(float(study.best_value), 4),
        "n_trials": n_trials,
    }
    _OPTUNA_FIELD_CACHE[n_trials] = pack
    return pack


def _optuna_predict_field(df: pd.DataFrame, n_trials: int = 3) -> dict[str, Any]:
    pack = _fit_optuna_field_model(n_trials=n_trials)
    if not pack.get("ok"):
        return {"ok": False, "error": pack.get("error", "optuna field model failed")}
    vec = _schema_feature_vector(df).reshape(1, -1)
    model = pack["model"]
    le: LabelEncoder = pack["label_encoder"]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        order = np.argsort(-proba)
        top_i = int(order[0])
        ranking = [
            {"domain": le.classes_[i], "prob": round(float(proba[i]), 4)}
            for i in order[:5]
        ]
        return {
            "ok": True,
            "domain": str(le.classes_[top_i]),
            "confidence": round(float(proba[top_i]), 4),
            "ranking": ranking,
            "best_params": pack["best_params"],
            "cv_accuracy": pack["cv_accuracy"],
            "proba_table": pd.DataFrame(ranking),
        }
    pred = model.predict(vec)[0]
    dom = str(le.inverse_transform([pred])[0])
    return {"ok": True, "domain": dom, "confidence": 0.7, "ranking": [{"domain": dom, "prob": 0.7}], "best_params": pack["best_params"], "cv_accuracy": pack["cv_accuracy"], "proba_table": pd.DataFrame([{"domain": dom, "prob": 0.7}])}


def _field_col_signature(df: pd.DataFrame) -> str:
    return f"{len(df)}|{df.shape[1]}|{','.join(str(c) for c in df.columns)}"


def _lock_field_to_user_override(result: dict[str, Any]) -> dict[str, Any]:
    if not st.session_state.get("domain_user_override"):
        return result
    app_dom = str(st.session_state.get("domain") or "generic")
    if app_dom not in DOMAIN_CATALOG:
        app_dom = "generic"
    out = dict(result)
    out["detected_domain"] = result.get("domain")
    out["domain"] = app_dom
    out["label"] = DOMAIN_CATALOG.get(app_dom, {}).get("label", app_dom)
    out["overridden"] = True
    out["forge_domain"] = st.session_state.get("forge_domain")
    return out


def apply_detected_domain(meta: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Write Field/app domain unless the user already overrode on Upload or Field."""
    meta = _lock_field_to_user_override(meta)
    if st.session_state.get("domain_user_override") and not force:
        return meta
    st.session_state.domain = meta.get("domain") or "generic"
    st.session_state.domain_meta = meta
    os_key = APP_TO_OS_DOMAIN.get(str(meta.get("domain") or ""), st.session_state.get("forge_domain") or "generic")
    st.session_state.forge_domain = os_key
    return meta


def detect_field(df: pd.DataFrame, use_gemini: bool = True, optuna_trials: int = 3) -> dict[str, Any]:
    """Heuristic O(columns) first; Optuna on a tiny fingerprint matrix; Gemini optional."""
    optuna_trials = max(1, min(FIELD_DETECT_MAX_TRIALS, int(optuna_trials or FIELD_DETECT_DEFAULT_TRIALS)))
    sig = f"{_field_col_signature(df)}|g{int(bool(use_gemini))}|t{optuna_trials}"
    cached = st.session_state.get("_field_detect_cache")
    if isinstance(cached, dict) and cached.get("sig") == sig and isinstance(cached.get("result"), dict):
        return _lock_field_to_user_override(cached["result"])

    scores, reasons, scoreboard = _heuristic_field_scores(df)
    if scores:
        heur = max(scores, key=scores.get)
        heur_conf = scores[heur] / max(1.0, max(scores.values()) or 1.0)
    else:
        heur, heur_conf = "generic", 0.2

    opt = _optuna_predict_field(df, n_trials=optuna_trials)
    opt_domain = opt.get("domain") if opt.get("ok") else None
    opt_conf = float(opt.get("confidence") or 0.0) if opt.get("ok") else 0.0

    gemini_domain = None
    gemini_raw = ""
    gemini_why = ""
    gemini_error = None
    gconf = 0.0
    gemini_attempted = bool(use_gemini and get_gemini_api_key())
    if gemini_attempted:
        schema = [
            {"column": str(c), "dtype": str(df[c].dtype), "sample": [str(x) for x in df[c].dropna().head(3).tolist()]}
            for c in df.columns[:40]
        ]
        prompt = (
            "Classify this dataset into ONE domain key from: "
            + ", ".join(DOMAIN_CATALOG.keys())
            + ".\nPrefer exclusive signals (e.g. churn→telecom_churn, student/gpa→education, "
            "patient/bmi→healthcare, revenue/sku→sales_forecasting, vibration/rul→predictive_maintenance). "
            "Do NOT pick healthcare from weak names like age or score.\n"
            "Return JSON only: {\"domain\": \"...\", \"confidence\": 0-1, \"why\": \"...\"}\n"
            f"Columns/dtypes/samples: {json.dumps(schema)[:4500]}"
        )
        gemini_raw = _gemini_answer(prompt)
        gemini_error = gemini_issue_from_raw(gemini_raw, attempted=True)
        try:
            start = gemini_raw.find("{")
            end = gemini_raw.rfind("}") + 1
            if start >= 0 and end > start:
                payload = json.loads(gemini_raw[start:end])
                gd = str(payload.get("domain", "")).strip()
                if gd in DOMAIN_CATALOG:
                    gemini_domain = gd
                    gconf = float(payload.get("confidence", 0.85))
                    gemini_why = str(payload.get("why", ""))
        except Exception:
            if gemini_error is None and gemini_raw and not gemini_raw.startswith("[Gemini error]"):
                gemini_error = f"Gemini response was not valid JSON: {gemini_raw[:240]}"
        if gemini_error:
            try:
                st.session_state.last_gemini_error = gemini_error
            except Exception:
                pass

    vote: dict[str, float] = {}
    for d, sc in scores.items():
        vote[d] = vote.get(d, 0.0) + 0.35 * (sc / max(1.0, max(scores.values()) or 1.0))
    if opt_domain and opt_domain in DOMAIN_CATALOG:
        vote[opt_domain] = vote.get(opt_domain, 0.0) + 0.40 * opt_conf
    if gemini_domain and gemini_domain in DOMAIN_CATALOG:
        vote[gemini_domain] = vote.get(gemini_domain, 0.0) + 0.25 * gconf

    for d, hits in reasons.items():
        excl = sum(1 for h in hits if str(h).startswith("EXCL:"))
        if excl >= 2:
            vote[d] = vote.get(d, 0.0) + 0.35
        elif d == "healthcare" and excl == 0:
            vote[d] = min(vote.get(d, 0.0), 0.15)

    if vote:
        final = max(vote, key=vote.get)
        conf = float(min(0.99, max(0.3, vote[final])))
    else:
        final, conf = "generic", 0.25

    if scores.get(heur, 0) >= 6 and (not opt_domain or scores.get(opt_domain, 0) < scores.get(heur, 0) * 0.6):
        if any(str(h).startswith("EXCL:") for h in reasons.get(heur, [])):
            final = heur
            conf = max(conf, min(0.97, 0.55 + 0.05 * scores[heur]))

    hc_excl = sum(1 for h in reasons.get("healthcare", []) if str(h).startswith("EXCL:"))
    ed_excl = sum(1 for h in reasons.get("education", []) if str(h).startswith("EXCL:"))
    if final == "healthcare" and hc_excl == 0:
        if ed_excl >= 1:
            final, conf = "education", max(conf, 0.7)
        else:
            final, conf = "generic", min(conf, 0.4)
    elif ed_excl >= 1 and scores.get("education", 0) >= scores.get(final, 0):
        final = "education"
        conf = max(conf, min(0.95, 0.55 + 0.05 * scores["education"]))

    prior = st.session_state.get("domain_meta") if isinstance(st.session_state.get("domain_meta"), dict) else {}
    forge_prior = st.session_state.get("forge_detect") if isinstance(st.session_state.get("forge_detect"), dict) else {}
    col_sig = _field_col_signature(df)
    if not st.session_state.get("domain_user_override"):
        prior_conf = float(prior.get("confidence") or 0)
        prior_dom = str(prior.get("domain") or "")
        if prior.get("col_sig") == col_sig and prior_conf >= FIELD_DETECT_HIGH_CONF and prior_dom in DOMAIN_CATALOG:
            final = prior_dom
            conf = max(conf, prior_conf)
        elif float(forge_prior.get("confidence") or 0) >= FIELD_DETECT_HIGH_CONF:
            fp = str(forge_prior.get("fingerprint") or "")
            if fp.startswith(f"{len(df)}:{df.shape[1]}:"):
                mapped = OS_TO_APP_DOMAIN.get(str(forge_prior.get("domain") or ""))
                if mapped and mapped in DOMAIN_CATALOG:
                    final = mapped
                    conf = max(conf, float(forge_prior.get("confidence") or 0))

    vote_df = pd.DataFrame(
        [{"domain": d, "ensemble_vote": round(v, 4), "label": DOMAIN_CATALOG[d]["label"]} for d, v in vote.items()]
    ).sort_values("ensemble_vote", ascending=False).reset_index(drop=True)

    result = {
        "domain": final,
        "label": DOMAIN_CATALOG.get(final, DOMAIN_CATALOG["generic"])["label"],
        "confidence": round(conf, 3),
        "heuristic": heur,
        "heuristic_scores": scores,
        "reasons": reasons.get(final) or reasons.get(heur) or [],
        "gemini_domain": gemini_domain,
        "gemini_why": gemini_why,
        "gemini_raw": gemini_raw[:500],
        "gemini_error": gemini_error,
        "gemini_attempted": gemini_attempted,
        "optuna": {
            "domain": opt_domain,
            "confidence": opt_conf,
            "cv_accuracy": opt.get("cv_accuracy"),
            "best_params": opt.get("best_params"),
            "ranking": opt.get("ranking"),
        },
        "scoreboard": scoreboard,
        "vote_table": vote_df,
        "optuna_proba_table": opt.get("proba_table"),
        "col_sig": col_sig,
    }
    result = _lock_field_to_user_override(result)
    try:
        st.session_state._field_detect_cache = {"sig": sig, "result": result}
    except Exception:
        pass
    return result

def discover_filter_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    """Find loc / date / time / people columns across any domain."""
    loc = _col(df, "location", "loc", "region", "city", "store", "warehouse", "site", "plant", "branch", "state", "country", "ward", "department")
    date = _col(df, "date", "timestamp", "datetime", "day", "order_date", "admit_date", "hire_date")
    time_c = _col(df, "time", "hour", "timestamp", "datetime")
    people = _col(df, "customer", "customer_id", "patient", "patient_id", "employee", "employee_id", "user", "user_id", "subscriber", "person", "name")
    return {"location": loc, "date": date, "time": time_c, "people": people}


def apply_analytic_filters(
    df: pd.DataFrame,
    location_vals: Optional[list[Any]] = None,
    people_vals: Optional[list[Any]] = None,
    date_start: Any = None,
    date_end: Any = None,
) -> tuple[pd.DataFrame, dict[str, Optional[str]]]:
    cols = discover_filter_columns(df)
    out = df.copy()
    if cols["location"] and location_vals:
        out = out[out[cols["location"]].astype(str).isin([str(v) for v in location_vals])]
    if cols["people"] and people_vals:
        out = out[out[cols["people"]].astype(str).isin([str(v) for v in people_vals])]
    dcol = cols["date"] or cols["time"]
    if dcol and (date_start is not None or date_end is not None):
        parsed = pd.to_datetime(out[dcol], errors="coerce")
        if date_start is not None:
            out = out[parsed >= pd.to_datetime(date_start)]
            parsed = pd.to_datetime(out[dcol], errors="coerce")
        if date_end is not None:
            out = out[parsed <= pd.to_datetime(date_end)]
    return out, cols


def render_filter_bar(df: pd.DataFrame, key_prefix: str = "kpi") -> pd.DataFrame:
    """Shared loc/date/time/people filters for KPIs and Charts."""
    cols = discover_filter_columns(df)
    st.markdown("##### Filters (loc · date · time · people)")
    c1, c2, c3, c4 = st.columns(4)
    location_vals = None
    people_vals = None
    date_start = date_end = None
    with c1:
        if cols["location"]:
            opts = sorted(df[cols["location"]].dropna().astype(str).unique().tolist())[:200]
            location_vals = st.multiselect("Location", opts, default=[], key=f"{key_prefix}_loc")
        else:
            st.caption("No location column")
    with c2:
        dcol = cols["date"] or cols["time"]
        if dcol:
            parsed = pd.to_datetime(df[dcol], errors="coerce").dropna()
            if len(parsed):
                mn, mx = parsed.min().date(), parsed.max().date()
                picked = st.date_input("Date range", value=(mn, mx), key=f"{key_prefix}_dates")
                if isinstance(picked, (list, tuple)) and len(picked) == 2:
                    date_start, date_end = picked[0], picked[1]
            else:
                st.caption("Date unparsable")
        else:
            st.caption("No date/time column")
    with c3:
        if cols["time"] and cols["time"] != cols["date"]:
            st.caption(f"Time col: `{cols['time']}`")
        else:
            st.caption("Time uses date/timestamp")
    with c4:
        if cols["people"]:
            opts = sorted(df[cols["people"]].dropna().astype(str).unique().tolist())[:200]
            people_vals = st.multiselect("People", opts, default=[], key=f"{key_prefix}_people")
        else:
            st.caption("No people column")
    filtered, _ = apply_analytic_filters(df, location_vals or None, people_vals or None, date_start, date_end)
    st.caption(f"Filtered rows: **{len(filtered):,}** / {len(df):,}")
    return filtered


def _mean(df: pd.DataFrame, *names: str) -> Any:
    c = _col(df, *names)
    if not c:
        return "—"
    s = pd.to_numeric(df[c], errors="coerce")
    return round(float(s.mean()), 3) if s.notna().any() else "—"


def _sum(df: pd.DataFrame, *names: str) -> Any:
    c = _col(df, *names)
    if not c:
        return "—"
    s = pd.to_numeric(df[c], errors="coerce")
    return round(float(s.sum()), 2) if s.notna().any() else "—"

def get_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Domain-aware KPI dictionary (numbers for square metric boxes)."""
    n_rows, n_cols = df.shape
    miss = round(float(df.isna().sum().sum() / max(1, df.size) * 100), 2)
    domain = st.session_state.get("domain")
    if not domain or (domain == "generic" and not st.session_state.get("domain_user_override") and not st.session_state.get("domain_meta")):
        domain = detect_field(df, use_gemini=False, optuna_trials=FIELD_DETECT_DEFAULT_TRIALS).get("domain", "generic")
    base = {"Rows": int(n_rows), "Cols": int(n_cols), "Missing%": miss, "Domain": DOMAIN_CATALOG.get(domain, {}).get("label", domain)}

    if domain == "predictive_maintenance":
        fcol = _col(df, "failure", "fault", "alarm")
        rcol = _col(df, "rul", "remaining_useful_life")
        base.update({
            "Mean_temp": _mean(df, "temperature", "temp"),
            "Mean_vib": _mean(df, "vibration", "vib"),
            "Mean_pressure": _mean(df, "pressure"),
            "Failure_Count": int(pd.to_numeric(df[fcol], errors="coerce").fillna(0).sum()) if fcol else 0,
            "Min_RUL": round(float(pd.to_numeric(df[rcol], errors="coerce").min()), 2) if rcol and pd.to_numeric(df[rcol], errors="coerce").notna().any() else "—",
        })
    elif domain == "healthcare":
        base.update({
            "Mean_Age": _mean(df, "age"),
            "Mean_BP": _mean(df, "bp", "blood_pressure", "systolic"),
            "Mean_Glucose": _mean(df, "glucose", "blood_sugar"),
            "Patients": int(df[_col(df, "patient", "patient_id")].nunique()) if _col(df, "patient", "patient_id") else n_rows,
        })
        w, h = _col(df, "weight"), _col(df, "height")
        if w and h:
            ww = pd.to_numeric(df[w], errors="coerce")
            hh = pd.to_numeric(df[h], errors="coerce")
            hh_m = np.where(hh > 3, hh / 100.0, hh)
            bmi = ww / np.square(np.where(hh_m == 0, np.nan, hh_m))
            base["Mean_BMI"] = round(float(np.nanmean(bmi)), 2) if np.isfinite(np.nanmean(bmi)) else "—"
        readm = _col(df, "readmission")
        if readm:
            base["Readmission%"] = round(float(pd.to_numeric(df[readm], errors="coerce").fillna(0).mean() * 100), 1)
    elif domain == "education":
        sid = _col(df, "student", "student_id")
        base.update({
            "Students": int(df[sid].nunique()) if sid else n_rows,
            "Mean_Score": _mean(df, "math_score", "score", "marks", "gpa"),
            "Mean_Attendance": _mean(df, "attendance"),
        })
    elif domain == "sales_forecasting":
        base.update({
            "Total_Revenue": _sum(df, "revenue", "sales", "gmv", "amount"),
            "Avg_Order": _mean(df, "revenue", "sales", "amount"),
            "Units": _sum(df, "units", "qty", "quantity"),
            "Customers": int(df[_col(df, "customer", "customer_id")].nunique()) if _col(df, "customer", "customer_id") else "—",
            "SKUs": int(df[_col(df, "sku", "product")].nunique()) if _col(df, "sku", "product") else "—",
        })
    elif domain == "telecom_churn":
        churn = _col(df, "churn")
        base.update({
            "Churn%": round(float(pd.to_numeric(df[churn], errors="coerce").fillna(0).mean() * 100), 2) if churn else "—",
            "Mean_Tenure": _mean(df, "tenure"),
            "Mean_ARPU": _mean(df, "arpu", "monthly_charges", "revenue"),
            "Subscribers": int(df[_col(df, "subscriber", "customer", "customer_id")].nunique()) if _col(df, "subscriber", "customer", "customer_id") else n_rows,
        })
    elif domain == "finance_risk":
        default_c = _col(df, "default", "delinquent", "fraud")
        base.update({
            "Default%": round(float(pd.to_numeric(df[default_c], errors="coerce").fillna(0).mean() * 100), 2) if default_c else "—",
            "Mean_Loan": _mean(df, "loan", "loan_amount", "amount", "balance"),
            "Mean_Income": _mean(df, "income", "annual_income"),
            "Mean_Score": _mean(df, "credit_score", "score"),
        })
    elif domain == "warehouse_logistics":
        base.update({
            "Total_Stock": _sum(df, "inventory", "stock", "on_hand"),
            "Mean_LeadTime": _mean(df, "lead_time", "leadtime"),
            "Shipments": _sum(df, "shipment", "shipments", "orders"),
            "SKUs": int(df[_col(df, "sku")].nunique()) if _col(df, "sku") else "—",
        })
    elif domain == "energy_utilities":
        base.update({
            "Mean_Load": _mean(df, "load", "consumption", "kwh", "mw", "power"),
            "Peak_Load": (
                round(float(pd.to_numeric(df[_col(df, "load", "consumption", "kwh", "mw", "power")], errors="coerce").max()), 2)
                if _col(df, "load", "consumption", "kwh", "mw", "power") else "—"
            ),
            "Mean_Voltage": _mean(df, "voltage"),
            "Mean_Current": _mean(df, "current"),
        })
    elif domain == "agriculture_iot":
        base.update({
            "Mean_Moisture": _mean(df, "moisture", "soil_moisture"),
            "Mean_Rainfall": _mean(df, "rainfall", "rain"),
            "Mean_pH": _mean(df, "ph"),
            "Mean_Yield": _mean(df, "yield", "crop_yield"),
        })
    elif domain == "hr_people":
        attr = _col(df, "attrition", "churn")
        base.update({
            "Headcount": int(df[_col(df, "employee", "employee_id")].nunique()) if _col(df, "employee", "employee_id") else n_rows,
            "Mean_Salary": _mean(df, "salary", "compensation", "pay"),
            "Attrition%": round(float(pd.to_numeric(df[attr], errors="coerce").fillna(0).mean() * 100), 2) if attr else "—",
            "Departments": int(df[_col(df, "department")].nunique()) if _col(df, "department") else "—",
        })
    else:
        # generic numeric summary
        nums = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        for c in nums:
            base[f"Mean_{c}"] = round(float(pd.to_numeric(df[c], errors="coerce").mean()), 3)
    return base

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
    elif domain == "hr_people":
        sal = _col(out, "salary", "compensation", "pay")
        if sal:
            out["salary_z"] = (
                pd.to_numeric(out[sal], errors="coerce") - pd.to_numeric(out[sal], errors="coerce").mean()
            ) / (pd.to_numeric(out[sal], errors="coerce").std() or 1)
        attr = _col(out, "attrition")
        if attr:
            out["attrition_flag"] = pd.to_numeric(out[attr], errors="coerce").fillna(0)
    return out



def kpi_group_comparisons(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Loc-to-loc and time-to-time average tables for Auto KPIs."""
    out: dict[str, pd.DataFrame] = {}
    cols = discover_filter_columns(df)
    domain = st.session_state.get("domain") or "generic"
    metric_candidates = {
        "sales_forecasting": ["revenue", "sales", "gmv", "amount", "units"],
        "healthcare": ["age", "bmi", "glucose", "bp", "systolic", "readmission"],
        "education": ["gpa", "marks", "score", "attendance", "grade"],
        "telecom_churn": ["churn", "arpu", "tenure", "monthly_charges"],
        "predictive_maintenance": ["temperature", "vibration", "pressure", "rul", "failure"],
        "finance_risk": ["loan_amount", "amount", "default", "credit_score", "income"],
        "warehouse_logistics": ["inventory", "stock", "lead_time", "shipments"],
        "energy_utilities": ["load", "kwh", "mw", "power", "voltage"],
        "agriculture_iot": ["moisture", "yield", "rainfall", "ph"],
        "hr_people": ["salary", "attrition", "performance"],
    }
    names = metric_candidates.get(domain, [])
    metric = None
    for n in names:
        metric = _col(df, n)
        if metric:
            break
    if metric is None:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        metric = nums[0] if nums else None
    if metric is None:
        return out

    if cols["location"]:
        g = df.groupby(cols["location"], dropna=False)[metric].agg(["mean", "sum", "count"]).reset_index()
        g.columns = [cols["location"], f"avg_{metric}", f"sum_{metric}", "rows"]
        g = g.sort_values(f"avg_{metric}", ascending=False)
        out["loc_to_loc"] = g
    dcol = cols["date"] or cols["time"]
    if dcol:
        tmp = df.copy()
        tmp["_period"] = pd.to_datetime(tmp[dcol], errors="coerce").dt.to_period("M").astype(str)
        g2 = tmp.dropna(subset=["_period"]).groupby("_period")[metric].agg(["mean", "sum", "count"]).reset_index()
        g2.columns = ["period", f"avg_{metric}", f"sum_{metric}", "rows"]
        out["time_to_time"] = g2
    return out


def kpi_model_insight(df: pd.DataFrame, comparisons: dict[str, pd.DataFrame]) -> str:
    """Simple model insight: weakest location / trend direction."""
    bits = []
    domain = st.session_state.get("domain") or "generic"
    if "loc_to_loc" in comparisons and len(comparisons["loc_to_loc"]) >= 2:
        tab = comparisons["loc_to_loc"]
        avg_col = [c for c in tab.columns if c.startswith("avg_")][0]
        worst = tab.iloc[-1]
        best = tab.iloc[0]
        bits.append(
            f"Loc-to-loc: **{best.iloc[0]}** leads ({avg_col}={best[avg_col]:.2f}); "
            f"**{worst.iloc[0]}** is lowest ({avg_col}={worst[avg_col]:.2f})."
        )
    if "time_to_time" in comparisons and len(comparisons["time_to_time"]) >= 3:
        tab = comparisons["time_to_time"]
        avg_col = [c for c in tab.columns if c.startswith("avg_")][0]
        y = pd.to_numeric(tab[avg_col], errors="coerce").dropna()
        if len(y) >= 3:
            slope = float(np.polyfit(np.arange(len(y)), y.to_numpy(), 1)[0])
            direction = "rising" if slope > 0 else "falling"
            bits.append(f"Time-to-time trend for `{avg_col}` is **{direction}** (slope={slope:.4f} per period).")
            # quick RF level check on last vs predicted next
            try:
                X = np.arange(len(y)).reshape(-1, 1)
                rf = RandomForestRegressor(n_estimators=80, random_state=42)
                rf.fit(X, y)
                nxt = float(rf.predict([[len(y)]])[0])
                pct = (nxt - float(y.iloc[-1])) / abs(float(y.iloc[-1]) or 1) * 100
                bits.append(f"RF next-period forecast ≈ **{nxt:.2f}** ({pct:+.1f}% vs last).")
            except Exception:
                pass
    if not bits:
        bits.append(f"Domain **{DOMAIN_CATALOG.get(domain, {}).get('label', domain)}** — add location/date columns for loc/time KPIs.")
    return " ".join(bits)


def render_kpi_boxes(kpis: dict[str, Any], per_row: int = 4) -> None:
    """Square-ish bordered metric boxes."""
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #f7fafc 0%, #eef2f7 100%);
            border: 1px solid #d0d7de;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(16,24,40,.04);
            min-height: 96px;
        }
        div[data-testid="stMetric"] label { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    items = [(k, v) for k, v in kpis.items() if k != "Domain"]
    if "Domain" in kpis:
        st.caption(f"KPI pack for: **{kpis['Domain']}**")
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for j, (k, v) in enumerate(items[i : i + per_row]):
            with cols[j]:
                st.metric(str(k).replace("_", " "), v)


def chart_business_insight(df: pd.DataFrame, x: str, y: str) -> str:
    """Plain-language insight under charts (e.g. revenue by location will drop)."""
    work = df[[x, y]].copy() if x in df.columns and y in df.columns else df.copy()
    if x not in work.columns or y not in work.columns:
        return "Select valid X/Y columns for insights."
    ynum = pd.to_numeric(work[y], errors="coerce")
    if ynum.notna().sum() < 3:
        return f"Not enough numeric values in `{y}` for prediction."

    # categorical X → compare groups
    if not pd.api.types.is_numeric_dtype(work[x]) or work[x].nunique() < max(3, len(work) // 10):
        g = work.assign(_y=ynum).groupby(x, dropna=False)["_y"].mean().sort_values()
        if len(g) >= 2:
            low, high = g.index[0], g.index[-1]
            msg = (
                f"**{y}** is lowest for **{low}** (avg={g.iloc[0]:.2f}) and highest for **{high}** "
                f"(avg={g.iloc[-1]:.2f}). Gap={g.iloc[-1]-g.iloc[0]:.2f}."
            )
            # forecast overall
            try:
                s = ynum.dropna()
                slope = float(np.polyfit(np.arange(len(s)), s.to_numpy(), 1)[0])
                future = float(s.iloc[-1] + slope * max(5, len(s) // 10))
                pct = (future - float(s.iloc[-1])) / abs(float(s.iloc[-1]) or 1) * 100
                direction = "increase" if pct >= 0 else "drop"
                msg += f" Overall `{y}` is likely to **{direction} ~{abs(pct):.1f}%** in the near future (trend model)."
            except Exception:
                pass
            return msg

    # numeric X/Y — correlation + trend
    tmp = pd.DataFrame({"x": pd.to_numeric(work[x], errors="coerce"), "y": ynum}).dropna()
    if len(tmp) < 5:
        return "Need more points for numeric insight."
    corr = float(tmp["x"].corr(tmp["y"]))
    slope = float(np.polyfit(tmp["x"], tmp["y"], 1)[0])
    direction = "rise" if slope > 0 else "fall"
    return (
        f"`{y}` vs `{x}`: correlation={corr:.2f}. As `{x}` grows, `{y}` tends to **{direction}** "
        f"(slope={slope:.4f})."
    )


def render_adaptive_chart(df: pd.DataFrame, x: str, y: str, chart_type: str, library: str, color: Optional[str] = None) -> None:
    plot_df = df.copy()
    if library == "plotly":
        if chart_type == "line":
            fig = px.line(plot_df, x=x, y=y, color=color, title=f"{y} by {x}")
        elif chart_type == "bar":
            fig = make_readable_bar(plot_df, x, y, color=color, title=f"{y} by {x}")
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color, title=f"{y} vs {x}")
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y, title=f"{y} share by {x}")
        else:
            # heatmap via pivot if possible
            if color and color in plot_df.columns:
                piv = plot_df.pivot_table(index=x, columns=color, values=y, aggfunc="mean")
                fig = px.imshow(piv, title=f"Heatmap {y}", aspect="auto")
            else:
                fig = px.density_heatmap(plot_df, x=x, y=y, title=f"Density {y} vs {x}")
        st.plotly_chart(fig, use_container_width=True)
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    n_cats = int(plot_df[x].nunique(dropna=False)) if x in plot_df.columns else 0
    categorical = x in plot_df.columns and not pd.api.types.is_numeric_dtype(plot_df[x])
    use_h = chart_type == "bar" and categorical and n_cats >= 8
    fig_w, fig_h = (9, max(4.8, 0.32 * n_cats + 1.6)) if use_h else (9, 5.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    try:
        if library == "seaborn":
            if chart_type == "line":
                sns.lineplot(data=plot_df, x=x, y=y, hue=color, ax=ax)
            elif chart_type == "bar":
                if use_h:
                    sns.barplot(data=plot_df, x=y, y=x, hue=color, ax=ax, errorbar=None, orient="h")
                else:
                    sns.barplot(data=plot_df, x=x, y=y, hue=color, ax=ax, errorbar=None)
            elif chart_type == "scatter":
                sns.scatterplot(data=plot_df, x=x, y=y, hue=color, ax=ax)
            elif chart_type == "pie":
                pie = plot_df.groupby(x, dropna=False)[y].sum()
                ax.pie(pie.values, labels=pie.index.astype(str), autopct="%1.1f%%")
                ax.set_ylabel("")
            else:
                if color and color in plot_df.columns:
                    piv = plot_df.pivot_table(index=x, columns=color, values=y, aggfunc="mean")
                    sns.heatmap(piv, ax=ax, cmap="mako")
                else:
                    sns.histplot(data=plot_df, x=y, ax=ax)
        else:  # matplotlib
            if chart_type == "line":
                ax.plot(plot_df[x], pd.to_numeric(plot_df[y], errors="coerce"))
            elif chart_type == "bar":
                g = plot_df.groupby(x, dropna=False)[y].mean()
                labels = [str(i) if len(str(i)) <= 18 else str(i)[:17] + "…" for i in g.index]
                if use_h:
                    ax.barh(labels, g.values)
                else:
                    ax.bar(labels, g.values)
            elif chart_type == "scatter":
                ax.scatter(pd.to_numeric(plot_df[x], errors="coerce"), pd.to_numeric(plot_df[y], errors="coerce"), alpha=0.7)
            elif chart_type == "pie":
                pie = plot_df.groupby(x, dropna=False)[y].sum()
                ax.pie(pie.values, labels=pie.index.astype(str), autopct="%1.1f%%")
            else:
                ax.hist(pd.to_numeric(plot_df[y], errors="coerce").dropna(), bins=20)
        ax.set_title(f"{y} by {x}")
        if chart_type == "bar":
            tick_labels = ax.get_yticklabels() if use_h else ax.get_xticklabels()
            for lbl in tick_labels:
                text = lbl.get_text()
                if len(text) > 18:
                    lbl.set_text(text[:17] + "…")
            if use_h:
                ax.tick_params(axis="y", labelsize=11)
                fig.subplots_adjust(left=0.28, bottom=0.12, right=0.98, top=0.90)
            else:
                plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=11)
                fig.subplots_adjust(bottom=0.32, left=0.10, right=0.98, top=0.90)
        else:
            fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    finally:
        plt.close(fig)


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
    domain = st.session_state.get("domain")
    if not domain:
        domain = detect_field(df, use_gemini=False, optuna_trials=FIELD_DETECT_DEFAULT_TRIALS)["domain"]
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
    if domain == "hr_people":
        attr = _col(work, "attrition", "churn")
        if attr is not None:
            rate = float(pd.to_numeric(work[attr], errors="coerce").fillna(0).mean())
            risk = max(risk, rate * 100)
            explanations.append(f"attrition_rate={rate*100:.1f}%")
        if "salary_z" in work.columns and pd.notna(work["salary_z"].iloc[-1]):
            explanations.append(f"salary_z={float(work['salary_z'].iloc[-1]):.2f}")
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
        # pandas 2.2+/3 dropped uppercase offset aliases (H → h); lowercase works on 2.0+.
        ds = pd.date_range(end=datetime.utcnow(), periods=len(df), freq="h")
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
    # Infer freq (pandas 2.0 may still return H/T; 2.2+/3 require h/min)
    freq = pd.infer_freq(tmp["ds"]) or "D"
    if freq in {"H", "h", "T", "min"}:
        periods = 90 * 24
        horizon_days = 90
        freq = {"H": "h", "T": "min"}.get(freq, freq)
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


def gemini_key_ui(context: str = "upload") -> None:
    render_gemini_key_ui(context)


def _gemini_answer(prompt: str) -> str:
    key = get_gemini_api_key()
    if not key:
        return ""
    model_name = get_gemini_model()
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", None) or str(resp)).strip()
    except Exception as exc:
        return f"[Gemini error] {exc}"


def rag_ask(question: str, df: pd.DataFrame) -> str:
    """Backward-compatible wrapper → LlamaIndex search + Gemini. """
    out = ask_llama_gemini(question, df)
    return out.get("answer") or ""


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


# === FORGE ML / LLAMA / EMAIL HELPERS ===
FORGE_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    # --- Classification (6-7+) ---
    "LogisticRegression": {"task": "classification", "library": "sklearn", "class": "sklearn.linear_model.LogisticRegression", "params": {"max_iter": 1000}, "what": "Linear classifier for failure/churn labels.", "why": "Fast baseline, readable coefficients."},
    "DecisionTreeClassifier": {"task": "classification", "library": "sklearn", "class": "sklearn.tree.DecisionTreeClassifier", "params": {"max_depth": 8, "random_state": 42}, "what": "Rule-based tree for class labels.", "why": "Easy to explain to non-data users."},
    "RandomForestClassifier": {"task": "classification", "library": "sklearn", "class": "sklearn.ensemble.RandomForestClassifier", "params": {"n_estimators": 120, "max_depth": 10, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"}, "what": "Ensemble of trees for failure/churn.", "why": "Strong accuracy on tabular plant/CRM data."},
    "GradientBoostingClassifier": {"task": "classification", "library": "sklearn", "class": "sklearn.ensemble.GradientBoostingClassifier", "params": {"n_estimators": 100, "max_depth": 3, "random_state": 42}, "what": "Boosted trees for tough class boundaries.", "why": "Often beats RF on imbalanced labels."},
    "KNeighborsClassifier": {"task": "classification", "library": "sklearn", "class": "sklearn.neighbors.KNeighborsClassifier", "params": {"n_neighbors": 5}, "what": "Classify by nearest similar rows.", "why": "Good for local sensor neighborhoods."},
    "SVC": {"task": "classification", "library": "sklearn", "class": "sklearn.svm.SVC", "params": {"C": 1.0, "kernel": "rbf"}, "what": "Support-vector classifier.", "why": "Handles non-linear splits."},
    "XGBClassifier": {"task": "classification", "library": "xgboost", "class": "xgboost.XGBClassifier", "params": {"n_estimators": 120, "max_depth": 5, "learning_rate": 0.08, "random_state": 42, "eval_metric": "logloss"}, "what": "XGBoost classifier (Forge favorite).", "why": "Top CV scores on industrial tabular data."},
    # --- Regression (5-6+) ---
    "LinearRegression": {"task": "regression", "library": "sklearn", "class": "sklearn.linear_model.LinearRegression", "params": {}, "what": "Straight-line predictor for RUL/revenue.", "why": "Simple baseline."},
    "Ridge": {"task": "regression", "library": "sklearn", "class": "sklearn.linear_model.Ridge", "params": {"alpha": 1.0}, "what": "Regularized linear regressor.", "why": "Stable when columns correlate."},
    "RandomForestRegressor": {"task": "regression", "library": "sklearn", "class": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 120, "max_depth": 10, "random_state": 42, "n_jobs": -1}, "what": "RF regressor for RUL/sales.", "why": "Robust default for managers."},
    "GradientBoostingRegressor": {"task": "regression", "library": "sklearn", "class": "sklearn.ensemble.GradientBoostingRegressor", "params": {"n_estimators": 100, "max_depth": 3, "random_state": 42}, "what": "Boosted regressor.", "why": "Strong holdout R²."},
    "SVR": {"task": "regression", "library": "sklearn", "class": "sklearn.svm.SVR", "params": {"C": 1.0, "kernel": "rbf"}, "what": "Support-vector regressor.", "why": "Non-linear continuous targets."},
    "XGBRegressor": {"task": "regression", "library": "xgboost", "class": "xgboost.XGBRegressor", "params": {"n_estimators": 140, "max_depth": 5, "learning_rate": 0.08, "random_state": 42}, "what": "XGBoost regressor.", "why": "Often best CV on RUL/revenue."},
    "LGBMRegressor": {"task": "regression", "library": "lightgbm", "class": "lightgbm.LGBMRegressor", "params": {"n_estimators": 140, "max_depth": 6, "learning_rate": 0.08, "random_state": 42, "verbosity": -1}, "what": "LightGBM regressor.", "why": "Fast on mid/large CSVs."},
    # --- Special ---
    "Prophet": {"task": "forecast", "library": "prophet", "class": "prophet.Prophet", "params": {}, "what": "Business time-series forecast.", "why": "Shows rise/drop % for managers."},
    "PCA": {"task": "dimensionality", "library": "sklearn", "class": "sklearn.decomposition.PCA", "params": {"n_components": 3}, "what": "Compress sensors into principal components.", "why": "Spot drift / dead sensors."},
    "StatsmodelsOLS": {"task": "regression", "library": "statsmodels", "class": "statsmodels.api.OLS", "params": {}, "what": "Classical OLS with p-values.", "why": "Stats-course friendly explainability."},
    "IsolationForest": {"task": "anomaly", "library": "sklearn", "class": "sklearn.ensemble.IsolationForest", "params": {"contamination": 0.08, "random_state": 42}, "what": "Unsupervised anomaly detector.", "why": "Flags weird sensor packets."},
}

DOMAIN_RECOMMENDED_MODELS: dict[str, list[str]] = {
    "predictive_maintenance": ["RandomForestRegressor", "XGBRegressor", "RandomForestClassifier", "XGBClassifier", "Prophet", "IsolationForest", "PCA"],
    "sales_forecasting": ["Prophet", "RandomForestRegressor", "XGBRegressor", "Ridge"],
    "telecom_churn": ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"],
    "healthcare": ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"],
    "education": ["RandomForestRegressor", "XGBRegressor", "RandomForestClassifier"],
    "finance_risk": ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"],
    "energy_utilities": ["Prophet", "RandomForestRegressor", "XGBRegressor", "PCA"],
    "warehouse_logistics": ["RandomForestRegressor", "XGBRegressor", "Prophet"],
    "agriculture_iot": ["RandomForestRegressor", "Prophet", "IsolationForest"],
    "hr_people": ["RandomForestClassifier", "XGBClassifier", "LogisticRegression"],
    "generic": ["RandomForestRegressor", "RandomForestClassifier", "Prophet", "PCA"],
}


_ML_LIB_READY_CACHE: dict[str, tuple[bool, str]] = {}


def _ml_library_ready(library: str) -> tuple[bool, str]:
    """
    True if native ML lib can actually load.
    Mac XGBoost often imports the Python package then fails on libomp.dylib —
    that raises XGBoostError / OSError, not ImportError (so bare ImportError guards miss it).
    """
    library = (library or "").lower()
    if library in _ML_LIB_READY_CACHE:
        return _ML_LIB_READY_CACHE[library]
    if library in {"sklearn", "prophet", "statsmodels", ""}:
        _ML_LIB_READY_CACHE[library] = (True, "")
        return True, ""
    if library == "xgboost":
        try:
            import xgboost as xgb  # noqa: F401
            # Force native lib load (import alone is not enough on macOS)
            _ = xgb.XGBRegressor(n_estimators=1, max_depth=1)
            result = (True, "")
        except Exception as exc:
            result = (
                False,
                (
                    f"XGBoost unavailable ({exc}). "
                    "On Mac: brew install libomp && pip install --force-reinstall xgboost. "
                    "Forge falls back to RandomForest."
                ),
            )
        _ML_LIB_READY_CACHE[library] = result
        return result
    if library == "lightgbm":
        try:
            import lightgbm as lgb  # noqa: F401
            _ = lgb.LGBMRegressor(n_estimators=1)
            result = (True, "")
        except Exception as exc:
            result = (False, f"LightGBM unavailable ({exc}). Using sklearn fallback.")
        _ML_LIB_READY_CACHE[library] = result
        return result
    _ML_LIB_READY_CACHE[library] = (True, "")
    return True, ""


def list_runnable_models() -> list[str]:
    """Catalog ids whose native libs load on this machine."""
    out = []
    for mid, meta in FORGE_MODEL_CATALOG.items():
        ok, _ = _ml_library_ready(meta.get("library", "sklearn"))
        if ok:
            out.append(mid)
    return out


def _resolve_estimator(dotted: str, params: dict[str, Any]):
    import importlib
    module_path, _, cls_name = dotted.rpartition(".")
    if module_path.startswith("xgboost"):
        ok, msg = _ml_library_ready("xgboost")
        if not ok:
            raise RuntimeError(msg)
    if module_path.startswith("lightgbm"):
        ok, msg = _ml_library_ready("lightgbm")
        if not ok:
            raise RuntimeError(msg)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    try:
        return cls(**params)
    except TypeError:
        return cls()


def _ml_pick_target(df: pd.DataFrame, task: str, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    pri_clf = ["failure", "fault", "churn", "default", "readmission", "attrition", "label", "class"]
    pri_reg = ["rul", "remaining_useful_life", "revenue", "sales", "amount", "gmv", "temperature", "load", "score"]
    pri = pri_clf if task == "classification" else pri_reg
    for n in pri:
        hit = _col(df, n)
        if hit:
            return hit
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if task == "classification":
        for c in df.columns:
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= 8 and not str(c).lower().endswith("_id"):
                return c
        return None
    return nums[-1] if nums else None


def _ml_pick_features(df: pd.DataFrame, target: str) -> list[str]:
    feats = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        cl = str(c).lower()
        if any(h in cl for h in ("timestamp", "datetime", "date", "time")) and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if cl in {"id", "uuid", "index"} or cl.endswith("_id") or "unnamed" in cl:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique(dropna=True) > min(40, max(15, int(n * 0.4))):
                continue
        feats.append(c)
    return feats


def run_forge_model(
    df: pd.DataFrame,
    model_id: str,
    target: Optional[str] = None,
    test_size: float = 0.2,
    time_series_split: bool = False,
) -> dict[str, Any]:
    """Forge Analytics-style model runner (class/reg/prophet/pca/statsmodels/anomaly)."""
    meta = FORGE_MODEL_CATALOG.get(model_id)
    if not meta:
        return {"ok": False, "error": f"Unknown model {model_id}", "model_id": model_id}
    task = meta["task"]
    library = meta["library"]
    ok_lib, lib_msg = _ml_library_ready(library)
    if not ok_lib:
        return {"ok": False, "error": lib_msg, "model_id": model_id, "task": task}

    # PCA
    if task == "dimensionality" or model_id == "PCA":
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        num = df.select_dtypes(include=[np.number]).dropna()
        if num.shape[1] < 2 or len(num) < 5:
            return {"ok": False, "error": "PCA needs ≥2 numeric cols and 5 rows.", "model_id": model_id, "task": task}
        X = StandardScaler().fit_transform(num.values)
        ncomp = min(int(meta["params"].get("n_components", 3)), X.shape[1])
        pca = PCA(n_components=ncomp, random_state=42)
        pcs = pca.fit_transform(X)
        preview = pd.DataFrame(pcs[:20], columns=[f"PC{i+1}" for i in range(ncomp)])
        return {
            "ok": True, "model_id": model_id, "task": task,
            "metrics": {
                "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
                "total_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
            },
            "predictions_preview": preview,
            "target": None,
        }

    # Anomaly
    if task == "anomaly" or model_id == "IsolationForest":
        num = df.select_dtypes(include=[np.number]).dropna()
        if num.shape[1] < 2 or len(num) < 10:
            return {"ok": False, "error": "IsolationForest needs numeric data.", "model_id": model_id, "task": task}
        est = _resolve_estimator(meta["class"], meta["params"])
        labels = est.fit_predict(num.values)
        rate = float((labels == -1).mean())
        preview = num.copy()
        preview["anomaly"] = labels
        return {
            "ok": True, "model_id": model_id, "task": task,
            "metrics": {"anomaly_rate": round(rate, 4), "anomalies": int((labels == -1).sum())},
            "predictions_preview": preview[preview["anomaly"] == -1].head(30),
            "target": None,
        }

    # Prophet
    if task == "forecast" or model_id == "Prophet":
        tgt = target or _ml_pick_target(df, "regression")
        if not tgt:
            return {"ok": False, "error": "Prophet needs a numeric target.", "model_id": model_id}
        text = prophet_forecast(df, tgt)
        # also structured metrics
        try:
            from prophet import Prophet
            date_col = _col(df, "timestamp", "date", "datetime", "time", "ds")
            if date_col:
                ds = pd.to_datetime(df[date_col], errors="coerce")
            else:
                ds = pd.date_range(end=datetime.utcnow(), periods=len(df), freq="D")
            tmp = pd.DataFrame({"ds": ds, "y": pd.to_numeric(df[tgt], errors="coerce")}).dropna().sort_values("ds")
            if len(tmp) < 10:
                return {"ok": False, "error": "Need ≥10 dated rows for Prophet.", "model_id": model_id, "task": task, "target": tgt}
            m = Prophet(uncertainty_samples=50)
            m.fit(tmp)
            future = m.make_future_dataframe(periods=30)
            fc = m.predict(future)
            last = float(tmp["y"].iloc[-1])
            end = float(fc["yhat"].iloc[-1])
            pct = (end - last) / abs(last or 1)
            return {
                "ok": True, "model_id": model_id, "task": "forecast", "target": tgt,
                "metrics": {
                    "last_actual": round(last, 3), "forecast_end": round(end, 3),
                    "pct_change": round(pct, 4), "horizon": 30, "mae": None,
                },
                "manager_briefing": text,
                "predictions_preview": fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Prophet failed: {exc}", "model_id": model_id, "task": task, "target": tgt}

    # Statsmodels OLS
    if library == "statsmodels" or model_id == "StatsmodelsOLS":
        try:
            import statsmodels.api as sm
        except ImportError as exc:
            return {"ok": False, "error": f"statsmodels missing: {exc}", "model_id": model_id}
        tgt = target or _ml_pick_target(df, "regression")
        feats = _ml_pick_features(df, tgt)[:8]
        work = df[feats + [tgt]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(work) < 15:
            return {"ok": False, "error": "Need ≥15 rows for OLS.", "model_id": model_id}
        y = work[tgt]
        X = sm.add_constant(work[feats])
        cut = int(len(work) * 0.8)
        model = sm.OLS(y.iloc[:cut], X.iloc[:cut]).fit()
        pred = model.predict(X.iloc[cut:])
        yt = y.iloc[cut:]
        metrics = {
            "r2": round(float(r2_score(yt, pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(yt, pred))), 4),
            "mae": round(float(mean_absolute_error(yt, pred)), 4),
            "aic": round(float(model.aic), 2),
            "bic": round(float(model.bic), 2),
        }
        return {
            "ok": True, "model_id": model_id, "task": "regression", "target": tgt,
            "metrics": metrics, "summary": str(model.summary()),
            "predictions_preview": pd.DataFrame({"y_true": yt.values, "y_pred": pred.values}).head(30),
        }

    # Supervised sklearn / xgboost / lightgbm
    tgt = target or _ml_pick_target(df, "classification" if task == "classification" else "regression")
    if not tgt:
        return {"ok": False, "error": "Could not pick target.", "model_id": model_id, "task": task}
    feats = _ml_pick_features(df, tgt)
    if not feats:
        return {"ok": False, "error": "No usable features.", "model_id": model_id, "target": tgt}
    work = df[feats + [tgt]].copy().dropna(subset=[tgt])
    if len(work) > 25000:
        work = work.sample(25000, random_state=42)
    if len(work) < 12:
        return {"ok": False, "error": "Need ≥12 rows.", "model_id": model_id}

    X = work[feats].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.factorize(X[c].astype(str))[0]
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    y = work[tgt]
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    else:
        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna()
        X, y = X.loc[mask], y.loc[mask]

    if time_series_split:
        cut = max(1, min(len(X) - 1, int(len(X) * (1 - test_size))))
        X_train, X_test = X.iloc[:cut], X.iloc[cut:]
        y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    else:
        strat = y if task == "classification" and y.nunique() > 1 and y.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    try:
        est = _resolve_estimator(meta["class"], dict(meta["params"]))
    except Exception as exc:
        return {"ok": False, "error": f"Import/build failed: {exc}", "model_id": model_id}

    # CV score
    cv_score = None
    try:
        scoring = "accuracy" if task == "classification" else "r2"
        cv_score = float(cross_val_score(est, X, y, cv=min(5, max(2, len(X) // 8)), scoring=scoring).mean())
    except Exception:
        pass

    try:
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
    except Exception as exc:
        return {"ok": False, "error": f"Fit failed: {exc}", "model_id": model_id, "target": tgt}

    metrics: dict[str, Any] = {}
    if task == "classification":
        metrics["accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
        metrics["f1"] = round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4)
        if cv_score is not None:
            metrics["cv_score"] = round(cv_score, 4)
    else:
        metrics["r2"] = round(float(r2_score(y_test, preds)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)
        if cv_score is not None:
            metrics["cv_score"] = round(cv_score, 4)

    preview = pd.DataFrame({"y_true": list(y_test)[:40], "y_pred": list(preds)[:40]})
    briefing = build_manager_briefing({"ok": True, "model_id": model_id, "task": task, "target": tgt, "metrics": metrics})
    return {
        "ok": True, "model_id": model_id, "task": task, "target": tgt, "features": feats[:20],
        "metrics": metrics, "predictions_preview": preview, "manager_briefing": briefing, "cv_score": cv_score,
    }


def build_manager_briefing(result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return ""
    model = result.get("model_id")
    target = result.get("target") or "target"
    m = result.get("metrics") or {}
    task = result.get("task")
    if task == "forecast":
        return result.get("manager_briefing") or f"Prophet outlook on `{target}` ready."
    if task == "classification":
        return (
            f"**Manager read — {model} on `{target}`**\n"
            f"- Accuracy **{m.get('accuracy', 'n/a')}**, F1 **{m.get('f1', 'n/a')}**, "
            f"CV **{m.get('cv_score', 'n/a')}**.\n"
            f"- Use this score to prioritize interventions on high-risk rows."
        )
    if task == "dimensionality":
        return f"**PCA** explains **{m.get('total_explained', 'n/a')}** of variance — check dead/correlated sensors."
    if task == "anomaly":
        return f"**IsolationForest** flagged **{m.get('anomaly_rate', 0)*100:.1f}%** anomaly rate ({m.get('anomalies')} rows)."
    return (
        f"**Manager read — {model} on `{target}`**\n"
        f"- R² **{m.get('r2', 'n/a')}**, RMSE **{m.get('rmse', 'n/a')}**, MAE **{m.get('mae', 'n/a')}**, "
        f"CV **{m.get('cv_score', 'n/a')}**.\n"
        f"- Higher R² / lower RMSE = more trustworthy predictions for planning."
    )


def domain_default_target(df: pd.DataFrame, domain: str) -> Optional[str]:
    mapping = {
        "predictive_maintenance": ["rul", "failure", "temperature"],
        "sales_forecasting": ["revenue", "sales", "gmv"],
        "telecom_churn": ["churn"],
        "healthcare": ["readmission", "bmi"],
        "education": ["gpa", "marks", "score", "grade"],
        "finance_risk": ["default", "loan_amount"],
        "energy_utilities": ["load", "kwh", "mw"],
        "warehouse_logistics": ["inventory", "lead_time"],
        "agriculture_iot": ["yield", "moisture"],
        "hr_people": ["attrition", "salary"],
    }
    for n in mapping.get(domain, ["rul", "failure", "revenue", "churn"]):
        hit = _col(df, n)
        if hit:
            return hit
    return _ml_pick_target(df, "regression") or _ml_pick_target(df, "classification")


def field_best_model_card(df: pd.DataFrame, domain: str) -> dict[str, Any]:
    """Auto-pick best model for domain target via quick CV bake-off (XGB if libomp OK, else RF)."""
    target = domain_default_target(df, domain)
    if not target:
        return {"ok": False, "error": "No target column found for this field."}
    y = df[target]
    y_num = pd.to_numeric(y, errors="coerce")
    task = "regression" if y_num.notna().sum() >= max(10, int(0.7 * len(y))) and y_num.nunique() > 8 else "classification"
    candidates = [m for m in DOMAIN_RECOMMENDED_MODELS.get(domain, []) if FORGE_MODEL_CATALOG.get(m, {}).get("task") == task]
    if not candidates:
        candidates = ["RandomForestRegressor", "XGBRegressor"] if task == "regression" else ["RandomForestClassifier", "XGBClassifier"]

    skipped: list[str] = []
    avail: list[str] = []
    for mid in candidates:
        lib = FORGE_MODEL_CATALOG[mid]["library"]
        ok, msg = _ml_library_ready(lib)
        if not ok:
            skipped.append(f"{mid}: {msg}")
            continue
        avail.append(mid)
    if not avail:
        avail = ["RandomForestRegressor"] if task == "regression" else ["RandomForestClassifier"]

    results = []
    failures = []
    for mid in avail[:4]:
        try:
            res = run_forge_model(df, mid, target=target, time_series_split=True)
        except Exception as exc:
            failures.append(f"{mid}: {exc}")
            continue
        if res.get("ok"):
            score = res.get("metrics", {}).get("cv_score")
            if score is None:
                score = res.get("metrics", {}).get("r2") or res.get("metrics", {}).get("accuracy") or 0
            results.append((float(score), mid, res))
        else:
            failures.append(f"{mid}: {res.get('error')}")
    if not results:
        # Absolute last resort — plain RF via run_forge_model
        fallback = "RandomForestRegressor" if task == "regression" else "RandomForestClassifier"
        try:
            res = run_forge_model(df, fallback, target=target, time_series_split=True)
            if res.get("ok"):
                results.append((float(res.get("metrics", {}).get("cv_score") or 0), fallback, res))
        except Exception as exc:
            return {
                "ok": False,
                "error": f"All candidate models failed. {exc}",
                "target": target,
                "skipped": skipped,
                "failures": failures,
            }
    if not results:
        return {"ok": False, "error": "All candidate models failed.", "target": target, "skipped": skipped, "failures": failures}
    results.sort(key=lambda z: -z[0])
    best_score, best_id, best_res = results[0]
    actions = perspective_actions(df, domain, best_res)
    note = ""
    if skipped:
        note = "Some models skipped (native lib missing — common on Mac without `brew install libomp`)."
    return {
        "ok": True,
        "domain": domain,
        "target": target,
        "best_model": best_id,
        "cv_score": round(best_score, 4),
        "result": best_res,
        "leaderboard": [{"model": m, "score": round(s, 4)} for s, m, _ in results],
        "actions": actions,
        "skipped": skipped[:3],
        "note": note,
    }


def perspective_actions(df: pd.DataFrame, domain: str, ml_result: dict[str, Any]) -> list[str]:
    """Plain-language alerts / recommended actions / downtime-cost style lines."""
    lines: list[str] = []
    domain_label = DOMAIN_CATALOG.get(domain, {}).get("label", domain)
    metrics = ml_result.get("metrics") or {}
    model = ml_result.get("model_id")
    target = ml_result.get("target")
    lines.append(f"Field **{domain_label}** · best model **{model}** on `{target}` · CV/score **{metrics.get('cv_score') or metrics.get('r2') or metrics.get('accuracy')}**.")

    if domain == "predictive_maintenance":
        v = _col(df, "vibration", "vib")
        t = _col(df, "temperature", "temp")
        mid = _col(df, "machine_id", "machine", "asset_id", "asset")
        alert_machine = None
        vib_val = None
        if v is not None:
            vs = pd.to_numeric(df[v], errors="coerce")
            thr = float(vs.mean() + 2 * (vs.std() or 1))
            hot = df.loc[vs >= thr]
            if len(hot) and mid:
                alert_machine = hot[mid].astype(str).iloc[-1]
                vib_val = float(vs.loc[hot.index[-1]])
            elif len(hot):
                vib_val = float(vs.loc[hot.index[-1]])
        if alert_machine and vib_val is not None:
            lines.append(f"🚨 **ALERT:** Bearing wear pattern on **{alert_machine}** — vibration spiked to **{vib_val:.2f}**.")
        elif vib_val is not None:
            lines.append(f"🚨 **ALERT:** Vibration spike detected at **{vib_val:.2f}** (sensor fault / bearing risk).")
        else:
            lines.append("Sensors within envelope — continue condition monitoring.")
        lines.append("**Recommended actions:** Schedule maintenance team within 8–12h · inspect bearings · verify temp gradient.")
        # downtime economics proxy
        risk = float(field_predict(df))
        downtime_hrs = round(2 + risk / 20.0, 1)
        saved_lakh = round(max(0.5, (100 - risk) / 100 * 3.5), 2)
        lines.append(f"**Expected downtime if ignored:** ~{downtime_hrs}h · **Downtime cost avoided if maintained:** ₹{saved_lakh}L (proxy).")
        if t:
            lines.append(f"Latest mean temperature context: **{_mean(df, 'temperature', 'temp')}**.")
    elif domain == "sales_forecasting":
        rev = _col(df, "revenue", "sales", "gmv")
        loc = _col(df, "location", "region", "store")
        if rev and loc:
            g = df.groupby(loc)[rev].mean().sort_values()
            lines.append(f"🚨 Revenue soft spot: **{g.index[0]}** (avg={g.iloc[0]:.2f}) vs leader **{g.index[-1]}** ({g.iloc[-1]:.2f}).")
        lines.append("**Recommended actions:** Promo push on weak region · re-check stockouts · run Prophet 30-day outlook in ML Studio.")
        pct = metrics.get("pct_change")
        if pct is not None:
            direction = "drop" if float(pct) < 0 else "rise"
            lines.append(f"**Forecast:** `{target}` likely to **{direction} ~{abs(float(pct))*100:.1f}%**.")
    elif domain == "telecom_churn":
        churn = _col(df, "churn")
        rate = float(pd.to_numeric(df[churn], errors="coerce").fillna(0).mean()) if churn else 0
        lines.append(f"🚨 Churn rate **{rate*100:.1f}%** — prioritize high-tenure / low-ARPU cohort.")
        lines.append("**Recommended actions:** Retention offers · call-drop RCA · win-back campaign on top-risk subscribers.")
        lines.append(f"**Revenue protected (proxy):** ₹{round(rate*2.5, 2)}L if churn cut 50%.")
    elif domain == "healthcare":
        lines.append("**Recommended actions:** Flag high BMI / glucose cohort · schedule follow-ups · monitor readmission risk.")
        lines.append("Model supports triage — not a medical diagnosis.")
    else:
        lines.append("**Recommended actions:** Review KPIs → pin charts to Dashboard → email weekly pack to stakeholders.")
    return lines


# -----------------------------------------------------------------------------
# LlamaIndex — build once per upload fingerprint
# -----------------------------------------------------------------------------

def _df_fingerprint(df: pd.DataFrame) -> str:
    name = str(st.session_state.get("manual_name") or st.session_state.get("mode") or "buf")
    return f"{name}|{len(df)}x{df.shape[1]}|{hash(tuple(df.columns))}"


def _row_documents(df: pd.DataFrame, max_rows: int = 400) -> list[str]:
    sample = df.head(max_rows)
    docs = []
    for i, row in sample.iterrows():
        parts = [f"row_id={i}"]
        for c in sample.columns:
            val = row[c]
            if pd.notna(val):
                parts.append(f"{c}={val}")
        docs.append(" | ".join(parts))
    return docs


def ensure_llama_index(df: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    """Build LlamaIndex (or keyword corpus) once per upload."""
    fp = _df_fingerprint(df)
    cached = st.session_state.get("llama_index_meta") or {}
    if not force and cached.get("fingerprint") == fp and st.session_state.get("llama_docs"):
        return {"ok": True, "cached": True, **cached}

    docs = _row_documents(df)
    index_obj = None
    mode = "keyword"
    try:
        from llama_index.core import Document, VectorStoreIndex
        documents = [Document(text=t) for t in docs]
        try:
            index_obj = VectorStoreIndex.from_documents(documents)
            mode = "llama_vector"
        except Exception:
            # no embedding model — keep docs for keyword search
            mode = "keyword"
    except Exception:
        mode = "keyword"

    meta = {
        "ok": True,
        "fingerprint": fp,
        "n_docs": len(docs),
        "mode": mode,
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    st.session_state.llama_docs = docs
    st.session_state.llama_index_obj = index_obj
    st.session_state.llama_index_meta = meta
    return meta


def llama_search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    """Search indexed CSV rows — works offline (keyword) or via LlamaIndex retriever."""
    docs: list[str] = st.session_state.get("llama_docs") or []
    index_obj = st.session_state.get("llama_index_obj")
    hits: list[dict[str, Any]] = []
    if index_obj is not None:
        try:
            engine = index_obj.as_query_engine(similarity_top_k=top_k)
            resp = engine.query(query)
            text = str(resp)
            hits.append({"score": 1.0, "text": text, "source": "llama_query_engine"})
            # also pull source nodes if present
            src = getattr(resp, "source_nodes", None) or []
            for n in src[:top_k]:
                hits.append({"score": float(getattr(n, "score", 0) or 0), "text": str(getattr(n, "node", n).get_content() if hasattr(getattr(n, "node", None), "get_content") else n), "source": "llama_node"})
            if hits:
                return hits[:top_k]
        except Exception:
            pass
    # keyword offline search
    q_tokens = set(re.findall(r"[a-zA-Z0-9_.]+", (query or "").lower()))
    scored = []
    for t in docs:
        toks = set(re.findall(r"[a-zA-Z0-9_.]+", t.lower()))
        score = len(q_tokens & toks)
        # boost numeric mentions when question asks why/fail/vibration
        if any(w in (query or "").lower() for w in ("fail", "vibration", "temp", "why", "spike")):
            if "vibration" in t.lower() or "fail" in t.lower():
                score += 2
        scored.append((score, t))
    scored.sort(key=lambda z: -z[0])
    for s, t in scored[:top_k]:
        if s > 0:
            hits.append({"score": float(s), "text": t, "source": "keyword"})
    if not hits:
        hits = [{"score": 0.0, "text": t, "source": "fallback"} for t in docs[:3]]
    return hits


def ask_llama_gemini(question: str, df: pd.DataFrame) -> dict[str, Any]:
    """Index → Search → Answer. Offline Llama/keyword first, Gemini grounded on hits."""
    ensure_llama_index(df, force=False)
    hits = llama_search(question, top_k=6)
    context = "\n".join(f"- {h['text']}" for h in hits)
    # Offline extractive answer: pull key numbers from hits
    offline = _offline_answer_from_hits(question, hits, df)
    gemini_ans = ""
    gemini_error = None
    gemini_attempted = bool(get_gemini_api_key())
    if gemini_attempted:
        prompt = (
            "You are Analytics Forge industrial OS. Answer ONLY using the retrieved CSV rows.\n"
            "Cite concrete numbers (vibration, temperature, machine_id, revenue, etc.).\n"
            f"Question: {question}\n\nRetrieved rows:\n{context}\n\n"
            "Give a short operational answer with recommended action."
        )
        gemini_ans = _gemini_answer(prompt)
        gemini_error = gemini_issue_from_raw(gemini_ans, attempted=True)
        if gemini_error:
            gemini_ans = ""
            try:
                st.session_state.last_gemini_error = gemini_error
            except Exception:
                pass
    final = gemini_ans.strip() if gemini_ans and not gemini_ans.startswith("[Gemini") else offline
    return {
        "answer": final,
        "offline_answer": offline,
        "gemini_answer": gemini_ans,
        "gemini_error": gemini_error,
        "used_offline_fallback": bool(gemini_error) or (gemini_attempted and not gemini_ans),
        "hits": hits,
        "index_meta": st.session_state.get("llama_index_meta"),
    }


def _offline_answer_from_hits(question: str, hits: list[dict[str, Any]], df: pd.DataFrame) -> str:
    q = (question or "").lower()
    # machine failure style
    mid = _col(df, "machine_id", "machine", "asset_id")
    v = _col(df, "vibration", "vib")
    t = _col(df, "temperature", "temp")
    machine_ask = None
    m = re.search(r"machine\s*([a-zA-Z0-9_-]+)", q)
    if m:
        machine_ask = m.group(1)
    if ("fail" in q or "why" in q) and (mid or v):
        subset = df
        if machine_ask and mid:
            subset = df[df[mid].astype(str).str.contains(machine_ask, case=False, na=False)]
            if subset.empty:
                subset = df
        vib_txt = ""
        temp_txt = ""
        if v is not None and len(subset):
            vv = float(pd.to_numeric(subset[v], errors="coerce").max())
            vib_txt = f"vibration reached **{vv:.2f}**"
        if t is not None and len(subset):
            tt = float(pd.to_numeric(subset[t], errors="coerce").max())
            temp_txt = f"temperature peaked at **{tt:.2f}**"
        who = f"Machine **{machine_ask}**" if machine_ask else "The asset"
        bits = ", ".join(x for x in (vib_txt, temp_txt) if x) or "sensor anomalies in retrieved rows"
        # enrich from hit text
        hit_snip = hits[0]["text"] if hits else ""
        return (
            f"{who} likely failed because {bits}. "
            f"LlamaIndex search evidence: `{hit_snip[:220]}`. "
            "Recommended: inspect bearings / schedule maintenance."
        )
    if hits:
        return f"Based on indexed CSV rows: {hits[0]['text'][:300]}"
    return f"Dataset has {len(df):,} rows. Ask about a machine, KPI, or column."


def build_html_report(
    domain: str,
    source_name: str,
    kpis: dict,
    insights: list,
    ml_result: Optional[dict],
    briefing: str,
) -> str:
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in (kpis or {}).items())
    insight_html = "".join(f"<li>{i}</li>" for i in (insights or []))
    ml = ml_result or {}
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Analytics Forge Report</title>
    <style>body{{font-family:Segoe UI,Arial;margin:24px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px 10px}}</style>
    </head><body>
    <h1>Analytics Forge v2 Report</h1>
    <p><b>Field:</b> {domain} · <b>Source:</b> {source_name}</p>
    <h2>Briefing</h2><p>{briefing}</p>
    <h2>KPIs</h2><table>{rows}</table>
    <h2>Insights</h2><ul>{insight_html}</ul>
    <h2>ML</h2><pre>{json.dumps({k: ml.get(k) for k in ('model_id','task','target','metrics')}, indent=2, default=str)}</pre>
    </body></html>"""


def send_forge_email_report(
    to_addr: str,
    subject: str,
    body: str,
    df: Optional[pd.DataFrame] = None,
    html_report: Optional[str] = None,
    html_filename: str = "forge_report.html",
    kpi_csv: Optional[bytes] = None,
) -> str:
    if not EMAIL_USER or not EMAIL_PASSWORD:
        raise RuntimeError(
            "Email not configured. Set EMAIL_USER and EMAIL_PASSWORD (Gmail App Password) in `.env`."
        )
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)
    if html_report:
        msg.add_attachment(
            html_report.encode("utf-8"),
            maintype="text",
            subtype="html",
            filename=html_filename,
        )
    if kpi_csv:
        msg.add_attachment(kpi_csv, maintype="text", subtype="csv", filename="forge-kpis.csv")
    if df is not None:
        msg.add_attachment(df.to_csv(index=False).encode("utf-8"), maintype="text", subtype="csv", filename="forge_data.csv")
    context = ssl.create_default_context()
    with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=30) as server:
        server.starttls(context=context)
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
    return f"Sent to {to_addr}"


def send_full_dashboard_email(
    to_addr: str,
    *,
    subject: str,
    body: str,
    html_report: str,
    kpi_csv: Optional[bytes] = None,
    df: Optional[pd.DataFrame] = None,
) -> str:
    """Email full dashboard pack (HTML charts + KPI CSV + optional data CSV)."""
    if not to_addr.strip():
        raise RuntimeError("Enter a recipient email.")
    return send_forge_email_report(
        to_addr.strip(),
        subject,
        body,
        df=df,
        html_report=html_report,
        html_filename="forge-dashboard-report.html",
        kpi_csv=kpi_csv,
    )


# =============================================================================
# PAGES
# =============================================================================


def page_live_console() -> None:
    """SCADA LIVE console — left: connection + metrics + buffer; right: auto insights / engine."""
    st.header("LIVE SCADA")
    st.caption(
        "One pipe, three layers: **OCP-U → pymodbus → (optional FastAPI) → data/live.csv → Streamlit**. "
        "Switch Direct / FastAPI / Buffer-only in `config.yaml` — Manual mode unchanged."
    )
    cfg = load_live_config()
    ctype = str(cfg.get("connection_type") or "direct").lower()

    left, right = st.columns([2.2, 1])
    with left:
        st.subheader("Connection")
        c1, c2, c3 = st.columns(3)
        with c1:
            ui_type = st.selectbox(
                "Connection type (session override)",
                ["direct", "fastapi", "buffer_only"],
                index=["direct", "fastapi", "buffer_only"].index(ctype) if ctype in ("direct", "fastapi", "buffer_only") else 0,
                help="Persists only this session; edit config.yaml for permanent.",
            )
        with c2:
            ip = st.text_input("OCP-U IP", value=str(cfg.get("ocp_u_ip") or "192.168.1.50"))
        with c3:
            port = st.number_input("Port", min_value=1, max_value=65535, value=int(cfg.get("ocp_u_port") or 502))
        fastapi_url = st.text_input("FastAPI /live URL", value=str(cfg.get("fastapi_url") or "http://127.0.0.1:8088/live"))
        # session overrides
        st.session_state.live_cfg_override = {
            **cfg,
            "connection_type": ui_type,
            "ocp_u_ip": ip,
            "ocp_u_port": int(port),
            "fastapi_url": fastapi_url,
        }

        # monkey-patch load for this session via wrapper stored
        def _cfg_override():
            return st.session_state.live_cfg_override

        # temporarily use override inside poll by writing a thin adapter
        b1, b2, b3 = st.columns(3)
        with b1:
            poll_now = st.button("Poll now", type="primary")
        with b2:
            auto = st.checkbox("Auto-poll (~5s on rerun)", value=bool(st.session_state.get("live_auto_poll")))
            st.session_state.live_auto_poll = auto
        with b3:
            if st.button("Reload buffer only"):
                buf = read_live_buffer()
                if buf is not None:
                    st.success(f"Buffer {len(buf):,} rows")
                else:
                    st.warning("No data/live.csv yet")

        # Apply override into ensure_live by patching load_live_config call site via session
        # We call fetch with override cfg directly for poll_now
        err = None
        df = None
        try:
            if poll_now or (auto and ui_type != "buffer_only"):
                if ui_type == "buffer_only":
                    df = read_live_buffer()
                    if df is None or df.empty:
                        raise RuntimeError("buffer_only: empty data/live.csv — run gateway.py on plant Pi.")
                    st.session_state.live_status = "buffer_only"
                    st.session_state.live_error = None
                else:
                    row = fetch_live_row(st.session_state.live_cfg_override)
                    df = append_live_csv(row)
                    st.session_state.live_last_poll = time.time()
                    st.session_state.live_status = f"connected:{ui_type}"
                    st.session_state.live_error = None
                    st.session_state.live_last_row = row
                    st.success(f"Polled OK via **{ui_type}**")
            else:
                df = read_live_buffer()
                if df is None:
                    # try soft ensure (may error)
                    try:
                        # use override
                        old = load_live_config
                        # call ensure with patched config by temporarily writing
                        row = None
                        if ui_type != "buffer_only":
                            pass
                        df = read_live_buffer()
                    except Exception:
                        df = None
        except Exception as exc:
            err = str(exc)
            st.session_state.live_status = "error"
            st.session_state.live_error = err
            df = read_live_buffer()

        st.write(
            f"Status: **{st.session_state.get('live_status', 'idle')}** · "
            f"config default `{ctype}` · buffer `{live_buffer_path(cfg)}`"
        )
        if err or st.session_state.get("live_error"):
            st.error(err or st.session_state.live_error)
            st.caption("Plant unreachable — showing last good buffer if any (no fake live data).")

        metrics = live_latest_metrics(df)
        if metrics:
            st.subheader("Live tags")
            keys = [k for k in ("temperature", "vibration", "pressure", "smps_voltage", "smps_current", "rul", "failure", "load") if k in metrics]
            if not keys:
                keys = list(metrics.keys())[:8]
            cols = st.columns(min(4, max(1, len(keys))))
            for i, k in enumerate(keys):
                cols[i % len(cols)].metric(k.replace("_", " ").title(), metrics[k])
        else:
            st.info(
                "No live tags yet. Set OCP-U IP (or FastAPI URL), click **Poll now**. "
                "For demo without plant: run gateway against a Modbus simulator, or copy a CSV to `data/live.csv`."
            )

        if df is not None and len(df):
            st.subheader("SCADA buffer (tail)")
            st.dataframe(df.tail(40), use_container_width=True)
            tcol = _col(df, "temperature")
            vcol = _col(df, "vibration")
            if tcol:
                st.line_chart(pd.to_numeric(df[tcol], errors="coerce").tail(60))
            if vcol:
                st.line_chart(pd.to_numeric(df[vcol], errors="coerce").tail(60))
        with st.expander("Register map (from config.yaml)"):
            st.json(cfg.get("registers") or {})
        with st.expander("How LIVE works"):
            st.markdown(
                """
**Direct (demo room):** `Streamlit → pymodbus → OCP-U @ IP:502`  
**FastAPI (factory):** `OCP-U → gateway.py (Pi) → /live` then `Streamlit → HTTP`  
**Buffer-only:** Gateway writes `data/live.csv` 24/7; Streamlit only reads (never crashes if plant drops).

Edit `config.yaml` → `LIVE_MODE.connection_type`.
                """
            )

    with right:
        st.subheader("Auto insights")
        engine = st.selectbox(
            "Insight engine",
            ["prophet", "pandas", "pyspark"],
            index=["prophet", "pandas", "pyspark"].index(str(cfg.get("default_insight_engine") or "prophet"))
            if str(cfg.get("default_insight_engine") or "prophet") in ("prophet", "pandas", "pyspark")
            else 0,
        )
        st.session_state.live_insight_engine = engine
        if df is not None and len(df) >= 5:
            if st.button("Generate LIVE insights", type="primary") or st.session_state.get("live_insights_auto"):
                with st.spinner("Analyzing buffer..."):
                    lines = live_auto_insights(df, engine=engine)
                    st.session_state.live_insight_lines = lines
                    st.session_state.dashboard_insights = list(
                        dict.fromkeys((st.session_state.get("dashboard_insights") or []) + lines[:4])
                    )
            for line in st.session_state.get("live_insight_lines") or live_auto_insights(df, engine=engine):
                st.markdown(f"- {line}")
            st.divider()
            st.subheader("Quick actions")
            if st.button("Detect field on live buffer"):
                with st.spinner("Detecting..."):
                    meta = detect_field(df, use_gemini=bool(get_gemini_api_key()), optuna_trials=FIELD_DETECT_DEFAULT_TRIALS)
                    apply_detected_domain(meta)
                    ensure_llama_index(df, force=True)
                st.success(f"Field → {meta['label']} ({meta['confidence']})")
            if st.button("Pin latest trend to Dashboard"):
                tcol = _col(df, "temperature", "vibration", "rul")
                if tcol:
                    charts = list(st.session_state.get("dashboard_charts") or [])
                    charts.append(
                        {
                            "title": f"LIVE {tcol} trend",
                            "chart_type": "line",
                            "lib": "plotly",
                            "x": "timestamp" if "timestamp" in df.columns else tcol,
                            "y": tcol,
                            "color": None,
                            "insight": f"Pinned from LIVE buffer ({len(df)} rows)",
                        }
                    )
                    st.session_state.dashboard_charts = charts
                    st.success("Pinned")
            if st.button("Run best live model (Field card)"):
                dom = st.session_state.get("domain") or "predictive_maintenance"
                try:
                    card = field_best_model_card(df, dom)
                except Exception as exc:
                    card = {"ok": False, "error": str(exc)}
                if card.get("ok"):
                    st.session_state.ml_result = card["result"]
                    if card.get("note"):
                        st.warning(card["note"])
                    for a in card.get("actions") or []:
                        st.markdown(f"- {a}")
                else:
                    st.warning(card.get("error"))
        else:
            st.caption("Poll plant first to unlock insights / Prophet / PySpark.")

    if auto and ui_type != "buffer_only":
        time.sleep(0.05)
        st.caption("Auto-poll armed — interact or refresh to pull next sample (Streamlit-safe, no blocking loop).")


def page_upload() -> None:
    st.header("Upload")
    st.caption(
        "MANUAL mode: choose cleaning engine (pandas / polars / pyspark) by size suggestion — never forced. "
        "LIVE mode ignores upload and uses Modbus SCADA buffer."
    )

    if st.session_state.mode == "LIVE CONNECT":
        page_live_console()
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
            reset_domain_pick_for_new_frame(df)
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

    detect = render_detection_ui(df, context="upload")
    chosen_domain = render_domain_selector(context="upload")
    if chosen_domain == "plant_oee":
        render_industry_banner(df)
    render_mapping_ui(df, context="upload")
    render_domain_hints(chosen_domain)

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

    with st.expander("Preview data (first 50 rows)"):
        st.caption(
            "Pipeline: **Detect → Map → Ask** (LlamaIndex/Gemini) or **Train** "
            "(Optuna on ML page). LlamaIndex Q&A and Optuna ML are separate steps."
        )
        if isinstance(detect, dict):
            show_gemini_issue(detect.get("gemini_error"))
        st.dataframe(df.head(50), use_container_width=True)


def page_data_integration() -> None:
    st.header("Data Integration")
    st.caption(
        "SQL-style joins across 2 or 3+ tables (INNER / LEFT / RIGHT / OUTER). "
        "The join result becomes the MANUAL working dataframe (`clean_df` → `get_data()`). "
        "LIVE CONNECT still reads the SCADA buffer."
    )
    if st.session_state.get("mode") == "LIVE CONNECT":
        st.info(
            "You are in **LIVE CONNECT**. Joins can use `live_buffer` as a table, but applying a join "
            "only updates MANUAL `clean_df`. Switch to **MANUAL UPLOAD** to analyze the joined frame."
        )

    tables = join_table_registry()
    extra = st.file_uploader(
        "Upload extra tables to join",
        type=["csv", "tsv", "xlsx", "xls", "xlsm", "json", "parquet"],
        accept_multiple_files=True,
        key="join_uploader",
    )
    if extra:
        for uf in extra:
            try:
                loaded = load_tabular_file(uf)
                if not isinstance(loaded, pd.DataFrame):
                    raise TypeError("file did not load as a DataFrame")
                stem = Path(uf.name).stem.replace(" ", "_") or "table"
                tables[stem] = loaded
            except Exception as exc:
                st.error(f"{uf.name}: {exc}")
        st.session_state.uploaded_tables = {
            **(st.session_state.get("uploaded_tables") or {}),
            **tables,
        }

    if len(tables) < 2:
        st.warning("Need at least 2 tables — upload more files or load pipeline data first.")
        return

    names = list(tables.keys())
    st.write("Tables:", ", ".join(f"`{n}` ({len(tables[n]):,} rows)" for n in names))
    c1, c2, c3 = st.columns(3)
    left = c1.selectbox("Left", names, key="f_left")
    right_opts = [n for n in names if n != left]
    if not right_opts:
        st.warning("Pick a different left table so a right table remains.")
        return
    right = c2.selectbox("Right", right_opts, key="f_right")
    how = c3.selectbox("Join type", list(JOIN_TYPES.keys()), format_func=lambda k: JOIN_TYPES[k], key="f_how")
    suggested = suggest_join_keys(tables[left], tables[right])
    keys = st.multiselect(
        "Join keys",
        suggested,
        default=suggested[:1],
        key="f_keys",
    )
    if st.button("Run join", type="primary", key="f_join"):
        try:
            merged, meta = join_two(tables[left], tables[right], how=how, on=keys or None)
            apply_joined_as_working(merged, tables, [meta])
            st.success(
                f"Joined → {len(merged):,} rows × {merged.shape[1]} cols "
                f"(MANUAL working set via clean_df)"
            )
            st.json(meta)
            st.dataframe(merged.head(30), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    if len(names) >= 3:
        st.subheader("Chain 3+ tables")
        r1 = st.selectbox("Join #1 right", [n for n in names if n != left], key="f_r1")
        r2_opts = [n for n in names if n not in (left, r1)]
        if not r2_opts:
            st.caption("Need a third distinct table for a chain.")
            return
        r2 = st.selectbox("Join #2 right", r2_opts, key="f_r2")
        how2 = st.selectbox("Join #2 type", list(JOIN_TYPES.keys()), format_func=lambda k: JOIN_TYPES[k], key="f_how2")
        k1 = suggest_join_keys(tables[left], tables[r1])[:1]
        if st.button("Run join chain", key="f_chain"):
            try:
                merged, logs = join_many(
                    tables,
                    [
                        {"left": left, "right": r1, "how": how, "on": k1 or None},
                        {"left": "_result", "right": r2, "how": how2},
                    ],
                )
                apply_joined_as_working(merged, tables, logs)
                st.success(f"Chain → {len(merged):,} rows × {merged.shape[1]} cols")
                st.json(logs)
                st.dataframe(merged.head(30), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    log = st.session_state.get("join_log")
    if log is not None:
        with st.expander("Last join log"):
            st.json(log)


def page_dwdm_sql() -> None:
    st.header("DWDM & SQL Lab")
    st.caption("Data warehousing / mining concepts + read-only SQL over registered tables.")
    st.dataframe(pd.DataFrame(DWDM_CONCEPTS), use_container_width=True)

    working: Optional[pd.DataFrame] = None
    if st.session_state.get("mode") == "LIVE CONNECT":
        buf = read_live_buffer()
        if _is_nonempty_frame(buf):
            working = buf
    if working is None:
        clean = st.session_state.get("clean_df")
        manual = st.session_state.get("manual_df")
        if _is_nonempty_frame(clean):
            working = clean
        elif _is_nonempty_frame(manual):
            working = manual

    if working is None:
        st.warning("Upload, clean, or connect LIVE first.")
        return

    nums = working.select_dtypes(include="number").columns.tolist()
    c1, c2, c3 = st.columns(3)
    bins = c1.multiselect("Bin columns", nums, default=nums[:1], key="dwdm_bin")
    smooth = c2.multiselect("Smooth columns", nums, default=nums[:1] if nums else [], key="dwdm_smooth")
    norm = c3.multiselect("Z-normalize", nums, key="dwdm_norm")
    if st.button("Apply DWDM transforms", key="dwdm_apply"):
        out, log = apply_dwdm_transforms(working, bin_cols=bins, smooth_cols=smooth, normalize_cols=norm)
        if st.session_state.get("mode") != "LIVE CONNECT":
            st.session_state.clean_df = out
            st.session_state.prefer_clean_df = True
        st.success("; ".join(log) or "done")
        st.dataframe(out.head(20), use_container_width=True)

    tables = join_table_registry()
    if _is_nonempty_frame(working):
        tables.setdefault("working", working)
    examples = default_sql_examples(list(tables.keys()))
    st.subheader("SQL Lab")
    st.code("\n\n".join(examples), language="sql")
    q = st.text_area("SQL", value=examples[0], height=120, key="forge_sql")
    if st.button("Run SQL", type="primary", key="forge_sql_run"):
        try:
            result, eng = run_sql(q, tables)
            st.session_state.sql_lab_result = result
            st.session_state.sql_lab_engine = eng
            st.session_state.sql_lab_query = q
        except Exception as exc:
            st.session_state.sql_lab_result = None
            st.error(str(exc))

    sql_result = st.session_state.get("sql_lab_result")
    if isinstance(sql_result, pd.DataFrame):
        st.caption(f"Engine: {st.session_state.get('sql_lab_engine')}")
        st.dataframe(sql_result, use_container_width=True)
        if st.session_state.get("mode") != "LIVE CONNECT":
            if st.button("Use SQL result as working dataframe", key="sql_as_working"):
                apply_joined_as_working(
                    sql_result,
                    tables,
                    {"sql": st.session_state.get("sql_lab_query"), "engine": st.session_state.get("sql_lab_engine")},
                )
                st.success(f"Working set → {len(sql_result):,} rows")


def _labs_working_df() -> Optional[pd.DataFrame]:
    if st.session_state.get("mode") == "LIVE CONNECT":
        buf = read_live_buffer()
        if _is_nonempty_frame(buf):
            return buf
    clean = st.session_state.get("clean_df")
    manual = st.session_state.get("manual_df")
    if _is_nonempty_frame(clean):
        return clean
    if _is_nonempty_frame(manual):
        return manual
    return None


def page_dwdm_labs() -> None:
    st.header("DWDM labs")
    st.caption(
        "Optional labs on the current working dataframe. "
        "Leaves **DWDM & SQL** (DuckDB / transforms) unchanged."
    )
    working = _labs_working_df()
    if working is None:
        st.warning("Upload, clean, or connect LIVE first.")
        return
    live = st.session_state.get("mode") == "LIVE CONNECT"
    nums = lab_numeric_columns(working)
    date_opts = [c for c in working.columns if "date" in str(c).lower() or "time" in str(c).lower()]
    for c in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[c]) and str(c) not in date_opts:
            date_opts.append(str(c))

    tab_star, tab_apr, tab_km, tab_mice = st.tabs(["Star / OLAP", "Apriori", "K-means", "MICE"])

    with tab_star:
        st.subheader("Star / OLAP-style")
        st.caption("Pick a date dim, entity dim (student / asset / …), and numeric facts. Pandas grain — not a cube server.")
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date dimension", ["(none)"] + date_opts, key="lab_star_date")
        entity_col = c2.selectbox("Entity dimension", ["(none)"] + list(map(str, working.columns)), key="lab_star_ent")
        facts = st.multiselect("Fact numeric columns", nums, default=nums[:2], key="lab_star_facts")
        if st.button("Build star tables", type="primary", key="lab_star_go"):
            pack = build_star_schema(
                working,
                date_col=None if date_col == "(none)" else date_col,
                entity_col=None if entity_col == "(none)" else entity_col,
                fact_cols=facts,
            )
            if not pack.get("ok"):
                st.warning(pack.get("error") or "Could not build star.")
            else:
                st.caption(pack.get("caption") or "")
                st.write("**Fact**")
                st.dataframe(pack["fact"].head(40), use_container_width=True)
                for name, dim in (pack.get("dims") or {}).items():
                    st.write(f"**{name}**")
                    st.dataframe(dim.head(40), use_container_width=True)

    with tab_apr:
        st.subheader("Apriori")
        txn = st.selectbox("Transaction id", ["(none)"] + list(map(str, working.columns)), key="lab_apr_txn")
        item = st.selectbox("Item column", ["(none)"] + list(map(str, working.columns)), key="lab_apr_item")
        row_bins = st.checkbox("Row-as-basket (high/low bins) — lab, not market-basket", value=False, key="lab_apr_row")
        bin_cols = st.multiselect("Numeric items (row-as-basket)", nums, default=nums[:3], key="lab_apr_bins")
        s1, s2 = st.columns(2)
        min_sup = s1.slider("Min support", 0.05, 0.5, 0.15, 0.05, key="lab_apr_sup")
        min_conf = s2.slider("Min confidence", 0.2, 0.9, 0.5, 0.05, key="lab_apr_conf")
        if st.button("Mine itemsets", type="primary", key="lab_apr_go"):
            mode = "row"
            if txn != "(none)" and item != "(none)":
                baskets = baskets_from_txn(working, txn, item)
                mode = "txn"
            elif row_bins:
                baskets = baskets_row_bins(working, bin_cols)
                st.warning("Row-as-basket is a lab, not market-basket.")
            else:
                baskets = []
                st.info(apriori_need_txn_hint())
            if baskets:
                mined = mine_apriori(baskets, min_support=min_sup, min_confidence=min_conf)
                if not mined.get("ok"):
                    st.info(mined.get("hint") or mined.get("error") or "Need transaction shape.")
                else:
                    st.caption(f"{mined.get('n_baskets')} baskets · {mined.get('n_rules')} rules · mode={mode}")
                    st.dataframe(mined["rules"], use_container_width=True)

    with tab_km:
        st.subheader("K-means clustering")
        k = st.slider("k", 2, 12, 3, key="lab_km_k")
        km_cols = st.multiselect("Numeric columns", nums, default=nums[: min(4, len(nums))], key="lab_km_cols")
        want_sil = st.checkbox("Silhouette score", value=True, key="lab_km_sil")
        if st.button("Run K-means", type="primary", key="lab_km_go"):
            packed = assign_kmeans(working, km_cols, k=k, silhouette=want_sil)
            if not packed.get("ok"):
                st.warning(packed.get("error") or "K-means failed.")
            else:
                st.session_state._lab_kmeans = packed
                bits = [f"assigned {packed.get('n_assigned')} rows"]
                if packed.get("silhouette") is not None:
                    bits.append(f"silhouette {packed['silhouette']}")
                st.success(" · ".join(bits))
                st.dataframe(packed["frame"][km_cols + ["cluster_id"]].head(30), use_container_width=True)
        packed = st.session_state.get("_lab_kmeans")
        if isinstance(packed, dict) and packed.get("ok") and not live:
            if st.button("Apply cluster_id to working df", key="lab_km_apply"):
                st.session_state.clean_df = packed["frame"]
                st.session_state.prefer_clean_df = True
                st.success("cluster_id written to working dataframe.")

    with tab_mice:
        st.subheader("MICE (IterativeImputer)")
        st.caption("Numeric columns only. Opt-in lab — not default Clean. Preview before apply.")
        mice_cols = st.multiselect("Columns to impute", nums, default=nums[: min(6, len(nums))], key="lab_mice_cols")
        iters = st.slider("Max iterations", 2, 20, 8, key="lab_mice_iter")
        if len(working) > 20000:
            st.warning("Large frame — IterativeImputer can be slow.")
        if st.button("Preview MICE", type="primary", key="lab_mice_go"):
            packed = mice_impute(working, mice_cols, max_iter=iters)
            st.session_state._lab_mice = packed
        packed = st.session_state.get("_lab_mice")
        if isinstance(packed, dict):
            if packed.get("warning"):
                st.warning(packed["warning"])
            if not packed.get("ok"):
                st.warning(packed.get("error") or "MICE failed.")
            else:
                st.caption(f"Imputed cells: {packed.get('n_imputed', 0)}")
                st.dataframe(packed.get("preview"), use_container_width=True)
                if packed.get("changed") and not live:
                    if st.button("Apply imputed values to working df", key="lab_mice_apply"):
                        st.session_state.clean_df = packed["frame"]
                        st.session_state.prefer_clean_df = True
                        st.success("MICE values written to working dataframe.")


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
        "Industry detection + LlamaIndex build-once + best predictive model (RF preferred; XGB if OpenMP OK) "
        "+ perspective alerts (bearing wear, downtime ₹ saved)."
    )
    gemini_key_ui("field")
    df = require_data()
    if df is None:
        return

    render_industry_banner(df)
    render_domain_selector(context="field")
    use_gem = st.checkbox("Use Gemini in domain ensemble", value=bool(get_gemini_api_key()))
    trials = st.slider(
        "Optuna trials for field detect",
        1,
        FIELD_DETECT_MAX_TRIALS,
        FIELD_DETECT_DEFAULT_TRIALS,
    )
    run = st.button("Detect field + build LlamaIndex + best model", type="primary")
    if run:
        with st.spinner("Domain detect → LlamaIndex → model bake-off..."):
            meta = apply_detected_domain(
                detect_field(df, use_gemini=use_gem, optuna_trials=trials)
            )
            try:
                llama_meta = ensure_llama_index(df, force=True)
            except Exception as exc:
                llama_meta = {"mode": "error", "n_docs": 0, "error": str(exc)}
            engineered = apply_domain_feature_engineering(df, meta["domain"])
            explain = field_risk_explain(engineered)
            try:
                card = field_best_model_card(df, meta["domain"])
            except Exception as exc:
                card = {
                    "ok": False,
                    "error": f"Bake-off soft-failed: {exc}",
                    "skipped": [],
                    "note": "Using domain detect only — open ML Studio and pick RandomForest.",
                }
            if card.get("ok"):
                st.session_state.ml_result = card["result"]
                st.session_state.dashboard_insights = list(
                    dict.fromkeys((st.session_state.get("dashboard_insights") or []) + card.get("actions", []))
                )
            st.session_state.field_result = {
                "meta": meta,
                "explain": explain,
                "llama": llama_meta,
                "model_card": card,
                "engineered_cols": [c for c in engineered.columns if c not in df.columns],
            }
            try:
                autosave_after_pipeline(title=f"Field · {st.session_state.get('manual_name') or meta.get('domain')}")
            except Exception:
                pass

    res = st.session_state.field_result
    if not res:
        guess = st.session_state.get("forge_detect") or st.session_state.get("domain_meta") or {}
        label = guess.get("label") or DOMAIN_CATALOG.get(st.session_state.get("domain") or "generic", {}).get("label", "Generic")
        conf = float(guess.get("confidence") or 0)
        st.info(
            f"Active pack **{label}** (`{st.session_state.get('domain')}`). "
            "Click Detect to run Optuna + Gemini ensemble (default 3 trials). "
            "Override above is kept."
        )
        if conf and conf < 0.7:
            st.caption("guess — override if wrong.")
        return
    meta, explain, card = res["meta"], res["explain"], res.get("model_card") or {}
    show_gemini_issue(meta.get("gemini_error"))
    st.subheader(f"Detected: {meta.get('label')} (`{meta.get('domain')}`)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Domain confidence", f"{float(meta.get('confidence', 0))*100:.1f}%")
    m2.metric("Risk", f"{explain.get('risk_pct')}%")
    m3.metric("LlamaIndex docs", (res.get("llama") or {}).get("n_docs", "—"))
    m4.metric("Index mode", (res.get("llama") or {}).get("mode", "—"))

    if card.get("ok"):
        st.success(
            f"Selected target **`{card.get('target')}`** · best model **{card.get('best_model')}** · "
            f"CV score **{card.get('cv_score')}**"
        )
        if card.get("note"):
            st.warning(card["note"])
        if card.get("skipped"):
            with st.expander("Skipped optional models (native libs)"):
                for line in card["skipped"]:
                    st.caption(line)
                st.caption("Mac fix: `brew install libomp` then `pip install --force-reinstall xgboost`.")
        st.dataframe(pd.DataFrame(card.get("leaderboard") or []), use_container_width=True)
        st.subheader("Perspective actions")
        for line in card.get("actions") or []:
            st.markdown(f"- {line}")
        with st.expander("Model metrics"):
            st.json(card.get("result", {}).get("metrics") or {})
    else:
        st.warning(card.get("error") or "Model card unavailable")
        if card.get("skipped"):
            st.caption("Skipped: " + " | ".join(card["skipped"][:2]))

    st.info(explain.get("explanation", ""))
    c1, c2 = st.columns(2)
    with c1:
        if isinstance(meta.get("scoreboard"), pd.DataFrame):
            st.caption("Domain scoreboard")
            st.dataframe(meta["scoreboard"], use_container_width=True)
    with c2:
        if isinstance(meta.get("vote_table"), pd.DataFrame):
            st.caption("Ensemble votes")
            st.dataframe(meta["vote_table"], use_container_width=True)
    render_kpi_boxes(get_kpis(df))


def page_kpis() -> None:
    st.header("Auto KPIs")
    st.caption("Domain-specific square KPI boxes + loc/date/people filters + model comparisons (loc↔loc, time↔time).")
    df = require_data()
    if df is None:
        return
    if st.session_state.get("domain_user_override"):
        meta = st.session_state.get("domain_meta") or {"domain": st.session_state.get("domain"), "confidence": 1.0, "overridden": True}
    elif not st.session_state.get("domain") or (st.session_state.domain == "generic" and not st.session_state.get("domain_meta")):
        with st.spinner("Detecting field for KPI pack..."):
            meta = apply_detected_domain(
                detect_field(df, use_gemini=bool(get_gemini_api_key()), optuna_trials=FIELD_DETECT_DEFAULT_TRIALS)
            )
    else:
        meta = st.session_state.domain_meta or {"domain": st.session_state.get("domain"), "confidence": 0}

    st.write(f"Active field: **{DOMAIN_CATALOG.get(st.session_state.domain, {}).get('label')}** "
             f"(confidence {float(meta.get('confidence', 0))*100:.1f}%)")
    show_gemini_issue(meta.get("gemini_error") if isinstance(meta, dict) else None)
    render_industry_banner(df)

    filtered = render_filter_bar(df, key_prefix="kpi")
    kpis = get_kpis(filtered)
    render_kpi_boxes(kpis, per_row=4)

    impact = render_dollar_impact(filtered, key_prefix="kpi")
    field_actions = []
    if st.session_state.get("field_result"):
        field_actions = list((st.session_state.field_result.get("model_card") or {}).get("actions") or [])
    brief = render_manager_brief(
        insights=st.session_state.get("dashboard_insights") or [],
        quality_checks=st.session_state.get("clean_checks"),
        ml_result=st.session_state.get("ml_result"),
        dollar_impact=impact,
        field_actions=field_actions,
        key_prefix="kpi",
    )
    domain_label = DOMAIN_CATALOG.get(st.session_state.get("domain") or "generic", {}).get(
        "label", st.session_state.get("domain") or "generic"
    )
    roles = dict(st.session_state.get("column_roles") or {})
    forge_domain = str(st.session_state.get("forge_domain") or st.session_state.get("domain") or "generic")
    export_pack = assemble_dashboard_export(
        filtered,
        kpis=kpis,
        insights=st.session_state.get("dashboard_insights") or [],
        actions=brief.get("actions") or [],
        briefing=brief.get("body") or "",
        domain=domain_label,
        chart_domain=forge_domain,
        source_name=str(st.session_state.get("manual_name") or "forge.csv"),
        roles=roles,
        pins=st.session_state.get("dashboard_charts") or [],
    )

    st.markdown("##### Share KPIs / insights only")
    if EMAIL_USER and EMAIL_PASSWORD and brief.get("body"):
        if st.button("Email this brief", key="kpi_email_brief"):
            try:
                html = build_html_report(
                    domain=st.session_state.get("domain") or "generic",
                    source_name=str(st.session_state.get("manual_name") or "forge.csv"),
                    kpis=kpis,
                    insights=brief.get("actions") or [],
                    ml_result=st.session_state.get("ml_result"),
                    briefing=brief.get("body") or "",
                )
                msg = send_forge_email_report(
                    OPERATOR_EMAIL,
                    f"[Analytics Forge v2] Top 3 actions",
                    brief.get("body") or "",
                    df=filtered,
                    html_report=html,
                )
                st.success(msg)
            except Exception as exc:
                st.error(str(exc))
    else:
        st.caption("Set EMAIL_USER + EMAIL_PASSWORD to email Top 3 only. Full dashboard export below works without SMTP.")

    def _email_full_kpi(to: str, body: str, html: str, kpi_csv: bytes) -> str:
        return send_full_dashboard_email(
            to,
            subject=f"[Analytics Forge v2] Dashboard — {domain_label}",
            body=body,
            html_report=html,
            kpi_csv=kpi_csv,
            df=filtered,
        )

    render_export_controls(
        html_report=export_pack["html"],
        kpi_csv=export_pack["kpi_csv"],
        email_body=export_pack["body"],
        smtp_ok=bool(EMAIL_USER and EMAIL_PASSWORD),
        default_to=OPERATOR_EMAIL,
        key_prefix="kpi",
        send_fn=_email_full_kpi,
    )

    comparisons = kpi_group_comparisons(filtered)
    st.subheader("Comparisons")
    insight = kpi_model_insight(filtered, comparisons)
    st.info(insight)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Location → location averages**")
        if "loc_to_loc" in comparisons:
            st.dataframe(comparisons["loc_to_loc"], use_container_width=True)
        else:
            st.caption("No location column detected.")
    with c2:
        st.markdown("**Time → time averages**")
        if "time_to_time" in comparisons:
            st.dataframe(comparisons["time_to_time"], use_container_width=True)
        else:
            st.caption("No date/time column detected.")

    explain = field_risk_explain(filtered)
    st.subheader("Model briefing")
    st.write(explain["explanation"])



def page_charts() -> None:
    st.header("Charts")
    st.caption("Adaptive charts — pin any view to Dashboard (Power BI style).")
    df0 = require_data()
    if df0 is None:
        return
    if st.session_state.get("domain_user_override"):
        pass
    elif not st.session_state.get("domain") or st.session_state.domain == "generic":
        meta = apply_detected_domain(
            detect_field(df0, use_gemini=False, optuna_trials=FIELD_DETECT_DEFAULT_TRIALS)
        )

    filtered = render_filter_bar(df0, key_prefix="chart")
    if filtered.empty:
        st.warning("Filters removed all rows.")
        return

    cols = list(filtered.columns)
    num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in cols if c not in num_cols]
    default_y = num_cols[0] if num_cols else cols[0]
    default_x = cat_cols[0] if cat_cols else cols[0]
    for pref_y in ("revenue", "sales", "temperature", "vibration", "churn", "load"):
        hit = _col(filtered, pref_y)
        if hit:
            default_y = hit
            break
    for pref_x in ("location", "region", "machine_id", "date", "timestamp", "store"):
        hit = _col(filtered, pref_x)
        if hit:
            default_x = hit
            break

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        library = st.selectbox("Library", ["plotly", "seaborn", "matplotlib"], index=0)
    with c2:
        chart_type = st.selectbox("Chart type", ["bar", "line", "scatter", "pie", "heatmap"], index=0)
    with c3:
        x = st.selectbox("X axis", cols, index=cols.index(default_x) if default_x in cols else 0)
    with c4:
        y_opts = cols if chart_type == "pie" else (num_cols or cols)
        y = st.selectbox("Y axis", y_opts, index=y_opts.index(default_y) if default_y in y_opts else 0)
    color = st.selectbox("Color / group (optional)", ["(none)"] + cat_cols, index=0)
    color_col = None if color == "(none)" else color

    plot_df = filtered
    if chart_type in ("bar", "pie") and x in filtered.columns and y in filtered.columns:
        if filtered[x].nunique() > 30 and not pd.api.types.is_numeric_dtype(filtered[x]):
            plot_df = filtered.groupby(x, dropna=False)[y].mean().reset_index()

    render_adaptive_chart(plot_df, x, y, chart_type, library, color_col)
    insight = chart_business_insight(filtered, x, y)
    st.success(insight)

    title = f"{chart_type}: {y} by {x}"
    if st.button("📌 Add to Dashboard", type="primary"):
        entry = {
            "title": title,
            "chart_type": chart_type,
            "lib": library,
            "x": x,
            "y": y,
            "color": color_col,
            "insight": insight,
        }
        charts = list(st.session_state.get("dashboard_charts") or [])
        charts.append(entry)
        st.session_state.dashboard_charts = charts
        insights = list(st.session_state.get("dashboard_insights") or [])
        if insight not in insights:
            insights.append(insight)
        st.session_state.dashboard_insights = insights
        st.success(f"Pinned to Dashboard ({len(charts)} charts)")


def page_ml() -> None:
    st.header("ML Studio")
    st.caption(
        "Forge Analytics catalog — choose model + target. "
        "Classifiers · Regressors · XGBoost (if OpenMP OK) · Prophet · PCA · StatsmodelsOLS · IsolationForest."
    )
    df = require_data()
    if df is None:
        return
    domain = st.session_state.get("domain")
    if not domain:
        domain = detect_field(df, use_gemini=False, optuna_trials=FIELD_DETECT_DEFAULT_TRIALS)["domain"]
        if not st.session_state.get("domain_user_override"):
            st.session_state.domain = domain
    st.session_state.domain = st.session_state.get("domain") or domain
    runnable = set(list_runnable_models())
    hidden = [m for m in FORGE_MODEL_CATALOG if m not in runnable]
    if hidden:
        st.caption(
            "Hidden (native lib missing on this machine): "
            + ", ".join(hidden)
            + " — Mac: `brew install libomp` then reinstall xgboost."
        )
    recommended = [m for m in DOMAIN_RECOMMENDED_MODELS.get(domain, []) if m in runnable]
    model_ids = recommended + [m for m in runnable if m not in recommended]
    if not model_ids:
        st.error("No runnable models found. Check sklearn install.")
        return
    model_id = st.selectbox(
        "Model",
        model_ids,
        format_func=lambda m: f"{m} [{FORGE_MODEL_CATALOG[m]['task']}]" + (" ★" if m in recommended else ""),
    )
    meta = FORGE_MODEL_CATALOG[model_id]
    st.info(f"**What:** {meta.get('what')}\n\n**Why:** {meta.get('why')}\n\nLibrary: `{meta.get('library')}`")

    cols = list(df.columns)
    auto_tgt = domain_default_target(df, domain)
    default_i = cols.index(auto_tgt) + 1 if auto_tgt in cols else 0
    target_sel = st.selectbox("Target (the number/label to predict)", ["(auto)"] + cols, index=default_i)
    target_arg = None if target_sel == "(auto)" else target_sel
    ts_split = st.checkbox("Time-series split (last 20% holdout)", value=True)
    compare = st.checkbox("Also run Optuna AutoML bake-off on same target", value=False)

    if st.button("Run model", type="primary"):
        with st.spinner(f"Training {model_id}..."):
            result = run_forge_model(df, model_id, target=target_arg, time_series_split=ts_split)
            if result.get("ok") and not result.get("manager_briefing"):
                result["manager_briefing"] = build_manager_briefing(result)
            st.session_state.ml_result = result
            if result.get("ok") and result.get("manager_briefing"):
                st.session_state.dashboard_insights = list(
                    dict.fromkeys((st.session_state.get("dashboard_insights") or []) + [result["manager_briefing"]])
                )
            if compare and result.get("ok") and result.get("target"):
                try:
                    best, metrics = run_automl(df, result["target"], n_trials=20, time_series_split=ts_split)
                    st.session_state.automl_result = {"best_model": best, "metrics": metrics}
                except Exception as exc:
                    st.session_state.automl_result = {"error": str(exc)}

    result = st.session_state.get("ml_result")
    if result:
        if result.get("ok"):
            st.success(f"Finished: **{result.get('model_id')}** on `{result.get('target')}`")
            metrics = result.get("metrics") or {}
            cols_m = st.columns(min(4, max(1, len(metrics))))
            for i, (k, v) in enumerate(list(metrics.items())[:4]):
                cols_m[i].metric(str(k), v)
            if result.get("manager_briefing"):
                st.markdown(result["manager_briefing"])
            preview = result.get("predictions_preview")
            if preview is not None:
                st.subheader("Predictions preview")
                st.dataframe(preview, use_container_width=True)
            if result.get("summary"):
                with st.expander("Statsmodels summary"):
                    st.text(result["summary"])
        else:
            st.error(result.get("error", "Failed"))

    if st.session_state.get("automl_result"):
        st.subheader("Optuna AutoML bake-off")
        st.json(st.session_state.automl_result)


def page_ask() -> None:
    st.header("Ask / AI")
    st.caption("LlamaIndex search on your CSV (offline) + Gemini grounded answer. Index builds once per upload (Field).")
    gem_ok = bool(get_gemini_api_key())
    c1, c2, c3 = st.columns(3)
    c1.metric("Gemini", "Ready" if gem_ok else "No key")
    c2.metric("LlamaIndex", "Ready" if st.session_state.get("llama_docs") else "Build in Field")
    c3.metric("Offline search", "Always on")
    gemini_key_ui("ask")

    df = require_data()
    if df is None:
        return

    if st.button("Rebuild LlamaIndex now"):
        with st.spinner("Indexing rows..."):
            meta = ensure_llama_index(df, force=True)
        st.success(f"Indexed {meta.get('n_docs')} docs · mode={meta.get('mode')}")
    elif not st.session_state.get("llama_docs"):
        ensure_llama_index(df, force=False)
        st.caption(f"Auto-indexed {(st.session_state.get('llama_index_meta') or {}).get('n_docs')} rows")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hits"):
                with st.expander("Retrieved rows"):
                    for h in msg["hits"][:4]:
                        st.code(h.get("text", "")[:400])

    q = st.chat_input("e.g. Why did machine 3 fail? What is avg vibration?")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("LlamaIndex search → Gemini answer..."):
                out = ask_llama_gemini(q, df)
            st.markdown(out["answer"])
            if out.get("gemini_error"):
                show_gemini_issue(out.get("gemini_error"))
            elif out.get("used_offline_fallback") and gem_ok:
                st.warning("Gemini did not return an answer — showing the offline Llama/keyword fallback.")
            with st.expander("Search hits (LlamaIndex / keyword)"):
                for h in out.get("hits") or []:
                    st.write(f"[{h.get('source')} · score={h.get('score')}] {h.get('text')[:350]}")
            if out.get("gemini_answer") and out.get("offline_answer") and out["gemini_answer"] != out["answer"]:
                st.caption("Offline fallback was also computed.")
        st.session_state.chat_history.append(
            {"role": "assistant", "content": out["answer"], "hits": out.get("hits")}
        )


def page_dashboard() -> None:
    st.header("Dashboard")
    st.caption(
        "Power BI / Tableau style — filters change the whole board. "
        "**Core** (4) + **Extended** (5) charts plus pinned views from Charts."
    )
    df0 = require_data()
    if df0 is None:
        return
    src, src_label = dashboard_source_frame(df0)
    if not st.session_state.get("domain") and not st.session_state.get("domain_user_override"):
        apply_detected_domain(detect_field(src, use_gemini=False, optuna_trials=FIELD_DETECT_DEFAULT_TRIALS))

    filtered = render_filter_bar(src, key_prefix="dash")
    if filtered.empty:
        st.warning("Filters removed all rows — clear location/people filters.")
        return
    if src_label != "working":
        st.caption(f"Charts use the **{src_label}** table from this session.")

    roles = dict(st.session_state.get("column_roles") or {})
    forge_domain = str(st.session_state.get("forge_domain") or st.session_state.get("domain") or "generic")
    domain_label = DOMAIN_CATALOG.get(st.session_state.get("domain") or "generic", {}).get(
        "label", st.session_state.get("domain") or "generic"
    )

    kpis_all = get_kpis(filtered)
    kpi_keys = [k for k in kpis_all.keys() if k != "Domain"]
    pick = st.multiselect("KPIs to show", kpi_keys, default=kpi_keys[:8], key="dash_kpi_pick")
    render_kpi_boxes({k: kpis_all[k] for k in pick} | {"Domain": kpis_all.get("Domain")}, per_row=4)

    left, right = st.columns([3, 1])
    with right:
        st.subheader("Insights")
        st.caption("SaaS login + Monday scheduled email = next phase.")
        for insight in st.session_state.get("dashboard_insights") or []:
            st.markdown(f"- {insight}")
        ml = st.session_state.get("ml_result")
        if ml and ml.get("ok"):
            st.caption(f"Last ML: {ml.get('model_id')} · {ml.get('metrics')}")
        risk = field_predict(filtered if len(filtered) >= 10 else src)
        st.metric("Live risk", f"{risk}%")
        brief = render_manager_brief(
            insights=st.session_state.get("dashboard_insights") or [],
            quality_checks=st.session_state.get("clean_checks"),
            ml_result=st.session_state.get("ml_result"),
            dollar_impact=None,
            field_actions=list(
                ((st.session_state.get("field_result") or {}).get("model_card") or {}).get("actions") or []
            ),
            key_prefix="dash",
        )

    with left:
        core_specs = render_core_charts(filtered, roles=roles, domain=forge_domain)

    extended_specs = render_extended_charts(filtered, roles=roles, domain=forge_domain)

    charts = list(st.session_state.get("dashboard_charts") or [])
    st.subheader("Pinned charts")
    if not charts:
        st.info("Go to **Charts**, build a view, click **Add to Dashboard**.")
    for i, meta in enumerate(charts):
        st.markdown(f"**{meta.get('title')}**")
        try:
            render_adaptive_chart(
                filtered,
                meta.get("x"),
                meta.get("y"),
                meta.get("chart_type", "bar"),
                meta.get("lib", "plotly"),
                meta.get("color"),
            )
        except Exception as exc:
            st.warning(f"Chart {i+1} failed on filtered data: {exc}")
        if meta.get("insight"):
            st.caption(meta["insight"])
        if st.button(f"Remove chart {i+1}", key=f"rm_chart_{i}"):
            charts.pop(i)
            st.session_state.dashboard_charts = charts
            st.rerun()

    export_pack = assemble_dashboard_export(
        filtered,
        kpis=kpis_all,
        insights=st.session_state.get("dashboard_insights") or [],
        actions=brief.get("actions") or [],
        briefing=brief.get("body") or "",
        domain=domain_label,
        chart_domain=forge_domain,
        source_name=str(st.session_state.get("manual_name") or "forge.csv"),
        roles=roles,
        pins=charts,
        core_specs=core_specs,
        extended_specs=extended_specs,
    )

    def _email_full(to: str, body: str, html: str, kpi_csv: bytes) -> str:
        return send_full_dashboard_email(
            to,
            subject=f"[Analytics Forge v2] Dashboard — {domain_label}",
            body=body,
            html_report=html,
            kpi_csv=kpi_csv,
            df=filtered,
        )

    st.divider()
    render_export_controls(
        html_report=export_pack["html"],
        kpi_csv=export_pack["kpi_csv"],
        email_body=export_pack["body"],
        smtp_ok=bool(EMAIL_USER and EMAIL_PASSWORD),
        default_to=OPERATOR_EMAIL,
        key_prefix="dash",
        send_fn=_email_full,
    )

    if st.button("Clear pinned dashboard charts"):
        st.session_state.dashboard_charts = []
        st.rerun()


def page_email() -> None:
    st.header("Email")
    st.caption(
        "Forge Analytics style — send **full dashboard** HTML (KPIs + insights + charts) + CSV, "
        "or use Auto KPIs for Top-3-only email."
    )
    status_ok = bool(EMAIL_USER and EMAIL_PASSWORD)
    if status_ok:
        st.success(f"SMTP ready · {EMAIL_SMTP_HOST}:{EMAIL_SMTP_PORT} · from {EMAIL_FROM or EMAIL_USER}")
    else:
        st.warning(
            "Set `EMAIL_USER` + `EMAIL_PASSWORD` (Gmail App Password) in `.env`.\n\n"
            "```\nEMAIL_USER=you@gmail.com\nEMAIL_PASSWORD=xxxx xxxx xxxx xxxx\nEMAIL_FROM=you@gmail.com\n```"
        )

    df = None
    try:
        df = get_data()
    except Exception:
        st.info("Load data (Upload / LIVE) to attach CSV + KPI report.")

    domain = st.session_state.get("domain") or "generic"
    domain_label = DOMAIN_CATALOG.get(domain, {}).get("label", domain)
    kpis = get_kpis(df) if df is not None else {}
    insights = st.session_state.get("dashboard_insights") or []
    ml = st.session_state.get("ml_result")
    cached_brief = st.session_state.get("kpi_manager_brief") or st.session_state.get("dash_manager_brief") or {}
    actions = list(cached_brief.get("actions") or [])
    briefing = str(cached_brief.get("body") or "")
    if not briefing:
        if st.session_state.get("field_result") and st.session_state.field_result.get("model_card", {}).get("actions"):
            briefing = " | ".join(st.session_state.field_result["model_card"]["actions"][:3])
        elif ml and ml.get("manager_briefing"):
            briefing = ml["manager_briefing"]
        else:
            briefing = f"Analytics Forge report for {domain_label}."

    export_pack = None
    if df is not None:
        export_pack = assemble_dashboard_export(
            df,
            kpis=kpis,
            insights=insights,
            actions=actions,
            briefing=briefing,
            domain=domain_label,
            chart_domain=str(st.session_state.get("forge_domain") or domain),
            source_name=str(st.session_state.get("manual_name") or "live.csv"),
            roles=dict(st.session_state.get("column_roles") or {}),
            pins=st.session_state.get("dashboard_charts") or [],
        )

    with st.form("email_send_form"):
        to_addr = st.text_input("Recipient", value=OPERATOR_EMAIL)
        subject = st.text_input("Subject", value=f"[Analytics Forge v2] Dashboard — {domain_label}")
        note = st.text_area(
            "Extra note",
            value="Attached: forge-dashboard-report.html (KPIs + insights + charts) + KPI CSV + data CSV.",
        )
        send_clicked = st.form_submit_button("Email full report", type="primary")
        if send_clicked:
            if not to_addr.strip():
                st.error("Enter recipient email.")
            elif df is None or export_pack is None:
                st.error("No data loaded.")
            else:
                try:
                    body = note + "\n\n" + export_pack["body"]
                    msg = send_full_dashboard_email(
                        to_addr.strip(),
                        subject=subject,
                        body=body,
                        html_report=export_pack["html"],
                        kpi_csv=export_pack["kpi_csv"],
                        df=df,
                    )
                    st.success(msg)
                except Exception as exc:
                    st.error(str(exc))

    if export_pack:
        render_export_controls(
            html_report=export_pack["html"],
            kpi_csv=export_pack["kpi_csv"],
            email_body=export_pack["body"],
            smtp_ok=bool(EMAIL_USER and EMAIL_PASSWORD),
            default_to=OPERATOR_EMAIL,
            key_prefix="email",
            send_fn=lambda to, body, html, kpi_csv: send_full_dashboard_email(
                to,
                subject=f"[Analytics Forge v2] Dashboard — {domain_label}",
                body=body,
                html_report=html,
                kpi_csv=kpi_csv,
                df=df,
            ),
        )

    st.subheader("Report preview KPIs")
    if kpis:
        render_kpi_boxes(kpis)
    st.subheader("Insights that will be emailed")
    for i in insights[:8]:
        st.markdown(f"- {i}")


PAGES = [
    "Upload",
    "Clean",
    "Data Integration",
    "DWDM & SQL",
    "DWDM labs",
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
            try:
                _lc = load_live_config()
                st.caption(
                    f"LIVE `{_lc.get('connection_type')}` · "
                    f"{_lc.get('ocp_u_ip')}:{_lc.get('ocp_u_port')} · buffer `{LIVE_CSV.name}`"
                )
            except Exception:
                st.caption(f"SCADA → `{LIVE_CSV.name}`")
            st.write(f"Link: **{st.session_state.get('live_status', 'idle')}**")
            if st.session_state.get("live_error"):
                st.warning(str(st.session_state.live_error)[:180])
        else:
            up = st.file_uploader(
                "Quick upload",
                type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
                key="sidebar_upload",
            )
            if up is not None:
                try:
                    st.session_state.manual_df = load_uploaded_file(up)
                    reset_domain_pick_for_new_frame(st.session_state.manual_df)
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
        render_session_sidebar()
        st.divider()
        if st.button("Start FORGE", type="primary"):
            st.session_state.pipeline_started = True
            try:
                autosave_after_pipeline(title=f"FORGE · {st.session_state.get('manual_name') or 'session'}")
            except Exception:
                pass
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
        try:
            _lc = load_live_config()
            st.info(
                f"MODE: LIVE CONNECT (`{_lc.get('connection_type')}`) — "
                f"OCP-U {_lc.get('ocp_u_ip')}:{_lc.get('ocp_u_port')} · "
                f"buffer `data/live.csv` · Upload page = SCADA console"
            )
        except Exception:
            st.info("MODE: LIVE CONNECT — all pages read `data/live.csv` buffer")
    else:
        name = st.session_state.manual_name or "none"
        st.info(f"MODE: MANUAL UPLOAD — all pages read uploaded file (`{name}`)")

    routers = {
        "Upload": page_upload,
        "Clean": page_clean,
        "Data Integration": page_data_integration,
        "DWDM & SQL": page_dwdm_sql,
        "DWDM labs": page_dwdm_labs,
        "Field": page_field,
        "Auto KPIs": page_kpis,
        "Charts": page_charts,
        "ML Studio": page_ml,
        "Ask / AI": page_ask,
        "Dashboard": page_dashboard,
        "Email": page_email,
    }
    try:
        handlers = routers.get(page)
        if handlers is None:
            st.error(f"Unknown page: {page}")
        else:
            handlers()
    except Exception as exc:
        st.error(f"Page `{page}` failed: {exc}")
        with st.expander("Technical details (for debugging)"):
            st.code(traceback.format_exc())
        st.info(
            "Tip: switch Mode (LIVE ↔ MANUAL), reload buffer/file, or check `config.yaml` / `.env`. "
            "Manual path does not need OCP-U."
        )


if __name__ == "__main__":
    main()
