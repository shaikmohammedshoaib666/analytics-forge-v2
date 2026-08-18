"""Forge OS domain detection + role packs — runs before column mapping.

Python 3.9 compatible. Heuristics work offline; optional Gemini boost in forge_os.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

# OS-level domains (column-mapping packs). Distinct from app.py DOMAIN_CATALOG keys.
FORGE_DOMAINS = (
    "generic",
    "sales",
    "forecasting",
    "churn",
    "predictive_maintenance",
    "plant_oee",
    "quality",
    "education",
    "health",
)

FORGE_DOMAIN_LABELS: dict[str, str] = {
    "generic": "Generic Analytics",
    "sales": "Sales / Revenue",
    "forecasting": "Forecasting / Time Series",
    "churn": "Churn / Retention",
    "predictive_maintenance": "Predictive Maintenance",
    "plant_oee": "Plant / OEE",
    "quality": "Quality / Defects",
    "education": "Education / Student",
    "health": "Health / Hospital",
}

# Bridge OS domain → app.py DOMAIN_CATALOG key for downstream KPIs / ML hints.
OS_TO_APP_DOMAIN: dict[str, str] = {
    "generic": "generic",
    "sales": "sales_forecasting",
    "forecasting": "sales_forecasting",
    "churn": "telecom_churn",
    "predictive_maintenance": "predictive_maintenance",
    "plant_oee": "generic",
    "quality": "warehouse_logistics",
    "education": "education",
    "health": "healthcare",
}

APP_TO_OS_DOMAIN: dict[str, str] = {
    "generic": "generic",
    "sales_forecasting": "sales",
    "telecom_churn": "churn",
    "predictive_maintenance": "predictive_maintenance",
    "warehouse_logistics": "quality",
    "healthcare": "health",
    "education": "education",
}

DOMAIN_OVERRIDE_LOW_CONF = 0.70

BASE_ROLES: tuple[str, ...] = ("id", "date", "category", "metric", "unused")

DOMAIN_ROLE_PACKS: dict[str, tuple[str, ...]] = {
    "generic": (),
    "sales": ("customer", "product", "region", "revenue", "quantity", "order_id"),
    "forecasting": ("target", "qty", "revenue", "seasonality_hint"),
    "churn": ("customer_id", "churn_flag", "tenure", "subscription", "support_tickets"),
    "predictive_maintenance": ("timestamp", "asset", "sensor", "failure_label", "rul_target"),
    "plant_oee": (
        "asset",
        "downtime",
        "loss",
        "scrap",
        "qty",
        "availability",
        "performance",
        "quality",
    ),
    "quality": ("scrap", "defect", "fpy", "batch", "spec_limit"),
    "education": ("student_id", "grade", "score", "course", "attendance"),
    "health": ("patient_id", "diagnosis", "bmi", "readmission"),
}

# Legacy alias — full role list for plant-heavy default (backward compat in tests).
COLUMN_ROLES: tuple[str, ...] = BASE_ROLES + DOMAIN_ROLE_PACKS["plant_oee"]


def roles_for_domain(domain: str) -> list[str]:
    """Base roles + domain pack (deduped, stable order)."""
    dom = domain if domain in FORGE_DOMAINS else "generic"
    seen: set[str] = set()
    out: list[str] = []
    for role in BASE_ROLES + DOMAIN_ROLE_PACKS.get(dom, ()):
        if role not in seen:
            seen.add(role)
            out.append(role)
    return out


def _norm_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _is_id_like(series: pd.Series, n_rows: int) -> bool:
    n = len(series.dropna())
    if n == 0:
        return False
    n_unique = series.nunique(dropna=True)
    name_hint = False  # set by caller via column name
    ratio = n_unique / max(1, n)
    if ratio > 0.85 and n_unique > 5:
        return True
    if n_unique == n and n > 3:
        return True
    return name_hint


def detect_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Per column: date | number | text | id_like | boolean."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    n_rows = len(df)
    out: dict[str, str] = {}
    for col in df.columns:
        col_str = str(col)
        n_lower = _norm_name(col_str)
        series = df[col]
        dtype = series.dtype

        if pd.api.types.is_bool_dtype(dtype):
            out[col_str] = "boolean"
            continue

        if pd.api.types.is_datetime64_any_dtype(dtype):
            out[col_str] = "date"
            continue

        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.6 and "date" in n_lower or "time" in n_lower or parsed.notna().mean() >= 0.85:
            if parsed.notna().sum() >= max(3, int(0.5 * n_rows)):
                out[col_str] = "date"
                continue

        if pd.api.types.is_numeric_dtype(dtype):
            uniq = series.dropna().nunique()
            if uniq <= 2 and uniq > 0:
                vals = set(pd.to_numeric(series, errors="coerce").dropna().unique())
                if vals <= {0, 1} or vals <= {0.0, 1.0}:
                    out[col_str] = "boolean"
                    continue
            if any(k in n_lower for k in ("id", "key", "code", "sku", "uuid")):
                out[col_str] = "id_like"
                continue
            if _is_id_like(series, n_rows) or (n_lower.endswith("_id") or n_lower == "id"):
                out[col_str] = "id_like"
                continue
            out[col_str] = "number"
            continue

        if any(k in n_lower for k in ("id", "key", "uuid", "customer", "order", "asset", "machine")):
            if _is_id_like(series, n_rows) or n_lower.endswith("_id"):
                out[col_str] = "id_like"
                continue

        out[col_str] = "text"
    return out


def _token_set(columns: list[str]) -> set[str]:
    toks: set[str] = set()
    for c in columns:
        n = _norm_name(c)
        toks.add(n)
        toks.update(p for p in n.split("_") if p)
    return toks


def _score_domain(
    dom: str,
    toks: set[str],
    col_join: str,
    column_types: dict[str, str],
    df: pd.DataFrame,
) -> tuple[float, list[str]]:
    """Heuristic score + human-readable reasons for one OS domain."""
    reasons: list[str] = []
    score = 0.0

    rules: dict[str, dict[str, Any]] = {
        "sales": {
            "strong": ("revenue", "sales", "gmv", "order_id", "sku", "customer", "units", "asp"),
            "weak": ("region", "channel", "store", "product", "discount", "quantity"),
            "type_bonus": {"number": 0.2},
        },
        "forecasting": {
            "strong": ("forecast", "target", "yhat", "seasonal", "trend", "demand"),
            "weak": ("date", "period", "week", "month", "qty", "revenue", "actual"),
            "needs_date": True,
        },
        "churn": {
            "strong": ("churn", "churned", "cancelled", "attrition", "is_churn"),
            "weak": ("tenure", "subscription", "contract", "arpu", "support", "ticket", "complaint"),
            "type_bonus": {"boolean": 0.3},
        },
        "predictive_maintenance": {
            "strong": ("vibration", "rul", "failure", "fault", "sensor", "bearing", "modbus", "opc"),
            "weak": ("temperature", "pressure", "torque", "rpm", "machine", "asset", "timestamp"),
            "numeric_ratio_min": 0.45,
        },
        "plant_oee": {
            "strong": ("oee", "availability", "downtime", "scrap", "performance"),
            "weak": ("asset", "line", "shift", "reject", "planned", "runtime", "good_units"),
        },
        "quality": {
            "strong": ("defect", "fpy", "first_pass", "reject", "ncr", "spec_limit", "batch"),
            "weak": ("scrap", "quality", "inspection", "ppm", "cpk"),
        },
        "education": {
            "strong": (
                "student", "gpa", "marks", "exam", "attendance", "assignment", "course",
                "math_score", "reading_score", "writing_score", "cgpa",
            ),
            "weak": ("grade", "school", "university", "subject", "credits"),
        },
        "health": {
            "strong": ("patient", "hospital", "bmi", "glucose", "readmission", "diagnosis", "icd", "ward", "spo2"),
            "weak": ("age", "weight", "height", "bp", "blood", "pulse"),
            "require_strong": True,
        },
    }

    if dom == "generic":
        return 0.1, ["fallback when no domain signals"]

    meta = rules.get(dom, {})
    for k in meta.get("strong", ()):
        if k in toks or k in col_join:
            score += 3.0
            reasons.append(f"column match: {k}")

    for k in meta.get("weak", ()):
        if k in toks or k in col_join:
            score += 0.6
            reasons.append(f"signal: {k}")

    if meta.get("require_strong"):
        strong_hit = any(k in toks or k in col_join for k in meta.get("strong", ()))
        if not strong_hit:
            score = min(score, 0.4)
            reasons.append("weak names only — not locked")

    if meta.get("needs_date"):
        date_cols = sum(1 for t in column_types.values() if t == "date")
        if date_cols >= 1:
            score += 2.0
            reasons.append(f"{date_cols} date column(s)")

    if meta.get("numeric_ratio_min"):
        num_ratio = sum(1 for t in column_types.values() if t == "number") / max(1, len(column_types))
        if num_ratio >= meta["numeric_ratio_min"]:
            score += 1.5
            reasons.append(f"numeric sensor ratio {num_ratio:.0%}")

    neg_rules = {
        "sales": ("vibration", "rul", "patient", "churn", "downtime", "oee", "student", "gpa"),
        "forecasting": ("vibration", "patient", "churn", "failure", "student"),
        "churn": ("vibration", "revenue", "sku", "oee", "sensor", "student", "gpa"),
        "predictive_maintenance": ("revenue", "sku", "churn", "patient", "order", "student", "gpa"),
        "plant_oee": ("churn", "patient", "revenue", "sku", "student", "gpa"),
        "quality": ("churn", "patient", "revenue", "student", "gpa"),
        "education": ("patient", "hospital", "bmi", "glucose", "readmission", "vibration", "oee", "rul"),
        "health": ("student", "gpa", "marks", "exam", "attendance", "assignment", "course", "sku", "churn"),
    }
    for k in neg_rules.get(dom, ()):
        if k in toks or k in col_join:
            score -= 2.0
            reasons.append(f"negative: {k}")

    if dom == "churn":
        for col, ctype in column_types.items():
            n = _norm_name(col)
            if "churn" in n and ctype in ("boolean", "number", "text"):
                score += 2.5
                reasons.append(f"churn column `{col}`")

    if dom == "predictive_maintenance" and df is not None and not df.empty:
        for hint, (lo, hi) in (("vibration", (0, 50)), ("temperature", (-40, 500)), ("rul", (0, 100000))):
            for col in df.columns:
                if hint in _norm_name(str(col)):
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    if len(s) >= 5:
                        med = float(s.median())
                        if lo <= med <= hi:
                            score += 1.5
                            reasons.append(f"{hint} median ~{med:.1f}")

    return max(0.0, score), reasons[:12]


def detect_domain(
    df: pd.DataFrame,
    column_types: Optional[dict[str, str]] = None,
    *,
    gemini_domain: Optional[str] = None,
    gemini_confidence: float = 0.0,
) -> dict[str, Any]:
    """Detect OS domain with confidence + reasons. Heuristics-first."""
    empty = {
        "domain": "generic",
        "label": FORGE_DOMAIN_LABELS["generic"],
        "confidence": 0.25,
        "reasons": ["no data"],
        "scores": {},
        "column_types": {},
        "app_domain": "generic",
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty

    ctypes = column_types or detect_column_types(df)
    cols = [str(c) for c in df.columns]
    toks = _token_set(cols)
    col_join = " ".join(_norm_name(c) for c in cols)

    scores: dict[str, float] = {}
    all_reasons: dict[str, list[str]] = {}
    for dom in FORGE_DOMAINS:
        if dom == "generic":
            continue
        sc, rs = _score_domain(dom, toks, col_join, ctypes, df)
        scores[dom] = sc
        all_reasons[dom] = rs

    if gemini_domain and gemini_domain in FORGE_DOMAINS:
        scores[gemini_domain] = scores.get(gemini_domain, 0.0) + 3.0 * max(0.0, min(1.0, gemini_confidence))
        all_reasons.setdefault(gemini_domain, []).append("Gemini suggestion")

    if scores and max(scores.values()) >= 2.0:
        best = max(scores, key=scores.get)
        top = scores[best]
        second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        margin = (top - second) / max(top, 1.0)
        confidence = round(min(0.97, 0.45 + 0.08 * top + 0.15 * margin), 3)
        reasons = all_reasons.get(best, [])
    else:
        best = "generic"
        confidence = 0.35
        reasons = ["no strong domain signals — using generic pack"]

    return {
        "domain": best,
        "label": FORGE_DOMAIN_LABELS[best],
        "confidence": confidence,
        "reasons": reasons,
        "scores": {k: round(v, 2) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
        "column_types": ctypes,
        "app_domain": OS_TO_APP_DOMAIN.get(best, "generic"),
    }


def suggest_roles(
    columns: list[str],
    *,
    domain: str = "generic",
    column_types: Optional[dict[str, str]] = None,
    df: Optional[pd.DataFrame] = None,
) -> dict[str, str]:
    """Domain-aware role suggestions using names + inferred column types."""
    dom = domain if domain in FORGE_DOMAINS else "generic"
    ctypes = column_types or {}
    if df is not None and not ctypes:
        ctypes = detect_column_types(df)

    allowed = set(roles_for_domain(dom))
    name_hints: dict[str, tuple[str, ...]] = {
        "date": ("date", "timestamp", "datetime", "day", "budat", "shift_date", "period"),
        "id": ("id", "uuid", "guid"),
        "customer_id": ("customer_id", "cust_id", "subscriber_id", "account_id"),
        "customer": ("customer", "client", "buyer", "account_name"),
        "order_id": ("order_id", "order_no", "invoice_id", "transaction_id"),
        "revenue": ("revenue", "sales", "amount", "gmv", "total", "net_sales"),
        "quantity": ("quantity", "qty", "units", "count", "volume"),
        "product": ("product", "sku", "item", "material"),
        "region": ("region", "territory", "market", "geo", "country", "state"),
        "target": ("target", "forecast", "y", "label", "actual"),
        "seasonality_hint": ("season", "holiday", "week", "month", "quarter"),
        "churn_flag": ("churn", "churned", "is_churn", "cancelled", "attrition"),
        "tenure": ("tenure", "months_active", "lifetime", "seniority"),
        "subscription": ("subscription", "plan", "tier", "package"),
        "support_tickets": ("ticket", "complaint", "support", "cases"),
        "timestamp": ("timestamp", "datetime", "time", "ts"),
        "asset": ("asset", "machine", "equipment", "line", "asset_id", "machine_id"),
        "sensor": ("temperature", "vibration", "pressure", "humidity", "sensor", "rpm", "torque"),
        "failure_label": ("failure", "fault", "alarm", "breakdown", "failed"),
        "rul_target": ("rul", "remaining_life", "ttf", "time_to_failure"),
        "downtime": ("downtime", "down_time", "stop_min", "idle_min", "duration"),
        "loss": ("loss", "lost", "waste_cost"),
        "scrap": ("scrap", "reject", "defect_qty"),
        "qty": ("qty", "produced", "good_units", "output"),
        "availability": ("availability", "avail", "uptime"),
        "performance": ("performance", "perf", "speed_loss"),
        "quality": ("quality", "yield", "fpy"),
        "defect": ("defect", "ncr", "nonconform"),
        "fpy": ("fpy", "first_pass", "first_pass_yield"),
        "batch": ("batch", "lot", "run_id"),
        "spec_limit": ("spec", "usl", "lsl", "tolerance", "limit"),
        "student_id": ("student_id", "student", "learner_id", "roll_no", "roll_number"),
        "grade": ("grade", "letter_grade", "final_grade"),
        "score": ("score", "marks", "gpa", "cgpa", "points"),
        "course": ("course", "subject", "class_name", "module"),
        "attendance": ("attendance", "absent", "present"),
        "patient_id": ("patient_id", "patient", "mrn"),
        "diagnosis": ("diagnosis", "icd", "condition"),
        "bmi": ("bmi",),
        "readmission": ("readmission", "readmit"),
        "category": ("category", "type", "class", "shift", "location", "department"),
        "metric": ("metric", "value", "score", "rate", "index"),
    }

    out: dict[str, str] = {}
    used_roles: set[str] = set()

    for col in columns:
        col_str = str(col)
        n = _norm_name(col_str)
        ctype = ctypes.get(col_str, "")

        if ctype == "date" and "date" in allowed and "date" not in used_roles:
            out[col_str] = "date"
            used_roles.add("date")
            continue

        matched = False
        for role, keys in name_hints.items():
            if role not in allowed:
                continue
            if any(k in n for k in keys):
                if role in used_roles and role not in (
                    "metric",
                    "sensor",
                    "category",
                    "quantity",
                    "qty",
                    "unused",
                ):
                    continue
                out[col_str] = role
                used_roles.add(role)
                matched = True
                break
        if matched:
            continue

        if ctype == "id_like" and "id" in allowed:
            out[col_str] = "customer_id" if dom == "churn" and "customer_id" in allowed else "id"
        elif ctype == "number" and "metric" in allowed:
            if dom == "predictive_maintenance" and "sensor" in allowed:
                out[col_str] = "sensor"
            elif dom in ("sales", "forecasting") and "revenue" in allowed and any(
                k in n for k in ("rev", "sales", "amount")
            ):
                out[col_str] = "revenue"
            else:
                out[col_str] = "metric"
        elif ctype == "boolean" and dom == "churn" and "churn_flag" in allowed:
            out[col_str] = "churn_flag"
        elif ctype == "text" and "category" in allowed:
            out[col_str] = "category"

    return out


def domain_pipeline_hint(domain: str) -> str:
    """Short UX hint for Ask vs ML routing."""
    hints = {
        "sales": "Use **Ask / AI** for revenue trends and segments; **ML page** + Optuna for forecast tuning.",
        "forecasting": "Use **Ask / AI** for seasonality Q&A; **ML page** + Optuna to tune forecast models.",
        "churn": "Map **churn_flag** and customer fields, then **ML page** for classification + Optuna.",
        "predictive_maintenance": "Map **failure_label** if available; **ML** for risk scores. Deep reliability → dedicated PdM app.",
        "plant_oee": "Plant / OEE pack — use **OEE Pulse** for weekly plant ritual; Forge handles clean + insights here.",
        "quality": "Map **defect** / **fpy** columns; **Ask** for defect patterns; **ML** for defect-rate models.",
        "education": "Map **student_id** / **score**; **Ask** for cohort questions; **ML** for grade models. Override if this is not student data.",
        "health": "Map **patient_id** / clinical fields; **ML** is triage support — not a diagnosis. Override if this is not health data.",
        "generic": "Pipeline: **Detect → Map → Ask** (LlamaIndex/Gemini) or **Train** (Optuna on ML page).",
    }
    return hints.get(domain, hints["generic"])
