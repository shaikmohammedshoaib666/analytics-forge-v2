"""Custom KPI Studio — Tableau/Power BI style calculated fields (safe, no eval).

Python 3.9 compatible. Streamlit is imported only inside UI helpers.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

AGG_OPS = ("mean", "sum", "count", "min", "max", "median")
FILTER_OPS = ("==", "!=", ">", ">=", "<", "<=", "contains")

SESSION_KEY = "custom_kpis"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_kpi_id() -> str:
    return str(uuid.uuid4())


def formula_label(spec: dict[str, Any]) -> str:
    """Human-readable formula string for UI / reports."""
    agg = str(spec.get("agg") or "mean")
    col = str(spec.get("column") or "?")
    base = f"{agg}({col})"
    filt = spec.get("filter") or {}
    if filt.get("column") and filt.get("op"):
        base += f" where {filt['column']} {filt['op']} {filt.get('value', '')}"
    ratio = spec.get("ratio") or {}
    if ratio.get("column"):
        ragg = str(ratio.get("agg") or "sum")
        base = f"({base}) / ({ragg}({ratio['column']})"
        rf = ratio.get("filter") or {}
        if rf.get("column") and rf.get("op"):
            base += f" where {rf['column']} {rf['op']} {rf.get('value', '')}"
        base += ")"
    return base


def _coerce_filter_value(series: pd.Series, raw: Any) -> Any:
    text = "" if raw is None else str(raw).strip()
    if text == "":
        return text
    if pd.api.types.is_numeric_dtype(series):
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return text
    if pd.api.types.is_bool_dtype(series):
        low = text.lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return text


def apply_simple_filter(df: pd.DataFrame, filt: Optional[dict[str, Any]]) -> pd.DataFrame:
    """Allowlisted column/op/value filter — never eval()."""
    if not filt or not filt.get("column") or not filt.get("op"):
        return df
    col = str(filt["column"])
    op = str(filt["op"])
    if col not in df.columns:
        raise ValueError(f"Filter column not found: {col}")
    if op not in FILTER_OPS:
        raise ValueError(f"Filter op not allowed: {op}")
    series = df[col]
    value = _coerce_filter_value(series, filt.get("value"))
    if op == "contains":
        mask = series.astype(str).str.contains(str(value), case=False, na=False)
    elif op == "==":
        mask = series == value
    elif op == "!=":
        mask = series != value
    elif op == ">":
        mask = pd.to_numeric(series, errors="coerce") > float(value)  # type: ignore[arg-type]
    elif op == ">=":
        mask = pd.to_numeric(series, errors="coerce") >= float(value)  # type: ignore[arg-type]
    elif op == "<":
        mask = pd.to_numeric(series, errors="coerce") < float(value)  # type: ignore[arg-type]
    elif op == "<=":
        mask = pd.to_numeric(series, errors="coerce") <= float(value)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Filter op not allowed: {op}")
    return df.loc[mask].copy()


def _run_agg(df: pd.DataFrame, agg: str, column: str) -> float:
    agg = str(agg or "").lower()
    if agg not in AGG_OPS:
        raise ValueError(f"Aggregation not allowed: {agg}")
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    if agg == "count":
        return float(df[column].notna().sum())
    series = pd.to_numeric(df[column], errors="coerce")
    if not series.notna().any():
        raise ValueError(f"No numeric values in column: {column}")
    if agg == "mean":
        return float(series.mean())
    if agg == "sum":
        return float(series.sum())
    if agg == "min":
        return float(series.min())
    if agg == "max":
        return float(series.max())
    if agg == "median":
        return float(series.median())
    raise ValueError(f"Aggregation not allowed: {agg}")


def evaluate_kpi(df: pd.DataFrame, spec: dict[str, Any]) -> Any:
    """Compute a custom KPI on a (already filtered) dataframe. Safe allowlist only."""
    if df is None or getattr(df, "empty", True):
        return "—"
    work = apply_simple_filter(df, spec.get("filter"))
    if work.empty:
        return "—"
    agg = str(spec.get("agg") or "mean")
    col = str(spec.get("column") or "")
    num = _run_agg(work, agg, col)
    ratio = spec.get("ratio") or {}
    if ratio.get("column"):
        denom_df = apply_simple_filter(df, ratio.get("filter"))
        if denom_df.empty:
            return "—"
        denom = _run_agg(denom_df, str(ratio.get("agg") or "sum"), str(ratio["column"]))
        if denom == 0:
            return "—"
        val = num / denom
    else:
        val = num
    if abs(val) >= 1000:
        return round(val, 2)
    if abs(val) >= 1:
        return round(val, 3)
    return round(val, 4)


def evaluate_all(df: pd.DataFrame, specs: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in specs or []:
        name = str(spec.get("name") or "Custom").strip() or "Custom"
        try:
            out[name] = evaluate_kpi(df, spec)
        except Exception as exc:
            out[name] = f"err: {exc}"
    return out


def merge_kpi_dicts(auto_kpis: dict[str, Any], custom: dict[str, Any]) -> dict[str, Any]:
    """Auto KPIs first; custom names overwrite only on exact collision (prefixed)."""
    merged = dict(auto_kpis or {})
    for k, v in (custom or {}).items():
        key = k if k not in merged else f"Custom_{k}"
        merged[key] = v
    return merged


def validate_spec(spec: dict[str, Any], columns: list[str]) -> Optional[str]:
    name = str(spec.get("name") or "").strip()
    if not name:
        return "Name is required."
    agg = str(spec.get("agg") or "")
    if agg not in AGG_OPS:
        return f"Aggregation must be one of {AGG_OPS}."
    col = str(spec.get("column") or "")
    if col not in columns:
        return f"Column not in dataframe: {col}"
    filt = spec.get("filter") or {}
    if filt.get("column"):
        if filt["column"] not in columns:
            return f"Filter column not found: {filt['column']}"
        if filt.get("op") not in FILTER_OPS:
            return f"Filter op must be one of {FILTER_OPS}."
    ratio = spec.get("ratio") or {}
    if ratio.get("column"):
        if ratio["column"] not in columns:
            return f"Ratio column not found: {ratio['column']}"
        if str(ratio.get("agg") or "sum") not in AGG_OPS:
            return "Ratio aggregation invalid."
        rf = ratio.get("filter") or {}
        if rf.get("column") and rf["column"] not in columns:
            return f"Ratio filter column not found: {rf['column']}"
    return None


def get_session_kpis() -> list[dict[str, Any]]:
    import streamlit as st

    raw = st.session_state.get(SESSION_KEY)
    if not isinstance(raw, list):
        st.session_state[SESSION_KEY] = []
        return []
    return list(raw)


def set_session_kpis(specs: list[dict[str, Any]]) -> None:
    import streamlit as st

    st.session_state[SESSION_KEY] = list(specs)


def ensure_custom_kpis_loaded(user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Hydrate session from tenant_store when logged in (once per session flag)."""
    import streamlit as st

    specs = get_session_kpis()
    if user_id and not st.session_state.get("_custom_kpis_hydrated"):
        try:
            from modules import tenant_store as ts

            remote = ts.list_custom_kpis(user_id)
            if remote:
                # Prefer remote when session empty; merge by id if session has local drafts
                by_id = {str(s.get("id")): s for s in remote if s.get("id")}
                for s in specs:
                    sid = str(s.get("id") or "")
                    if sid and sid not in by_id:
                        by_id[sid] = s
                specs = list(by_id.values()) if by_id else remote
                set_session_kpis(specs)
        except Exception:
            pass
        st.session_state["_custom_kpis_hydrated"] = True
    return get_session_kpis()


def save_custom_kpi(spec: dict[str, Any], user_id: Optional[str] = None) -> dict[str, Any]:
    specs = get_session_kpis()
    sid = str(spec.get("id") or new_kpi_id())
    payload = dict(spec)
    payload["id"] = sid
    payload["updated_at"] = _utcnow()
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    replaced = False
    for i, existing in enumerate(specs):
        if str(existing.get("id")) == sid:
            specs[i] = payload
            replaced = True
            break
    if not replaced:
        specs.append(payload)
    set_session_kpis(specs)
    if user_id:
        try:
            from modules import tenant_store as ts

            ts.save_custom_kpi(user_id, payload)
        except Exception:
            pass
    return payload


def delete_custom_kpi(kpi_id: str, user_id: Optional[str] = None) -> None:
    specs = [s for s in get_session_kpis() if str(s.get("id")) != str(kpi_id)]
    set_session_kpis(specs)
    if user_id:
        try:
            from modules import tenant_store as ts

            ts.delete_custom_kpi(user_id, kpi_id)
        except Exception:
            pass


def _filter_ui(prefix: str, columns: list[str], label: str) -> Optional[dict[str, Any]]:
    import streamlit as st

    enable = st.checkbox(f"Filter {label}", value=False, key=f"{prefix}_filt_on")
    if not enable:
        return None
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        fcol = st.selectbox("Column", columns, key=f"{prefix}_filt_col")
    with c2:
        fop = st.selectbox("Op", list(FILTER_OPS), key=f"{prefix}_filt_op")
    with c3:
        fval = st.text_input("Value", value="", key=f"{prefix}_filt_val")
    return {"column": fcol, "op": fop, "value": fval}


def render_kpi_studio(df: pd.DataFrame, *, user_id: Optional[str] = None, key_prefix: str = "kpi_studio") -> list[dict[str, Any]]:
    """Streamlit UI: build / preview / save / delete custom KPIs."""
    import streamlit as st

    st.markdown("### KPI Studio")
    st.caption(
        "Build calculated fields like Tableau / Power BI — aggregation + column + optional filter + optional ratio. "
        "No free-text eval; only allowlisted pandas aggregations."
    )
    specs = ensure_custom_kpis_loaded(user_id)
    columns = [str(c) for c in df.columns]
    num_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    pick_cols = num_cols or columns
    if not pick_cols:
        st.warning("No columns available for custom KPIs.")
        return specs

    with st.expander("Create custom KPI", expanded=not bool(specs)):
        name = st.text_input("Name", value="", key=f"{key_prefix}_name", placeholder="e.g. Avg revenue East")
        c1, c2 = st.columns(2)
        with c1:
            agg = st.selectbox("Aggregation", list(AGG_OPS), index=0, key=f"{key_prefix}_agg")
        with c2:
            col = st.selectbox("Column", pick_cols, key=f"{key_prefix}_col")
        filt = _filter_ui(f"{key_prefix}_main", columns, "(numerator)")
        use_ratio = st.checkbox("Ratio of two aggregations", value=False, key=f"{key_prefix}_ratio_on")
        ratio: Optional[dict[str, Any]] = None
        if use_ratio:
            r1, r2 = st.columns(2)
            with r1:
                ragg = st.selectbox("Denominator aggregation", list(AGG_OPS), index=1, key=f"{key_prefix}_ragg")
            with r2:
                rcol = st.selectbox("Denominator column", pick_cols, key=f"{key_prefix}_rcol")
            rfilt = _filter_ui(f"{key_prefix}_ratio", columns, "(denominator)")
            ratio = {"agg": ragg, "column": rcol, "filter": rfilt}

        draft = {
            "name": name.strip(),
            "agg": agg,
            "column": col,
            "filter": filt,
            "ratio": ratio,
        }
        st.caption(f"Formula: `{formula_label(draft)}`")
        try:
            preview = evaluate_kpi(df, draft) if name.strip() else "—"
        except Exception as exc:
            preview = f"err: {exc}"
        st.metric("Preview (current filtered data)", preview)

        if st.button("Save custom KPI", type="primary", key=f"{key_prefix}_save"):
            err = validate_spec(draft, columns)
            if err:
                st.error(err)
            else:
                save_custom_kpi(draft, user_id=user_id)
                st.success(f"Saved **{draft['name']}**")
                st.rerun()

    if specs:
        st.markdown("##### Saved custom KPIs")
        custom_vals = evaluate_all(df, specs)
        for spec in specs:
            sid = str(spec.get("id") or "")
            nm = str(spec.get("name") or "Custom")
            val = custom_vals.get(nm, "—")
            row1, row2 = st.columns([4, 1])
            with row1:
                st.write(f"**{nm}** = `{val}` · `{formula_label(spec)}`")
            with row2:
                if st.button("Delete", key=f"{key_prefix}_del_{sid}"):
                    delete_custom_kpi(sid, user_id=user_id)
                    st.rerun()
    else:
        st.caption("No custom KPIs yet — create one above.")

    return get_session_kpis()
