from __future__ import annotations

# Recovered upgraded page snippets from transcript messages 602, 665, and 668.
# This file is splice-ready and expects the original app.py imports/globals and helpers.

# Supporting Gemini helpers recovered from messages 635 and 642.
# These expect os, st, ROOT, and GEMINI_MODEL from app.py.
_ENV_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()


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


# =============================================================================
# UPGRADED PAGES
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

