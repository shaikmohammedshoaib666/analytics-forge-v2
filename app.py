"""
Analytics Forge — Phase 1 Streamlit app.
Upload → Clean → Field → KPIs → Charts → ML → AI → Dashboard → Email
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import SAMPLES_DIR, UPLOADS_DIR
from core import db
from core.classify import classify, load_domains
from core.kpis import compute_kpis
from core.pack import build_html_pack
from core.pipeline import run_pipeline
from modules.ai_guide import ask_ai
from modules.charts import build_chart, load_charts_catalog
from modules.ml_registry import list_models
from modules.ml_runner import run_model
from ui.auth_gate import render_user_sidebar, require_login
from ui.components import (
    download_df_button,
    download_html_pack_button,
    kpi_cards,
    show_ml_metrics,
)
from ui.session import init_session_state, reset_analysis_state
from ui.theme import inject_css, page_hero
from modules.manager_insights import build_manager_insight

st.set_page_config(
    page_title="Analytics Forge",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def ensure_samples() -> None:
    """Create sample CSVs if missing (also ships with repo samples)."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    pdm = SAMPLES_DIR / "sample_predictive_maintenance.csv"
    sales = SAMPLES_DIR / "sample_sales.csv"
    if not pdm.exists():
        # Minimal regenerate
        rows = []
        for mid in range(1, 6):
            for h in range(8):
                rows.append(
                    {
                        "machine_id": f"M-{mid:03d}",
                        "timestamp": f"2024-01-0{1 + h // 4} {h % 4 * 6:02d}:00:00",
                        "temperature": 70 + mid * 2 + h,
                        "vibration": 0.3 + mid * 0.05 + h * 0.08,
                        "pressure": 102 - h * 0.8,
                        "failure": 1 if h == 7 and mid % 2 else 0,
                        "rul": 0 if h == 7 and mid % 2 else 200 - h * 10,
                    }
                )
        pd.DataFrame(rows).to_csv(pdm, index=False)
    if not sales.exists():
        import numpy as np

        rng = np.random.default_rng(42)
        regions = ["North", "South", "East", "West"]
        cats = ["Electronics", "Furniture", "Office Supplies"]
        rows = []
        for d in range(1, 31):
            rows.append(
                {
                    "order_date": f"2024-01-{d:02d}",
                    "region": regions[d % 4],
                    "category": cats[d % 3],
                    "revenue": float(rng.integers(100, 5000)),
                    "units": int(rng.integers(1, 30)),
                }
            )
        pd.DataFrame(rows).to_csv(sales, index=False)


def apply_pipeline_to_state(result: dict) -> None:
    st.session_state.messy_df = result["messy_df"]
    st.session_state.clean_df = result["clean_df"]
    st.session_state.clean_log = result["clean_log"]
    st.session_state.source_name = result["source_name"]
    st.session_state.domain = result["domain"]
    st.session_state.classification = result["classification"]
    st.session_state.kpis = result["kpis"]
    st.session_state.briefing = result["briefing"]
    st.session_state.schema = result["schema"]
    st.session_state.run_id = result.get("run_id")
    st.session_state.pipeline_done = True
    st.session_state.dashboard_insights = [result["briefing"]]
    st.session_state.ml_result = None
    st.session_state.chat_history = []


def _current_user_id() -> int | None:
    user = st.session_state.get("user")
    if not user:
        return None
    try:
        return int(user["id"])
    except (KeyError, TypeError, ValueError):
        return None


def load_saved_project(run_id: int) -> bool:
    """Reload a past project from SQLite + saved clean CSV."""
    from pathlib import Path

    from core.briefing import build_briefing
    from core.ingest import schema_summary

    row = db.get_run(run_id)
    if not row:
        st.error("Project not found.")
        return False
    uid = _current_user_id()
    if uid is not None and row.get("user_id") not in (None, uid):
        st.error("That project belongs to another account.")
        return False

    clean_path = row.get("clean_path") or ""
    if not clean_path or not Path(clean_path).exists():
        st.error("Saved data file is missing on this machine. Re-upload the CSV.")
        return False

    clean_df = pd.read_csv(clean_path)
    domain = row.get("domain") or "generic"
    classification = {"domain": domain, "confidence": row.get("domain_confidence") or 0.0}
    kpis = compute_kpis(clean_df, domain=domain)
    briefing = build_briefing(domain, clean_df.shape, kpis=kpis, classification=classification)
    schema = schema_summary(clean_df)

    reset_analysis_state()
    st.session_state.messy_df = clean_df.copy()
    st.session_state.clean_df = clean_df
    st.session_state.clean_log = []
    st.session_state.source_name = row.get("source_name") or row.get("title") or f"run_{run_id}"
    st.session_state.domain = domain
    st.session_state.classification = classification
    st.session_state.kpis = kpis
    st.session_state.briefing = briefing
    st.session_state.schema = schema
    st.session_state.run_id = run_id
    st.session_state.pipeline_done = True
    st.session_state.dashboard_insights = [briefing]

    ml = db.get_latest_ml_for_run(run_id)
    if ml:
        st.session_state.ml_result = {
            "ok": True,
            "model_id": ml.get("model_id"),
            "task": ml.get("task"),
            "target": ml.get("target_col"),
            "metrics": ml.get("metrics") or {},
            "manager_briefing": ml.get("manager_briefing") or "",
        }

    if uid is not None:
        chats = db.list_chat_messages(uid, run_id=run_id, limit=40)
        st.session_state.chat_history = [
            {"role": c["role"], "content": c["content"]} for c in chats
        ]
    return True


def render_recent_projects() -> None:
    uid = _current_user_id()
    if uid is None:
        return
    st.sidebar.markdown("#### Recent projects")
    rows = db.list_recent_runs(uid, limit=10)
    if not rows:
        st.sidebar.caption("No saved projects yet — upload data to create one.")
        return
    for r in rows:
        label = r.get("title") or r.get("source_name") or f"Run {r['id']}"
        meta = f"{r.get('domain', '?')} · {r.get('row_count') or 0:,} rows"
        if st.sidebar.button(
            f"{label}\n{meta}",
            key=f"recent_run_{r['id']}",
            use_container_width=True,
        ):
            if load_saved_project(int(r["id"])):
                st.session_state.page = "Dashboard"
                st.rerun()


def page_upload() -> None:
    page_hero(
        "Upload",
        "Drop a table file (or a ZIP that contains one). Any industry data works — not only sales or PdM.",
        st.session_state.get("domain"),
    )
    st.write(
        "Supported formats: **CSV, TSV, Excel (.xlsx / .xls), JSON, Parquet**, "
        "or a **ZIP** that contains one of those. "
        "ZIP is only a container — after extract we still run the same cleaning pipeline."
    )
    st.caption(
        "Tip for managers: export from Excel / Google Sheets / your ERP as CSV or Excel, "
        "or zip that file and upload the zip."
    )

    ensure_samples()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    c1, c2 = st.columns(2)
    with c1:
        from core.ingest import ZipNoTabularError, list_zip_tabular_members

        uploaded = st.file_uploader(
            "Upload data file",
            type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet", "zip"],
            help=(
                "CSV / TSV / Excel / JSON / Parquet, or a ZIP containing one of those. "
                "If the ZIP has several tables, you can pick which one to load."
            ),
        )
        zip_member = None
        if uploaded is not None:
            data = uploaded.getvalue()
            dest = UPLOADS_DIR / uploaded.name
            dest.write_bytes(data)

            if uploaded.name.lower().endswith(".zip"):
                try:
                    members = list_zip_tabular_members(data)
                except Exception as exc:
                    st.error(f"Could not read ZIP: {exc}")
                    members = []
                if not members:
                    st.error(
                        "This ZIP has no tabular files. "
                        "Include at least one CSV, TSV, Excel (.xlsx/.xls), JSON, or Parquet file."
                    )
                elif len(members) == 1:
                    zip_member = members[0]
                    st.caption(f"ZIP contains: `{zip_member}`")
                else:
                    zip_member = st.selectbox(
                        "This ZIP has several tables — pick one",
                        members,
                        help="We load one sheet/table into the cleaning pipeline.",
                    )

            can_run = True
            if uploaded.name.lower().endswith(".zip") and not zip_member:
                can_run = False

            if can_run and st.button("Run pipeline on upload", type="primary", key="run_upload"):
                try:
                    with st.spinner("Running pipeline…"):
                        result = run_pipeline(
                            file_bytes=data,
                            filename=uploaded.name,
                            domain_override=st.session_state.domain_override,
                            persist=True,
                            zip_member=zip_member,
                            user_id=_current_user_id(),
                        )
                    apply_pipeline_to_state(result)
                    st.success(f"Loaded **{result['source_name']}** · domain `{result['domain']}`")
                    st.dataframe(result["clean_df"].head(20), use_container_width=True)
                except ZipNoTabularError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Could not load file: {exc}")

    with c2:
        sample_choice = st.selectbox(
            "Or use sample data",
            [
                "sample_predictive_maintenance.csv",
                "sample_sales.csv",
            ],
        )
        if st.button("Load sample & run pipeline", key="run_sample"):
            path = SAMPLES_DIR / sample_choice
            with st.spinner("Running pipeline…"):
                result = run_pipeline(
                    source=path,
                    domain_override=st.session_state.domain_override,
                    persist=True,
                    user_id=_current_user_id(),
                )
            apply_pipeline_to_state(result)
            st.success(f"Sample loaded · domain `{result['domain']}`")
            st.dataframe(result["clean_df"].head(20), use_container_width=True)

    if st.session_state.pipeline_done:
        st.divider()
        st.subheader("Current dataset")
        st.write(
            f"**{st.session_state.source_name}** — "
            f"{len(st.session_state.clean_df):,} rows × {st.session_state.clean_df.shape[1]} cols · "
            f"domain `{st.session_state.domain}` · run_id `{st.session_state.run_id}`"
        )
        if st.button("Reset analysis", key="reset_all"):
            reset_analysis_state()
            st.rerun()


def page_clean() -> None:
    page_hero(
        "Clean",
        "Messy vs clean side-by-side — see exactly what pandas cleaned for you.",
        st.session_state.get("domain"),
    )
    if st.session_state.messy_df is None:
        st.warning("Upload or load a sample first.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Messy (raw)")
        st.dataframe(st.session_state.messy_df.head(50), use_container_width=True)
        st.caption(f"{len(st.session_state.messy_df):,} rows")
        download_df_button(
            st.session_state.messy_df,
            "Download raw CSV",
            "messy.csv",
            key="dl_messy",
        )
    with right:
        st.subheader("Clean")
        st.dataframe(st.session_state.clean_df.head(50), use_container_width=True)
        st.caption(f"{len(st.session_state.clean_df):,} rows")
        download_df_button(
            st.session_state.clean_df,
            "Download clean CSV",
            "clean.csv",
            key="dl_clean",
        )

    st.subheader("Cleaning log (pandas ops)")
    log = st.session_state.clean_log or []
    st.dataframe(pd.DataFrame(log), use_container_width=True)


def page_field() -> None:
    page_hero(
        "Field detection",
        "Auto-detects warehouse, sales, PdM, hospital, and more — then suggests models & next steps.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    domains = load_domains()
    labels = {k: v.get("label", k) for k, v in domains.items()}
    clf = st.session_state.classification or classify(st.session_state.clean_df)

    st.write(f"**Auto-detected:** `{clf.get('domain')}` — {clf.get('label')}")
    scores = clf.get("scores") or {}
    score_df = (
        pd.DataFrame(
            [{"domain": d, "label": labels.get(d, d), "score": round(s, 3)} for d, s in scores.items()]
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(score_df, use_container_width=True)

    override = st.selectbox(
        "Override domain",
        options=list(domains.keys()),
        format_func=lambda x: f"{labels.get(x, x)} ({x})",
        index=list(domains.keys()).index(st.session_state.domain)
        if st.session_state.domain in domains
        else list(domains.keys()).index("generic"),
    )
    if st.button("Apply domain override", type="primary"):
        st.session_state.domain_override = override
        clf2 = classify(st.session_state.clean_df, override=override)
        st.session_state.classification = clf2
        st.session_state.domain = clf2["domain"]
        st.session_state.kpis = compute_kpis(
            st.session_state.clean_df,
            domain=clf2["domain"],
            ml_metrics=(st.session_state.ml_result or {}).get("metrics"),
        )
        from core.briefing import build_briefing

        st.session_state.briefing = build_briefing(
            clf2["domain"],
            st.session_state.clean_df.shape,
            kpis=st.session_state.kpis,
            classification=clf2,
        )
        st.success(f"Domain set to `{clf2['domain']}`")
        st.rerun()

    st.info("Recommended models: " + ", ".join(clf.get("recommended_models") or []))


def page_kpis() -> None:
    page_hero(
        "Auto KPIs",
        "Scoreboard numbers for your detected field — including sales-on-latest-day style metrics.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    if st.button("Recompute KPIs"):
        ml_m = None
        if st.session_state.ml_result and st.session_state.ml_result.get("ok"):
            ml_m = st.session_state.ml_result
        st.session_state.kpis = compute_kpis(
            st.session_state.clean_df,
            domain=st.session_state.domain,
            ml_metrics=ml_m,
        )

    kpi_cards(st.session_state.kpis or {})
    st.subheader("All KPIs")
    rows = []
    for kid, item in (st.session_state.kpis or {}).items():
        if isinstance(item, dict):
            rows.append(
                {
                    "id": kid,
                    "name": item.get("name"),
                    "value": item.get("value"),
                    "formula": item.get("formula", ""),
                }
            )
        else:
            rows.append({"id": kid, "name": kid, "value": item, "formula": ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.session_state.domain == "predictive_maintenance" and st.session_state.ml_result:
        st.subheader("PdM ML metrics")
        show_ml_metrics(st.session_state.ml_result)

    st.markdown(st.session_state.briefing or "")


def page_charts() -> None:
    page_hero(
        "Charts",
        "Build colorful views, download one chart, or pin several into your final dashboard.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    df = st.session_state.clean_df
    catalog = load_charts_catalog()
    chart_types = list(catalog.keys())
    cols = list(df.columns)

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox("Chart type", chart_types, format_func=lambda t: catalog[t].get("label", t))
    with c2:
        libs = catalog.get(chart_type, {}).get("libs", ["plotly", "matplotlib", "seaborn"])
        lib = st.selectbox("Library", libs)
    with c3:
        title = st.text_input("Title", value=f"{catalog[chart_type].get('label', chart_type)}")

    needs = catalog.get(chart_type, {}).get("needs", [])
    x = y = names = values = None
    r1, r2 = st.columns(2)
    with r1:
        if "x" in needs or chart_type in ("bar", "line", "scatter", "histogram", "box", "area"):
            x = st.selectbox("X column", cols, key="chart_x")
        if "names" in needs:
            names = st.selectbox("Names", cols, key="chart_names")
    with r2:
        num_cols = df.select_dtypes("number").columns.tolist() or cols
        if "y" in needs or chart_type in ("bar", "line", "scatter", "box", "area"):
            y = st.selectbox("Y column", num_cols if num_cols else cols, key="chart_y")
        if "values" in needs:
            values = st.selectbox("Values", num_cols if num_cols else cols, key="chart_vals")

    if st.button("Render chart", type="primary"):
        try:
            fig = build_chart(
                df,
                chart_type=chart_type,
                lib=lib,
                x=x,
                y=y,
                names=names,
                values=values,
                title=title,
            )
            st.session_state["_last_fig"] = fig
            st.session_state["_last_fig_meta"] = {
                "chart_type": chart_type,
                "lib": lib,
                "title": title,
                "x": x,
                "y": y,
                "names": names,
                "values": values,
            }
        except Exception as exc:
            st.error(str(exc))

    fig = st.session_state.get("_last_fig")
    meta = st.session_state.get("_last_fig_meta")
    if fig is not None and meta:
        if meta.get("lib") == "plotly":
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.pyplot(fig)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Add to dashboard"):
                entry = {**meta}
                st.session_state.dashboard_charts.append(entry)
                if st.session_state.run_id:
                    db.save_chart(
                        st.session_state.run_id,
                        chart_type=meta["chart_type"],
                        lib=meta["lib"],
                        title=meta["title"],
                        config=meta,
                    )
                st.success("Added to dashboard")
        with b2:
            # download: plotly html or note
            if meta.get("lib") == "plotly":
                try:
                    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
                    st.download_button(
                        "Download chart HTML",
                        data=html_bytes,
                        file_name=f"{meta['chart_type']}_chart.html",
                        mime="text/html",
                        key="dl_chart_html",
                    )
                except Exception:
                    st.caption("Chart download unavailable for this figure.")


def page_ml() -> None:
    page_hero(
        "ML Studio",
        "Pick a model, read plain-English What / Why / What it does, choose a target, then see R² / RMSE / MAE.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    from modules.ml_runner import model_guidance

    models = list_models(include_soft_fail=True)
    domain = st.session_state.domain
    domains = load_domains()
    recommended = domains.get(domain, {}).get("recommended_models", [])
    model_ids = list(models.keys())
    # put recommended first
    model_ids = [m for m in recommended if m in models] + [m for m in model_ids if m not in recommended]

    model_id = st.selectbox(
        "Model",
        model_ids,
        format_func=lambda m: f"{models[m].get('label', m)} [{models[m].get('task')}]"
        + (" ★" if m in recommended else ""),
    )
    meta = models[model_id]
    guide = model_guidance(model_id)
    st.info(
        f"**What:** {guide.get('what') or guide.get('good_for', '')}\n\n"
        f"**Why:** {guide.get('why', '')}\n\n"
        f"**What it does:** {guide.get('what_it_does', '')}\n\n"
        f"**Target to pick:** {guide.get('target', '')}"
    )
    if meta.get("note"):
        st.caption(meta["note"])
    else:
        st.caption(f"Library: {meta.get('library')}")

    df = st.session_state.clean_df
    cols = list(df.columns)
    target_label = "Target (the number or label to predict)"
    if model_id == "Prophet":
        target_label = "Target = the number to forecast (revenue / sales / RUL)"
    elif meta.get("task") in {"clustering", "anomaly", "dimensionality", "optimization"}:
        target_label = "Target (usually leave as auto — not required)"

    target = st.selectbox(
        target_label,
        options=["(auto)"] + cols,
        help=guide.get("target", "Pick the column you want to predict."),
    )
    target_arg = None if target == "(auto)" else target

    if model_id == "Prophet":
        st.success(
            "Prophet tip for managers: pick **Target** = revenue, sales, or RUL (the number). "
            "The **date column is found automatically** — you do not pick it here. "
            "Why: forecasting needs a timeline plus a number to predict."
        )

    if st.button("Run model", type="primary"):
        with st.spinner("Training…"):
            result = run_model(df, model_id=model_id, target=target_arg)
        if result.get("ok"):
            briefing = build_manager_insight(result)
            result["manager_briefing"] = briefing
            if briefing:
                st.session_state.dashboard_insights = list(
                    dict.fromkeys(
                        (st.session_state.dashboard_insights or []) + [briefing]
                    )
                )
        st.session_state.ml_result = result
        if result.get("ok"):
            st.session_state.kpis = compute_kpis(
                df, domain=domain, ml_metrics=result
            )
            if st.session_state.run_id:
                db.save_ml_run(
                    st.session_state.run_id,
                    model_id=model_id,
                    task=result.get("task", ""),
                    target_col=str(result.get("target") or ""),
                    metrics=result.get("metrics") or {},
                    manager_briefing=result.get("manager_briefing") or "",
                )
                if result.get("manager_briefing"):
                    db.save_insight(
                        st.session_state.run_id,
                        f"manager:{model_id}",
                        result["manager_briefing"],
                    )
            st.success("Model finished")
        else:
            st.error(result.get("error", "Failed"))

    show_ml_metrics(st.session_state.ml_result)

    if st.session_state.ml_result and st.session_state.ml_result.get("ok"):
        if st.session_state.ml_result.get("model_id") == "Prophet":
            m = st.session_state.ml_result.get("metrics") or {}
            nums = st.columns(4)
            if m.get("last_actual") is not None:
                nums[0].metric("Last actual", f"{m['last_actual']:,.2f}")
            if m.get("forecast_end") is not None:
                nums[1].metric("Forecast end", f"{m['forecast_end']:,.2f}")
            if m.get("forecast_mean") is not None:
                nums[2].metric("Avg forecast", f"{m['forecast_mean']:,.2f}")
            if m.get("pct_change") is not None:
                nums[3].metric("Change", f"{m['pct_change'] * 100:.1f}%")

        preview = st.session_state.ml_result.get("predictions_preview")
        if preview is not None:
            st.subheader("Predictions preview")
            st.dataframe(preview, use_container_width=True)
            download_df_button(
                preview,
                "Download predictions CSV",
                "ml_predictions_preview.csv",
                key="dl_preds",
            )

        # Adaptive PdM tables
        if domain == "predictive_maintenance":
            st.subheader("Adaptive PdM view")
            metrics = st.session_state.ml_result.get("metrics") or {}
            mcols = st.columns(3)
            if "r2" in metrics:
                mcols[0].metric("R²", f"{metrics['r2']:.4f}")
            if "rmse" in metrics:
                mcols[1].metric("RMSE", f"{metrics['rmse']:.4f}")
            if "mae" in metrics:
                mcols[2].metric("MAE", f"{metrics['mae']:.4f}")
            if "accuracy" in metrics:
                mcols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")

            sensor_cols = [
                c
                for c in df.columns
                if str(c).lower() in {"temperature", "vibration", "pressure", "rul", "failure"}
            ]
            if sensor_cols:
                st.dataframe(df[sensor_cols].describe(), use_container_width=True)
            if "machine_id" in [c.lower() for c in df.columns]:
                mid = next(c for c in df.columns if str(c).lower() == "machine_id")
                fail_col = next(
                    (c for c in df.columns if str(c).lower() == "failure"), None
                )
                if fail_col:
                    agg = (
                        df.groupby(mid)
                        .agg(
                            rows=(mid, "count"),
                            failures=(fail_col, "sum"),
                            avg_temp=(
                                next(
                                    (c for c in df.columns if "temp" in str(c).lower()),
                                    fail_col,
                                ),
                                "mean",
                            ),
                        )
                        .reset_index()
                    )
                    st.dataframe(agg, use_container_width=True)


def page_ai() -> None:
    page_hero(
        "Ask / AI Guide",
        "Ask in plain English — e.g. “what are sales today?” — and get guided next steps.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    from modules.ai_guide import gemini_configured, openai_configured, provider_status

    status = provider_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Gemini", "Ready" if status["gemini"] else "No key")
    c2.metric("OpenAI", "Ready" if status["openai"] else "No key")
    c3.metric("Offline", "Always on")

    if status["gemini"] or status["openai"]:
        st.success("Cloud AI available — pick a provider below.")
    else:
        st.warning(
            "No cloud AI keys yet — offline answers still work.\n\n"
            "Add to `.env` (local) or Streamlit Secrets (cloud), then restart:\n\n"
            "```\n"
            "GEMINI_API_KEY=your_key_from_Google_AI_Studio\n"
            "GEMINI_MODEL=gemini-2.0-flash\n"
            "AI_DEFAULT_PROVIDER=gemini\n"
            "# optional:\n"
            "OPENAI_API_KEY=sk-...\n"
            "OPENAI_MODEL=gpt-4o-mini\n"
            "```\n\n"
            "Gemini free tier: https://aistudio.google.com/apikey — never paste keys into chat."
        )

    selectable = ["auto"]
    if gemini_configured():
        selectable.append("gemini")
    if openai_configured():
        selectable.append("openai")
    selectable.append("offline")
    provider = st.selectbox("AI provider", selectable, index=0, help="Gemini often has a free tier.")

    st.caption(
        "Try: `which model did I use?` · `show kpis` · `which machine will fail?` · `how to reduce machine failure?`"
    )
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about your data…")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        uid = _current_user_id()
        if uid is not None:
            db.save_chat_message(
                uid, "user", prompt, run_id=st.session_state.get("run_id")
            )
        result = ask_ai(
            prompt,
            domain=st.session_state.domain,
            schema=st.session_state.schema,
            kpis=st.session_state.kpis,
            df=st.session_state.clean_df,
            briefing=st.session_state.briefing or "",
            history=st.session_state.chat_history[:-1],
            ml_result=st.session_state.ml_result,
            provider=provider,
        )
        answer = result.get("answer", "")
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if uid is not None:
            db.save_chat_message(
                uid, "assistant", answer, run_id=st.session_state.get("run_id")
            )
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"source: {result.get('source')}")


def page_dashboard() -> None:
    page_hero(
        "Dashboard",
        "Your selected charts + insights. Auto KPIs stay on the right. Download the final pack anytime.",
        st.session_state.get("domain"),
    )
    if st.session_state.clean_df is None:
        st.warning("Upload or load a sample first.")
        return

    center, right = st.columns([3, 1])
    with right:
        st.subheader("Auto KPIs")
        kpi_cards(st.session_state.kpis or {}, max_cards=10)

    with center:
        st.subheader("Insights")
        for insight in st.session_state.dashboard_insights or [st.session_state.briefing]:
            st.markdown(insight or "_No insights yet._")

        st.subheader("Charts")
        charts = st.session_state.dashboard_charts or []
        if not charts:
            st.info("Add charts from the Charts page.")
        df = st.session_state.clean_df
        for i, meta in enumerate(charts):
            st.markdown(f"**{meta.get('title', meta.get('chart_type'))}**")
            try:
                fig = build_chart(
                    df,
                    chart_type=meta.get("chart_type", "bar"),
                    lib=meta.get("lib", "plotly"),
                    x=meta.get("x"),
                    y=meta.get("y"),
                    names=meta.get("names"),
                    values=meta.get("values"),
                    title=meta.get("title"),
                )
                if meta.get("lib") == "plotly":
                    st.plotly_chart(fig, use_container_width=True, key=f"dash_plotly_{i}")
                else:
                    st.pyplot(fig)
            except Exception as exc:
                st.warning(f"Could not render chart: {exc}")

        if st.session_state.domain == "predictive_maintenance" and st.session_state.ml_result:
            st.subheader("PdM model quality")
            show_ml_metrics(st.session_state.ml_result)

    st.divider()
    pack = build_html_pack(
        domain=st.session_state.domain,
        source_name=st.session_state.source_name or "",
        clean_log=st.session_state.clean_log,
        kpis=st.session_state.kpis,
        insights=st.session_state.dashboard_insights,
        charts=st.session_state.dashboard_charts,
        ml_metrics=st.session_state.ml_result,
        briefing=st.session_state.briefing or "",
    )
    download_html_pack_button(pack, key="dash_pack")
    if st.session_state.run_id and st.button("Save dashboard layout to SQLite"):
        db.save_dashboard_layout(
            st.session_state.run_id,
            name="default",
            layout={
                "charts": st.session_state.dashboard_charts,
                "insights": st.session_state.dashboard_insights,
            },
        )
        st.success("Layout saved")


def page_email() -> None:
    page_hero(
        "Email automation",
        "Send the report pack now, or auto-process inbound CSV emails and reply with insights.",
        st.session_state.get("domain"),
    )
    st.caption(
        "Send the dashboard/report pack to any email. "
        "Or click Check inbox: unread emails with CSV/Excel are cleaned, analyzed, and auto-replied with the report."
    )

    from modules.email_automation import (
        EmailConfigError,
        config_status,
        email_configured,
        process_inbound_mailbox,
        send_current_report,
    )

    status = config_status()
    if status["configured"]:
        st.success(f"Email ready · SMTP {status['smtp']} · IMAP {status['imap']} · as {status['from']}")
    else:
        missing = status.get("missing") or ["EMAIL_USER", "EMAIL_PASSWORD"]
        st.warning(
            "Email not configured yet — this is expected until you add credentials locally.\n\n"
            "**Do not paste your real password into chat.** Put secrets in `.env` (local) or "
            "Streamlit Cloud → Settings → Secrets.\n\n"
            "Minimum for Gmail send/inbox:\n\n"
            "```\n"
            "EMAIL_USER=you@gmail.com\n"
            "EMAIL_PASSWORD=your_16_char_app_password\n"
            "EMAIL_FROM=you@gmail.com\n"
            "EMAIL_SMTP_HOST=smtp.gmail.com\n"
            "EMAIL_SMTP_PORT=587\n"
            "EMAIL_IMAP_HOST=imap.gmail.com\n"
            "EMAIL_IMAP_PORT=993\n"
            "```\n\n"
            f"Currently missing: {', '.join(missing)}\n\n"
            "Gmail: enable 2-Step Verification → create an **App Password** "
            "(Google Account → Security). That App Password is what EMAIL_PASSWORD means — "
            "not your normal Gmail login password. Aliases `SMTP_USER` / `SMTP_PASSWORD` / "
            "`SMTP_HOST` also work."
        )

    st.subheader("1) Send current report to me / anyone")
    if not st.session_state.get("pipeline_done"):
        st.info("Load and clean a dataset first (Upload), then you can email the report.")
    with st.form("email_send_form"):
        to_addr = st.text_input("Recipient email", placeholder="you@gmail.com")
        subject = st.text_input(
            "Subject",
            value=f"[Analytics Forge] Report — {st.session_state.get('domain') or 'analytics'}",
        )
        note = st.text_area("Extra note (optional)", value="")
        send_clicked = st.form_submit_button("Send report now (HTML pack + clean CSV)", type="primary")
        if send_clicked:
            if not st.session_state.get("pipeline_done"):
                st.error("No analysis loaded. Upload/clean data first.")
            elif not to_addr.strip():
                st.error("Enter a recipient email.")
            else:
                try:
                    briefing = st.session_state.briefing or ""
                    if not isinstance(briefing, str):
                        import json as _json

                        briefing = _json.dumps(briefing, default=str)
                    result = send_current_report(
                        to_addr.strip(),
                        domain=st.session_state.domain or "generic",
                        source_name=st.session_state.source_name or "dataset",
                        clean_log=st.session_state.clean_log or [],
                        kpis=st.session_state.kpis or {},
                        insights=st.session_state.dashboard_insights or [],
                        charts=st.session_state.dashboard_charts or [],
                        ml_metrics=st.session_state.ml_result,
                        briefing=briefing,
                        clean_df=st.session_state.clean_df,
                        run_id=st.session_state.run_id,
                        subject=subject,
                        extra_body=note,
                    )
                    st.success(
                        f"Sent to {result['to']} (email_id={result['email_id']}). "
                        f"Attachments: {', '.join(result.get('attachments') or [])}"
                    )
                except EmailConfigError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Send failed: {exc}")

    st.subheader("2) Inbound automation (CSV email → auto report reply)")
    st.markdown(
        "Email a **CSV or Excel** file to your configured inbox (`EMAIL_USER`). "
        "Then click the button below. Forge will: clean → detect field → KPIs → baseline ML → "
        "email the HTML report + clean CSV back to the sender."
    )
    if st.button("Check inbox & auto-process unread CSV emails", type="primary"):
        try:
            with st.spinner("Checking IMAP inbox..."):
                out = process_inbound_mailbox(limit=10)
            st.success(f"Processed {out.get('count', 0)} message(s).")
            if out.get("processed"):
                st.dataframe(pd.DataFrame(out["processed"]), use_container_width=True)
            else:
                st.info("No new unread CSV emails found.")
        except EmailConfigError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inbox processing failed: {exc}")

    st.subheader("Email log (SQLite)")
    try:
        rows = db.list_emails(100)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No emails logged yet.")
    except Exception:
        st.info("Email log empty.")


def main() -> None:
    init_session_state()
    inject_css()
    db.init_db()
    ensure_samples()

    if not require_login():
        return

    render_user_sidebar()
    st.sidebar.markdown("### Analytics Forge")
    st.sidebar.caption("Your data workspace — projects stay after refresh")
    page = st.sidebar.radio(
        "Navigate",
        PAGES,
        index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
    )
    st.session_state.page = page
    render_recent_projects()

    if st.session_state.pipeline_done:
        st.sidebar.success(
            f"{st.session_state.source_name or 'dataset'}\n\n"
            f"`{st.session_state.domain}` · {len(st.session_state.clean_df):,} rows"
        )
    else:
        st.sidebar.info("Load data on Upload to begin.")

    try:
        from modules.email_automation import email_configured as _email_ok

        st.sidebar.caption(
            "Email: configured" if _email_ok() else "Email: set .env to enable send/inbox"
        )
    except Exception:
        pass

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
        page_ai()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Email":
        page_email()


if __name__ == "__main__":
    main()
