#!/usr/bin/env python3
"""Forge v2 integration smoke test — Manual + LIVE (no Streamlit UI)."""
from __future__ import annotations

import traceback
import numpy as np
import pandas as pd


class SS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as exc:
            raise AttributeError(k) from exc

    def __setattr__(self, k, v):
        self[k] = v


def main() -> int:
    import app as A

    errors: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            print(f"FAIL  {name}: {exc}")
            print(traceback.format_exc().splitlines()[-3:])

    def reset(mode: str = "MANUAL UPLOAD") -> None:
        A.st.session_state = SS(
            signed_in=True,
            mode=mode,
            page="Upload",
            manual_df=None,
            manual_name=None,
            clean_engine="pandas",
            gemini_api_key_override="",
            live_last_poll=0.0,
            live_status="idle",
            live_error=None,
            clean_df=None,
            clean_checks=None,
            clean_report=None,
            field_result=None,
            domain="generic",
            domain_meta=None,
            automl_result=None,
            forecast_text=None,
            chat_history=[],
            pipeline_started=False,
            prefer_clean_df=False,
            ml_result=None,
            dashboard_charts=[],
            dashboard_insights=[],
            llama_docs=None,
            llama_index_obj=None,
            llama_index_meta=None,
            live_cfg_override=None,
            live_auto_poll=False,
            live_last_row=None,
            live_insight_lines=None,
            live_insight_engine="prophet",
            uploaded_tables={},
            join_log=None,
            sql_lab_result=None,
            sql_lab_engine=None,
            sql_lab_query=None,
            usd_per_hour=0.0,
            usd_per_unit=0.0,
            column_roles={},
            forge_domain="generic",
            column_types={},
            forge_detect=None,
            forge_session_id=None,
            forge_session_title="",
            last_gemini_error="",
            domain_user_override=False,
        )

    rng = np.random.default_rng(0)
    pdm = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=80, freq="h"),
            "machine_id": rng.choice(["M1", "M2", "M3"], 80),
            "location": rng.choice(["Hyderabad", "Pune"], 80),
            "temperature": 75 + rng.normal(0, 4, 80),
            "vibration": 0.5 + rng.normal(0, 0.08, 80),
            "pressure": 5 + rng.normal(0, 0.2, 80),
            "rul": np.linspace(100, 40, 80),
            "failure": (rng.random(80) > 0.88).astype(int),
        }
    )
    sales = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "region": rng.choice(["East", "West"], 40),
            "revenue": rng.normal(500, 80, 40),
            "sales": rng.normal(500, 80, 40),
            "sku": [f"S{i%5}" for i in range(40)],
            "units": rng.integers(1, 20, 40),
            "customer_id": rng.integers(1, 15, 40),
        }
    )

    check("config", lambda: A.load_live_config())
    check("gateway", lambda: __import__("gateway").health())

    reset()
    A.st.session_state.manual_df = pdm

    def clean():
        _, checks = A.clean_data(pdm, engine="pandas")
        assert len(checks) >= 15

    check("clean", clean)
    check("field PdM", lambda: (_ for _ in ()).throw(AssertionError(A.detect_field(pdm, False, 3)["domain"])) if A.detect_field(pdm, False, 3)["domain"] != "predictive_maintenance" else None)
    check("field sales", lambda: (_ for _ in ()).throw(AssertionError(A.detect_field(sales, False, 3)["domain"])) if A.detect_field(sales, False, 3)["domain"] != "sales_forecasting" else None)

    students = pd.DataFrame(
        {
            "student_id": [f"S{i}" for i in range(30)],
            "age": rng.integers(16, 22, 30),
            "math_score": rng.integers(40, 100, 30),
            "reading_score": rng.integers(40, 100, 30),
            "writing_score": rng.integers(40, 100, 30),
            "gender": rng.choice(["M", "F"], 30),
        }
    )

    def field_student():
        import time as _t
        t0 = _t.perf_counter()
        meta = A.detect_field(students, False, 3)
        elapsed = _t.perf_counter() - t0
        assert meta["domain"] == "education", meta["domain"]
        assert meta["domain"] != "healthcare"
        assert elapsed < 12, elapsed
        t1 = _t.perf_counter()
        A.detect_field(students, False, 3)
        assert (_t.perf_counter() - t1) < 1.5

    check("field student fast", field_student)

    def dwdm_labs():
        from modules import dwdm_labs as L

        star = L.build_star_schema(
            students,
            date_col=None,
            entity_col="student_id",
            fact_cols=["math_score", "reading_score"],
        )
        assert star["ok"] and len(star["fact"]) >= 1 and "entity_dim" in star["dims"]
        txn = pd.DataFrame(
            {
                "order_id": [f"o{i//2}" for i in range(24)],
                "item": (["A", "B", "A", "C", "B", "C"] * 4),
            }
        )
        mined = L.mine_apriori(L.baskets_from_txn(txn, "order_id", "item"), min_support=0.2, min_confidence=0.4)
        assert mined["ok"] and len(mined["rules"]) >= 1
        empty = L.mine_apriori([])
        assert not empty["ok"]
        km = L.assign_kmeans(students, ["math_score", "reading_score"], k=3, silhouette=True)
        assert km["ok"] and "cluster_id" in km["frame"].columns
        dirty = students.copy()
        dirty.loc[0:3, "math_score"] = np.nan
        mice = L.mice_impute(dirty, ["math_score", "reading_score"], max_iter=4)
        assert mice["ok"] and mice["n_imputed"] >= 1 and mice["frame"]["math_score"].isna().sum() == 0

    check("dwdm labs helpers", dwdm_labs)

    def override_wins():
        reset()
        A.st.session_state.domain_user_override = True
        A.st.session_state.domain = "education"
        A.st.session_state.forge_domain = "education"
        meta = A.detect_field(pdm, False, 3)
        assert meta["domain"] == "education"
        assert meta.get("overridden")

    check("domain override wins", override_wins)
    check("ml RF", lambda: (_ for _ in ()).throw(AssertionError("ml")) if not A.run_forge_model(pdm, "RandomForestRegressor", target="rul").get("ok") else None)
    check("llama", lambda: A.ensure_llama_index(pdm, True))

    maint = pd.DataFrame(
        {
            "machine_id": ["M1", "M2", "M3"],
            "last_service_days": [12, 40, 7],
            "tech": ["A", "B", "A"],
        }
    )
    costs = pd.DataFrame(
        {
            "machine_id": ["M1", "M2", "M3"],
            "parts_cost": [120.0, 80.0, 200.0],
        }
    )
    empty_right = pd.DataFrame(columns=["machine_id", "note"])

    def joins():
        inner, m_in = A.join_two(pdm, maint, how="inner", on=["machine_id"])
        leftj, m_left = A.join_two(pdm, maint, how="left", on=["machine_id"])
        rightj, _ = A.join_two(pdm, maint, how="right", on=["machine_id"])
        outer, _ = A.join_two(pdm, maint, how="outer", on=["machine_id"])
        assert len(inner) >= 1 and len(leftj) >= len(pdm)
        assert len(rightj) >= 1 and len(outer) >= len(pdm)
        assert m_in["how"] == "inner" and m_left["keys"] == ["machine_id"]
        chained, logs = A.join_many(
            {"sensors": pdm, "maint": maint, "costs": costs},
            [
                {"left": "sensors", "right": "maint", "how": "left", "on": ["machine_id"]},
                {"left": "_result", "right": "costs", "how": "inner", "on": ["machine_id"]},
            ],
        )
        assert len(chained) >= 1 and "parts_cost" in chained.columns and len(logs) == 2
        empty_join, meta_e = A.join_two(pdm, empty_right, how="left", on=["machine_id"])
        assert isinstance(empty_join, pd.DataFrame) and meta_e["right_rows"] == 0
        out, dlog = A.apply_dwdm_transforms(pdm, bin_cols=["temperature"], smooth_cols=["vibration"])
        assert "temperature_dwdm_bin" in out.columns and dlog
        sql_df, eng = A.run_sql("SELECT * FROM sensors LIMIT 5", {"sensors": pdm})
        assert isinstance(sql_df, pd.DataFrame) and len(sql_df) == 5 and eng in {"duckdb", "pandas-fallback"}

    check("joins+sql", joins)

    reset()
    A.st.session_state.manual_df = pdm
    A.st.session_state.prefer_clean_df = True
    merged, _ = A.join_two(pdm, maint, how="left", on=["machine_id"])
    A.apply_joined_as_working(merged, {"manual_df": pdm, "maint": maint}, [{"how": "left"}])

    def manual_uses_join():
        got = A.get_data()
        assert isinstance(got, pd.DataFrame) and not got.empty
        assert "last_service_days" in got.columns
        assert len(got) == len(merged)

    check("manual get_data uses join", manual_uses_join)

    path = A.live_buffer_path()
    pdm.to_csv(path, index=False)
    reset("LIVE CONNECT")
    A.st.session_state.live_cfg_override = {**A.load_live_config(), "connection_type": "buffer_only"}
    A.st.session_state.clean_df = merged
    A.st.session_state.prefer_clean_df = True
    check("live buffer", lambda: (_ for _ in ()).throw(AssertionError("empty")) if A.ensure_live_poll(True).empty else None)

    def live_ignores_join():
        got = A.get_data()
        assert isinstance(got, pd.DataFrame) and not got.empty
        assert "last_service_days" not in got.columns

    check("LIVE ignores joined clean_df", live_ignores_join)

    def forge_os_helpers():
        import os
        from modules import forge_os as F
        from modules import domain_detect as D

        assert F.get_gemini_model()
        prev = os.environ.get("GEMINI_MODEL")
        try:
            for alias in (
                "gemini-flash-latest",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-pro",
            ):
                os.environ["GEMINI_MODEL"] = alias
                assert F.get_gemini_model() == "gemini-3.6-flash"
        finally:
            if prev is None:
                os.environ.pop("GEMINI_MODEL", None)
            else:
                os.environ["GEMINI_MODEL"] = prev
        assert F.gemini_issue_from_raw("", attempted=False) is None
        assert F.gemini_issue_from_raw("", attempted=True)
        assert str(F.gemini_issue_from_raw("[Gemini error] quota", attempted=True)).startswith("[Gemini error]")

        plant = pd.DataFrame(
            {
                "availability": [0.8],
                "downtime_minutes": [40],
                "scrap": [3],
                "asset_id": ["A1"],
                "oee": [0.6],
            }
        )
        assert F.looks_like_plant_oee(plant)["match"]
        assert not F.looks_like_plant_oee(sales)["match"]
        ctypes = D.detect_column_types(sales)
        assert ctypes.get("date") == "date"
        dmeta = D.detect_domain(sales, ctypes)
        assert dmeta["domain"] in {"sales", "forecasting"}
        assert "revenue" in D.roles_for_domain("sales")
        s_roles = D.suggest_roles(list(sales.columns), domain="sales", column_types=ctypes, df=sales)
        assert s_roles.get("revenue") in {"revenue", "metric"}

        impact = F.estimate_dollar_impact(plant, usd_per_hour=120.0, usd_per_unit=5.0)
        assert impact["ok"] and impact["total_usd"] > 0
        sales_impact = F.estimate_dollar_impact(
            sales,
            roles={"revenue": "revenue"},
            domain="sales",
        )
        assert sales_impact["ok"] and sales_impact["total_usd"] > 0

        F.save_named_mapping(
            "smoke_map",
            {"downtime_minutes": "downtime", "asset_id": "asset"},
            source_columns=list(plant.columns),
            domain="plant_oee",
        )
        loaded = F.load_named_mapping("smoke_map")
        assert loaded and loaded.get("downtime_minutes") == "downtime"
        applied = F.resolve_mapping_to_frame(["Downtime Minutes", "Asset", "qty"], loaded)
        assert applied
        recs = F.list_named_mappings()
        assert any(r.get("name") == "smoke_map" and r.get("domain") == "plant_oee" for r in recs)

        sid = F.new_session_id()
        F.save_session(sid, title="smoke", source_name="pdm.csv", frames={"clean_df": pdm}, meta={"domain": "predictive_maintenance"})
        frames = F.load_frames(sid)
        assert isinstance(frames.get("clean_df"), pd.DataFrame) and len(frames["clean_df"]) == len(pdm)
        meta = F.load_session_meta(sid)
        assert meta.get("domain") == "predictive_maintenance"
        brief = F.build_top3_actions(
            insights=["Vibration spike on M1"],
            dollar_impact=impact,
            use_gemini=False,
        )
        assert len(brief["actions"]) == 3

        students = pd.DataFrame(
            {
                "student_id": [f"S{i}" for i in range(24)],
                "age": rng.integers(16, 22, 24),
                "math_score": rng.integers(40, 100, 24),
                "reading_score": rng.integers(40, 100, 24),
                "gender": rng.choice(["M", "F"], 24),
            }
        )
        stypes = D.detect_column_types(students)
        smeta = D.detect_domain(students, stypes)
        assert smeta["domain"] == "education", smeta
        weak = pd.DataFrame({"age": rng.integers(10, 18, 12), "score": rng.integers(50, 90, 12)})
        wmeta = D.detect_domain(weak, D.detect_column_types(weak))
        assert wmeta["domain"] not in {"health", "healthcare"}

        gone = F.new_session_id()
        F.save_session(gone, title="to_delete", source_name="x.csv", frames={"clean_df": pdm}, meta={})
        assert F.session_exists(gone)
        assert F.delete_session(gone)
        assert not F.session_exists(gone)

        prev_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "forge-env-key-smoke"
        try:
            assert F.get_gemini_api_key() == "forge-env-key-smoke"
        finally:
            if prev_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = prev_key

    check("forge_os helpers", forge_os_helpers)

    def dashboard_charts_helpers():
        from modules import dashboard_charts as DC

        rng = np.random.default_rng(0)

        roles = {"date": "date", "revenue": "revenue", "region": "region", "customer_id": "customer_id"}
        core = DC.build_core_charts(sales, roles=roles, domain="sales")
        ext = DC.build_extended_charts(sales, roles=roles, domain="sales")
        assert len(core) == 4 and len(ext) == 5
        rendered = sum(1 for s in core + ext if s.get("fig") is not None)
        assert rendered >= 6
        pack = DC.assemble_dashboard_export(
            sales,
            kpis={"Rows": len(sales), "Total_Revenue": float(sales["revenue"].sum())},
            insights=["East region leads"],
            actions=["Review West region"],
            briefing="Smoke pack",
            domain="Sales",
            chart_domain="sales",
            source_name="sales.csv",
            roles=roles,
        )
        assert b"<!DOCTYPE html>" in pack["html"].encode("utf-8")
        assert b"plotly" in pack["html"].encode("utf-8").lower()
        assert pack["kpi_csv"]
        assert "forge-dashboard-report" in pack["body"] or "HTML" in pack["body"]

        plant = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "asset_id": rng.choice(["A1", "A2", "A3"], 30),
                "downtime_minutes": rng.integers(5, 120, 30),
                "scrap": rng.integers(0, 8, 30),
                "oee": rng.uniform(0.5, 0.9, 30),
            }
        )
        plant_roles = {"date": "date", "asset_id": "asset", "downtime_minutes": "downtime", "scrap": "scrap"}
        plant_ext = DC.build_extended_charts(plant, roles=plant_roles, domain="plant_oee")
        assert len(plant_ext) == 5
        assert any(s.get("fig") is not None for s in plant_ext)

        messy = pd.DataFrame(
            {
                "sold_on": pd.date_range("2024-01-01", periods=40, freq="D").astype(str),
                "amount_usd": rng.integers(10, 500, 40),
                "qty_units": rng.integers(1, 20, 40),
                "margin_pct": rng.uniform(0.1, 0.4, 40),
                "zone": rng.choice(["N", "S", "E", "W"], 40),
                "channel": rng.choice(["web", "store"], 40),
                "sales_rep": rng.choice(["Ana", "Bo", "Cy"], 40),
            }
        )
        ext_free = DC.build_extended_charts(messy, roles={}, domain="generic")
        assert len(ext_free) == 5
        assert sum(1 for s in ext_free if s.get("fig") is not None) == 5
        assert DC._metric_col(messy, {}, "sales") != "sales_rep"
        assert DC._date_col(messy, {}) == "sold_on"

        # Short hourly span uses resample("h") — uppercase "H" crashes pandas 2.2+/3.
        hourly = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
                "vibration": rng.normal(0.5, 0.08, 48),
                "machine_id": rng.choice(["M1", "M2"], 48),
            }
        )
        hourly_roles = {"timestamp": "timestamp", "vibration": "sensor", "machine_id": "asset"}
        hourly_ext = DC.build_extended_charts(
            hourly, roles=hourly_roles, domain="predictive_maintenance"
        )
        hourly_ts = next(s for s in hourly_ext if s["id"] == "ext_timeseries")
        assert hourly_ts.get("fig") is not None, hourly_ts.get("skip_reason")
        pd.date_range("2024-01-01", periods=3, freq="h")
        hourly.set_index("timestamp")["vibration"].resample("h").mean()

        few = pd.DataFrame(
            {
                "region": ["East-Region-Name-Long", "West"],
                "revenue": [10.0, 20.0],
            }
        )
        bar_v = DC.make_readable_bar(few, "region", "revenue", title="rev by region")
        assert bar_v.data[0].type == "bar"
        assert getattr(bar_v.data[0], "orientation", None) != "h"
        assert float(bar_v.layout.xaxis.tickangle) == -40
        assert int(bar_v.layout.margin.b or 0) >= 120
        assert bar_v.layout.xaxis.automargin is True
        ticktext = [str(t) for t in (bar_v.layout.xaxis.ticktext or [])]
        assert any("…" in t for t in ticktext)
        assert "East-Region-Name-Long" in [str(v) for v in bar_v.data[0].x]
        hover = str(bar_v.data[0].hovertemplate or "")
        assert "customdata" in hover

        many = pd.DataFrame(
            {
                "product": [f"Very Long Product Name {i} Extra" for i in range(12)],
                "revenue": list(range(12, 0, -1)),
            }
        )
        bar_h = DC.make_readable_bar(many, "product", "revenue", title="rev by product")
        assert getattr(bar_h.data[0], "orientation", None) == "h"
        assert int(bar_h.layout.margin.l or 0) >= 96
        assert "Very Long Product Name 0 Extra" in [str(v) for v in bar_h.data[0].y]

        vol = next(s for s in core if s["id"] == "core_volume")
        assert vol.get("fig") is not None
        pulse = next(s for s in core if s["id"] == "core_pulse")
        assert pulse.get("fig") is not None
        assert float(pulse["fig"].layout.xaxis.tickangle) == -40

        pin_fig = DC.fig_from_pin(few, {"chart_type": "bar", "x": "region", "y": "revenue", "title": "pin bar"})
        assert pin_fig is not None
        assert int(pin_fig.layout.margin.b or 0) >= 120

        joined = messy.rename(columns={"zone": "warehouse_zone"})
        reset()
        A.st.session_state.manual_df = messy
        A.st.session_state.clean_df = joined
        A.st.session_state.uploaded_tables = {"joined": joined}
        src, label = A.dashboard_source_frame(messy)
        assert "warehouse_zone" in src.columns
        assert "cleaned" in label or "joined" in label

    check("dashboard_charts helpers", dashboard_charts_helpers)

    if errors:
        print(f"\n{len(errors)} FAILURE(S)")
        for e in errors:
            print(" -", e)
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
