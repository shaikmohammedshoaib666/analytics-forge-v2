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
            forge_session_id=None,
            forge_session_title="",
            last_gemini_error="",
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
    check("field PdM", lambda: (_ for _ in ()).throw(AssertionError(A.detect_field(pdm, False, 8)["domain"])) if A.detect_field(pdm, False, 8)["domain"] != "predictive_maintenance" else None)
    check("field sales", lambda: (_ for _ in ()).throw(AssertionError(A.detect_field(sales, False, 8)["domain"])) if A.detect_field(sales, False, 8)["domain"] != "sales_forecasting" else None)
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

        assert F.get_gemini_model()
        prev = os.environ.get("GEMINI_MODEL")
        os.environ["GEMINI_MODEL"] = "gemini-flash-latest"
        try:
            assert F.get_gemini_model() == "gemini-2.0-flash"
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

        impact = F.estimate_dollar_impact(plant, usd_per_hour=120.0, usd_per_unit=5.0)
        assert impact["ok"] and impact["total_usd"] > 0

        F.save_named_mapping("smoke_map", {"downtime_minutes": "downtime", "asset_id": "asset"}, source_columns=list(plant.columns))
        loaded = F.load_named_mapping("smoke_map")
        assert loaded and loaded.get("downtime_minutes") == "downtime"
        applied = F.resolve_mapping_to_frame(["Downtime Minutes", "Asset", "qty"], loaded)
        assert applied

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

    if errors:
        print(f"\n{len(errors)} FAILURE(S)")
        for e in errors:
            print(" -", e)
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
