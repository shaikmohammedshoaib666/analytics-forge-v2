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

    path = A.live_buffer_path()
    pdm.to_csv(path, index=False)
    reset("LIVE CONNECT")
    A.st.session_state.live_cfg_override = {**A.load_live_config(), "connection_type": "buffer_only"}
    check("live buffer", lambda: (_ for _ in ()).throw(AssertionError("empty")) if A.ensure_live_poll(True).empty else None)

    if errors:
        print(f"\n{len(errors)} FAILURE(S)")
        for e in errors:
            print(" -", e)
        return 1
    print("\nALL SMOKE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
