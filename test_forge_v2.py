"""FORGE v2 comprehensive test — dual-mode, live sim, filters, engines, quality, ARIMA, insights, alerts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


def test_mode():
    from core.mode import DataMode, parse_mode
    assert parse_mode("manual") == DataMode.MANUAL
    assert parse_mode("Live") == DataMode.LIVE
    print("  mode: OK")


def test_filters():
    from core.filters import TopFilters, apply_buffer_filters, MAX_ROWS_HARD_CAP, filter_schema_for_domain
    f = TopFilters(values={"site": "Site-1"}, max_rows=100, domain="predictive_maintenance")
    assert f.effective_max_rows() == 100
    f2 = TopFilters(max_rows=999999)
    assert f2.effective_max_rows() == MAX_ROWS_HARD_CAP
    # Buffer filter
    df = pd.DataFrame({"site": ["A", "B", "A"], "val": [1, 2, 3]})
    out = apply_buffer_filters(df, TopFilters(values={"site": "A"}, max_rows=10))
    assert len(out) == 2
    # Industry schemas differ
    factory = [x["key"] for x in filter_schema_for_domain("predictive_maintenance")]
    health = [x["key"] for x in filter_schema_for_domain("healthcare")]
    assert "machine" in factory
    assert "hospital" in health
    assert "machine" not in health
    print("  filters: OK (industry-adaptive)")


def test_live_simulator():
    from core.filters import TopFilters
    from core.live.stubs import fetch_live_data, list_connectors, VIRTUAL_UNIVERSE_SIZE
    connectors = list_connectors()
    assert "demo_simulator" in connectors
    assert "azure" in connectors
    assert connectors["pymodbus"]["capability"] == "capable"
    assert connectors["demo_simulator"]["capability"] == "sim"
    assert VIRTUAL_UNIVERSE_SIZE > 1_000_000
    filters = TopFilters(values={"site": "Site-1"}, max_rows=50, domain="predictive_maintenance")
    df = fetch_live_data("demo_simulator", filters)
    assert len(df) <= 50
    # Healthcare filters produce healthcare columns
    hf = TopFilters(values={"hospital": "City General"}, max_rows=30, domain="healthcare")
    hdf = fetch_live_data("api", hf)
    assert "hospital" in hdf.columns or "wait_minutes" in hdf.columns
    print(f"  live_simulator: OK ({len(df)} rows; connectors={list(connectors)})")


def test_engine_polars():
    from core.engine import available_engines, clean_with_engine
    engines = available_engines()
    assert "pandas" in engines
    assert "polars" in engines
    df = pd.DataFrame({"a": [1, 2, None, 2], "b": ["x", "y", None, "x"]})
    clean_df, log = clean_with_engine(df, "polars")
    assert len(clean_df) > 0
    print(f"  engine_polars: OK (engines={engines})")


def test_quality_pipeline():
    from core.clean_quality import run_quality_pipeline
    df = pd.DataFrame({"a": [1, 2, 3, 100], "b": [4, 5, 6, 7]})
    report = run_quality_pipeline(df)
    assert "basic" in report
    assert report["basic"]["rows"] == 4
    assert "great_expectations" in report
    assert "cleanlab" in report
    assert "summary" in report
    print(f"  quality: OK (GE={report['great_expectations'].get('available')}, "
          f"ydata={report['ydata_profiling'].get('available')}, "
          f"cleanlab={report['cleanlab'].get('available')})")


def test_arima():
    from modules.ml_runner import run_model
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=30), "revenue": range(30)})
    result = run_model(df, model_id="ARIMA", target="revenue")
    assert result["ok"], result.get("error")
    assert "r2" in result["metrics"]
    print(f"  ARIMA: OK (R²={result['metrics']['r2']})")


def test_data_insights():
    from modules.ml_runner import run_model
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    result = run_model(df, model_id="DataInsights")
    assert result["ok"], result.get("error")
    assert "quality_report" in result
    print("  DataInsights: OK")


def test_optuna():
    from modules.optuna_tuner import optuna_available, tune_model
    assert optuna_available()
    df = pd.DataFrame({"a": range(50), "b": range(50, 100), "target": range(50)})
    result = tune_model(df, "RandomForestRegressor", "target", ["a", "b"], n_trials=5, domain="sales_forecasting")
    assert result["ok"], result.get("error")
    assert result.get("business_insight")
    print(f"  Optuna: OK (best={result['best_score']:.3f}; insight={result['business_insight'][:60]}...)")


def test_rag():
    from modules.rag import rag_available, build_index, query_rag
    df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [90, 85]})
    if rag_available():
        idx = build_index(df)
        result = query_rag(idx, "who scored highest?")
        print(f"  RAG: OK (source={result['source']})")
    else:
        result = query_rag(None, "test")
        assert "stub" in result["source"]
        print("  RAG: OK (keyword stub)")


def test_auto_dashboard():
    from modules.auto_dashboard import build_auto_dashboard
    df = pd.DataFrame({"temp": [70, 80, 90], "pressure": [100, 101, 102], "site": ["A", "B", "A"]})
    result = build_auto_dashboard(df, "predictive_maintenance")
    assert len(result["kpi_cards"]) > 0
    assert len(result["charts"]) > 0
    print("  auto_dashboard: OK")


def test_business_alerts():
    from modules.business_alerts import generate_alerts
    df = pd.DataFrame({"failure": [0, 0, 1, 1, 1], "rul": [200, 30, 10, 5, 150], "temperature": [70, 80, 99, 100, 60]})
    alerts = generate_alerts(df, "predictive_maintenance")
    assert len(alerts) > 0
    print(f"  business_alerts: OK ({len(alerts)} alerts)")


def test_templates():
    from core.templates import load_templates, get_template
    templates = load_templates()
    assert "predictive_maintenance" in templates
    assert "healthcare" in templates
    assert "erp_cloud" in templates
    t = get_template("healthcare")
    assert any(f["key"] == "hospital" for f in t.get("filter_fields", []))
    print("  templates: OK")


def test_live_pipeline_e2e():
    from core.filters import TopFilters
    from core.live.stubs import fetch_live_data
    from core.pipeline import run_pipeline
    filters = TopFilters(values={"site": "Site-1"}, max_rows=100, domain="predictive_maintenance")
    raw = fetch_live_data("demo_simulator", filters)
    result = run_pipeline(raw_df=raw, filename="live_test", persist=False, clean_engine="pandas", data_mode="live")
    assert result["clean_df"] is not None
    assert result["data_mode"] == "live"
    print(f"  live_pipeline_e2e: OK ({len(result['clean_df'])} rows, domain={result['domain']})")


if __name__ == "__main__":
    print("FORGE v2 Tests")
    print("=" * 40)
    test_mode()
    test_filters()
    test_live_simulator()
    test_engine_polars()
    test_quality_pipeline()
    test_arima()
    test_data_insights()
    test_optuna()
    test_rag()
    test_auto_dashboard()
    test_business_alerts()
    test_templates()
    test_live_pipeline_e2e()
    print("=" * 40)
    print("ALL FORGE v2 TESTS PASSED ✓")
