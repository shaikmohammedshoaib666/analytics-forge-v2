"""Smoke test pipeline without Streamlit UI."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.db import init_db, queue_email, list_queued_emails
from core.pipeline import run_pipeline
from modules.ml_runner import run_model


def main() -> None:
    init_db()
    for sample in [
        ROOT / "data" / "samples" / "sample_predictive_maintenance.csv",
        ROOT / "data" / "samples" / "sample_sales.csv",
    ]:
        print(f"\n=== {sample.name} ===")
        result = run_pipeline(source=sample, persist=True)
        print(f"domain={result['domain']} rows={len(result['clean_df'])} run_id={result['run_id']}")
        print(f"kpis={list(result['kpis'].keys())[:6]}")
        model_id = "random_forest_regressor"
        if result["domain"] == "predictive_maintenance":
            model_id = "random_forest_regressor"
        ml = run_model(result["clean_df"], model_id=model_id)
        print(f"ml ok={ml.get('ok')} metrics={ml.get('metrics') or ml.get('error')}")

    eid = queue_email("test@example.com", "Smoke test", "Hello from Analytics Forge")
    print(f"\nqueued email id={eid} count={len(list_queued_emails())}")
    assert eid > 0
    assert list_queued_emails()
    print("SMOKE OK")


if __name__ == "__main__":
    main()
