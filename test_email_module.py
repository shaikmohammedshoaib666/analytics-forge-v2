"""Offline checks for email automation helpers (no real SMTP needed)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.db import init_db, is_imap_processed, mark_imap_processed, update_email_status, queue_email, list_emails
from core.pipeline import run_pipeline
from modules.email_automation import build_report_attachments, config_status, email_configured


def main() -> None:
    init_db()
    status = config_status()
    print("email configured:", status["configured"], status)

    sample = ROOT / "data" / "samples" / "sample_sales.csv"
    pipe = run_pipeline(source=sample, persist=True)
    atts = build_report_attachments(
        domain=pipe["domain"],
        source_name=pipe["source_name"],
        clean_log=pipe["clean_log"],
        kpis=pipe["kpis"],
        insights=[str(pipe.get("briefing") or "")],
        charts=[],
        ml_metrics=None,
        briefing=str(pipe.get("briefing") or ""),
        clean_df=pipe["clean_df"],
    )
    assert any(name.endswith(".html") for name, _, _ in atts)
    assert any(name.endswith(".csv") for name, _, _ in atts)
    print("attachments:", [a[0] for a in atts], "html_bytes=", len(atts[0][1]))

    eid = queue_email(to_addr="demo@example.com", subject="t", body="b", run_id=pipe["run_id"])
    update_email_status(eid, "sent")
    mark_imap_processed("test-msg-1", eid, note="unit")
    assert is_imap_processed("test-msg-1")
    assert list_emails(5)
    print("EMAIL MODULE OK (SMTP live send requires .env credentials)")
    if not email_configured():
        print("NOTE: fill .env EMAIL_* to enable live send/inbox")


if __name__ == "__main__":
    main()
