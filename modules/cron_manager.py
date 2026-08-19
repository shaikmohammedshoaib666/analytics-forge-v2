"""
Cron job management — Monday report settings per user.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _use_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _get_client():
    from modules.supabase_auth import init_supabase_client
    return init_supabase_client()


def get_cron_config(user_id: str) -> Optional[dict]:
    if _use_supabase():
        client = _get_client()
        res = client.table("cron_jobs").select("*").eq("user_id", user_id).eq("job_type", "monday_report").limit(1).execute()
        return res.data[0] if res.data else None
    return st.session_state.get("_cron_local")


def save_cron_config(user_id: str, enabled: bool, email: str) -> bool:
    data = {
        "user_id": user_id,
        "job_type": "monday_report",
        "schedule": "0 7 * * 1",
        "enabled": enabled,
        "config": {"email": email},
        "updated_at": datetime.utcnow().isoformat(),
    }
    if _use_supabase():
        client = _get_client()
        client.table("cron_jobs").upsert(data, on_conflict="user_id,job_type").execute()
        return True
    st.session_state["_cron_local"] = data
    return True


def render_cron_settings(user_id: str):
    """Render Monday report cron settings UI."""
    st.subheader("📅 Monday Report (Cron)")
    st.caption("Your Monday brief will include: Top 3 actions, $ impact, KPI summary")

    existing = get_cron_config(user_id) or {}
    config = existing.get("config", {})

    enabled = st.toggle("Enable Monday email report", value=existing.get("enabled", False))
    email = st.text_input("Delivery email", value=config.get("email", ""), placeholder="you@company.com")

    if st.button("Save Cron Settings"):
        if enabled and not email:
            st.warning("Enter an email address for delivery")
        else:
            save_cron_config(user_id, enabled, email)
            st.success("Monday report settings saved.")

    st.info("When enabled, you'll receive a weekly brief every Monday at 7:00 AM UTC with your latest KPIs and top actions.")
