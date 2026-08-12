"""Streamlit session state helpers."""
from __future__ import annotations

import streamlit as st


DEFAULTS = {
    "page": "Upload",
    "messy_df": None,
    "clean_df": None,
    "clean_log": [],
    "source_name": None,
    "domain": "generic",
    "domain_override": None,
    "classification": None,
    "kpis": {},
    "briefing": "",
    "schema": None,
    "run_id": None,
    "ml_result": None,
    "chat_history": [],
    "dashboard_charts": [],
    "dashboard_insights": [],
    "pipeline_done": False,
    "user": None,
    "data_mode": "manual",
    "quality_report": None,
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value if not isinstance(value, (list, dict)) else (
                value.copy() if isinstance(value, dict) else list(value)
            )


def reset_analysis_state() -> None:
    for key in (
        "messy_df",
        "clean_df",
        "clean_log",
        "source_name",
        "domain",
        "domain_override",
        "classification",
        "kpis",
        "briefing",
        "schema",
        "run_id",
        "ml_result",
        "pipeline_done",
    ):
        st.session_state[key] = DEFAULTS[key] if key != "clean_log" else []
    st.session_state["chat_history"] = []
    # keep dashboard optional — clear on new upload
    st.session_state["dashboard_charts"] = []
    st.session_state["dashboard_insights"] = []
