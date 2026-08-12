"""Reusable Streamlit UI helpers."""
from __future__ import annotations

from html import escape
from typing import Any, Optional

import streamlit as st

from core.kpis import format_kpi_value
from ui.theme import KPI_PALETTE


def kpi_cards(kpis: dict[str, Any], max_cards: int = 8) -> None:
    if not kpis:
        st.info("No KPIs yet — run the pipeline first.")
        return
    items = list(kpis.items())[:max_cards]
    cards = []
    for i, (kid, item) in enumerate(items):
        if isinstance(item, dict):
            name = item.get("name", kid)
            val = format_kpi_value(item.get("value"))
        else:
            name, val = str(kid).replace("_", " ").title(), format_kpi_value(item)
        fg, bg = KPI_PALETTE[i % len(KPI_PALETTE)]
        cards.append(
            f"""
            <div class="af-kpi" style="background:{bg}; border-top:4px solid {fg};">
              <div class="label" style="color:{fg}">{escape(str(name))}</div>
              <div class="value" style="color:#0F172A">{escape(str(val))}</div>
            </div>
            """
        )
    st.markdown(f'<div class="af-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def download_df_button(df, label: str, file_name: str, key: str) -> None:
    if df is None:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=file_name, mime="text/csv", key=key)


def download_bytes_button(
    data: bytes,
    label: str,
    file_name: str,
    mime: str = "application/octet-stream",
    key: str = "dl",
) -> None:
    st.download_button(label, data=data, file_name=file_name, mime=mime, key=key)


def download_html_pack_button(pack_bytes: bytes, key: str = "pack_dl") -> None:
    download_bytes_button(
        pack_bytes,
        label="Download Final Pack (HTML)",
        file_name="analytics_forge_pack.html",
        mime="text/html",
        key=key,
    )


def show_ml_metrics(ml_result: Optional[dict]) -> None:
    if not ml_result:
        st.info("No ML results yet.")
        return
    if not ml_result.get("ok"):
        st.error(ml_result.get("error", "ML run failed"))
        return
    metrics = ml_result.get("metrics") or {}
    # colorful metric strip (skip nested/list payloads)
    fake_kpis = {
        str(k).upper(): v
        for k, v in metrics.items()
        if v is not None and not isinstance(v, (list, dict, tuple))
    }
    if fake_kpis:
        kpi_cards(fake_kpis, max_cards=8)
    st.caption(
        f"Model: `{ml_result.get('model_id')}` · Task: `{ml_result.get('task')}` · "
        f"Target: `{ml_result.get('target')}`"
    )

    briefing = ml_result.get("manager_briefing")
    if not briefing:
        try:
            from modules.manager_insights import build_manager_insight

            briefing = build_manager_insight(ml_result)
        except Exception:
            briefing = ""
    if briefing:
        st.markdown("#### For managers")
        st.info(briefing)
