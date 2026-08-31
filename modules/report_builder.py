"""Report Builder — checklist assemble + colored HTML download (Plotly CDN).

Python 3.9 compatible. Reuses dashboard_charts HTML/plotly pack patterns.
"""
from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from modules.dashboard_charts import (
    build_core_charts,
    build_extended_charts,
    chart_specs_to_html,
    email_body_text,
    kpis_to_csv,
)

TILE_AUTO_KPIS = "auto_kpis"
TILE_CUSTOM_KPIS = "custom_kpis"
TILE_CORE = "core_charts"
TILE_EXTENDED = "extended_charts"
TILE_INSIGHTS = "insights"

TILE_LABELS = {
    TILE_AUTO_KPIS: "Auto KPIs",
    TILE_CUSTOM_KPIS: "Custom KPIs",
    TILE_CORE: "Core dashboard charts (4)",
    TILE_EXTENDED: "Extended charts (5)",
    TILE_INSIGHTS: "Insights / Top 3 actions",
}

DEFAULT_ORDER = [
    TILE_AUTO_KPIS,
    TILE_CUSTOM_KPIS,
    TILE_INSIGHTS,
    TILE_CORE,
    TILE_EXTENDED,
]


def _kpi_cards_html(kpis: Optional[dict[str, Any]], title: str = "KPIs") -> str:
    items = [(k, v) for k, v in (kpis or {}).items() if k != "Domain"]
    if not items:
        return (
            f"<section class='block'><h2>{html_lib.escape(title)}</h2>"
            f"<p>No KPIs selected.</p></section>"
        )
    cards = []
    for k, v in items:
        cards.append(
            "<div class='kpi-card'><div class='kpi-label'>{}</div>"
            "<div class='kpi-value'>{}</div></div>".format(
                html_lib.escape(str(k).replace("_", " ")),
                html_lib.escape(str(v)),
            )
        )
    domain = (kpis or {}).get("Domain")
    domain_bit = (
        f"<p class='meta'>Pack: {html_lib.escape(str(domain))}</p>" if domain else ""
    )
    return (
        f"<section class='block'><h2>{html_lib.escape(title)}</h2>{domain_bit}"
        f"<div class='kpi-grid'>{''.join(cards)}</div></section>"
    )


def _list_html(items: Optional[list[Any]], empty: str) -> str:
    cleaned = [str(i).strip() for i in (items or []) if str(i).strip()]
    if not cleaned:
        return f"<p>{html_lib.escape(empty)}</p>"
    return "<ol>" + "".join(f"<li>{html_lib.escape(x)}</li>" for x in cleaned) + "</ol>"


def build_report_html(
    *,
    tile_order: list[str],
    columns: int = 1,
    domain: str = "generic",
    source_name: str = "",
    auto_kpis: Optional[dict[str, Any]] = None,
    custom_kpis: Optional[dict[str, Any]] = None,
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
    core_specs: Optional[list[dict[str, Any]]] = None,
    extended_specs: Optional[list[dict[str, Any]]] = None,
    generated_at: Optional[str] = None,
) -> str:
    """Colored HTML report matching dashboard aesthetic (KPI cards + Plotly CDN)."""
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols = 2 if int(columns) == 2 else 1
    sections: list[str] = []
    plotly_js: Any = "cdn"
    for tile in tile_order:
        if tile == TILE_AUTO_KPIS:
            sections.append(_kpi_cards_html(auto_kpis, "Auto KPIs"))
        elif tile == TILE_CUSTOM_KPIS:
            sections.append(_kpi_cards_html(custom_kpis, "Custom KPIs"))
        elif tile == TILE_INSIGHTS:
            briefing_html = f"<p>{html_lib.escape(str(briefing))}</p>" if briefing else ""
            sections.append(
                "<section class='block'><h2>Insights / actions</h2>"
                f"{briefing_html}"
                f"<h3>Top 3 actions</h3>{_list_html(actions, 'No actions in session yet.')}"
                f"<h3>Insights</h3>{_list_html(insights, 'No insights pinned yet.')}"
                "</section>"
            )
        elif tile == TILE_CORE:
            charts = chart_specs_to_html(core_specs or [], include_plotlyjs=plotly_js)
            if any(s.get("fig") is not None for s in (core_specs or [])):
                plotly_js = False
            sections.append(
                f"<section class='block'><h2>Core charts</h2>"
                f"{charts or '<p>No core charts.</p>'}</section>"
            )
        elif tile == TILE_EXTENDED:
            charts = chart_specs_to_html(extended_specs or [], include_plotlyjs=plotly_js)
            if any(s.get("fig") is not None for s in (extended_specs or [])):
                plotly_js = False
            sections.append(
                f"<section class='block'><h2>Extended charts</h2>"
                f"{charts or '<p>No extended charts.</p>'}</section>"
            )

    grid_class = "report-grid cols-2" if cols == 2 else "report-grid cols-1"
    body_sections = f"<div class='{grid_class}'>" + "\n".join(sections) + "</div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Analytics Forge report</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 0; padding: 24px;
  color: #12263f; background: linear-gradient(180deg, #f0f4f8 0%, #fafbfc 40%, #fff 100%); }}
h1 {{ margin: 0 0 8px; font-size: 1.6rem; }}
h2 {{ color: #12263f; font-size: 1.15rem; margin: 0 0 12px; }}
h3 {{ font-size: 1rem; margin: 12px 0 6px; color: #244061; }}
.meta {{ color: #57606a; font-size: 0.92rem; }}
.report-grid.cols-1 {{ display: block; }}
.report-grid.cols-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media (max-width: 900px) {{ .report-grid.cols-2 {{ grid-template-columns: 1fr; }} }}
section.block {{ background: #fff; border: 1px solid #e6edf2; border-radius: 12px;
  padding: 16px; margin: 0 0 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }}
.kpi-card {{ background: linear-gradient(180deg, #f7fafc 0%, #eef2f7 100%);
  border: 1px solid #d0d7de; border-radius: 12px; padding: 14px 16px; min-height: 88px; }}
.kpi-label {{ font-weight: 600; font-size: 0.85rem; color: #57606a; margin-bottom: 6px; }}
.kpi-value {{ font-size: 1.35rem; font-weight: 700; color: #12263f; }}
section.chart {{ background: #fff; border: 1px solid #e6edf2; border-radius: 10px;
  padding: 12px; margin: 12px 0; }}
ol {{ margin: 0; padding-left: 1.2rem; }}
</style>
</head>
<body>
<header>
<h1>Analytics Forge v2 — custom report</h1>
<p class="meta"><b>Field:</b> {html_lib.escape(str(domain))} · <b>Source:</b> {html_lib.escape(str(source_name) or "session")} · <b>Generated:</b> {html_lib.escape(stamp)}</p>
<p class="meta">Colored KPI cards + interactive Plotly charts (CDN). Open in a browser.</p>
</header>
{body_sections}
</body>
</html>"""


def normalize_tile_order(selected: list[str], order: Optional[list[str]] = None) -> list[str]:
    """Keep selected tiles in preferred order; append any extras."""
    preferred = list(order or DEFAULT_ORDER)
    selected_set = set(selected)
    out = [t for t in preferred if t in selected_set]
    for t in selected:
        if t not in out:
            out.append(t)
    return out


def move_tile(order: list[str], tile: str, direction: str) -> list[str]:
    """Move tile up/down in order list."""
    items = list(order)
    if tile not in items:
        return items
    i = items.index(tile)
    if direction == "up" and i > 0:
        items[i - 1], items[i] = items[i], items[i - 1]
    elif direction == "down" and i < len(items) - 1:
        items[i + 1], items[i] = items[i], items[i + 1]
    return items


def assemble_custom_report(
    df: pd.DataFrame,
    *,
    selected_tiles: list[str],
    tile_order: Optional[list[str]] = None,
    columns: int = 1,
    auto_kpis: Optional[dict[str, Any]] = None,
    custom_kpis: Optional[dict[str, Any]] = None,
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
    domain: str = "generic",
    chart_domain: Optional[str] = None,
    source_name: str = "",
    roles: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build HTML + KPI CSV + email body for selected tiles only."""
    roles = roles or {}
    chart_dom = chart_domain if chart_domain is not None else domain
    selected = normalize_tile_order(list(selected_tiles), tile_order)

    core_specs: list[dict[str, Any]] = []
    extended_specs: list[dict[str, Any]] = []
    if TILE_CORE in selected:
        core_specs = build_core_charts(df, roles, chart_dom)
    if TILE_EXTENDED in selected:
        extended_specs = build_extended_charts(df, roles, chart_dom)

    html_report = build_report_html(
        tile_order=selected,
        columns=columns,
        domain=domain,
        source_name=source_name,
        auto_kpis=auto_kpis if TILE_AUTO_KPIS in selected else None,
        custom_kpis=custom_kpis if TILE_CUSTOM_KPIS in selected else None,
        insights=insights if TILE_INSIGHTS in selected else None,
        actions=actions if TILE_INSIGHTS in selected else None,
        briefing=briefing if TILE_INSIGHTS in selected else "",
        core_specs=core_specs,
        extended_specs=extended_specs,
    )

    merged_kpis: dict[str, Any] = {}
    if TILE_AUTO_KPIS in selected and auto_kpis:
        merged_kpis.update(auto_kpis)
    if TILE_CUSTOM_KPIS in selected and custom_kpis:
        for k, v in custom_kpis.items():
            merged_kpis[k if k not in merged_kpis else f"Custom_{k}"] = v

    return {
        "html": html_report,
        "kpi_csv": kpis_to_csv(merged_kpis),
        "body": email_body_text(
            kpis=merged_kpis,
            insights=insights if TILE_INSIGHTS in selected else None,
            actions=actions if TILE_INSIGHTS in selected else None,
            briefing=briefing if TILE_INSIGHTS in selected else "",
        ),
        "tile_order": selected,
        "core": core_specs,
        "extended": extended_specs,
    }


def render_report_builder_page(
    df: pd.DataFrame,
    *,
    auto_kpis: dict[str, Any],
    custom_kpis: dict[str, Any],
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
    domain_label: str = "generic",
    chart_domain: str = "generic",
    source_name: str = "",
    roles: Optional[dict[str, str]] = None,
    smtp_ok: bool = False,
    default_to: str = "",
    send_fn: Optional[Callable[..., str]] = None,
    key_prefix: str = "rb",
) -> None:
    """Streamlit UI for checklist, arrange, preview, download, optional email."""
    import streamlit as st

    from modules.dashboard_charts import render_chart_specs

    st.header("Report Builder")
    st.caption(
        "Pick tiles → arrange order → preview → download colored HTML (matches dashboard). "
        "CSV clean-data download stays on Clean / Email pages."
    )

    available = list(DEFAULT_ORDER)
    if not custom_kpis:
        st.caption("No custom KPIs yet — create some on **Auto KPIs → KPI Studio**.")
    if not (insights or actions or briefing):
        st.caption("Insights / Top 3 appear after you run Auto KPIs or pin charts.")

    selected = st.multiselect(
        "Include in report",
        options=available,
        default=[TILE_AUTO_KPIS, TILE_CORE, TILE_INSIGHTS],
        format_func=lambda t: TILE_LABELS.get(t, t),
        key=f"{key_prefix}_tiles",
    )
    if not selected:
        st.warning("Select at least one tile.")
        return

    order_key = f"{key_prefix}_order"
    if order_key not in st.session_state:
        st.session_state[order_key] = normalize_tile_order(selected)
    st.session_state[order_key] = normalize_tile_order(selected, st.session_state[order_key])

    st.markdown("##### Arrange tiles")
    for tile in list(st.session_state[order_key]):
        c1, c2, c3 = st.columns([4, 1, 1])
        with c1:
            st.write(TILE_LABELS.get(tile, tile))
        with c2:
            if st.button("↑", key=f"{key_prefix}_up_{tile}"):
                st.session_state[order_key] = move_tile(st.session_state[order_key], tile, "up")
                st.rerun()
        with c3:
            if st.button("↓", key=f"{key_prefix}_down_{tile}"):
                st.session_state[order_key] = move_tile(st.session_state[order_key], tile, "down")
                st.rerun()

    layout_cols = st.radio("Layout columns", [1, 2], index=0, horizontal=True, key=f"{key_prefix}_cols")

    pack = assemble_custom_report(
        df,
        selected_tiles=selected,
        tile_order=st.session_state[order_key],
        columns=int(layout_cols),
        auto_kpis=auto_kpis,
        custom_kpis=custom_kpis,
        insights=insights or [],
        actions=actions or [],
        briefing=briefing or "",
        domain=domain_label,
        chart_domain=chart_domain,
        source_name=source_name,
        roles=roles or {},
    )

    st.markdown("##### Preview")
    for tile in pack["tile_order"]:
        if tile == TILE_AUTO_KPIS:
            st.subheader("Auto KPIs")
            items = [(k, v) for k, v in (auto_kpis or {}).items() if k != "Domain"]
            for i in range(0, len(items), 4):
                cols_ui = st.columns(4)
                for j, (k, v) in enumerate(items[i : i + 4]):
                    with cols_ui[j]:
                        st.metric(str(k).replace("_", " "), v)
        elif tile == TILE_CUSTOM_KPIS:
            st.subheader("Custom KPIs")
            if not custom_kpis:
                st.caption("None saved.")
            else:
                items = list(custom_kpis.items())
                for i in range(0, len(items), 4):
                    cols_ui = st.columns(4)
                    for j, (k, v) in enumerate(items[i : i + 4]):
                        with cols_ui[j]:
                            st.metric(str(k).replace("_", " "), v)
        elif tile == TILE_INSIGHTS:
            st.subheader("Insights / Top 3")
            if briefing:
                st.info(briefing)
            for a in (actions or [])[:3]:
                st.markdown(f"- {a}")
            for ins in (insights or [])[:8]:
                st.markdown(f"- {ins}")
            if not (briefing or actions or insights):
                st.caption("No insights in session.")
        elif tile == TILE_CORE:
            st.subheader("Core charts")
            render_chart_specs(pack["core"], key_prefix=f"{key_prefix}_core")
        elif tile == TILE_EXTENDED:
            st.subheader("Extended charts")
            render_chart_specs(pack["extended"], key_prefix=f"{key_prefix}_ext")

    st.divider()
    st.subheader("Download / email")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.download_button(
            "Download colored HTML report",
            data=pack["html"].encode("utf-8"),
            file_name="forge-custom-report.html",
            mime="text/html",
            key=f"{key_prefix}_dl_html",
            type="primary",
        )
    with c2:
        st.download_button(
            "KPI CSV (selected)",
            data=pack["kpi_csv"],
            file_name="forge-report-kpis.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
        )
    with c3:
        to_addr = st.text_input("Email report to", value=default_to, key=f"{key_prefix}_email_to")
        if smtp_ok and send_fn is not None:
            if st.button("Email HTML report", key=f"{key_prefix}_email_btn"):
                try:
                    msg = send_fn(to_addr.strip(), pack["body"], pack["html"], pack["kpi_csv"])
                    st.success(msg)
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("Set EMAIL_USER + EMAIL_PASSWORD to email. HTML download still works.")
