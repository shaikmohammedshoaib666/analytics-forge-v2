"""Build downloadable HTML final pack."""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Optional


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _kpi_rows(kpis: Optional[dict]) -> str:
    if not kpis:
        return "<tr><td colspan='2'>No KPIs</td></tr>"
    rows = []
    for kid, item in kpis.items():
        if isinstance(item, dict):
            name = item.get("name", kid)
            val = item.get("value")
        else:
            name, val = kid, item
        rows.append(f"<tr><td>{_esc(name)}</td><td>{_esc(val)}</td></tr>")
    return "\n".join(rows)


def _list_items(items: Optional[list], empty="None") -> str:
    if not items:
        return f"<li>{_esc(empty)}</li>"
    return "\n".join(f"<li>{_esc(i)}</li>" for i in items)


def build_html_pack(
    domain: str = "generic",
    source_name: str = "",
    clean_log: Optional[list] = None,
    kpis: Optional[dict] = None,
    insights: Optional[list] = None,
    charts: Optional[list] = None,
    ml_metrics: Optional[dict] = None,
    briefing: str = "",
) -> bytes:
    """Return HTML pack as UTF-8 bytes for download."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    clean_rows = []
    for step in clean_log or []:
        clean_rows.append(
            "<tr>"
            f"<td>{_esc(step.get('operation'))}</td>"
            f"<td>{_esc(step.get('detail'))}</td>"
            f"<td>{_esc(step.get('rows_before'))}</td>"
            f"<td>{_esc(step.get('rows_after'))}</td>"
            "</tr>"
        )
    clean_html = "\n".join(clean_rows) or "<tr><td colspan='4'>No steps</td></tr>"

    chart_items = []
    for ch in charts or []:
        if isinstance(ch, dict):
            chart_items.append(
                f"{ch.get('title', ch.get('chart_type', 'chart'))} "
                f"({ch.get('lib', '')} / {ch.get('chart_type', '')})"
            )
        else:
            chart_items.append(str(ch))

    insight_items = insights or ([] if not briefing else [briefing])

    ml_rows = []
    if ml_metrics:
        metrics = ml_metrics.get("metrics", ml_metrics)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                ml_rows.append(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>")
        ml_meta = (
            f"<p>Model: <b>{_esc(ml_metrics.get('model_id', ''))}</b> — "
            f"Target: <b>{_esc(ml_metrics.get('target', ml_metrics.get('target_col', '')))}</b></p>"
        )
    else:
        ml_meta = "<p>No ML run included.</p>"
    ml_html = "\n".join(ml_rows) or "<tr><td colspan='2'>—</td></tr>"

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Analytics Forge Pack — {_esc(domain)}</title>
<style>
  body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ color: #0b3d5c; }}
  h2 {{ border-bottom: 2px solid #0b3d5c; padding-bottom: .25rem; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: .5rem .75rem; text-align: left; }}
  th {{ background: #e8f1f8; }}
  .meta {{ color: #555; }}
  pre {{ background: #f6f8fa; padding: 1rem; overflow: auto; white-space: pre-wrap; }}
</style>
</head>
<body>
  <h1>Analytics Forge — Final Pack</h1>
  <p class="meta">Generated {ts}</p>
  <p><b>Source:</b> {_esc(source_name)} &nbsp;|&nbsp; <b>Domain:</b> {_esc(domain)}</p>

  <h2>Briefing</h2>
  <pre>{_esc(briefing)}</pre>

  <h2>Cleaning Guide</h2>
  <table>
    <thead><tr><th>Operation</th><th>Pandas detail</th><th>Rows before</th><th>Rows after</th></tr></thead>
    <tbody>
    {clean_html}
    </tbody>
  </table>

  <h2>KPIs</h2>
  <table>
    <thead><tr><th>KPI</th><th>Value</th></tr></thead>
    <tbody>
    {_kpi_rows(kpis)}
    </tbody>
  </table>

  <h2>Insights</h2>
  <ul>
  {_list_items(insight_items)}
  </ul>

  <h2>Charts</h2>
  <ul>
  {_list_items(chart_items, empty="No charts added")}
  </ul>

  <h2>ML Metrics</h2>
  {ml_meta}
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
    {ml_html}
    </tbody>
  </table>

  <p class="meta">Analytics Forge Phase 1 — local SQLite-backed run pack</p>
</body>
</html>
"""
    return doc.encode("utf-8")


def build_html_pack_str(**kwargs) -> str:
    return build_html_pack(**kwargs).decode("utf-8")
