"""Chart builders for plotly / matplotlib / seaborn."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from config.settings import CHARTS_CATALOG_YAML


def load_charts_catalog(path: Optional[Path] = None) -> dict:
    """Return chart catalog keyed by chart id for the Streamlit UI."""
    p = path or CHARTS_CATALOG_YAML
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw = data.get("charts", data)
    out: dict[str, dict] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            cid = entry.get("id") or entry.get("name")
            if not cid:
                continue
            out[str(cid)] = {
                "id": cid,
                "label": entry.get("name") or entry.get("label") or cid,
                "needs": entry.get("needs", []),
                "libs": entry.get("libs", ["plotly", "matplotlib", "seaborn"]),
            }
    elif isinstance(raw, dict):
        for cid, meta in raw.items():
            if isinstance(meta, dict):
                out[str(cid)] = {
                    "id": cid,
                    "label": meta.get("name") or meta.get("label") or cid,
                    "needs": meta.get("needs", []),
                    "libs": meta.get("libs", ["plotly", "matplotlib", "seaborn"]),
                }
            else:
                out[str(cid)] = {"id": cid, "label": str(cid), "needs": [], "libs": ["plotly"]}
    return out


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    lib: str = "plotly",
    x: Optional[str] = None,
    y: Optional[str] = None,
    names: Optional[str] = None,
    values: Optional[str] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
) -> Any:
    """
    Build a chart figure from the catalog.
    Returns a plotly Figure or matplotlib Figure depending on lib.
    """
    lib = (lib or "plotly").lower()
    chart_type = (chart_type or "bar").lower()
    title = title or f"{chart_type.title()} Chart"

    if lib == "plotly":
        return _plotly(df, chart_type, x, y, names, values, title, color)
    if lib == "seaborn":
        return _seaborn(df, chart_type, x, y, names, values, title, color)
    return _matplotlib(df, chart_type, x, y, names, values, title, color)


def _plotly(df, chart_type, x, y, names, values, title, color):
    import plotly.express as px
    import plotly.io as pio

    palette = ["#0D9488", "#EA580C", "#2563EB", "#DB2777", "#CA8A04", "#0891B2", "#B91C1C", "#4F46E5"]
    pio.templates.default = "plotly_white"

    def _style(fig):
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(240,253,250,0.65)",
            font=dict(family="DM Sans, sans-serif", color="#134E4A"),
            title_font=dict(family="Space Grotesk, sans-serif", size=18, color="#0F766E"),
            colorway=palette,
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    if chart_type in ("line", "time_series"):
        return _style(px.line(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "bar" or chart_type == "kpi_card":
        return _style(px.bar(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "area":
        return _style(px.area(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "scatter":
        return _style(px.scatter(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type in ("hist", "histogram"):
        return _style(px.histogram(df, x=x or y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "box":
        return _style(px.box(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "violin":
        return _style(px.violin(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))
    if chart_type == "pie":
        return _style(px.pie(df, names=names or x, values=values or y, title=title, color_discrete_sequence=palette))
    if chart_type == "heatmap":
        nums = df.select_dtypes(include=[np.number])
        if nums.empty:
            raise ValueError("No numeric columns for heatmap")
        corr = nums.corr(numeric_only=True)
        return _style(px.imshow(corr, title=title or "Correlation heatmap", text_auto=True, color_continuous_scale="Tealgrn"))
    if chart_type == "table":
        import plotly.graph_objects as go

        return go.Figure(
            data=[
                go.Table(
                    header=dict(values=list(df.columns), fill_color="#0D9488", font=dict(color="white")),
                    cells=dict(values=[df[c].head(50) for c in df.columns], fill_color="#F0FDFA"),
                )
            ]
        )
    return _style(px.scatter(df, x=x, y=y, color=color, title=title, color_discrete_sequence=palette))


def _matplotlib(df, chart_type, x, y, names, values, title, color):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if chart_type in ("line", "time_series"):
        ax.plot(df[x], df[y])
    elif chart_type in ("bar", "kpi_card"):
        ax.bar(df[x].astype(str).head(30), pd.to_numeric(df[y], errors="coerce").head(30))
        ax.tick_params(axis="x", rotation=45)
    elif chart_type == "scatter":
        ax.scatter(df[x], df[y])
    elif chart_type in ("hist", "histogram"):
        ax.hist(pd.to_numeric(df[x or y], errors="coerce").dropna(), bins=20)
    elif chart_type == "box":
        ax.boxplot(pd.to_numeric(df[y or x], errors="coerce").dropna())
    elif chart_type == "heatmap":
        nums = df.select_dtypes(include=[np.number])
        im = ax.imshow(nums.corr(numeric_only=True), aspect="auto")
        fig.colorbar(im, ax=ax)
    elif chart_type == "area":
        ax.fill_between(range(len(df)), pd.to_numeric(df[y], errors="coerce").fillna(0))
    else:
        ax.plot(pd.to_numeric(df[y or x], errors="coerce").fillna(0))
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _seaborn(df, chart_type, x, y, names, values, title, color):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if chart_type in ("line", "time_series"):
        sns.lineplot(data=df, x=x, y=y, hue=color, ax=ax)
    elif chart_type in ("bar", "kpi_card"):
        sns.barplot(data=df.head(30), x=x, y=y, hue=color, ax=ax)
        ax.tick_params(axis="x", rotation=45)
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax)
    elif chart_type in ("hist", "histogram"):
        sns.histplot(data=df, x=x or y, hue=color, ax=ax)
    elif chart_type == "box":
        sns.boxplot(data=df, x=x, y=y, hue=color, ax=ax)
    elif chart_type == "violin":
        sns.violinplot(data=df, x=x, y=y, hue=color, ax=ax)
    elif chart_type == "heatmap":
        nums = df.select_dtypes(include=[np.number])
        sns.heatmap(nums.corr(numeric_only=True), ax=ax, annot=False)
    else:
        sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig
