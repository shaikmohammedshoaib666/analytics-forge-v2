"""Domain-aware Dashboard charts + full HTML report pack.

Python 3.9 compatible. Streamlit is imported only inside render helpers.
"""
from __future__ import annotations

import html as html_lib
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.domain_detect import FORGE_DOMAINS

# App.py DOMAIN_CATALOG keys → Forge OS packs
_APP_TO_FORGE = {
    "sales_forecasting": "sales",
    "telecom_churn": "churn",
    "predictive_maintenance": "predictive_maintenance",
    "warehouse_logistics": "quality",
    "finance_risk": "generic",
    "healthcare": "health",
    "education": "education",
    "energy_utilities": "generic",
    "agriculture_iot": "generic",
    "hr_people": "churn",
    "plant": "plant_oee",
    "oee": "plant_oee",
    "pdm": "predictive_maintenance",
}

_METRIC_ROLES = (
    "metric",
    "revenue",
    "qty",
    "quantity",
    "downtime",
    "loss",
    "scrap",
    "sensor",
    "target",
    "availability",
    "performance",
    "quality",
    "rul_target",
)
_DATE_ROLES = ("date", "timestamp")
_CAT_ROLES = (
    "category",
    "product",
    "customer",
    "customer_id",
    "asset",
    "region",
    "batch",
    "subscription",
)
_LOSS_ROLES = ("downtime", "loss", "scrap", "defect")


def normalize_domain(domain: Optional[str]) -> str:
    raw = str(domain or "generic").strip().lower()
    mapped = _APP_TO_FORGE.get(raw, raw)
    return mapped if mapped in FORGE_DOMAINS else "generic"


def _spec(chart_id: str, title: str, fig: Optional[go.Figure] = None, skip_reason: Optional[str] = None) -> dict[str, Any]:
    return {"id": chart_id, "title": title, "fig": fig, "skip_reason": skip_reason}


def _col_from_roles(df: pd.DataFrame, roles: dict[str, str], *wanted: str) -> Optional[str]:
    wanted_set = set(wanted)
    for col, mapped in (roles or {}).items():
        if mapped in wanted_set and col in df.columns:
            return col
    return None


def _find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    return _find_col_among([str(c) for c in df.columns], *names)


def _find_col_among(candidates: list[str], *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in candidates}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        needle = n.lower()
        for k, real in lower.items():
            if needle in k:
                return real
    return None


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        n = str(c).lower()
        if n in {"id", "index"} or n.endswith("_id"):
            continue
        out.append(str(c))
    return out


def _is_valid_metric(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not col or col not in df.columns:
        return False
    if pd.api.types.is_numeric_dtype(df[col]):
        return bool(_to_numeric(df[col]).notna().any())
    return bool(_to_numeric(df[col]).notna().sum() >= 5)


def _is_valid_category(df: pd.DataFrame, col: Optional[str], max_unique: int = 40) -> bool:
    if not col or col not in df.columns:
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return False
    nuniq = int(df[col].nunique(dropna=True))
    n_rows = max(len(df), 1)
    if nuniq < 2 or nuniq > max_unique:
        return False
    if nuniq >= n_rows * 0.9:
        return False
    return True


def _category_cols(df: pd.DataFrame, max_unique: int = 40) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if _is_valid_category(df, str(c), max_unique=max_unique):
            out.append(str(c))
    return out


def _looks_like_dates(series: pd.Series) -> bool:
    if series is None or getattr(series, "empty", True):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna()
    if sample.empty:
        return False
    if len(sample) > 80:
        sample = sample.iloc[:80]
    parsed = pd.to_datetime(sample, errors="coerce")
    if float(parsed.notna().mean()) < 0.6:
        return False
    if int(parsed.nunique(dropna=True)) < 3:
        return False
    years = parsed.dt.year.dropna()
    if years.empty:
        return False
    if int(years.min()) < 1970 or int(years.max()) > 2100:
        return False
    span = parsed.max() - parsed.min()
    return bool(pd.notna(span) and span >= pd.Timedelta(hours=1))


def _date_col(df: pd.DataFrame, roles: dict[str, str]) -> Optional[str]:
    """Mapped date role if valid, else datetime dtype / parseable strings. No Field lock."""
    hit = _col_from_roles(df, roles, *_DATE_ROLES)
    if hit and _looks_like_dates(df[hit]):
        return hit
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return str(c)
    scored: list[tuple[float, str]] = []
    for c in df.columns:
        series = df[c]
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not _looks_like_dates(series):
            continue
        sample = series.dropna()
        if len(sample) > 80:
            sample = sample.iloc[:80]
        rate = float(pd.to_datetime(sample, errors="coerce").notna().mean())
        scored.append((rate, str(c)))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    named = _find_col(df, "timestamp", "datetime", "date", "time", "month")
    if named and _looks_like_dates(df[named]):
        return named
    return None


def _metric_col(df: pd.DataFrame, roles: dict[str, str], domain: str) -> Optional[str]:
    """Mapped metric if numeric, else first numeric column. Names are hints only."""
    hit = _col_from_roles(df, roles, *_METRIC_ROLES)
    if hit and _is_valid_metric(df, hit):
        return hit
    nums = _numeric_cols(df)
    if not nums:
        for c in df.columns:
            if str(c) not in nums and _is_valid_metric(df, str(c)):
                nums.append(str(c))
    if not nums:
        return None
    prefs: dict[str, tuple[str, ...]] = {
        "plant_oee": ("oee", "availability", "downtime_hours", "downtime_minutes", "downtime", "scrap"),
        "quality": ("scrap", "defect", "reject", "fpy"),
        "sales": ("revenue", "sales", "gmv", "amount", "net_sales"),
        "forecasting": ("revenue", "sales", "qty", "quantity", "units"),
        "churn": ("churn", "arpu", "monthly_charges", "support_tickets", "tenure"),
        "predictive_maintenance": ("vibration", "temperature", "rul", "pressure", "failure"),
        "generic": ("revenue", "sales", "amount", "qty", "value"),
    }
    names = prefs.get(domain, ()) + prefs["generic"]
    found = _find_col_among(nums, *names)
    if found:
        return found
    return nums[0]


def _category_col(df: pd.DataFrame, roles: dict[str, str], domain: str) -> Optional[str]:
    """Mapped category if low-cardinality, else dtype inference. Names are hints only."""
    hit = _col_from_roles(df, roles, *_CAT_ROLES)
    if hit and _is_valid_category(df, hit):
        return hit
    cats = _category_cols(df)
    if not cats:
        return None
    cats_text = [c for c in cats if not pd.api.types.is_numeric_dtype(df[c])]
    pool = cats_text or cats
    prefs: dict[str, tuple[str, ...]] = {
        "plant_oee": ("asset_id", "asset", "machine_id", "machine", "line", "shift"),
        "quality": ("batch", "sku", "line", "defect_type"),
        "sales": ("customer_id", "customer", "product", "sku", "region"),
        "forecasting": ("sku", "product", "region"),
        "churn": ("customer_id", "customer", "segment", "plan"),
        "predictive_maintenance": ("machine_id", "asset_id", "asset", "machine", "location"),
        "generic": ("region", "location", "category", "product", "store"),
    }
    found = _find_col_among(pool, *prefs.get(domain, ()), *prefs["generic"])
    if found:
        return found
    return pool[0]


def _second_category(df: pd.DataFrame, roles: dict[str, str], domain: str, used: Optional[str]) -> Optional[str]:
    extra_roles = ("region", "product", "shift", "batch", "subscription")
    hit = _col_from_roles(df, roles, *extra_roles)
    if hit and hit != used and _is_valid_category(df, hit):
        return hit
    cats = [c for c in _category_cols(df) if c != used]
    if not cats:
        return None
    prefs = ("shift", "region", "location", "line", "month", "product", "sku")
    found = _find_col_among(cats, *prefs)
    if found:
        return found
    return cats[0]


def _loss_col(df: pd.DataFrame, roles: dict[str, str], domain: str) -> Optional[str]:
    hit = _col_from_roles(df, roles, *_LOSS_ROLES)
    if hit and _is_valid_metric(df, hit):
        return hit
    if domain in ("plant_oee", "quality", "predictive_maintenance"):
        found = _find_col(
            df,
            "downtime_minutes",
            "downtime_hours",
            "downtime",
            "scrap",
            "reject",
            "defect",
            "idle_hours",
        )
        if found and _is_valid_metric(df, found):
            return found
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return pd.to_numeric(series, errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = series if pd.api.types.is_datetime64_any_dtype(series) else pd.to_datetime(series, errors="coerce")
    tz = getattr(parsed.dt, "tz", None)
    if tz is not None:
        try:
            parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        except (TypeError, ValueError, AttributeError):
            try:
                parsed = parsed.dt.tz_localize(None)
            except Exception:
                pass
    return parsed


def _sample(df: pd.DataFrame, n: int = 3500) -> pd.DataFrame:
    if df is None or len(df) <= n:
        return df
    return df.sample(n, random_state=42)


def _short_tick(val: Any, max_len: int = 18) -> str:
    text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _unique_tick_vals(values: Any) -> list[Any]:
    out: list[Any] = []
    seen = set()
    if values is None:
        return out
    for v in list(values):
        key = "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
        if key in seen:
            continue
        seen.add(key)
        out.append("" if key == "" else v)
    return out


def _apply_layout(fig: go.Figure, title: str, *, kind: str = "default", n_cats: int = 0) -> go.Figure:
    margin = dict(l=48, r=28, t=56, b=56)
    height = 400
    if kind == "bar":
        margin = dict(l=56, r=28, t=56, b=140)
        height = 480
    elif kind == "bar_h":
        margin = dict(l=168, r=28, t=56, b=56)
        height = max(420, 30 * max(int(n_cats), 4) + 100)
    fig.update_layout(
        title=title,
        margin=margin,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=height,
        bargap=0.25,
    )
    tickfont = dict(size=12)
    if kind == "bar":
        x_kwargs: dict[str, Any] = dict(tickangle=-40, tickfont=tickfont, automargin=True)
        try:
            fig.update_xaxes(ticklabeloverflow="allow", **x_kwargs)
        except (ValueError, TypeError):
            fig.update_xaxes(**x_kwargs)
        fig.update_yaxes(tickfont=tickfont, automargin=True)
    elif kind == "bar_h":
        fig.update_yaxes(tickfont=tickfont, automargin=True)
        fig.update_xaxes(tickfont=tickfont, automargin=True)
    else:
        fig.update_xaxes(automargin=True, tickfont=tickfont)
        fig.update_yaxes(automargin=True, tickfont=tickfont)
    return fig


def style_bar_figure(
    fig: go.Figure,
    *,
    n_cats: Optional[int] = None,
    horizontal: Optional[bool] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Readable bar ticks: rotate / pad, truncate labels, keep full name on hover."""
    if fig is None:
        return fig
    bar_traces = [tr for tr in fig.data if getattr(tr, "type", None) == "bar"]
    if not bar_traces:
        return fig
    if horizontal is None:
        horizontal = any(getattr(tr, "orientation", None) == "h" for tr in bar_traces)

    labels: list[str] = []
    for tr in fig.data:
        seq = tr.y if horizontal else tr.x
        labels.extend(_unique_tick_vals(seq))
    labels = _unique_tick_vals(labels)
    shorts = [_short_tick(v) for v in labels]
    n = int(n_cats or len(labels) or 0)

    for tr in bar_traces:
        seq = tr.y if horizontal else tr.x
        if seq is None:
            continue
        full = ["" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v) for v in list(seq)]
        tr.customdata = np.array(full).reshape(-1, 1)
        if horizontal:
            tr.hovertemplate = "%{customdata[0]}<br>%{x}<extra></extra>"
        else:
            tr.hovertemplate = "%{customdata[0]}<br>%{y}<extra></extra>"

    resolved_title = title
    if resolved_title is None:
        try:
            resolved_title = str(fig.layout.title.text or "")
        except Exception:
            resolved_title = ""
    fig = _apply_layout(fig, resolved_title or "", kind="bar_h" if horizontal else "bar", n_cats=n)
    if labels:
        tick_kwargs = dict(tickmode="array", tickvals=labels, ticktext=shorts, automargin=True, tickfont=dict(size=12))
        if horizontal:
            fig.update_yaxes(**tick_kwargs)
        else:
            fig.update_xaxes(tickangle=-40, **tick_kwargs)
    return fig


def make_readable_bar(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    title: str = "",
    barmode: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
) -> go.Figure:
    """Charts page / pins / Dashboard bars: horizontal when many categories."""
    nuniq = 0
    categorical = False
    if plot_df is not None and x in plot_df.columns:
        nuniq = int(plot_df[x].nunique(dropna=False))
        categorical = not pd.api.types.is_numeric_dtype(plot_df[x])
    kwargs: dict[str, Any] = {"title": title or f"{y} by {x}"}
    if color and plot_df is not None and color in plot_df.columns:
        kwargs["color"] = color
    if barmode:
        kwargs["barmode"] = barmode
    if labels:
        kwargs["labels"] = labels
    if categorical and nuniq >= 8:
        fig = px.bar(plot_df, x=y, y=x, orientation="h", **kwargs)
        return style_bar_figure(fig, n_cats=nuniq, horizontal=True, title=kwargs["title"])
    fig = px.bar(plot_df, x=x, y=y, **kwargs)
    return style_bar_figure(fig, n_cats=nuniq, horizontal=False, title=kwargs["title"])


# -----------------------------------------------------------------------------
# Core (4) — always-on overview
# -----------------------------------------------------------------------------

def _core_pulse(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Core · Metric pulse (column means)"
    if domain == "plant_oee":
        oee_names = []
        for name in ("availability", "performance", "quality", "oee"):
            col = _col_from_roles(df, roles, name) or _find_col(df, name)
            if col:
                oee_names.append(col)
        if len(oee_names) >= 2:
            means = {c: float(_to_numeric(df[c]).mean()) for c in oee_names if _to_numeric(df[c]).notna().any()}
            if means:
                plot = pd.DataFrame({"metric": list(means.keys()), "mean": list(means.values())})
                fig = px.bar(plot, x="metric", y="mean", title="OEE-style component means")
                return _spec(
                    "core_pulse",
                    "Core · OEE-style bars",
                    style_bar_figure(fig, n_cats=len(plot), title="OEE-style component means"),
                )
    nums = _numeric_cols(df)[:6]
    metric = _metric_col(df, roles, domain)
    if metric and metric not in nums:
        nums = [metric] + nums
        nums = list(dict.fromkeys(nums))[:6]
    if not nums:
        return _spec("core_pulse", title, skip_reason="No numeric columns for a metric pulse.")
    means = {c: float(_to_numeric(df[c]).mean()) for c in nums if _to_numeric(df[c]).notna().any()}
    if not means:
        return _spec("core_pulse", title, skip_reason="Numeric columns have no finite values.")
    plot = pd.DataFrame({"metric": list(means.keys()), "mean": list(means.values())})
    fig = px.bar(plot, x="metric", y="mean")
    return _spec("core_pulse", title, style_bar_figure(fig, n_cats=len(plot), title=title))


def _core_volume(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Core · Volume by category"
    cat = _category_col(df, roles, domain)
    metric = _metric_col(df, roles, domain)
    if cat is None:
        return _spec("core_volume", title, skip_reason="No category column (product / customer / asset / region).")
    work = df[[cat] + ([metric] if metric else [])].copy()
    if metric:
        work["_v"] = _to_numeric(work[metric]).fillna(0)
        g = work.groupby(cat, dropna=False)["_v"].sum().reset_index()
        g.columns = [cat, "value"]
        ylab = f"sum({metric})"
    else:
        g = work.groupby(cat, dropna=False).size().reset_index(name="value")
        ylab = "row count"
    g = g.sort_values("value", ascending=False).head(12)
    g[cat] = g[cat].astype(str)
    n_cats = len(g)
    if n_cats >= 8:
        g = g.sort_values("value", ascending=True)
        fig = px.bar(g, x="value", y=cat, orientation="h", labels={"value": ylab})
        fig = style_bar_figure(fig, n_cats=n_cats, horizontal=True, title=f"{ylab} by {cat}")
    else:
        fig = px.bar(g, x=cat, y="value", labels={"value": ylab})
        fig = style_bar_figure(fig, n_cats=n_cats, horizontal=False, title=f"{ylab} by {cat}")
    return _spec("core_volume", title, fig)


def _core_scatter(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Core · Relationship scatter"
    nums = _numeric_cols(df)
    color = None
    x = y = None
    if domain == "predictive_maintenance":
        x = _find_col(df, "vibration", "vib") or (nums[0] if nums else None)
        y = _find_col(df, "temperature", "temp") or (nums[1] if len(nums) > 1 else None)
        color = _find_col(df, "failure", "fault", "alarm")
        title = "Core · Asset risk scatter (sensor vs sensor)"
    elif domain == "churn":
        x = _find_col(df, "tenure") or (nums[0] if nums else None)
        y = _find_col(df, "support_tickets", "tickets", "arpu", "monthly_charges", "churn") or (
            nums[1] if len(nums) > 1 else None
        )
        color = _find_col(df, "churn", "churn_flag")
        title = "Core · Tenure vs churn / tickets"
    else:
        metric = _metric_col(df, roles, domain)
        rest = [c for c in nums if c != metric]
        x = rest[0] if rest else (nums[0] if nums else None)
        y = metric or (nums[1] if len(nums) > 1 else None)
        color = _category_col(df, roles, domain)
        if color and df[color].nunique(dropna=True) > 12:
            color = None
    if not x or not y or x == y:
        return _spec("core_scatter", title, skip_reason="Need two numeric columns for a scatter.")
    plot = _sample(df[[c for c in (x, y, color) if c]].dropna(subset=[x, y]))
    plot[x] = _to_numeric(plot[x])
    plot[y] = _to_numeric(plot[y])
    fig = px.scatter(plot, x=x, y=y, color=color, opacity=0.7)
    return _spec("core_scatter", title, _apply_layout(fig, f"{y} vs {x}"))


def _core_share(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Core · Share of primary metric"
    cat = _category_col(df, roles, domain)
    metric = _metric_col(df, roles, domain)
    if cat is None:
        return _spec("core_share", title, skip_reason="No category column for a share / pie chart.")
    work = df[[cat] + ([metric] if metric else [])].copy()
    if metric:
        work["_v"] = _to_numeric(work[metric]).fillna(0)
        g = work.groupby(cat, dropna=False)["_v"].sum()
    else:
        g = work.groupby(cat, dropna=False).size()
    g = g.sort_values(ascending=False)
    if g.sum() <= 0:
        return _spec("core_share", title, skip_reason="Category totals are zero — nothing to share.")
    top = g.head(8)
    other = g.iloc[8:].sum() if len(g) > 8 else 0
    labels = [str(i) for i in top.index] + (["Other"] if other else [])
    values = list(top.values) + ([other] if other else [])
    fig = px.pie(names=labels, values=values, hole=0.35)
    return _spec("core_share", title, _apply_layout(fig, f"Share by {cat}"))


def build_core_charts(df: pd.DataFrame, roles: Optional[dict[str, str]] = None, domain: Optional[str] = None) -> list[dict[str, Any]]:
    roles = roles or {}
    dom = normalize_domain(domain)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        reason = "No rows to chart."
        return [
            _spec("core_pulse", "Core · Metric pulse", skip_reason=reason),
            _spec("core_volume", "Core · Volume by category", skip_reason=reason),
            _spec("core_scatter", "Core · Relationship scatter", skip_reason=reason),
            _spec("core_share", "Core · Share of primary metric", skip_reason=reason),
        ]
    return [
        _core_pulse(df, roles, dom),
        _core_volume(df, roles, dom),
        _core_scatter(df, roles, dom),
        _core_share(df, roles, dom),
    ]


# -----------------------------------------------------------------------------
# Extended (5) — analytical board
# -----------------------------------------------------------------------------

def _ext_timeseries(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Extended · Primary metric over time"
    dcol = _date_col(df, roles)
    metric = _metric_col(df, roles, domain)
    if domain == "plant_oee":
        metric = _loss_col(df, roles, domain) or _find_col(df, "scrap") or metric
        title = "Extended · Scrap / downtime over time"
    elif domain in ("sales", "forecasting"):
        title = "Extended · Revenue over time"
    elif domain == "predictive_maintenance":
        metric = _find_col(df, "vibration", "temperature", "sensor") or metric
        title = "Extended · Sensor over time"
    if dcol is None:
        return _spec("ext_timeseries", title, skip_reason="No date / timestamp column for a time series.")
    if metric is None:
        return _spec("ext_timeseries", title, skip_reason="No numeric metric (revenue / qty / sensor / downtime).")
    work = df[[dcol, metric]].copy()
    work["_dt"] = _parse_dates(work[dcol])
    work["_v"] = _to_numeric(work[metric])
    work = work.dropna(subset=["_dt", "_v"])
    if work.empty:
        return _spec("ext_timeseries", title, skip_reason=f"`{dcol}` did not parse as dates or `{metric}` is empty.")
    span = work["_dt"].max() - work["_dt"].min()
    # pandas 2.2+/3 dropped uppercase offset aliases (H → h); lowercase works on 2.0+.
    rule = "h" if pd.notna(span) and span <= pd.Timedelta(days=3) else "D"
    grouped = work.set_index("_dt")["_v"].resample(rule).sum().reset_index()
    grouped.columns = ["period", metric]
    if grouped[metric].fillna(0).sum() == 0:
        grouped = work.set_index("_dt")["_v"].resample(rule).mean().reset_index()
        grouped.columns = ["period", metric]
    fig = px.line(grouped, x="period", y=metric, markers=len(grouped) <= 60)
    return _spec("ext_timeseries", title, _apply_layout(fig, f"{metric} by {rule} ({dcol})"))


def _ext_topn(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Extended · Top-N category"
    cat = _category_col(df, roles, domain)
    metric = _metric_col(df, roles, domain)
    if domain in ("sales", "forecasting"):
        cat = _col_from_roles(df, roles, "customer", "customer_id", "product") or cat
        title = "Extended · Top customers / products"
    elif domain in ("plant_oee", "predictive_maintenance"):
        cat = _col_from_roles(df, roles, "asset") or cat
        title = "Extended · Top assets"
    if cat is None:
        return _spec("ext_topn", title, skip_reason="No category column for a Top-N bar (product / customer / asset / region).")
    n = 10
    work = df[[cat] + ([metric] if metric else [])].copy()
    if metric:
        work["_v"] = _to_numeric(work[metric]).fillna(0)
        g = work.groupby(cat, dropna=False)["_v"].sum().reset_index()
        g.columns = [cat, "value"]
        ylab = f"sum({metric})"
    else:
        g = work.groupby(cat, dropna=False).size().reset_index(name="value")
        ylab = "rows"
    g = g.sort_values("value", ascending=True).tail(n)
    g[cat] = g[cat].astype(str)
    fig = px.bar(g, x="value", y=cat, orientation="h", labels={"value": ylab})
    fig = style_bar_figure(fig, n_cats=len(g), horizontal=True, title=f"Top {n} {cat} by {ylab}")
    return _spec("ext_topn", title, fig)


def _pareto_frame(labels: pd.Series, values: pd.Series) -> Optional[pd.DataFrame]:
    g = pd.DataFrame({"label": labels.astype(str), "value": _to_numeric(values).fillna(0)})
    g = g.groupby("label", dropna=False)["value"].sum().sort_values(ascending=False).reset_index()
    g = g[g["value"] > 0]
    if g.empty or len(g) < 2:
        return None
    g = g.head(20)
    total = float(g["value"].sum()) or 1.0
    g["cum_pct"] = g["value"].cumsum() / total * 100.0
    return g


def _ext_pareto(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Extended · Pareto (80/20)"
    cat = _category_col(df, roles, domain)
    if domain in ("plant_oee", "quality"):
        metric = _loss_col(df, roles, domain) or _metric_col(df, roles, domain)
        title = "Extended · Downtime / scrap Pareto"
    elif domain in ("sales", "forecasting"):
        metric = _metric_col(df, roles, domain)
        title = "Extended · Revenue concentration (80/20)"
    else:
        metric = _loss_col(df, roles, domain) or _metric_col(df, roles, domain)
    if cat is None:
        return _spec("ext_pareto", title, skip_reason="No category column for a Pareto (need asset / customer / product / region).")
    if metric is None:
        return _spec("ext_pareto", title, skip_reason="No numeric loss or revenue column for a Pareto.")
    g = _pareto_frame(df[cat], df[metric])
    if g is None:
        return _spec("ext_pareto", title, skip_reason=f"`{metric}` by `{cat}` has fewer than 2 positive groups.")
    ticks = g["label"].astype(str)
    fig = go.Figure()
    fig.add_bar(x=ticks, y=g["value"], name=str(metric))
    fig.add_scatter(x=ticks, y=g["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers")
    fig.add_scatter(
        x=ticks,
        y=[80] * len(g),
        name="80%",
        yaxis="y2",
        mode="lines",
        line=dict(dash="dash"),
    )
    fig.update_layout(
        yaxis=dict(title=str(metric)),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
        barmode="group",
    )
    fig = style_bar_figure(fig, n_cats=len(g), horizontal=False, title=f"Pareto of {metric} by {cat}")
    return _spec("ext_pareto", title, fig)


def _ext_hist(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Extended · Distribution of key numeric"
    metric = _metric_col(df, roles, domain)
    if domain == "churn":
        metric = _find_col(df, "tenure", "arpu", "monthly_charges") or metric
        title = "Extended · Tenure / ARPU distribution"
    elif domain == "predictive_maintenance":
        metric = _find_col(df, "vibration", "temperature", "rul") or metric
    if metric is None:
        return _spec("ext_hist", title, skip_reason="No numeric column for a histogram.")
    series = _to_numeric(df[metric]).dropna()
    if len(series) < 5:
        return _spec("ext_hist", title, skip_reason=f"`{metric}` has fewer than 5 numeric values.")
    plot = pd.DataFrame({metric: series})
    fig = px.histogram(plot, x=metric, nbins=min(40, max(10, int(np.sqrt(len(series))))))
    return _spec("ext_hist", title, _apply_layout(fig, f"Distribution of {metric}"))


def _ext_heatmap(df: pd.DataFrame, roles: dict[str, str], domain: str) -> dict[str, Any]:
    title = "Extended · Metric by two categories"
    metric = _metric_col(df, roles, domain)
    cat1 = _category_col(df, roles, domain)
    dcol = _date_col(df, roles)
    cat2 = _second_category(df, roles, domain, cat1)

    if dcol is not None and cat1 is not None and metric is not None:
        work = df[[dcol, cat1, metric]].copy()
        work["_dt"] = _parse_dates(work[dcol])
        work["_v"] = _to_numeric(work[metric])
        work = work.dropna(subset=["_dt", "_v"])
        if not work.empty and work[cat1].nunique(dropna=True) <= 24:
            work["_period"] = work["_dt"].dt.to_period("M").astype(str)
            piv = work.pivot_table(index="_period", columns=cat1, values="_v", aggfunc="mean")
            if piv.shape[0] >= 2 and piv.shape[1] >= 2:
                fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", labels={"color": metric})
                return _spec("ext_heatmap", title, _apply_layout(fig, f"{metric} · month × {cat1}"))

    if cat1 and cat2 and metric and cat1 != cat2:
        work = df[[cat1, cat2, metric]].copy()
        work["_v"] = _to_numeric(work[metric])
        n1, n2 = work[cat1].nunique(dropna=True), work[cat2].nunique(dropna=True)
        if 2 <= n1 <= 20 and 2 <= n2 <= 20:
            piv = work.pivot_table(index=cat1, columns=cat2, values="_v", aggfunc="mean")
            if piv.shape[0] >= 2 and piv.shape[1] >= 2:
                fig = px.imshow(piv, aspect="auto", color_continuous_scale="Teal", labels={"color": metric})
                return _spec("ext_heatmap", f"Extended · {cat1} × {cat2}", _apply_layout(fig, f"{metric} by {cat1} × {cat2}"))
            grouped = work.groupby([cat1, cat2], dropna=False)["_v"].mean().reset_index()
            fig = px.bar(grouped, x=cat1, y="_v", color=cat2, barmode="group", labels={"_v": metric})
            fig = style_bar_figure(fig, n_cats=int(n1), title=f"{metric} by {cat1} × {cat2}")
            return _spec("ext_heatmap", title, fig)

    nums = _numeric_cols(df)[:8]
    if len(nums) >= 3:
        corr = df[nums].apply(_to_numeric).corr()
        fig = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu", zmin=-1, zmax=1)
        return _spec("ext_heatmap", "Extended · Numeric correlation heatmap", _apply_layout(fig, "Correlation of numeric columns"))
    return _spec(
        "ext_heatmap",
        title,
        skip_reason="Need month×category, two categories, or 3+ numerics for a heatmap.",
    )


def build_extended_charts(df: pd.DataFrame, roles: Optional[dict[str, str]] = None, domain: Optional[str] = None) -> list[dict[str, Any]]:
    roles = roles or {}
    dom = normalize_domain(domain)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        reason = "No rows to chart."
        return [
            _spec("ext_timeseries", "Extended · Primary metric over time", skip_reason=reason),
            _spec("ext_topn", "Extended · Top-N category", skip_reason=reason),
            _spec("ext_pareto", "Extended · Pareto (80/20)", skip_reason=reason),
            _spec("ext_hist", "Extended · Distribution of key numeric", skip_reason=reason),
            _spec("ext_heatmap", "Extended · Metric by two categories", skip_reason=reason),
        ]
    return [
        _ext_timeseries(df, roles, dom),
        _ext_topn(df, roles, dom),
        _ext_pareto(df, roles, dom),
        _ext_hist(df, roles, dom),
        _ext_heatmap(df, roles, dom),
    ]


def build_all_dashboard_charts(
    df: pd.DataFrame,
    roles: Optional[dict[str, str]] = None,
    domain: Optional[str] = None,
) -> list[dict[str, Any]]:
    """4 core + 5 extended = 9 chart slots (skipped slots still returned)."""
    return build_core_charts(df, roles, domain) + build_extended_charts(df, roles, domain)


def fig_from_pin(df: pd.DataFrame, meta: dict[str, Any]) -> Optional[go.Figure]:
    """Rebuild a Plotly figure from a Charts-page pin (best-effort)."""
    if not meta or df is None or df.empty:
        return None
    x = meta.get("x")
    y = meta.get("y")
    chart_type = str(meta.get("chart_type") or "bar")
    color = meta.get("color")
    if x not in df.columns or y not in df.columns:
        return None
    plot_df = df
    if chart_type in ("bar", "pie") and df[x].nunique() > 30 and not pd.api.types.is_numeric_dtype(df[x]):
        ynum = _to_numeric(df[y])
        plot_df = df.assign(_y=ynum).groupby(x, dropna=False)["_y"].mean().reset_index()
        plot_df = plot_df.rename(columns={"_y": y})
    try:
        if chart_type == "line":
            fig = px.line(plot_df, x=x, y=y, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x, y=y, color=color)
        elif chart_type == "pie":
            fig = px.pie(plot_df, names=x, values=y)
        elif chart_type == "heatmap":
            if color and color in plot_df.columns:
                piv = plot_df.pivot_table(index=x, columns=color, values=y, aggfunc="mean")
                fig = px.imshow(piv, aspect="auto")
            else:
                fig = px.density_heatmap(plot_df, x=x, y=y)
        else:
            fig = make_readable_bar(plot_df, x, y, color=color, title=str(meta.get("title") or f"{y} by {x}"))
            return fig
        return _apply_layout(fig, str(meta.get("title") or f"{y} by {x}"))
    except Exception:
        return None


def pins_to_specs(df: pd.DataFrame, pins: Optional[list[Any]], limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, meta in enumerate(list(pins or [])[:limit]):
        if not isinstance(meta, dict):
            continue
        title = str(meta.get("title") or f"Pinned {i + 1}")
        fig = fig_from_pin(df, meta)
        if fig is None:
            out.append(_spec(f"pin_{i}", f"Pinned · {title}", skip_reason="Could not rebuild this pin on the filtered frame."))
        else:
            out.append(_spec(f"pin_{i}", f"Pinned · {title}", fig))
    return out


# -----------------------------------------------------------------------------
# Streamlit render
# -----------------------------------------------------------------------------

def render_chart_specs(
    specs: list[dict[str, Any]],
    key_prefix: str = "dash",
    *,
    silent_skip: bool = False,
) -> None:
    import streamlit as st

    visible = [s for s in specs if s.get("fig") is not None] if silent_skip else list(specs)
    if not visible:
        return
    for i in range(0, len(visible), 2):
        cols = st.columns(2)
        pair = visible[i : i + 2]
        for j, spec in enumerate(pair):
            with cols[j]:
                st.markdown(f"**{spec.get('title') or spec.get('id')}**")
                fig = spec.get("fig")
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{spec.get('id')}")
                elif not silent_skip:
                    st.caption(spec.get("skip_reason") or "Skipped — required columns are missing.")


def render_core_charts(df: pd.DataFrame, roles: Optional[dict[str, str]] = None, domain: Optional[str] = None) -> list[dict[str, Any]]:
    import streamlit as st

    specs = build_core_charts(df, roles, domain)
    st.subheader("Core charts")
    st.caption("Four overview views from mapped roles / detected columns.")
    render_chart_specs(specs, key_prefix="core")
    return specs


def render_extended_charts(df: pd.DataFrame, roles: Optional[dict[str, str]] = None, domain: Optional[str] = None) -> list[dict[str, Any]]:
    """Five extra business charts. Heading always visible; skip a slot silently if it cannot be built."""
    import streamlit as st

    specs = build_extended_charts(df, roles, domain)
    st.subheader("Extended charts")
    st.caption(
        "Time series, Top-N, Pareto, distribution, heatmap — inferred from date / numeric / category columns. "
        "Column mapping is optional."
    )
    render_chart_specs(specs, key_prefix="ext", silent_skip=True)
    return specs


# -----------------------------------------------------------------------------
# Export pack (HTML + KPI CSV + email body)
# -----------------------------------------------------------------------------

def kpis_to_csv(kpis: Optional[dict[str, Any]]) -> bytes:
    rows = [{"kpi": str(k), "value": v} for k, v in (kpis or {}).items()]
    if not rows:
        rows = [{"kpi": "(none)", "value": ""}]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def email_body_text(
    *,
    kpis: Optional[dict[str, Any]] = None,
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
) -> str:
    lines = [
        "Analytics Forge v2 — full dashboard report",
        "",
        "Charts are in the attached HTML file (forge-dashboard-report.html).",
        "Open that file in a browser to see KPIs, insights, and interactive Plotly charts.",
        "",
    ]
    if briefing:
        lines.extend(["Briefing", str(briefing).strip(), ""])
    actions = [str(a).strip() for a in (actions or []) if str(a).strip()]
    if actions:
        lines.append("Top 3 actions")
        for i, a in enumerate(actions[:3], 1):
            lines.append(f"{i}. {a}")
        lines.append("")
    insights = [str(i).strip() for i in (insights or []) if str(i).strip()]
    if insights:
        lines.append("Insights")
        for item in insights[:8]:
            lines.append(f"- {item}")
        lines.append("")
    if kpis:
        lines.append("KPI summary")
        for k, v in list(kpis.items())[:16]:
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _kpi_table_html(kpis: Optional[dict[str, Any]]) -> str:
    rows = []
    for k, v in (kpis or {}).items():
        rows.append(
            "<tr><th>{}</th><td>{}</td></tr>".format(
                html_lib.escape(str(k)),
                html_lib.escape(str(v)),
            )
        )
    if not rows:
        rows.append("<tr><td colspan='2'>No KPIs</td></tr>")
    return "<table><thead><tr><th>KPI</th><th>Value</th></tr></thead><tbody>{}</tbody></table>".format(
        "".join(rows)
    )


def _list_html(items: Optional[list[Any]], empty: str) -> str:
    cleaned = [str(i).strip() for i in (items or []) if str(i).strip()]
    if not cleaned:
        return f"<p>{html_lib.escape(empty)}</p>"
    return "<ol>" + "".join(f"<li>{html_lib.escape(x)}</li>" for x in cleaned) + "</ol>"


def chart_specs_to_html(specs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    js_mode: Any = "cdn"
    for spec in specs:
        title = html_lib.escape(str(spec.get("title") or spec.get("id") or "Chart"))
        fig = spec.get("fig")
        if fig is None:
            continue
        fragment = fig.to_html(include_plotlyjs=js_mode, full_html=False, config={"displayModeBar": False})
        js_mode = False
        chunks.append(f"<section class='chart'><h3>{title}</h3>{fragment}</section>")
    return "\n".join(chunks)


def assemble_dashboard_export(
    df: pd.DataFrame,
    *,
    kpis: Optional[dict[str, Any]] = None,
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
    domain: str = "generic",
    chart_domain: Optional[str] = None,
    source_name: str = "",
    roles: Optional[dict[str, str]] = None,
    pins: Optional[list[Any]] = None,
    core_specs: Optional[list[dict[str, Any]]] = None,
    extended_specs: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build HTML + KPI CSV + email body from the same chart specs shown on Dashboard."""
    roles = roles or {}
    chart_dom = chart_domain if chart_domain is not None else domain
    core_specs = core_specs if core_specs is not None else build_core_charts(df, roles, chart_dom)
    extended_specs = extended_specs if extended_specs is not None else build_extended_charts(df, roles, chart_dom)
    pin_specs = pins_to_specs(df, pins)
    specs = list(core_specs) + list(extended_specs) + pin_specs
    html_report = build_dashboard_report_html(
        domain=domain,
        source_name=source_name,
        kpis=kpis,
        insights=insights,
        actions=actions,
        briefing=briefing,
        chart_specs=specs,
    )
    return {
        "html": html_report,
        "kpi_csv": kpis_to_csv(kpis),
        "body": email_body_text(kpis=kpis, insights=insights, actions=actions, briefing=briefing),
        "core": core_specs,
        "extended": extended_specs,
        "specs": specs,
    }


def render_export_controls(
    *,
    html_report: str,
    kpi_csv: bytes,
    email_body: str,
    smtp_ok: bool,
    default_to: str = "",
    key_prefix: str = "dash",
    send_fn: Optional[Any] = None,
) -> None:
    """Download HTML pack + optional KPI CSV; email full report when SMTP is configured."""
    import streamlit as st

    st.subheader("Export / share")
    st.caption("Full pack = KPIs + Top 3 insights + all dashboard charts (HTML). KPI-only email stays on Auto KPIs.")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        st.download_button(
            "Export dashboard (KPIs + insights + charts)",
            data=html_report.encode("utf-8"),
            file_name="forge-dashboard-report.html",
            mime="text/html",
            key=f"{key_prefix}_dl_html",
            type="primary",
        )
    with c2:
        st.download_button(
            "KPI CSV",
            data=kpi_csv,
            file_name="forge-kpis.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv",
        )
    with c3:
        to_addr = st.text_input("Email full report to", value=default_to, key=f"{key_prefix}_email_to")
        if smtp_ok and send_fn is not None:
            if st.button("Email full report", key=f"{key_prefix}_email_btn"):
                try:
                    msg = send_fn(to_addr.strip(), email_body, html_report, kpi_csv)
                    st.success(msg)
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.caption("Set EMAIL_USER and EMAIL_PASSWORD in env to email. HTML download still works.")


def build_dashboard_report_html(
    *,
    domain: str = "generic",
    source_name: str = "",
    kpis: Optional[dict[str, Any]] = None,
    insights: Optional[list[Any]] = None,
    actions: Optional[list[Any]] = None,
    briefing: str = "",
    chart_specs: Optional[list[dict[str, Any]]] = None,
    generated_at: Optional[str] = None,
) -> str:
    stamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    charts_html = chart_specs_to_html(chart_specs or [])
    briefing_html = f"<p>{html_lib.escape(briefing)}</p>" if briefing else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Analytics Forge dashboard report</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 24px; color: #1a1a1a; background: #fafbfc; }}
h1,h2,h3 {{ color: #12263f; }}
table {{ border-collapse: collapse; background: #fff; }}
th,td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left; }}
section.chart {{ background: #fff; border: 1px solid #e6edf2; border-radius: 10px; padding: 12px; margin: 16px 0; }}
section.skip p {{ color: #57606a; }}
.meta {{ color: #57606a; }}
</style>
</head>
<body>
<h1>Analytics Forge v2 — dashboard report</h1>
<p class="meta"><b>Field:</b> {html_lib.escape(str(domain))} · <b>Source:</b> {html_lib.escape(str(source_name) or "session")} · <b>Generated:</b> {html_lib.escape(stamp)}</p>
<p class="meta">This file includes KPIs, Top 3 actions / insights, and Plotly charts (CDN). Open in a browser.</p>
<h2>Briefing</h2>
{briefing_html}
<h2>Top 3 actions</h2>
{_list_html(actions, "No manager actions yet — run Auto KPIs / Field.")}
<h2>Insights</h2>
{_list_html(insights, "No pinned insights yet.")}
<h2>KPIs</h2>
{_kpi_table_html(kpis)}
<h2>Charts</h2>
{charts_html or "<p>No charts available.</p>"}
</body>
</html>"""
