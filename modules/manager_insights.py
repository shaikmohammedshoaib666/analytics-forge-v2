"""Plain-English manager briefings from ML metrics (no jargon-first)."""
from __future__ import annotations

from typing import Any, Optional


def _fmt(n: Any, digits: int = 2) -> str:
    try:
        x = float(n)
    except (TypeError, ValueError):
        return str(n)
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 10:
        return f"{x:,.1f}"
    return f"{x:.{digits}f}"


def _pct(n: Any) -> str:
    try:
        return f"{float(n) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_manager_insight(result: Optional[dict[str, Any]]) -> str:
    """
    One short briefing a manager can read under the metrics.
    Covers Prophet forecasts and the usual sklearn-style models.
    """
    if not result or not result.get("ok"):
        return ""

    model = str(result.get("model_id") or "Model")
    task = str(result.get("task") or "")
    target = str(result.get("target") or "the target")
    metrics = result.get("metrics") or {}

    if task == "forecast" or model == "Prophet":
        return _prophet_brief(model, target, metrics, result)

    if task in {"classification", "binary_classification"} or any(
        k in metrics for k in ("accuracy", "f1", "precision", "recall")
    ):
        return _classification_brief(model, target, metrics)

    if task in {"clustering", "anomaly", "dimensionality", "optimization"}:
        return _special_brief(model, task, target, metrics, result)

    return _regression_brief(model, target, metrics)


def _prophet_brief(model: str, target: str, metrics: dict, result: dict) -> str:
    horizon = metrics.get("horizon") or 7
    last_actual = metrics.get("last_actual")
    forecast_end = metrics.get("forecast_end")
    forecast_mean = metrics.get("forecast_mean")
    lo = metrics.get("forecast_lower")
    hi = metrics.get("forecast_upper")
    pct = metrics.get("pct_change")
    mae = metrics.get("mae")
    r2 = metrics.get("r2")

    lines = [
        f"**Manager read — {model} on `{target}`**",
        "",
    ]

    if forecast_end is not None and last_actual is not None:
        direction = "up" if float(forecast_end) >= float(last_actual) else "down"
        change_txt = ""
        if pct is not None:
            change_txt = f" ({direction} about {_pct(abs(float(pct)))} vs last observed)"
        lines.append(
            f"- **Outlook:** Over the next **{horizon}** periods, `{target}` is projected "
            f"near **{_fmt(forecast_end)}**{change_txt}."
        )
    elif forecast_mean is not None:
        lines.append(
            f"- **Outlook:** Average forecast for the next **{horizon}** periods is "
            f"**{_fmt(forecast_mean)}**."
        )

    if lo is not None and hi is not None:
        lines.append(
            f"- **Range managers should plan for:** about **{_fmt(lo)}** to **{_fmt(hi)}** "
            f"(uncertainty band)."
        )

    if pct is not None:
        p = float(pct)
        if p > 0.03:
            lines.append(
                "- **Business takeaway:** Trend looks **positive** — this can support higher "
                "revenue / demand planning if operations can keep up."
            )
        elif p < -0.03:
            lines.append(
                "- **Business takeaway:** Trend looks **soft** — review pricing, demand, or "
                "capacity before the dip hits operations."
            )
        else:
            lines.append(
                "- **Business takeaway:** Near-term outlook is **stable** — good for steady "
                "scheduling; watch for shocks outside the band."
            )

    if mae is not None:
        lines.append(
            f"- **Trust check:** Holdout error (MAE) ≈ **{_fmt(mae)}**"
            + (f"; R² ≈ **{_fmt(r2, 3)}**" if r2 is not None else "")
            + ". Use the band, not a single point, for decisions."
        )

    preview = result.get("predictions_preview")
    if preview is not None and hasattr(preview, "tail") and len(preview) > 0:
        try:
            row = preview.tail(1).iloc[0]
            if "yhat" in preview.columns:
                lines.append(
                    f"- **Next checkpoint number:** **{_fmt(row['yhat'])}** "
                    f"(see table below for the full list)."
                )
        except Exception:
            pass

    lines.append(
        "- **What to do next:** Share this forecast with ops/finance; compare to last quarter "
        "and set one action (stock, staffing, or campaign) inside the range."
    )
    return "\n".join(lines)


def _regression_brief(model: str, target: str, metrics: dict) -> str:
    r2 = metrics.get("r2")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    lines = [
        f"**Manager read — {model} predicting `{target}`**",
        "",
    ]

    if r2 is not None:
        r = float(r2)
        if r >= 0.7:
            quality = "strong — useful for planning"
        elif r >= 0.4:
            quality = "moderate — good directional guide"
        elif r >= 0.15:
            quality = "weak — treat as a hint, not a commitment"
        else:
            quality = "poor — do not base big spend on this alone"
        lines.append(f"- **Fit quality (R² {_fmt(r, 3)}):** {quality}.")

    if mae is not None:
        lines.append(
            f"- **Typical miss:** about **{_fmt(mae)}** units of `{target}` "
            f"(MAE" + (f", RMSE {_fmt(rmse)}" if rmse is not None else "") + ")."
        )

    if r2 is not None and float(r2) >= 0.4:
        lines.append(
            "- **Business takeaway:** Model can help estimate outcomes that affect "
            "**cost, revenue, or risk** — use it to prioritize where humans dig deeper."
        )
    else:
        lines.append(
            "- **Business takeaway:** Numbers are directional only. Improve data quality "
            "or try another model before locking a budget decision."
        )

    lines.append(
        "- **What to do next:** Pick the top drivers / worst predictions and assign an owner "
        "to verify on the shop floor or in finance."
    )
    return "\n".join(lines)


def _classification_brief(model: str, target: str, metrics: dict) -> str:
    acc = metrics.get("accuracy")
    f1 = metrics.get("f1")
    lines = [
        f"**Manager read — {model} classifying `{target}`**",
        "",
    ]
    if acc is not None:
        a = float(acc)
        if a >= 0.85:
            tone = "reliable enough for triage"
        elif a >= 0.7:
            tone = "useful with human review"
        else:
            tone = "too noisy for automated decisions"
        lines.append(f"- **Accuracy {_pct(a)}:** {tone}.")
    if f1 is not None:
        lines.append(f"- **Balance (F1):** {_fmt(f1, 3)} — checks both catch-rate and false alarms.")

    lines.append(
        "- **Business takeaway:** Use scores to **flag high-risk / high-value cases** "
        "(failures, churn, quality issues) so teams spend time where it pays off."
    )
    lines.append(
        "- **What to do next:** Review the worst false alarms this week; that usually "
        "improves process faster than swapping models."
    )
    return "\n".join(lines)


def _special_brief(model: str, task: str, target: str, metrics: dict, result: dict) -> str:
    lines = [f"**Manager read — {model} ({task})**", ""]

    if task == "clustering":
        n = metrics.get("n_clusters") or metrics.get("clusters")
        lines.append(
            f"- **Segments found:** {n if n is not None else 'see metrics'} — "
            "group customers/machines/SKUs for different playbooks."
        )
        lines.append(
            "- **Business takeaway:** One average plan leaves money on the table; "
            "treat each segment differently (offer, maintenance, stock)."
        )
    elif task == "anomaly":
        n = metrics.get("n_anomalies") or metrics.get("anomaly_count")
        lines.append(
            f"- **Unusual rows flagged:** {n if n is not None else 'see metrics'}."
        )
        lines.append(
            "- **Business takeaway:** Anomalies often mean **waste, fraud, or early failure** — "
            "investigate the list before they hit revenue or uptime."
        )
    elif task == "dimensionality":
        lines.append(
            "- **Business takeaway:** Compresses many sensors/features into a simpler view "
            "so dashboards stay readable for non-data people."
        )
    elif task == "optimization":
        obj = metrics.get("objective") or metrics.get("status")
        lines.append(
            f"- **Optimization result:** {obj if obj is not None else 'see metrics'}."
        )
        lines.append(
            "- **Business takeaway:** This is a **plan suggestion** (cost/time/resources) — "
            "validate constraints with ops before locking schedules."
        )
    else:
        lines.append("- **Business takeaway:** Review metrics below with the process owner.")

    lines.append("- **What to do next:** Export the table, pick 3 actions, assign owners.")
    if result.get("n_train"):
        lines.append(f"- Trained on **{result.get('n_train')}** rows"
                      + (f", tested on **{result.get('n_test')}**." if result.get("n_test") else "."))
    return "\n".join(lines)
