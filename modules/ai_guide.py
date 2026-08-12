"""OpenAI chat guide with soft-fail + strong offline analytics answers."""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from config.settings import ROOT


def _schema_blurb(schema: Optional[dict]) -> str:
    if not schema:
        return "No schema."
    cols = schema.get("columns", {})
    lines = [f"- {c}: {m.get('dtype')} (nulls={m.get('nulls')})" for c, m in list(cols.items())[:40]]
    return f"Rows={schema.get('n_rows')}, Cols={schema.get('n_cols')}\n" + "\n".join(lines)


def _kpi_blurb(kpis: Optional[dict]) -> str:
    if not kpis:
        return "No KPIs."
    lines = []
    for kid, item in list(kpis.items())[:20]:
        if isinstance(item, dict):
            lines.append(f"- {item.get('name', kid)}: {item.get('value')}")
        else:
            lines.append(f"- {kid}: {item}")
    return "\n".join(lines)


def _ml_blurb(ml_result: Optional[dict]) -> str:
    if not ml_result:
        return "No ML model has been run yet in ML Studio."
    if not ml_result.get("ok"):
        return f"Last ML attempt failed: {ml_result.get('error') or ml_result.get('message')}"
    metrics = ml_result.get("metrics") or {}
    metric_txt = ", ".join(f"{k}={v}" for k, v in metrics.items())
    return (
        f"Model used: {ml_result.get('model_id')} | task={ml_result.get('task')} | "
        f"target={ml_result.get('target') or ml_result.get('target_col')} | metrics: {metric_txt}"
    )


def _kpi_get(kpis: Optional[dict], *keys: str):
    if not kpis:
        return None
    lower = {str(k).lower(): v for k, v in kpis.items()}
    for key in keys:
        if key.lower() in lower:
            v = lower[key.lower()]
            return v.get("value") if isinstance(v, dict) else v
        for k, v in lower.items():
            if key.lower() in k:
                return v.get("value") if isinstance(v, dict) else v
    return None


def _machine_risk_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = {c.lower(): c for c in df.columns}
    mid = next((cols[k] for k in cols if "machine" in k or k in {"asset_id", "asset", "equipment"}), None)
    if not mid:
        return None
    fail = next((cols[k] for k in cols if k in {"failure", "fault", "broken"}), None)
    rul = next((cols[k] for k in cols if "rul" in k), None)
    temp = next((cols[k] for k in cols if "temp" in k), None)
    vib = next((cols[k] for k in cols if "vib" in k), None)

    g = df.groupby(mid, dropna=False)
    out = pd.DataFrame({mid: g.size().index})
    out = out.set_index(mid)
    if fail:
        out["failure_rate"] = g[fail].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        out["failures"] = g[fail].apply(lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum())
    if rul:
        out["avg_rul"] = g[rul].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    if temp:
        out["avg_temp"] = g[temp].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    if vib:
        out["avg_vibration"] = g[vib].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    # risk score: high failure, low RUL, high vib/temp
    score = pd.Series(0.0, index=out.index)
    if "failure_rate" in out:
        score = score + out["failure_rate"].fillna(0) * 100
    if "avg_rul" in out:
        score = score + (1.0 / (out["avg_rul"].fillna(out["avg_rul"].median() or 1) + 1e-6)) * 50
    if "avg_vibration" in out:
        score = score + out["avg_vibration"].fillna(0) * 10
    if "avg_temp" in out:
        score = score + (out["avg_temp"].fillna(0) / 10.0)
    out["risk_score"] = score
    out = out.sort_values("risk_score", ascending=False).reset_index()
    return out


def rule_based_answer(
    question: str,
    df: Optional[pd.DataFrame] = None,
    kpis: Optional[dict] = None,
    domain: str = "generic",
    briefing: str = "",
    ml_result: Optional[dict] = None,
) -> Optional[str]:
    q = (question or "").strip().lower()
    if not q:
        return None

    if any(w in q for w in ("hello", "hi", "help", "what can you")):
        return (
            "Ask me about:\n"
            "- which model I used / R² RMSE\n"
            "- KPIs / sales today / failure rate\n"
            "- which machine will fail / highest risk machines\n"
            "- how to reduce machine failures\n"
            "- what to do next\n\n"
            f"Current field: **{domain}**."
        )

    # ---- models used ----
    if any(p in q for p in ("which model", "what model", "model used", "models used", "r2", "rmse", "mae", "accuracy")):
        if not ml_result:
            return (
                "No model has been run yet. Go to **ML Studio**, pick a model "
                "(e.g. RandomForestRegressor), click Run, then ask again."
            )
        return _ml_blurb(ml_result)

    # ---- KPIs ----
    if "kpi" in q or "scoreboard" in q:
        if not kpis:
            return "No KPIs yet — run Upload/Clean first."
        parts = []
        for kid, item in list(kpis.items())[:10]:
            if isinstance(item, dict):
                parts.append(f"**{item.get('name', kid)}** = {item.get('value')}")
            else:
                parts.append(f"**{kid}** = {item}")
        extra = ""
        if ml_result and ml_result.get("ok"):
            extra = f"\n\nAlso: {_ml_blurb(ml_result)}"
        return "Auto KPIs:\n- " + "\n- ".join(parts) + extra

    # ---- sales today ----
    if any(p in q for p in ("sales today", "revenue today", "sales latest", "today's sales", "todays sales")):
        val = _kpi_get(kpis, "sales_latest_day", "sales_on_latest_day", "total_revenue")
        if val is not None:
            latest = _kpi_get(kpis, "latest_day")
            suffix = f" (date {latest})" if latest is not None else " (latest day in data)"
            return f"Sales on the latest day in your dataset{suffix}: **{val}**."
        if df is not None:
            rev = next((c for c in df.columns if str(c).lower() in {"revenue", "sales", "amount"}), None)
            date_col = next((c for c in df.columns if any(h in str(c).lower() for h in ("date", "time"))), None)
            if rev and date_col:
                tmp = df[[date_col, rev]].copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp[rev] = pd.to_numeric(tmp[rev], errors="coerce")
                latest = tmp[date_col].max()
                if pd.notna(latest):
                    day = tmp[tmp[date_col].dt.date == latest.date()]
                    return f"Sales on {latest.date()}: **{float(day[rev].sum()):,.2f}**."
        return "Need sales CSV with revenue + date columns."

    # ---- which machine will fail / risk ranking ----
    if any(
        p in q
        for p in (
            "which machine",
            "machine will fail",
            "machines will fail",
            "highest risk",
            "most likely to fail",
            "fail first",
            "risk machine",
        )
    ):
        if df is None:
            return "Load PdM data first."
        risk = _machine_risk_table(df)
        if risk is None or risk.empty:
            return (
                "I need a machine id column (e.g. machine_id) plus failure/RUL/sensor columns. "
                "Load the Predictive Maintenance sample or your PdM CSV."
            )
        top = risk.head(5)
        lines = ["Highest-risk machines (from current data):"]
        for _, row in top.iterrows():
            bits = [f"**{row.iloc[0]}**"]
            for col in ("failure_rate", "avg_rul", "avg_vibration", "avg_temp", "risk_score"):
                if col in row and pd.notna(row[col]):
                    bits.append(f"{col}={float(row[col]):.3g}")
            lines.append("- " + " | ".join(bits))
        lines.append(
            "\nThis ranks machines using failure rate, low RUL, vibration, and temperature in your file. "
            "Run ML Studio for predictive scores, then ask again."
        )
        if ml_result and ml_result.get("ok"):
            lines.append(f"\nModel context: {_ml_blurb(ml_result)}")
        return "\n".join(lines)

    # ---- how to reduce machine failure ----
    if any(p in q for p in ("reduce failure", "reduce machine", "prevent failure", "how to reduce", "avoid failure")):
        tips = [
            "1. Prioritize the highest-risk machines (ask: `which machine will fail?`).",
            "2. Watch vibration/temperature spikes — schedule inspection when both rise together.",
            "3. Service assets with low RUL first; don't wait for failure=1 events.",
            "4. In ML Studio, train RandomForest on failure/RUL and track RMSE/R² quality.",
            "5. Put failure-rate + RUL KPIs on the Dashboard and email weekly packs to maintenance.",
        ]
        risk_line = ""
        if df is not None:
            risk = _machine_risk_table(df)
            if risk is not None and not risk.empty:
                top_id = risk.iloc[0, 0]
                risk_line = f"\n\nStart with machine **{top_id}** (currently highest risk in this dataset)."
        kpi_line = ""
        fr = _kpi_get(kpis, "failure_rate", "failure_rate_pct")
        if fr is not None:
            kpi_line = f"\nCurrent failure KPI: **{fr}**."
        ml_line = f"\n{_ml_blurb(ml_result)}" if ml_result else "\nNo ML model run yet — run one in ML Studio for stronger predictions."
        return "How to reduce machine failures:\n" + "\n".join(tips) + risk_line + kpi_line + ml_line

    if "failure" in q:
        val = _kpi_get(kpis, "failure_rate", "failure_rate_pct")
        if val is not None:
            return f"Failure KPI: **{val}**. Ask `which machine will fail?` for ranking."
        if df is not None:
            col = next((c for c in df.columns if str(c).lower() == "failure"), None)
            if col:
                return f"Failure rate: **{pd.to_numeric(df[col], errors='coerce').mean():.4f}**."

    if "rul" in q:
        val = _kpi_get(kpis, "avg_rul", "mean_rul")
        if val is not None:
            return f"Average RUL: **{val}**."

    if re.search(r"\b(row|rows|how many rows|shape)\b", q) and df is not None:
        return f"Dataset has **{len(df):,}** rows and **{df.shape[1]}** columns. Domain: `{domain}`."

    if "column" in q and df is not None:
        return "Columns: " + ", ".join(f"`{c}`" for c in df.columns)

    if "domain" in q or "field" in q:
        return f"Current detected field: **{domain}**."

    if any(p in q for p in ("next", "what should", "recommend", "process")):
        base = briefing.strip() if briefing else ""
        tip = {
            "predictive_maintenance": "Next: ask which machine will fail → ML Studio (RandomForest) → check R²/RMSE → Dashboard + email.",
            "sales_forecasting": "Next: Ask sales today → chart revenue → forecast model → Dashboard pack.",
        }.get(domain, "Next: review KPIs → Charts → ML Studio → Dashboard pack.")
        ml = f"\n{_ml_blurb(ml_result)}" if ml_result else ""
        return (base + "\n\n" if base else "") + tip + ml

    if df is not None:
        return (
            f"Loaded **{domain}** ({len(df):,} rows). "
            f"{_ml_blurb(ml_result)} "
            "Try: `which model did I use?`, `show kpis`, `which machine will fail?`, "
            "`how to reduce machine failure?`"
        )
    return None


def openai_configured() -> bool:
    from config.settings import get_openai_api_key, refresh_settings

    refresh_settings()
    return bool(get_openai_api_key())


def gemini_configured() -> bool:
    from config.settings import get_gemini_api_key, refresh_settings

    refresh_settings()
    return bool(get_gemini_api_key())


def provider_status() -> dict[str, bool]:
    return {"openai": openai_configured(), "gemini": gemini_configured(), "offline": True}


def _build_context(
    *,
    domain: str,
    schema: Optional[dict],
    kpis: Optional[dict],
    df: Optional[pd.DataFrame],
    briefing: str,
    ml_result: Optional[dict],
) -> str:
    sample = ""
    if df is not None and not df.empty:
        sample = df.head(8).to_csv(index=False)
    risk_txt = ""
    if df is not None:
        risk = _machine_risk_table(df)
        if risk is not None and not risk.empty:
            risk_txt = risk.head(5).to_csv(index=False)
    return (
        f"Domain: {domain}\n\nSchema:\n{_schema_blurb(schema)}\n\n"
        f"KPIs:\n{_kpi_blurb(kpis)}\n\nML:\n{_ml_blurb(ml_result)}\n\n"
        f"Briefing:\n{(briefing or '')[:1200]}\n\n"
        f"Top machine risk table:\n{risk_txt}\n\nSample rows:\n{sample}"
    )


SYSTEM_PROMPT = (
    "You are Analytics Forge, an industrial analytics co-pilot. "
    "Answer using the provided KPIs, ML metrics, and data context. "
    "For predictive maintenance: explain risk, which assets look worst, and practical actions. "
    "Be concise and actionable for students and plant engineers."
)


def _call_openai(question: str, context: str, history: Optional[list]) -> str:
    from openai import OpenAI

    from config.settings import get_openai_api_key, get_openai_model, refresh_settings

    refresh_settings()
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}"},
    ]
    for h in (history or [])[-6:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=get_openai_model() or "gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return resp.choices[0].message.content or ""


def _call_gemini(question: str, context: str, history: Optional[list]) -> str:
    from config.settings import get_gemini_api_key, get_gemini_model, refresh_settings

    refresh_settings()
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing")
    model_name = get_gemini_model() or "gemini-2.0-flash"

    try:
        from google import genai
    except Exception:
        # older package name fallback
        import google.generativeai as genai_old  # type: ignore

        genai_old.configure(api_key=api_key)
        model = genai_old.GenerativeModel(model_name)
        hist = ""
        for h in (history or [])[-6:]:
            hist += f"{h.get('role', 'user')}: {h.get('content', '')}\n"
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nChat:\n{hist}\nuser: {question}"
        resp = model.generate_content(prompt)
        return getattr(resp, "text", None) or str(resp)

    client = genai.Client(api_key=api_key)
    hist = ""
    for h in (history or [])[-6:]:
        hist += f"{h.get('role', 'user')}: {h.get('content', '')}\n"
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nChat:\n{hist}\nuser: {question}"
    resp = client.models.generate_content(model=model_name, contents=prompt)
    return getattr(resp, "text", None) or str(resp)


def ask_ai(
    question: str,
    *,
    domain: str = "generic",
    schema: Optional[dict] = None,
    kpis: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    briefing: str = "",
    history: Optional[list] = None,
    ml_result: Optional[dict] = None,
    provider: str = "auto",
) -> dict[str, Any]:
    from config.settings import AI_DEFAULT_PROVIDER, get_ai_default_provider, refresh_settings

    refresh_settings()
    rb = rule_based_answer(
        question,
        df=df,
        kpis=kpis,
        domain=domain,
        briefing=briefing or "",
        ml_result=ml_result,
    )

    provider = (provider or "auto").lower().strip()
    default_provider = get_ai_default_provider() or AI_DEFAULT_PROVIDER
    if provider == "auto":
        if gemini_configured():
            provider = "gemini"
        elif openai_configured():
            provider = "openai"
        else:
            provider = default_provider if (default_provider in {"gemini", "openai"} and (
                (default_provider == "gemini" and gemini_configured())
                or (default_provider == "openai" and openai_configured())
            )) else "offline"

    if provider in {"gemini", "openai"}:
        if provider == "gemini" and not gemini_configured():
            provider = "openai" if openai_configured() else "offline"
        if provider == "openai" and not openai_configured():
            provider = "gemini" if gemini_configured() else "offline"

    if provider == "offline":
        if rb:
            return {
                "ok": True,
                "source": "offline",
                "answer": rb
                + "\n\n_Offline mode. Add GEMINI_API_KEY (free tier) or OPENAI_API_KEY in `.env` / Streamlit secrets._",
            }
        return {
            "ok": True,
            "source": "offline",
            "answer": (
                "No AI key configured.\n\n"
                f"Edit `{ROOT / '.env'}`:\n"
                "- `GEMINI_API_KEY=...` (free tier from Google AI Studio)\n"
                "- or `OPENAI_API_KEY=sk-...`\n"
                "Then restart Streamlit.\n\n"
                "Meanwhile ask: `which model did I use?`, `show kpis`, "
                "`which machine will fail?`, `how to reduce machine failure?`"
            ),
        }

    context = _build_context(
        domain=domain,
        schema=schema,
        kpis=kpis,
        df=df,
        briefing=briefing or "",
        ml_result=ml_result,
    )

    try:
        if provider == "gemini":
            answer = _call_gemini(question, context, history)
            source = "gemini"
        else:
            answer = _call_openai(question, context, history)
            source = "openai"

        if rb and any(
            x in question.lower()
            for x in ("machine", "failure", "model", "kpi", "rmse", "r2")
        ):
            answer = f"{answer}\n\n---\nData check:\n{rb}"
        return {"ok": True, "source": source, "answer": answer}
    except Exception as exc:
        if rb:
            return {"ok": True, "source": "offline", "answer": f"{rb}\n\n_({provider} error: {exc})_"}
        return {
            "ok": False,
            "source": "error",
            "answer": f"{provider} error: {exc}. Check API key in `.env` or Streamlit secrets.",
        }
