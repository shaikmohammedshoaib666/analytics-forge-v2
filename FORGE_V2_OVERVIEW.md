# Analytics Forge v2.0 — Overview (for builders)

This repo is **FORGE v2**. It started from a copy of `analytics-forge` (v1 demo).

## Keep separate

| Repo | Role |
|------|------|
| **analytics-forge** (v1) | Leave alone. Streamlit Cloud demo. Do not break it. |
| **analytics-forge-v2** (this) | Industry-grade upgrades. Build here. |

## Product idea

Plug data in → clean (one engine) → detect industry → predict in plain English → RAG chat on *your* data.

## Dual mode, one brain

- **Manual:** CSV / zip / files. Keep click-to-build charts & models.
- **Live:** pymodbus / OPC-UA / API / SMPS plugs (capability + sim on Mac first; real plant later). **SCADA-style auto** KPIs, charts, insights.

After data is in, shared core: KPIs · filters · charts · clean · Optuna/ML · LlamaIndex RAG.

## Live data rule (important)

Do **not** load all 1M+ live rows into the UI. Use **top filters** industries actually use (machine/line, date range, product, site, etc.) to pull **only the slice** you need, then work on that buffer (e.g. last N rows).

## Cleaning engines (pick one)

- Default: **pandas**
- Bigger data (~100k+): user chooses **Polars** *or* **PySpark** — never all three at once
- Later: ydata-profiling, Great Expectations, rapidfuzz, Cleanlab, DWDM/SQL-style EDA/DQC

## ML / AI

- Keep ML Studio; add Optuna, Prophet, ARIMA/SARIMA/sktime, MLflow
- Skip PyTorch for now
- Output: business alert (*fail in 3 days / revenue ±X%*) **plus** metrics (R² etc.)
- Ask/AI → offline **LlamaIndex RAG** (not OpenAI-as-product)
- Industry auto-detect → templates (manufacturing, healthcare, sales/churn, …)

## How we build

1. Fully functional prototype on **localhost (Mac)**
2. Host later only if needed (Azure Student / etc.)
3. No payments in scope

## Source of truth for “why”

User + Meta chat context was integrated into this overview. Do not copy Meta chat as a literal week-by-week script — implement against this overview and the existing v1 code in this repo.
