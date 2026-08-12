# Analytics Forge v2

Industry dual-mode analytics OS — **Manual upload** + **LIVE SCADA** — shared core for clean → field → KPIs → charts → ML → Ask AI → dashboard → email.

Repo: `shaikmohammedshoaib666/analytics-forge-v2` (keep separate from `analytics-forge` v1 demo).

## Try now (localhost)

```bash
cd /Users/sk.md.shoaib.raza/Projects/analytics-forge-v2
git fetch origin
git checkout cursor/forge-v2-foundation-f3f9
git reset --hard origin/cursor/forge-v2-foundation-f3f9

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put keys in .env (gitignored)
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-flash-latest

streamlit run app.py
# open http://127.0.0.1:8501
```

Smoke check (no UI):

```bash
python smoke_test.py
```

## Modes

| Mode | What happens |
|------|----------------|
| **MANUAL UPLOAD** | CSV/Excel → Clean (pandas/polars/pyspark) → Field (Optuna+Gemini) → KPIs/Charts/ML/Ask/Dashboard/Email |
| **LIVE CONNECT** | `config.yaml` pipe: OCP-U → pymodbus → optional FastAPI gateway → `data/live.csv` → SCADA console + same analytics pages |

### LIVE config (`config.yaml`)

```yaml
LIVE_MODE:
  connection_type: direct   # direct | fastapi | buffer_only
  ocp_u_ip: "192.168.1.50"
  ocp_u_port: 502
  fastapi_url: "http://127.0.0.1:8088/live"
```

Optional plant gateway (Pi / same LAN):

```bash
uvicorn gateway:app --host 0.0.0.0 --port 8088
```

## What each page does

1. **Upload** — Manual file + engine picker · LIVE = SCADA console (poll, metrics, right-rail insights)
2. **Clean** — 15+ DWDM / GE / Cleanlab / OPC checks
3. **Field** — Domain detect + LlamaIndex build + best model card + actions
4. **Auto KPIs** — Domain boxes + loc/date/people filters
5. **Charts** — Plotly/Seaborn/Matplotlib · pin to Dashboard
6. **ML Studio** — RF/XGB/Prophet/PCA/Statsmodels… choose model + target
7. **Ask / AI** — LlamaIndex search + Gemini answer
8. **Dashboard** — Power BI-style filters + pinned charts/KPIs
9. **Email** — HTML pack + CSV (needs Gmail App Password)

## .env

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-latest
EMAIL_USER=
EMAIL_PASSWORD=
# optional LIVE overrides:
# LIVE_CONNECTION_TYPE=direct
# OCP_U_IP=192.168.1.50
# FASTAPI_LIVE_URL=http://127.0.0.1:8088/live
```

## Deploy

Not in this step — confirm git + localhost first, then we pick Streamlit Cloud / Docker / Azure.
