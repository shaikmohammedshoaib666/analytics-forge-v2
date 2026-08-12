# Analytics Forge v2 — Single-file Industrial Dual-Mode OS

**One file only:** `app.py`

## Modes
- **LIVE CONNECT** — real Modbus TCP SCADA (`MODBUS_HOST:MODBUS_PORT`, default `192.168.1.100:502`), polls holding registers every 5s into `data/live.csv`. No DemoSimulator.
- **MANUAL UPLOAD** — CSV/Excel/JSON/Parquet drives every page.

## Shared core
`get_data()` → all pages. Clean = PySpark/pandas + Great Expectations + ydata + Cleanlab. Field = RF+GB+IsolationForest risk %. ML Studio = Optuna AutoML (no manual model pick) + Prophet 90-day business forecast. Ask/AI = LlamaIndex RAG + Gemini.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env must contain GEMINI_API_KEY=...
streamlit run app.py
```

## .env
```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-latest
MODBUS_HOST=192.168.1.100
MODBUS_PORT=502
EMAIL_USER=
EMAIL_PASSWORD=
```
