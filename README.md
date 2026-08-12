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

# macOS only — XGBoost needs OpenMP (fixes Field "libomp.dylib" crash).
# Without this, Forge still runs and falls back to RandomForest.
# brew install libomp && pip install --force-reinstall xgboost

# .env (gitignored)
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-flash-latest

streamlit run app.py
# http://127.0.0.1:8501
```

```bash
python smoke_test.py
```

---

## Deploy: pick a host (Streamlit Cloud is slow — use these)

| Platform | Public URL? | Speed | Best for |
|----------|-------------|-------|----------|
| **Hugging Face Spaces** | Yes | Usually faster first build | **Recommended free demo link** — see `deploy/HUGGINGFACE.md` |
| **Render** (Blueprint) | Yes (`render.yaml`) | Medium; free tier sleeps | Stable demo — see `deploy/RENDER.md` |
| **Streamlit Cloud** | Yes | Often 15–40+ min / OOM | Only if you already use it + `requirements-cloud.txt` |
| **Oracle Free / Azure / VPS + Docker** | Yes (your IP:8501) | Fast once VM exists | Full stack + LIVE gateway — `deploy/ORACLE_FREE.md` |

**I cannot log into your HF / Render / Streamlit account** — you click once; then you own the permanent URL.

**PySpark on free cloud hosts will usually fail or OOM.** Always use **`requirements-cloud.txt`** for public demos.

### Option A — Hugging Face Spaces (preferred free link)

1. https://huggingface.co/new-space → SDK **Streamlit** → link this GitHub repo  
2. Branch `cursor/forge-v2-foundation-f3f9`, app file `app.py`  
3. Secret: `GEMINI_API_KEY`  
4. Details: **`deploy/HUGGINGFACE.md`**

### Option B — Render Blueprint

1. https://dashboard.render.com/select-repo?type=blueprint  
2. Repo uses root **`render.yaml`** (installs `requirements-cloud.txt`)  
3. Details: **`deploy/RENDER.md`**

### Option C — Streamlit Cloud (slower)

1. [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repo: `shaikmohammedshoaib666/analytics-forge-v2`
3. Branch: `cursor/forge-v2-foundation-f3f9` · Main file: `app.py`
4. Requirements file: **`requirements-cloud.txt`**
5. Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-flash-latest`

Cloud demos: use **MANUAL** or LIVE **`buffer_only`**. Private plant `192.168.x` Modbus will not reach from the internet.

### Option D — Docker on your VM (full heavy app)

```bash
docker compose up --build
# http://127.0.0.1:8501  (or http://VM_PUBLIC_IP:8501)
```

See `deploy/ORACLE_FREE.md` / `deploy/AZURE_STUDENT.md`.

---

## Modes

| Mode | What happens |
|------|----------------|
| **MANUAL UPLOAD** | CSV/Excel → Clean → Field → KPIs/Charts/ML/Ask/Dashboard/Email |
| **LIVE CONNECT** | `config.yaml`: OCP-U → pymodbus → optional FastAPI → `data/live.csv` → SCADA console |

```bash
# optional plant gateway
uvicorn gateway:app --host 0.0.0.0 --port 8088
```

## .env

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-flash-latest
EMAIL_USER=
EMAIL_PASSWORD=
```
