---
title: Analytics Forge v2
emoji: 🏭
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Dual-mode industrial analytics — Manual upload + LIVE SCADA buffer
---

# Analytics Forge v2

Industry dual-mode analytics OS — **Manual upload** + **LIVE SCADA** — shared core for clean → field → KPIs → charts → ML → Ask AI → dashboard → email.

Repo: `shaikmohammedshoaib666/analytics-forge-v2` (keep separate from `analytics-forge` v1 demo).

> **Deploy:** Free public host = **Render** (`deploy/RENDER.md`).  
> Hugging Face free tier is **Static only** now — Docker needs **HF PRO** and cannot run Forge on free Static. See `deploy/HUGGINGFACE.md`.

## Try now (localhost)

```bash
cd /Users/sk.md.shoaib.raza/Projects/analytics-forge-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS only — XGBoost needs OpenMP (fixes Field "libomp.dylib" crash).
# Without this, Forge still runs and falls back to RandomForest.
# brew install libomp && pip install --force-reinstall xgboost

# .env (gitignored)
# GEMINI_API_KEY=...
# GEMINI_MODEL=gemini-3.6-flash

streamlit run app.py
# http://127.0.0.1:8501
```

```bash
python smoke_test.py
```

---

## Deploy: pick a host

| Platform | Free public URL? | Notes |
|----------|------------------|-------|
| **Render** (Blueprint) | **Yes — use this** | Root `render.yaml` + `deploy/RENDER.md` |
| **Streamlit Cloud** | Yes | Often slow / OOM — use `requirements-cloud.txt` |
| **Oracle Free / Azure / VPS + Docker** | Yes (your IP) | Full stack — `deploy/ORACLE_FREE.md` |
| **Hugging Face Spaces** | **Docker = paid (PRO)** | Free = **Static only** — cannot run Streamlit Forge. `deploy/HUGGINGFACE.md` |

**PySpark on free cloud hosts will usually fail or OOM.** Public demos use **`requirements-cloud.txt`**.

### Option A — Render (recommended free link)

1. https://dashboard.render.com → sign up with GitHub  
2. **New → Blueprint** → repo `shaikmohammedshoaib666/analytics-forge-v2`  
3. Uses root **`render.yaml`** (branch **`main`**)  
4. Secret: `GEMINI_API_KEY`  
5. Details: **`deploy/RENDER.md`**

If an existing Render service is still pinned to `cursor/forge-v2-foundation-f3f9`: Dashboard → service → **Settings → Build & Deploy → Branch → `main`**.

### Option B — Streamlit Cloud

1. [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Repo: `shaikmohammedshoaib666/analytics-forge-v2`
3. Branch: `main` · Main file: `app.py`
4. Requirements file: **`requirements-cloud.txt`**
5. Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.6-flash`

Cloud demos: use **MANUAL** or LIVE **`buffer_only`**. Private plant `192.168.x` Modbus will not reach from the internet.

### Option C — Hugging Face (only if you pay PRO)

Free HF accounts can only create **Static** Spaces — those cannot run this Python app.  
Docker Spaces need [HF PRO](https://huggingface.co/pricing). Details: **`deploy/HUGGINGFACE.md`**.

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
| **MANUAL UPLOAD** | CSV/Excel → Clean → Data Integration (SQL joins) → Field → KPIs/Charts/ML/Ask/Dashboard/Email |
| **LIVE CONNECT** | `config.yaml`: OCP-U → pymodbus → optional FastAPI → `data/live.csv` → SCADA console |

```bash
# optional plant gateway
uvicorn gateway:app --host 0.0.0.0 --port 8088
```

## .env

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.6-flash
EMAIL_USER=
EMAIL_PASSWORD=
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=<anon-or-publishable-key>
# Optional but recommended for OAuth callbacks:
APP_BASE_URL=https://<your-render-service>.onrender.com
```

Google OAuth notes:
- Use `SUPABASE_KEY` as anon/publishable key only (never service role key in app auth flow).
- In Supabase Dashboard → Authentication → Providers → Google: enable provider and set Google client ID/secret.
- In Supabase Dashboard → Authentication → URL Configuration: add `https://<your-render-service>.onrender.com` to Redirect URLs / Additional Redirect URLs.
