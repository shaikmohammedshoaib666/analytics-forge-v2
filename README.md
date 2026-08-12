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

## Deploy: Streamlit Cloud vs Docker (read this)

### Short answer
| | **Streamlit Cloud** | **Docker** |
|--|---------------------|------------|
| What it is | Hosted website for Streamlit apps (share a URL) | A **box** that packs Python + libs so the same app runs anywhere |
| Like Streamlit? | Yes — it's Streamlit's own host | No — Docker is packaging/runtime; you still run Streamlit *inside* the box |
| Free tier | Yes, but **~1GB RAM**, slow installs | Needs a machine (your PC, Oracle free VM, Railway, Azure, …) |
| This app full stack? | **Too heavy** if you install PySpark + GE + ydata + everything | **Yes** — that's what Docker is for |
| Best for Forge v2 | **Demo / share link** with `requirements-cloud.txt` | **Full industrial OS** (Spark optional, gateway, persistent `data/`) |

**PySpark on free Streamlit Cloud will usually fail or OOM.** Polars + pandas + XGBoost + Prophet is the realistic Cloud set.

### Option A — Streamlit Cloud (public URL, lighter)

1. Push branch to GitHub (already done: `analytics-forge-v2`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repo: `shaikmohammedshoaib666/analytics-forge-v2`
4. Branch: `cursor/forge-v2-foundation-f3f9` (or `main` after merge)
5. Main file: `app.py`
6. **Important:** In app settings / advanced, set requirements file to **`requirements-cloud.txt`** (no PySpark).  
   If the UI only reads `requirements.txt`, temporarily rename or copy cloud file over for that deploy.
7. **Secrets** (Manage app → Secrets):

```toml
GEMINI_API_KEY = "your_key"
GEMINI_MODEL = "gemini-flash-latest"
```

8. Deploy. First build can take 10–20+ minutes.

Cloud demo will show **pandas + polars** engines (not Spark). LIVE Modbus to a factory PLC from Cloud usually **won't** reach private `192.168.x` — use `buffer_only` / sample CSV / FastAPI on a public URL for demos.

### Option B — Docker (full heavy app)

Docker ≠ Streamlit Cloud. Think: **shipping container**. You put Forge + all libraries inside; then run that container on a laptop or cloud VM. Same Streamlit UI, but you control RAM/CPU.

```bash
# build (uses full requirements.txt — includes Spark if needed)
docker compose up --build

# open http://127.0.0.1:8501
```

Or:

```bash
docker build -t analytics-forge-v2 .
docker run --rm -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" analytics-forge-v2
```

Put Docker on Oracle Free / Azure Student / a ₹500–1000 VPS when you want full Spark + gateway.

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
