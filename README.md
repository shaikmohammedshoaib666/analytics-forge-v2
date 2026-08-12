# Analytics Forge v2.0

Industry-grade fork of [analytics-forge](https://github.com/shaikmohammedshoaib666/analytics-forge).

- **v1 (`analytics-forge`)** — keep as Streamlit Cloud demo; do not break it.
- **v2 (this repo)** — dual mode (manual + live SCADA-style), engine choice, Optuna, LlamaIndex RAG, etc.

Read **[FORGE_V2_OVERVIEW.md](FORGE_V2_OVERVIEW.md)** before building.

---

# Analytics Forge (base)

Reusable Streamlit analytics OS (8 fields) with OpenAI + Gemini Ask/AI, ML studio, dashboard pack, and email automation.

## Auth (new)

On launch you **sign up / sign in** with email + password. Passwords are stored as **PBKDF2 hashes only** (never plain text) in SQLite (`data/analytics_forge.db`). Each upload is saved under your user with a **Recent projects** sidebar so history survives refresh. Same app later runs on **Oracle Free** with the DB file on the VM (swap to Postgres when you outgrow SQLite).

## Local run

```bash
cd analytics-forge-v2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
streamlit run app.py
```

Open http://localhost:8501

## AI keys (OpenAI + Gemini)

In `.env` (local) or Streamlit Cloud **Secrets**:

```
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-2.0-flash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
AI_DEFAULT_PROVIDER=gemini
```

- Gemini key: https://aistudio.google.com/apikey (free tier available)
- OpenAI key: https://platform.openai.com/api-keys

**Never commit real `.env` or secrets to GitHub.**

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io → New app (or open an existing app’s **⚙️ settings**)
3. Select repo, branch `main`, main file `app.py`
4. **Required — pin Python 3.12:** open **Advanced settings** → set **Python version** to **3.12** → Save  
   (Community Cloud does **not** honor `runtime.txt` for the runtime; the UI dropdown is the source of truth. This repo still ships `runtime.txt` with `python-3.12.8` as a local/docs signal.)
5. Add secrets from `.streamlit/secrets.toml.example`
6. Deploy / **Reboot** the app once after changing Python or `requirements.txt`

If logs show `Using Python 3.13` / `3.14` and install hangs after `uv pip install` / `Resolved … packages`, switch the UI to **3.12** and reboot — many ML wheels are unreliable on bleeding-edge Python.

## Deploy (Oracle Cloud Always Free VM)

Full steps: [deploy/ORACLE_FREE.md](deploy/ORACLE_FREE.md)

Short path after the VM + port **8501** ingress exist:

```bash
git clone https://github.com/shaikmohammedshoaib666/analytics-forge.git
cd analytics-forge
chmod +x deploy/setup-oracle.sh && ./deploy/setup-oracle.sh
# open http://YOUR_PUBLIC_IP:8501
```

## Deploy (Azure for Students — recommended while Oracle is stuck)

Full steps: [deploy/AZURE_STUDENT.md](deploy/AZURE_STUDENT.md)

1. Claim [GitHub Student Pack](https://education.github.com/pack) → Azure credits  
2. Create Ubuntu 22.04 VM (B2s/B2ms), open NSG port **8501**  
3. SSH in and run:

```bash
git clone https://github.com/shaikmohammedshoaib666/analytics-forge.git
cd analytics-forge
chmod +x deploy/setup-vm.sh && ./deploy/setup-vm.sh
# open http://YOUR_PUBLIC_IP:8501
```

### Updates after Azure is live

On your PC: change code → `git push`. On the VM:

```bash
cd ~/analytics-forge && git pull && docker compose up -d --build
```

## Hosts & data limits

| Host | Good for | Typical upload / data size |
|------|----------|----------------------------|
| **Streamlit Community Cloud (free)** | This app: CSV analytics, sklearn/XGBoost/LightGBM (Prophet optional locally) | Usually **tens of MB CSV** per session (memory ~1GB class). Avoid multi-GB files |
| **Streamlit Cloud / paid tier** | Same app, more RAM | Larger CSVs (hundreds of MB) depending on plan |
| **Oracle Always Free VM** | This app with login + DB persistence | Hundreds of MB depending on RAM (prefer 8–12 GB) |
| **AWS / GCP / Azure VM or GPU** | PyTorch deep learning | GBs + GPU training |
| **Databricks / EMR / Spark cluster** | PySpark big data | GBs–TBs across cluster |
| **Local strong PC** | Gurobi (with license), heavier models | Depends on your RAM |

### Why PyTorch / PySpark are listed but not installed here
They are **too heavy** for Streamlit free cloud and need special infrastructure. They appear in ML Studio as **enterprise stubs** with guidance.

### I4.0 models now runnable in this app
RandomForest, ExtraTrees, **IsolationForest**, GradientBoosting, KMeans, DBSCAN, **PCA**, sklearn baselines.

**Separate packages in main `requirements.txt`** (college / Streamlit Cloud after push + reboot): **XGBoost**, **LightGBM**, **statsmodels OLS**, **PuLP**. Soft-fail remains if import still fails. **Prophet** lives in `requirements-optional.txt` so Cloud builds do not hang on cmdstan; install locally with `pip install -r requirements-optional.txt` if you need forecasts.

Gurobi / OR-Tools / PyTorch / PySpark = stronger host + license/cluster (not in requirements).
