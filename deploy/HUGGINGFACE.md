# Hugging Face Spaces — status for Forge v2

## Short answer

**You understood correctly.** As of mid‑2026 Hugging Face changed free Spaces:

| SDK | Free? | Can run Forge (Streamlit)? |
|-----|-------|----------------------------|
| **Static** (Transformers / Svelte / React / Paper…) | Yes | **No** — HTML/JS only, no Python Streamlit |
| **Docker** | Needs **HF PRO** (~paid) | Yes |
| **Gradio** | Needs **HF PRO** (ZeroGPU exception only for Gradio demos) | No (wrong UI framework) |

So: **do not create a Static Space** for this project. It will never run `app.py`.

Official note: [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) — *“Static Spaces are free… Gradio and Docker Spaces … require a paid plan.”*

---

## What you should do instead (free)

Use one of these — **not** Hugging Face free Static:

### 1) Render (best free public link right now)

1. Open https://dashboard.render.com → sign up with GitHub  
2. **New → Blueprint** → connect `shaikmohammedshoaib666/analytics-forge-v2`  
3. It reads root **`render.yaml`** (already in repo)  
4. Set secret `GEMINI_API_KEY`  
5. Deploy → URL like `https://analytics-forge-v2.onrender.com`

Guide: **`deploy/RENDER.md`**

Free tier sleeps after idle; first open after sleep ~30–60s.

### 2) Streamlit Cloud (free, often slow)

https://share.streamlit.io → New app → this repo → `app.py` → requirements **`requirements-cloud.txt`**

### 3) Local (always works)

```bash
streamlit run app.py
# http://127.0.0.1:8501
```

### 4) Hugging Face PRO (only if you want to pay)

https://huggingface.co/pricing → PRO → then create Space with SDK **Docker** → push this repo (see older Docker steps below).

---

## If you already opened Static templates

Close that form. Those templates (Transformers, Gradio Lite, Svelte, React, Paper) are **websites**, not our industrial Streamlit OS.

---

## Optional: Docker Space steps (HF PRO only)

```bash
# after PRO is active
huggingface-cli repo create analytics-forge-v2 --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_HF_USER/analytics-forge-v2
git push hf cursor/forge-v2-foundation-f3f9:main --force
```

Secrets: `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.6-flash`
