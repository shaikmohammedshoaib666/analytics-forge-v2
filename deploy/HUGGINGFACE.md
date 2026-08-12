# Deploy Analytics Forge v2 on Hugging Face Spaces

> **Important (2025+):** Hugging Face **deprecated the native Streamlit SDK**.  
> New Spaces must use **Docker** (there is an official Streamlit-on-Docker template).  
> Do **NOT** pick **Static** / Transformers / Gradio Lite / Svelte / React / Paper — those are wrong for this app.

Your URL will be:

- Space: `https://huggingface.co/spaces/<YOUR_HF_USER>/analytics-forge-v2`
- App: `https://<YOUR_HF_USER>-analytics-forge-v2.hf.space`

This repo is already Docker-Space ready (`sdk: docker`, `app_port: 8501`, light `requirements-cloud.txt`).

---

## What you are seeing (and what to click)

On **https://huggingface.co/new-space** you get:

| Field | Choose |
|--------|--------|
| Space name | `analytics-forge-v2` |
| License | MIT (or any) |
| **SDK** | **Docker** ← not Static |
| Template (if asked) | **Blank** or **Streamlit** / `streamlit/streamlit-template-space` if listed under Docker |
| Hardware | CPU basic (free) |
| Visibility | Public |

### If the page only shows “Static” templates

1. Click the **SDK** dropdown again (above the templates).  
2. Change **Static** → **Docker** (or **Gradio** will also appear — ignore Gradio).  
3. Templates should switch away from Transformers / Svelte / React.  
4. Pick **Blank Docker** / empty Docker Space (or Streamlit Docker template).

**Never choose:** Static, Transformers, Gradio Lite, Svelte, Paper project, React.

---

## Path A — Create empty Docker Space, then push this repo (recommended)

### 1) Create Space
https://huggingface.co/new-space → SDK **Docker** → name `analytics-forge-v2` → Create.

### 2) On your Mac, push Forge into the Space

```bash
cd /Users/sk.md.shoaib.raza/Projects/analytics-forge-v2
git fetch origin
git reset --hard origin/cursor/forge-v2-foundation-f3f9

# YOUR_HF_USER = your Hugging Face username
git remote remove hf 2>/dev/null
git remote add hf https://huggingface.co/spaces/YOUR_HF_USER/analytics-forge-v2
git push hf cursor/forge-v2-foundation-f3f9:main --force
```

Password = **HF write token**: https://huggingface.co/settings/tokens  
(Username = your HF username; password = `hf_...` token)

### 3) Secrets
Space → **Settings** → **Variables and secrets**:

| Name | Value |
|------|--------|
| `GEMINI_API_KEY` | your key |
| `GEMINI_MODEL` | `gemini-flash-latest` |

### 4) Wait for Docker build
Open the Space **App** / **Logs** tab. First build ~5–15 min. Then open the `.hf.space` URL.

---

## Path B — CLI create (if UI is confusing)

```bash
pip install -U huggingface_hub
huggingface-cli login   # paste write token

# Create a Docker Space (no Static template)
huggingface-cli repo create analytics-forge-v2 --type space --space_sdk docker

cd /Users/sk.md.shoaib.raza/Projects/analytics-forge-v2
git remote remove hf 2>/dev/null
git remote add hf https://huggingface.co/spaces/YOUR_HF_USER/analytics-forge-v2
git push hf cursor/forge-v2-foundation-f3f9:main --force
```

---

## Demo inside the Space

- **MANUAL UPLOAD** → sample CSV under `data/samples/`
- LIVE without PLC → connection **`buffer_only`**
- Plant Modbus `192.168.x` will **not** work from Hugging Face

---

## If build fails

- Confirm Space README YAML has `sdk: docker` and `app_port: 8501` (already in this repo).
- Settings → **Factory rebuild**.
- Do not install PySpark on the Space.
