# Deploy Analytics Forge v2 on Hugging Face Spaces

Repo is **Docker Space ready** (`sdk: docker` in README + light `requirements-cloud.txt` in Dockerfile).

Your permanent URL will be:

- Space page: `https://huggingface.co/spaces/<YOUR_HF_USER>/analytics-forge-v2`
- App: `https://<YOUR_HF_USER>-analytics-forge-v2.hf.space`

---

## Do this now (you are logged into HF)

### 1) Create the Space from GitHub

1. Open: **https://huggingface.co/new-space**
2. **Owner:** your account  
3. **Space name:** `analytics-forge-v2`  
4. **License:** MIT (or any)  
5. **Select the SDK:** **Docker**  
6. **Hardware:** CPU basic (free)  
7. **Visibility:** Public  
8. Click **Create Space**

### 2) Connect this GitHub repo

On the new Space:

1. **Settings** → **Repository** / **Connected services** / **Factory reboot** area  
2. Or from the Space Files view: use **Add files → Import from Git repository** if shown  
3. Easiest path used by most people:
   - Space **Settings** → scroll to **GitHub** / clone instructions  
   - Or: delete default README if HF created an empty Space, then:

```bash
# On YOUR Mac (after Space exists) — push this branch into the Space
git remote add hf https://huggingface.co/spaces/<YOUR_HF_USER>/analytics-forge-v2
git fetch origin
git push hf cursor/forge-v2-foundation-f3f9:main
```

When prompted for password, use an **HF Access Token** (not account password):  
https://huggingface.co/settings/tokens → **New token** → write access → paste as password.

### 3) Alternate: duplicate via HF “GitHub” template

Some accounts see **Create from GitHub repository** on the new-space form:

- Repository: `shaikmohammedshoaib666/analytics-forge-v2`
- Branch: `cursor/forge-v2-foundation-f3f9`
- SDK: Docker

Use that if the button is visible — fewer steps.

### 4) Add Gemini secret (Ask AI)

Space → **Settings** → **Variables and secrets** → **New secret**:

| Name | Value |
|------|--------|
| `GEMINI_API_KEY` | your Gemini key |
| `GEMINI_MODEL` | `gemini-flash-latest` |

### 5) Wait for build

Open the Space → **Logs** / **Build**. First Docker build can take **5–15 minutes** (Prophet/xgboost wheels). When status is **Running**, open the `.hf.space` URL.

### 6) How to demo inside the Space

- Mode: **MANUAL UPLOAD** → upload `data/samples/sample_predictive_maintenance.csv`
- LIVE without PLC: connection **`buffer_only`** → Reload buffer (`data/live.csv` is seeded from the sample)

Private plant Modbus (`192.168.x`) **will not work** from Hugging Face.

---

## If build fails / OOM

- Confirm Dockerfile uses `requirements-cloud.txt` (default in this repo).
- Hardware: stay on CPU basic; do not install PySpark on the Space.
- Factory reboot: Space Settings → **Factory reboot**.

## Local check before push (optional)

```bash
docker build -t forge-hf .
docker run --rm -p 8501:8501 forge-hf
# http://127.0.0.1:8501
```
