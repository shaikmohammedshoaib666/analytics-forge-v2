# Deploy on Render (Blueprint)

Live URL: **https://analytics-forge-v2.onrender.com**

`main` now has the full v2 app (including `render.yaml`). New Blueprint instances should track **`main`**.

An existing service created from the old pin (`cursor/forge-v2-foundation-f3f9`) will keep auto-deploying that feature branch until you switch it.

## If the service already exists (typical)

Pushing `cursor/forge-v2-foundation-f3f9` auto-redeploys **until** you change the branch.

To follow `main` going forward:

1. Open **https://dashboard.render.com**
2. Click the **analytics-forge-v2** web service
3. **Settings** → **Build & Deploy**
4. **Branch** → change `cursor/forge-v2-foundation-f3f9` → **`main`** → Save
5. If it does not start a deploy: **Manual Deploy** → **Deploy latest commit**

Optional: **Environment** → confirm `GEMINI_API_KEY` (and `GEMINI_MODEL=gemini-3.6-flash`). Old values such as `gemini-2.0-flash` are remapped in code.
Paste-in-UI keys are session-only on Render — they do not persist. Use **Environment** + Manual Deploy.

## New Blueprint (only if the service does not exist)

1. Open: **https://dashboard.render.com/blueprints**
2. **New Blueprint Instance** → connect GitHub if asked
3. Select repo: **`shaikmohammedshoaib666/analytics-forge-v2`**
4. **Branch:** `main`
5. Blueprint path: `render.yaml` (repo root)
6. Apply / Create — paste **`GEMINI_API_KEY`** when asked (or add later)
7. Wait 5–15 minutes on the free plan

## After it is live

- Demos: mode **MANUAL UPLOAD** (joins, Clean, Field, Optuna, LlamaIndex)
- LIVE without PLC → connection **`buffer_only`**
- Free tier sleeps after ~15 min idle; first open after sleep can take ~30–60s

## If build fails

- Branch must be **`main`** (or the feature branch if you have not switched yet)
- Build command: `pip install -r requirements-cloud.txt`
- Logs → OOM / missing module → Clear build cache / Manual Deploy
