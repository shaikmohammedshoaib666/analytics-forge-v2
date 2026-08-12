# Deploy on Render (Blueprint) — do this now

## Git status (already OK on remote)

Branch `cursor/forge-v2-foundation-f3f9` is pushed and includes `render.yaml`.  
**Do not deploy from `main`** — `main` is an older layout and has no Blueprint file.

## Clicks (you are logged into Render)

1. Open: **https://dashboard.render.com/blueprints**  
   (or https://dashboard.render.com/select-repo?type=blueprint)
2. **New Blueprint Instance** → connect GitHub if asked  
3. Select repo: **`shaikmohammedshoaib666/analytics-forge-v2`**
4. **Branch:** `cursor/forge-v2-foundation-f3f9`  
5. Blueprint path: `render.yaml` (default root)  
6. Apply / Create — when prompted for **`GEMINI_API_KEY`**, paste your key (or leave blank and add later in Environment)  
7. Wait for first deploy (pip install + Streamlit). Free builds often take **5–15 minutes**.

## Your URL

After deploy succeeds:

`https://analytics-forge-v2.onrender.com`

(Exact subdomain is shown on the service page if Render renames it.)

## After it is live

- Use mode **MANUAL UPLOAD** for demos  
- LIVE without PLC → connection **`buffer_only`**  
- Free tier sleeps after ~15 min idle; first open after sleep can take ~30–60s

## If build fails

- Confirm branch is `cursor/forge-v2-foundation-f3f9` (not `main`)  
- Build command must be: `pip install -r requirements-cloud.txt`  
- Logs → look for OOM / missing module → Factory clear cache / Manual Deploy
