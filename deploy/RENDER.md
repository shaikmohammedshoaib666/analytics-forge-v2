# Deploy on Render (Blueprint)

1. Open https://dashboard.render.com/select-repo?type=blueprint  
2. Connect GitHub → `shaikmohammedshoaib666/analytics-forge-v2`  
3. Render reads root `render.yaml`  
4. Set secret `GEMINI_API_KEY` when prompted  
5. Deploy → you get `https://analytics-forge-v2.onrender.com` (name may vary)

**Note:** Free Render spins down after idle (~15 min). First hit after sleep can take ~30–60s. Still often less painful than Streamlit Cloud’s long first install.

LIVE plant Modbus from Render will not reach `192.168.x` — use MANUAL or `buffer_only`.
