# Deploy on Hugging Face Spaces (usually faster than Streamlit Cloud)

HF Spaces builds a Streamlit app from GitHub and gives you a stable URL like:

`https://<your-user>-analytics-forge-v2.hf.space`

## Why HF instead of Streamlit Cloud?

| | Streamlit Cloud | Hugging Face Spaces |
|--|-----------------|---------------------|
| Free public URL | Yes | Yes |
| First build | Often 15–40+ min / OOM on heavy deps | Usually quicker on Docker/Space |
| Requirements | Use `requirements-cloud.txt` | Same — light deps |
| LIVE Modbus to factory | No (no private LAN) | No — use MANUAL / `buffer_only` |

## Steps (you click — needs your HF account)

1. Open https://huggingface.co/new-space  
2. **Space name:** `analytics-forge-v2`  
3. **SDK:** Streamlit  
4. **Hardware:** CPU basic (free)  
5. Create Space → **Settings → Repository** → link GitHub  
   `shaikmohammedshoaib666/analytics-forge-v2`  
   Branch: `cursor/forge-v2-foundation-f3f9` (or `main`)  
6. In Space **Files**, ensure app entry is `app.py`  
7. Add Space secret: `GEMINI_API_KEY` (Settings → Repository secrets / Variables)  
8. If the Space installs full `requirements.txt` and OOMs, replace Space requirements with the contents of `requirements-cloud.txt` (or set the Space to use that file).

## After deploy

Open the Space URL → Mode **MANUAL UPLOAD** for demos.  
For LIVE without a PLC: Upload page → connection **`buffer_only`** → reload `data/live.csv`.
