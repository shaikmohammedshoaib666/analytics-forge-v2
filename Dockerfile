# Analytics Forge — Oracle Free / any Linux VM
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System libs for scientific wheels (LightGBM / XGBoost / matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional: uncomment on a strong VM if you want Prophet in the image
# COPY requirements-optional.txt .
# RUN pip install -r requirements-optional.txt

COPY . .

# Persist DB + uploads outside the container via volume mounts
RUN mkdir -p /app/data/uploads /app/data/clean /app/data/runs /app/data/samples /app/data/raw

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
