# Analytics Forge — cloud/HF Spaces by default (light deps).
# Full local/Oracle image: docker compose build --build-arg REQUIREMENTS_FILE=requirements.txt
FROM python:3.12-slim-bookworm

ARG REQUIREMENTS_FILE=requirements-cloud.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloud.txt requirements.txt requirements-optional.txt ./
RUN pip install --upgrade pip && pip install -r ${REQUIREMENTS_FILE}

COPY . .

RUN mkdir -p /app/data/uploads /app/data/clean /app/data/runs /app/data/samples /app/data/raw \
    && if [ ! -f /app/data/live.csv ] && [ -f /app/data/samples/sample_predictive_maintenance.csv ]; then \
         cp /app/data/samples/sample_predictive_maintenance.csv /app/data/live.csv; \
       fi

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://127.0.0.1:8501/_stcore/health || exit 1

CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
