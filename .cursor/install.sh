#!/usr/bin/env bash
# Idempotent dependency refresh for the Analytics Forge v2 Cloud Agent environment.
# Creates an isolated virtualenv and installs the light "cloud" profile (no PySpark/Java),
# plus FastAPI/uvicorn so the LIVE SCADA gateway (gateway.py) is usable in development.
set -euo pipefail

# Resolve repo root (parent of this script's .cursor directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

_ensure_venv_module() {
  if python3 -c "import venv" 2>/dev/null; then
    return 0
  fi
  echo "python3-venv missing — installing system package…"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv python3-pip
  else
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-venv python3-pip
  fi
}

if [ ! -d .venv ]; then
  _ensure_venv_module
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-cloud.txt
# gateway.py (LIVE FastAPI mode) needs these; kept out of requirements-cloud.txt so the
# free-tier Streamlit Cloud deploy profile stays lean.
pip install "fastapi>=0.110" "uvicorn>=0.27"

# Runtime data directories the app expects (gitignored; recreated on every setup).
mkdir -p data/uploads data/clean data/runs data/samples data/raw

echo "Analytics Forge v2 environment ready (.venv)."
