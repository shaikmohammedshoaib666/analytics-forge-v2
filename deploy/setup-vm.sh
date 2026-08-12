#!/usr/bin/env bash
# Run ON any Ubuntu VM (Azure Student, Oracle, etc.) after SSH access works.
# Usage:
#   chmod +x deploy/setup-vm.sh
#   ./deploy/setup-vm.sh
# Optional:
#   REPO_URL=https://github.com/shaikmohammedshoaib666/analytics-forge.git ./deploy/setup-vm.sh
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/analytics-forge}"
REPO_URL="${REPO_URL:-https://github.com/shaikmohammedshoaib666/analytics-forge.git}"

echo "==> Updating system"
sudo apt-get update -y
sudo apt-get install -y git curl ca-certificates

echo "==> Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "==> Installing Docker Compose plugin (if missing)"
sudo apt-get install -y docker-compose-plugin || true

echo "==> Opening firewall ports (ufw)"
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow OpenSSH || true
  sudo ufw allow 8501/tcp || true
  sudo ufw --force enable || true
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "==> Cloning $REPO_URL -> $APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"

if [[ ! -f .env ]]; then
  echo "==> Creating .env from .env.example (edit secrets later)"
  cp -n .env.example .env || true
fi

mkdir -p data/uploads data/clean data/runs data/samples data/raw

echo "==> Building and starting Analytics Forge"
if docker info >/dev/null 2>&1; then
  docker compose up -d --build
else
  echo "Docker needs a new login session for your user group."
  echo "Run: newgrp docker   OR log out/in, then:"
  echo "  cd $APP_DIR && docker compose up -d --build"
  sudo docker compose up -d --build
fi

PUBLIC_IP="$(curl -s ifconfig.me || hostname -I | awk '{print $1}')"
echo ""
echo "============================================"
echo " Analytics Forge should be running."
echo " Open: http://${PUBLIC_IP}:8501"
echo ""
echo " Azure: also allow TCP 8501 in NSG inbound rules."
echo " Updates later:"
echo "   cd $APP_DIR && git pull && docker compose up -d --build"
echo "============================================"
