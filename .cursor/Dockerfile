# Cloud Agent base image for Analytics Forge v2.
# System-level dependencies only; Python packages are installed by .cursor/install.sh
# against the checked-out repo (see .cursor/environment.json).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        python3-dev \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ubuntu:24.04 ships a non-root `ubuntu` user (uid 1000); run as it to match Cursor defaults.
USER ubuntu
