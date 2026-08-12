"""Analytics Forge settings and paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOADS_DIR = UPLOAD_DIR  # alias used by app.py
RUNS_DIR = DATA_DIR / "runs"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "analytics_forge.db"
CONFIG_DIR = ROOT / "config"
MODELS_CATALOG_YAML = CONFIG_DIR / "models_catalog.yaml"
CHARTS_CATALOG_YAML = CONFIG_DIR / "charts_catalog.yaml"

for d in (RAW_DIR, CLEAN_DIR, UPLOAD_DIR, RUNS_DIR, SAMPLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Alternate env names students sometimes use
_SECRET_ALIASES: dict[str, tuple[str, ...]] = {
    "EMAIL_USER": ("SMTP_USER", "SMTP_USERNAME", "MAIL_USER"),
    "EMAIL_PASSWORD": ("SMTP_PASSWORD", "MAIL_PASSWORD", "EMAIL_PASS"),
    "EMAIL_FROM": ("SMTP_FROM", "MAIL_FROM"),
    "EMAIL_SMTP_HOST": ("SMTP_HOST", "MAIL_SMTP_HOST"),
    "EMAIL_SMTP_PORT": ("SMTP_PORT", "MAIL_SMTP_PORT"),
    "EMAIL_IMAP_HOST": ("IMAP_HOST", "MAIL_IMAP_HOST"),
    "EMAIL_IMAP_PORT": ("IMAP_PORT", "MAIL_IMAP_PORT"),
}


def _from_streamlit(name: str) -> Optional[str]:
    """Read Streamlit secrets lazily (works after app boot / on Cloud)."""
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return None
        # Top-level key
        try:
            if name in secrets:
                val = secrets[name]
                if val is None:
                    return None
                text = str(val).strip()
                return text or None
        except Exception:
            pass
        # Nested [email] / [ai] style tables
        for section in ("email", "ai", "secrets", "general"):
            try:
                block = secrets.get(section) if hasattr(secrets, "get") else secrets[section]
            except Exception:
                continue
            if block is None:
                continue
            try:
                if name in block:
                    text = str(block[name]).strip()
                    return text or None
            except Exception:
                continue
    except Exception:
        return None
    return None


def _secret(name: str, default: str = "") -> str:
    """Read from env, then aliases, then Streamlit secrets. Always live (not cached)."""
    val = os.getenv(name, "").strip()
    if val:
        return val
    for alt in _SECRET_ALIASES.get(name, ()):
        alt_val = os.getenv(alt, "").strip()
        if alt_val:
            return alt_val
    st_val = _from_streamlit(name)
    if st_val:
        return st_val
    for alt in _SECRET_ALIASES.get(name, ()):
        st_alt = _from_streamlit(alt)
        if st_alt:
            return st_alt
    return default


def get_openai_api_key() -> str:
    return _secret("OPENAI_API_KEY")


def get_openai_model() -> str:
    return _secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


def get_gemini_api_key() -> str:
    return _secret("GEMINI_API_KEY")


def get_gemini_model() -> str:
    return _secret("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash"


def get_ai_default_provider() -> str:
    return (_secret("AI_DEFAULT_PROVIDER", "gemini") or "gemini").lower()


def get_email_user() -> str:
    return _secret("EMAIL_USER")


def get_email_password() -> str:
    return _secret("EMAIL_PASSWORD")


def get_email_from() -> str:
    return _secret("EMAIL_FROM") or get_email_user()


def get_email_smtp_host() -> str:
    return _secret("EMAIL_SMTP_HOST", "smtp.gmail.com") or "smtp.gmail.com"


def get_email_smtp_port() -> int:
    return int(_secret("EMAIL_SMTP_PORT", "587") or 587)


def get_email_smtp_use_tls() -> bool:
    return _secret("EMAIL_SMTP_USE_TLS", "true").lower() in {"1", "true", "yes"}


def get_email_imap_host() -> str:
    return _secret("EMAIL_IMAP_HOST", "imap.gmail.com") or "imap.gmail.com"


def get_email_imap_port() -> int:
    return int(_secret("EMAIL_IMAP_PORT", "993") or 993)


def get_email_imap_folder() -> str:
    return _secret("EMAIL_IMAP_FOLDER", "INBOX") or "INBOX"


# Backward-compatible module attributes:
# Prefer getters in new code. These snapshots are refreshed via refresh_settings().
OPENAI_API_KEY = get_openai_api_key()
OPENAI_MODEL = get_openai_model()
GEMINI_API_KEY = get_gemini_api_key()
GEMINI_MODEL = get_gemini_model()
AI_DEFAULT_PROVIDER = get_ai_default_provider()

EMAIL_USER = get_email_user()
EMAIL_PASSWORD = get_email_password()
EMAIL_FROM = get_email_from()
EMAIL_SMTP_HOST = get_email_smtp_host()
EMAIL_SMTP_PORT = get_email_smtp_port()
EMAIL_SMTP_USE_TLS = get_email_smtp_use_tls()
EMAIL_IMAP_HOST = get_email_imap_host()
EMAIL_IMAP_PORT = get_email_imap_port()
EMAIL_IMAP_FOLDER = get_email_imap_folder()
INBOUND_DIR = DATA_DIR / "inbound"
INBOUND_DIR.mkdir(parents=True, exist_ok=True)


def refresh_settings() -> None:
    """Re-read env / Streamlit secrets into module-level names."""
    global OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL, AI_DEFAULT_PROVIDER
    global EMAIL_USER, EMAIL_PASSWORD, EMAIL_FROM
    global EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, EMAIL_SMTP_USE_TLS
    global EMAIL_IMAP_HOST, EMAIL_IMAP_PORT, EMAIL_IMAP_FOLDER

    load_dotenv(ROOT / ".env", override=False)
    OPENAI_API_KEY = get_openai_api_key()
    OPENAI_MODEL = get_openai_model()
    GEMINI_API_KEY = get_gemini_api_key()
    GEMINI_MODEL = get_gemini_model()
    AI_DEFAULT_PROVIDER = get_ai_default_provider()
    EMAIL_USER = get_email_user()
    EMAIL_PASSWORD = get_email_password()
    EMAIL_FROM = get_email_from()
    EMAIL_SMTP_HOST = get_email_smtp_host()
    EMAIL_SMTP_PORT = get_email_smtp_port()
    EMAIL_SMTP_USE_TLS = get_email_smtp_use_tls()
    EMAIL_IMAP_HOST = get_email_imap_host()
    EMAIL_IMAP_PORT = get_email_imap_port()
    EMAIL_IMAP_FOLDER = get_email_imap_folder()
