"""Email/password auth — passwords stored as PBKDF2 hashes only (never plaintext)."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from typing import Any, Optional

from core import db

_ITERATIONS = 120_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def validate_email(email: str) -> Optional[str]:
    e = normalize_email(email)
    if not e or not _EMAIL_RE.match(e):
        return "Enter a valid email address."
    if len(e) > 254:
        return "Email is too long."
    return None


def validate_password(password: str) -> Optional[str]:
    if not password or len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 128:
        return "Password is too long."
    return None


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _ITERATIONS,
    )
    return f"pbkdf2_sha256${_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt, hexdigest = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iters,
        )
        return hmac.compare_digest(digest.hex(), hexdigest)
    except Exception:
        return False


def register_user(email: str, password: str, display_name: str = "") -> dict[str, Any]:
    err = validate_email(email)
    if err:
        return {"ok": False, "error": err}
    err = validate_password(password)
    if err:
        return {"ok": False, "error": err}

    email_n = normalize_email(email)
    if db.get_user_by_email(email_n):
        return {"ok": False, "error": "An account with this email already exists."}

    name = (display_name or "").strip() or email_n.split("@")[0]
    user_id = db.create_user(
        email=email_n,
        password_hash=hash_password(password),
        display_name=name,
    )
    user = db.get_user_by_id(user_id)
    return {"ok": True, "user": _public_user(user)}


def login_user(email: str, password: str) -> dict[str, Any]:
    err = validate_email(email)
    if err:
        return {"ok": False, "error": err}
    if not password:
        return {"ok": False, "error": "Enter your password."}

    user = db.get_user_by_email(normalize_email(email))
    if not user or not verify_password(password, user["password_hash"]):
        return {"ok": False, "error": "Wrong email or password."}

    return {"ok": True, "user": _public_user(user)}


def _public_user(user: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not user:
        return None
    return {
        "id": int(user["id"]),
        "email": user["email"],
        "display_name": user.get("display_name") or user["email"],
    }
