"""
Supabase Auth integration for Analytics Forge v2.
Falls back to stub auth if SUPABASE_URL/SUPABASE_KEY not set.
"""
from __future__ import annotations

import base64
import os
from importlib import import_module
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()

_client = None
_client_error = ""


def _supabase_available() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _is_likely_service_role_key(key: str) -> bool:
    """Best-effort guard to prevent using service_role key in end-user auth flow."""
    try:
        parts = key.split(".")
        if len(parts) != 3:
            return False
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        return '"role":"service_role"' in decoded or '"role": "service_role"' in decoded
    except Exception:
        return False


def _load_create_client():
    """Load supabase create_client safely and return (callable, error)."""
    try:
        module = import_module("supabase")
    except Exception as exc:
        return None, f"Supabase package import failed: {exc}"

    create_client = getattr(module, "create_client", None)
    if create_client is None:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return None, (
                "Supabase package is missing create_client. "
                "A local folder named 'supabase' may be shadowing the pip package."
            )
        return None, "Supabase package is incompatible (missing create_client)."
    return create_client, ""


def init_supabase_client():
    global _client, _client_error
    if _client is not None:
        return _client
    if not _supabase_available():
        _client_error = "Supabase env vars are not configured."
        return None

    create_client, err = _load_create_client()
    if not create_client:
        _client_error = err or "Supabase import failed."
        return None

    try:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        _client_error = ""
    except Exception as exc:
        _client = None
        _client_error = f"Supabase client init failed: {exc}"
    return _client


def supabase_status_message() -> str:
    if _supabase_available():
        if _is_likely_service_role_key(SUPABASE_KEY):
            return "SUPABASE_KEY must be anon/publishable key, not service_role key."
        if init_supabase_client() is None:
            return _client_error or "Supabase is unavailable."
        return ""
    return "Set SUPABASE_URL + SUPABASE_KEY env vars for real auth."


def sign_up(email: str, password: str) -> dict:
    client = init_supabase_client()
    if not client:
        return {"error": "Supabase not configured"}
    try:
        res = client.auth.sign_up({"email": email, "password": password})
        if res.user:
            st.session_state["supabase_user"] = {
                "id": res.user.id,
                "email": res.user.email,
            }
            st.session_state["supabase_session"] = res.session
            st.session_state["signed_in"] = True
            return {"user": res.user}
        return {"error": "Sign-up failed"}
    except Exception as e:
        return {"error": str(e)}


def sign_in(email: str, password: str) -> dict:
    client = init_supabase_client()
    if not client:
        return {"error": "Supabase not configured"}
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            st.session_state["supabase_user"] = {
                "id": res.user.id,
                "email": res.user.email,
            }
            st.session_state["supabase_session"] = res.session
            st.session_state["signed_in"] = True
            return {"user": res.user}
        return {"error": "Sign-in failed"}
    except Exception as e:
        return {"error": str(e)}


def sign_out():
    client = init_supabase_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    st.session_state["supabase_user"] = None
    st.session_state["supabase_session"] = None
    st.session_state["signed_in"] = False


def get_user() -> Optional[dict]:
    return st.session_state.get("supabase_user")


def get_user_id() -> Optional[str]:
    user = get_user()
    return user["id"] if user else None


def _build_google_authorize_fallback_url() -> str:
    base = SUPABASE_URL.rstrip("/")
    url = f"{base}/auth/v1/authorize?provider=google&apikey={quote(SUPABASE_KEY, safe='')}"
    if APP_BASE_URL:
        redirect_to = APP_BASE_URL.rstrip("/")
        url += f"&redirect_to={quote(redirect_to, safe='')}"
    return url


def _ensure_authorize_url_params(url: str) -> str:
    """Ensure browser authorize URL includes required apikey and optional redirect."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("provider", "google")
    query.setdefault("apikey", SUPABASE_KEY)
    if APP_BASE_URL:
        query.setdefault("redirect_to", APP_BASE_URL.rstrip("/"))
    encoded_query = urlencode(query, doseq=True, safe=":/")
    return urlunparse(parsed._replace(query=encoded_query))


def get_google_oauth_url() -> tuple[Optional[str], Optional[str]]:
    client = init_supabase_client()
    if not client:
        return None, _client_error or "Supabase auth client is unavailable."
    try:
        payload = {"provider": "google"}
        if APP_BASE_URL:
            payload["options"] = {"redirect_to": APP_BASE_URL.rstrip("/")}
        res = client.auth.sign_in_with_oauth(payload)
        oauth_url = getattr(res, "url", None) if res else None
        if oauth_url:
            return _ensure_authorize_url_params(oauth_url), None
    except Exception as exc:
        message = str(exc)
        if "provider is not enabled" in message.lower() or "unsupported provider" in message.lower():
            return None, (
                "Google provider is not enabled in Supabase Auth. "
                "Enable Google provider and configure redirect URL in Supabase dashboard."
            )

    # Fallback for environments where library response does not include URL.
    return _build_google_authorize_fallback_url(), None


def render_auth_page() -> bool:
    """Render login/register UI. Returns True if authenticated."""
    status = supabase_status_message()
    if status:
        st.warning(f"Auth fallback mode: {status}")
        st.session_state["signed_in"] = True
        return True

    if st.session_state.get("signed_in") and get_user():
        return True

    st.title("🔐 Analytics Forge v2")
    st.markdown("Sign in or create an account to continue.")

    tab_login, tab_register = st.tabs(["Sign In", "Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Sign In", type="primary", key="btn_signin"):
            if email and password:
                res = sign_in(email, password)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.rerun()
            else:
                st.warning("Enter email and password")

    with tab_register:
        reg_email = st.text_input("Email", key="reg_email")
        reg_pw = st.text_input("Password", type="password", key="reg_pw")
        reg_pw2 = st.text_input("Confirm password", type="password", key="reg_pw2")
        if st.button("Create Account", type="primary", key="btn_register"):
            if not reg_email or not reg_pw:
                st.warning("Fill all fields")
            elif reg_pw != reg_pw2:
                st.error("Passwords don't match")
            elif len(reg_pw) < 6:
                st.error("Password must be at least 6 characters")
            else:
                res = sign_up(reg_email, reg_pw)
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.success("Account created! Check email for confirmation.")
                    st.rerun()

    oauth_url, oauth_error = get_google_oauth_url()
    if oauth_url:
        st.markdown(f"[Sign in with Google]({oauth_url})")
    elif oauth_error:
        st.info(oauth_error)

    return False
