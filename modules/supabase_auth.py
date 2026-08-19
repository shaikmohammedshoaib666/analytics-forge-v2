"""
Supabase Auth integration for Analytics Forge v2.
Falls back to stub auth if SUPABASE_URL/SUPABASE_KEY not set.
"""
from __future__ import annotations

import base64
import logging
import os
import warnings
from importlib import import_module
from typing import Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()
_INVALID_URL_SEGMENTS = ("/rest/v1", "/auth/v1")

_client = None
_client_error = ""


def _supabase_available() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def normalize_supabase_url(raw_url: str) -> str:
    """
    Normalize Supabase base URL from env input.

    Accepts values that may accidentally include /rest/v1 or /auth/v1 and
    always returns the host base URL without trailing slash.
    """
    candidate = (raw_url or "").strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if not parsed.scheme or not parsed.netloc:
        return candidate.rstrip("/")

    path = parsed.path.rstrip("/")
    lowered = path.lower()
    # Users sometimes set SUPABASE_URL with API paths; peel those off.
    while any(lowered.endswith(suffix) for suffix in _INVALID_URL_SEGMENTS):
        for suffix in _INVALID_URL_SEGMENTS:
            if lowered.endswith(suffix):
                path = path[: -len(suffix)]
                lowered = path.lower()
                break

    path = path.rstrip("/")
    normalized = parsed._replace(path=path, params="", query="", fragment="")
    return urlunparse(normalized).rstrip("/")


def _looks_like_invalid_supabase_path(raw_url: str) -> bool:
    parsed = urlparse((raw_url or "").strip())
    path = parsed.path.lower().rstrip("/")
    return any(path.endswith(seg) for seg in _INVALID_URL_SEGMENTS)


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

    normalized_url = normalize_supabase_url(SUPABASE_URL)
    if not normalized_url:
        _client_error = "Supabase URL is empty after normalization."
        return None

    if _looks_like_invalid_supabase_path(SUPABASE_URL):
        warning_msg = (
            "SUPABASE_URL contains API path suffix. "
            f"Using normalized base URL: {normalized_url}"
        )
        logging.warning(warning_msg)
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    try:
        _client = create_client(normalized_url, SUPABASE_KEY)
        _client_error = ""
    except Exception as exc:
        _client = None
        message = str(exc)
        if "pgrst125" in message.lower() or "invalid path specified" in message.lower():
            _client_error = (
                "Supabase URL path appears invalid. "
                f"Resolved base URL: {normalized_url}. "
                f"Original error: {message}"
            )
        else:
            _client_error = f"Supabase client init failed: {message}"
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
    base = normalize_supabase_url(SUPABASE_URL)
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


def _set_signed_in_user(user, session) -> bool:
    if not user:
        return False
    st.session_state["supabase_user"] = {"id": user.id, "email": user.email}
    st.session_state["supabase_session"] = session
    st.session_state["signed_in"] = True
    return True


def _query_params_dict() -> dict[str, str]:
    try:
        params_obj = getattr(st, "query_params", None)
        if params_obj is not None:
            return {k: params_obj.get(k, "") for k in params_obj.keys()}
    except Exception:
        pass
    try:
        raw = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in raw.items()}
    except Exception:
        return {}


def _clear_auth_query_params():
    auth_keys = {
        "code",
        "error",
        "error_code",
        "error_description",
        "access_token",
        "refresh_token",
        "token_type",
        "expires_in",
        "provider_token",
        "provider_refresh_token",
    }
    try:
        params_obj = getattr(st, "query_params", None)
        if params_obj is not None:
            for key in list(params_obj.keys()):
                if key in auth_keys:
                    del params_obj[key]
            return
    except Exception:
        pass
    # Fallback API: rewrite with non-auth params.
    try:
        current = st.experimental_get_query_params()
        cleaned = {k: v for k, v in current.items() if k not in auth_keys}
        st.experimental_set_query_params(**cleaned)
    except Exception:
        pass


def _exchange_auth_code(client, auth_code: str):
    # supabase-py has used both dict and positional signatures across versions.
    last_error = None
    try:
        return client.auth.exchange_code_for_session({"auth_code": auth_code})
    except Exception as exc:
        last_error = exc
    try:
        return client.auth.exchange_code_for_session(auth_code)
    except Exception:
        raise last_error


def _handle_oauth_callback(client) -> tuple[bool, Optional[str]]:
    params = _query_params_dict()
    if not params:
        return False, None

    if params.get("error"):
        description = params.get("error_description") or "OAuth sign-in was cancelled or denied."
        _clear_auth_query_params()
        return False, description

    code = (params.get("code") or "").strip()
    if code:
        if st.session_state.get("_last_oauth_code") == code and st.session_state.get("signed_in") and get_user():
            return True, None
        try:
            res = _exchange_auth_code(client, code)
            user = getattr(res, "user", None) if res else None
            session = getattr(res, "session", None) if res else None
            if _set_signed_in_user(user, session):
                st.session_state["_last_oauth_code"] = code
                _clear_auth_query_params()
                return True, None
            return False, "Google sign-in callback was received, but no user session was returned."
        except Exception as exc:
            _clear_auth_query_params()
            return False, f"Google sign-in failed. The callback may be expired. ({exc})"

    access_token = (params.get("access_token") or "").strip()
    refresh_token = (params.get("refresh_token") or "").strip()
    if access_token and refresh_token:
        try:
            res = client.auth.set_session(access_token, refresh_token)
            user = getattr(res, "user", None) if res else None
            session = getattr(res, "session", None) if res else None
            if _set_signed_in_user(user, session):
                _clear_auth_query_params()
                return True, None
        except Exception:
            pass

    if any(k in params for k in ("provider_token", "provider_refresh_token", "access_token")):
        # Hash fragments are not available to Streamlit server callbacks.
        return (
            False,
            "Google callback tokens were not readable server-side. Please retry Google sign-in to complete code exchange.",
        )
    return False, None


def render_auth_page() -> bool:
    """Render login/register UI. Returns True if authenticated."""
    status = supabase_status_message()
    if status:
        st.warning(f"Auth fallback mode: {status}")
        st.session_state["signed_in"] = True
        return True

    client = init_supabase_client()
    if client:
        callback_authenticated, callback_error = _handle_oauth_callback(client)
        if callback_authenticated:
            st.rerun()
        if callback_error:
            st.error(callback_error)

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
