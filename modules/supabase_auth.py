"""
Supabase Auth integration for Analytics Forge v2.
Falls back to stub auth if SUPABASE_URL/SUPABASE_KEY not set.
"""
from __future__ import annotations

import base64
import logging
import os
import socket
import warnings
from importlib import import_module
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

import requests
import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip()
_INVALID_URL_SEGMENTS = ("/rest/v1", "/auth/v1")

_client = None
_client_error = ""
_connectivity_cache: dict[str, Any] = {"host": "", "checked": False, "reachable": True, "error": ""}


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


def _supabase_host() -> str:
    normalized = normalize_supabase_url(SUPABASE_URL)
    return urlparse(normalized).netloc if normalized else ""


def _unreachable_supabase_message(host: str) -> str:
    display_host = host or "unknown host"
    return (
        f"Cannot reach Supabase at {display_host}. "
        "Check Render SUPABASE_URL — project may be paused or deleted. "
        "Open Supabase dashboard and update env vars."
    )


def _looks_like_connectivity_error(message: str) -> bool:
    lowered = (message or "").lower()
    markers = (
        "nxdomain",
        "name or service not known",
        "failed to resolve",
        "nodename nor servname provided",
        "getaddrinfo failed",
        "connection refused",
        "connection error",
        "connection aborted",
        "network is unreachable",
        "temporary failure in name resolution",
        "max retries exceeded",
        "name resolution",
        "errno -2",
        "errno -3",
        "errno 110",
        "errno 111",
    )
    return any(marker in lowered for marker in markers)


def check_supabase_connectivity(*, force: bool = False) -> tuple[bool, str]:
    """
    Lightweight DNS + HTTP check against Supabase Auth health endpoint.
    Returns (reachable, error_message).
    """
    global _connectivity_cache

    if not _supabase_available():
        return False, "Supabase env vars are not configured."

    host = _supabase_host()
    if not host:
        return False, "Supabase URL is empty after normalization."

    if (
        not force
        and _connectivity_cache["checked"]
        and _connectivity_cache["host"] == host
    ):
        if _connectivity_cache["reachable"]:
            return True, ""
        return False, str(_connectivity_cache["error"])

    normalized = normalize_supabase_url(SUPABASE_URL)
    health_url = f"{normalized}/auth/v1/health"
    error = ""

    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        error = _unreachable_supabase_message(host)
        _connectivity_cache = {
            "host": host,
            "checked": True,
            "reachable": False,
            "error": error,
        }
        return False, error

    try:
        resp = requests.head(health_url, timeout=5, allow_redirects=True)
        # Any HTTP response means the host exists; auth may still fail for other reasons.
        if resp.status_code >= 500:
            logging.warning(
                "Supabase health check returned %s for %s", resp.status_code, health_url
            )
        _connectivity_cache = {"host": host, "checked": True, "reachable": True, "error": ""}
        return True, ""
    except requests.exceptions.Timeout:
        error = _unreachable_supabase_message(host)
    except requests.exceptions.RequestException as exc:
        if _looks_like_connectivity_error(str(exc)):
            error = _unreachable_supabase_message(host)
        else:
            error = _unreachable_supabase_message(host)
            logging.warning("Supabase connectivity check failed for %s: %s", health_url, exc)

    _connectivity_cache = {"host": host, "checked": True, "reachable": False, "error": error}
    return False, error


def _ensure_supabase_reachable() -> Optional[str]:
    """Return a user-facing error when Supabase host is unreachable."""
    reachable, error = check_supabase_connectivity()
    if reachable:
        return None
    st.session_state["_last_auth_error"] = error
    return error


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


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a pydantic model, plain object, or dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    if value is not None:
        return value
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump().get(name, default)
        except Exception:
            pass
    return default


def _parse_auth_response(res: Any) -> tuple[Any, Any]:
    """Extract user/session from AuthResponse across supabase-py versions."""
    if res is None:
        return None, None
    if isinstance(res, dict):
        return res.get("user"), res.get("session")
    user = _field(res, "user")
    session = _field(res, "session")
    if user is None and session is None:
        model_dump = getattr(res, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
                if isinstance(dumped, dict):
                    user = dumped.get("user")
                    session = dumped.get("session")
            except Exception:
                pass
    return user, session


def _normalize_user(user: Any) -> Optional[dict]:
    if not user:
        return None
    user_id = _field(user, "id")
    if not user_id:
        return None
    return {"id": str(user_id), "email": _field(user, "email") or ""}


def _user_is_confirmed(user: Any) -> bool:
    return bool(_field(user, "email_confirmed_at") or _field(user, "confirmed_at"))


def _session_tokens(session: Any) -> tuple[Optional[str], Optional[str]]:
    if not session:
        return None, None
    access = _field(session, "access_token")
    refresh = _field(session, "refresh_token")
    return (
        str(access).strip() if access else None,
        str(refresh).strip() if refresh else None,
    )


def _format_auth_error(exc: Exception) -> str:
    """Map supabase-py / GoTrue errors to user-friendly messages."""
    message = str(exc).strip()
    lowered = message.lower()
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    name = getattr(exc, "name", "") or exc.__class__.__name__

    if name == "AuthInvalidCredentialsError":
        return "Invalid email or password."
    if name == "AuthWeakPasswordError":
        return message or "Password is too weak. Use at least 6 characters with mixed characters."

    if code in ("email_not_confirmed", "user_not_confirmed") or "email not confirmed" in lowered:
        return "Please confirm your email before signing in. Check your inbox for the confirmation link."
    if code in ("invalid_credentials", "invalid_grant") or "invalid login credentials" in lowered:
        return "Invalid email or password."
    if code == "user_already_registered" or "already registered" in lowered or "already been registered" in lowered:
        return "An account with this email already exists. Try signing in instead."
    if code == "signup_disabled" or "signups not allowed" in lowered:
        return "New registrations are disabled in Supabase Auth settings."
    if status == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return "Too many attempts. Please wait a few minutes and try again."
    if "provider is not enabled" in lowered or "unsupported provider" in lowered:
        return (
            "Google sign-in is not enabled in Supabase Auth. "
            "Enable the Google provider and add your Render URL to redirect URLs."
        )
    if "invalid api key" in lowered or "invalid jwt" in lowered:
        return "Supabase API key appears invalid. Use the anon/publishable key from Project Settings → API."
    if _looks_like_connectivity_error(message):
        return _unreachable_supabase_message(_supabase_host())
    if message:
        return message
    return "Authentication failed. Please try again."


def _auth_action_failure(action: str, res: Any) -> str:
    user, session = _parse_auth_response(res)
    if user and not session:
        if action == "sign-up":
            return (
                "Account may have been created, but no session was returned. "
                "If email confirmation is enabled, check your inbox and confirm before signing in."
            )
        if not _user_is_confirmed(user):
            return "Please confirm your email before signing in."
        return f"{action.title()} succeeded but no session was returned. Check Supabase Auth settings."
    return f"{action.title()} failed. No user was returned from Supabase."


def _supabase_package_version() -> str:
    try:
        module = import_module("supabase")
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not installed"


def auth_health_diagnostic() -> dict:
    """Safe auth health snapshot for troubleshooting (no secrets)."""
    normalized = normalize_supabase_url(SUPABASE_URL)
    host = urlparse(normalized).netloc if normalized else ""
    key_hint = "missing"
    if SUPABASE_KEY:
        if _is_likely_service_role_key(SUPABASE_KEY):
            key_hint = "service_role (invalid for browser auth)"
        else:
            key_hint = "anon_or_publishable"
    client_ready = init_supabase_client() is not None
    reachable, connectivity_error = (
        check_supabase_connectivity() if _supabase_available() else (False, None)
    )
    return {
        "supabase_env_configured": _supabase_available(),
        "client_ready": client_ready,
        "client_error": _client_error or None,
        "supabase_host": host or None,
        "supabase_reachable": reachable if _supabase_available() else None,
        "supabase_connectivity_error": connectivity_error or None,
        "url_had_api_path_suffix": _looks_like_invalid_supabase_path(SUPABASE_URL),
        "app_base_url_set": bool(APP_BASE_URL),
        "app_base_url_host": urlparse(APP_BASE_URL).netloc if APP_BASE_URL else None,
        "key_type_hint": key_hint,
        "supabase_py_version": _supabase_package_version(),
        "last_auth_error": st.session_state.get("_last_auth_error"),
    }


def sign_up(email: str, password: str) -> dict:
    connectivity_error = _ensure_supabase_reachable()
    if connectivity_error:
        return {"error": connectivity_error}
    client = init_supabase_client()
    if not client:
        err = _client_error or "Supabase auth client failed to initialize."
        st.session_state["_last_auth_error"] = err
        return {"error": err}
    try:
        res = client.auth.sign_up({"email": email, "password": password})
        user, session = _parse_auth_response(res)
        if user and session and _set_signed_in_user(user, session):
            st.session_state["_last_auth_error"] = None
            return {"user": _normalize_user(user), "signed_in": True}
        if user and not session:
            st.session_state["_last_auth_error"] = None
            return {
                "user": _normalize_user(user),
                "needs_confirmation": not _user_is_confirmed(user),
                "signed_in": False,
                "message": (
                    "Account created. Check your email for a confirmation link, "
                    "then return here to sign in."
                ),
            }
        err = _auth_action_failure("sign-up", res)
        st.session_state["_last_auth_error"] = err
        return {"error": err}
    except Exception as exc:
        err = _format_auth_error(exc)
        st.session_state["_last_auth_error"] = err
        return {"error": err}


def sign_in(email: str, password: str) -> dict:
    connectivity_error = _ensure_supabase_reachable()
    if connectivity_error:
        return {"error": connectivity_error}
    client = init_supabase_client()
    if not client:
        err = _client_error or "Supabase auth client failed to initialize."
        st.session_state["_last_auth_error"] = err
        return {"error": err}
    try:
        res = client.auth.sign_in_with_password({"email": email, "password": password})
        user, session = _parse_auth_response(res)
        if user and session and _set_signed_in_user(user, session):
            st.session_state["_last_auth_error"] = None
            return {"user": _normalize_user(user), "signed_in": True}
        err = _auth_action_failure("sign-in", res)
        st.session_state["_last_auth_error"] = err
        return {"error": err}
    except Exception as exc:
        err = _format_auth_error(exc)
        st.session_state["_last_auth_error"] = err
        return {"error": err}


def sign_out():
    client = init_supabase_client()
    if client:
        try:
            client.auth.sign_out()
        except Exception:
            pass
    st.session_state["supabase_user"] = None
    st.session_state["supabase_session"] = None
    st.session_state["supabase_access_token"] = None
    st.session_state["supabase_refresh_token"] = None
    st.session_state["signed_in"] = False
    st.session_state["_last_oauth_code"] = None
    st.session_state["_oauth_failed_code"] = None
    st.session_state["_oauth_failed_error"] = None


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
    url += "&prompt=select_account"
    return url


def _ensure_authorize_url_params(url: str) -> str:
    """Ensure browser authorize URL includes required apikey and optional redirect."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("provider", "google")
    query.setdefault("apikey", SUPABASE_KEY)
    if APP_BASE_URL:
        query["redirect_to"] = APP_BASE_URL.rstrip("/")
    query["prompt"] = "select_account"
    encoded_query = urlencode(query, doseq=True, safe=":/")
    return urlunparse(parsed._replace(query=encoded_query))


def get_google_oauth_url() -> tuple[Optional[str], Optional[str]]:
    connectivity_error = _ensure_supabase_reachable()
    if connectivity_error:
        return None, connectivity_error
    client = init_supabase_client()
    if not client:
        return None, _client_error or "Supabase auth client is unavailable."
    try:
        payload = {"provider": "google"}
        if APP_BASE_URL:
            payload["options"] = {
                "redirect_to": APP_BASE_URL.rstrip("/"),
                "query_params": {"prompt": "select_account"},
            }
        else:
            payload["options"] = {"query_params": {"prompt": "select_account"}}
        res = client.auth.sign_in_with_oauth(payload)
        oauth_url = _field(res, "url")
        if oauth_url:
            return _ensure_authorize_url_params(str(oauth_url)), None
    except Exception as exc:
        message = _format_auth_error(exc)
        st.session_state["_last_auth_error"] = message
        if "google" in message.lower() or "provider" in message.lower():
            return None, message

    # Fallback for environments where library response does not include URL.
    return _build_google_authorize_fallback_url(), None


def _set_signed_in_user(user, session, *, require_session: bool = False) -> bool:
    normalized = _normalize_user(user)
    if not normalized:
        return False
    access_token, refresh_token = _session_tokens(session)
    if require_session and not (access_token and refresh_token):
        return False
    st.session_state["supabase_user"] = normalized
    st.session_state["supabase_session"] = session
    st.session_state["supabase_access_token"] = access_token
    st.session_state["supabase_refresh_token"] = refresh_token
    st.session_state["signed_in"] = True
    return True


def _restore_signed_in_user(client) -> bool:
    """Restore an authenticated user from saved session tokens or client session."""
    if st.session_state.get("signed_in") and get_user():
        return True

    access_token = (st.session_state.get("supabase_access_token") or "").strip()
    refresh_token = (st.session_state.get("supabase_refresh_token") or "").strip()
    if access_token and refresh_token:
        try:
            res = client.auth.set_session(access_token, refresh_token)
            user, session = _parse_auth_response(res)
            if _set_signed_in_user(user, session, require_session=True):
                return True
        except Exception:
            pass

    try:
        session_res = client.auth.get_session()
        session = _field(session_res, "session")
        user_res = client.auth.get_user()
        user = _field(user_res, "user")
        if _set_signed_in_user(user, session, require_session=True):
            return True
    except Exception:
        pass
    return False


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
        return False, description

    code = (params.get("code") or "").strip()
    if code:
        if st.session_state.get("_oauth_failed_code") == code:
            cached_error = st.session_state.get("_oauth_failed_error") or "Google sign-in failed."
            return False, f"{cached_error} Remove the stale callback URL params and try Google sign-in again."
        if st.session_state.get("_last_oauth_code") == code and st.session_state.get("signed_in") and get_user():
            return True, None
        try:
            res = _exchange_auth_code(client, code)
            user, session = _parse_auth_response(res)
            if _set_signed_in_user(user, session, require_session=True):
                st.session_state["_last_oauth_code"] = code
                st.session_state["_oauth_failed_code"] = None
                st.session_state["_oauth_failed_error"] = None
                st.session_state["_last_auth_error"] = None
                _clear_auth_query_params()
                return True, None
            error_message = "Google sign-in callback was received, but no user session was returned."
            st.session_state["_oauth_failed_code"] = code
            st.session_state["_oauth_failed_error"] = error_message
            st.session_state["_last_auth_error"] = error_message
            return False, error_message
        except Exception as exc:
            error_message = f"Google sign-in failed: {_format_auth_error(exc)}"
            st.session_state["_oauth_failed_code"] = code
            st.session_state["_oauth_failed_error"] = error_message
            st.session_state["_last_auth_error"] = error_message
            return False, error_message

    access_token = (params.get("access_token") or "").strip()
    refresh_token = (params.get("refresh_token") or "").strip()
    if access_token and refresh_token:
        try:
            res = client.auth.set_session(access_token, refresh_token)
            user, session = _parse_auth_response(res)
            if _set_signed_in_user(user, session, require_session=True):
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


def _render_auth_diagnostic():
    with st.expander("Auth diagnostic (no secrets)", expanded=False):
        st.json(auth_health_diagnostic())


def render_auth_page() -> bool:
    """Render login/register UI. Returns True if authenticated."""
    status = supabase_status_message()
    if status:
        st.warning(f"Auth fallback mode: {status}")
        st.session_state["signed_in"] = True
        return True

    client = init_supabase_client()
    connectivity_error = None
    if _supabase_available():
        _, connectivity_error = check_supabase_connectivity()
    if client:
        if _restore_signed_in_user(client):
            return True
        callback_authenticated, callback_error = _handle_oauth_callback(client)
        if callback_authenticated:
            st.rerun()
        if callback_error:
            st.error(callback_error)
            _render_auth_diagnostic()

    if st.session_state.get("signed_in") and get_user():
        return True

    st.title("🔐 Analytics Forge v2")
    st.markdown("Sign in or create an account to continue.")

    if connectivity_error:
        st.error(connectivity_error)
        _render_auth_diagnostic()

    tab_login, tab_register = st.tabs(["Sign In", "Register"])
    auth_error_shown = False

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Sign In", type="primary", key="btn_signin"):
            if email and password:
                res = sign_in(email, password)
                if "error" in res:
                    st.error(res["error"])
                    auth_error_shown = True
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
                auth_error_shown = True
            elif len(reg_pw) < 6:
                st.error("Password must be at least 6 characters")
                auth_error_shown = True
            else:
                res = sign_up(reg_email, reg_pw)
                if "error" in res:
                    st.error(res["error"])
                    auth_error_shown = True
                elif res.get("signed_in"):
                    st.rerun()
                else:
                    st.success(res.get("message") or "Account created! Check your email for confirmation.")

    oauth_url, oauth_error = get_google_oauth_url()
    if oauth_url:
        st.link_button("Sign in with Google (choose account)", oauth_url, type="secondary")
        if not APP_BASE_URL:
            st.caption(
                "Tip: set **APP_BASE_URL** to your Render URL "
                "(e.g. `https://analytics-forge-v2.onrender.com`) so Google can return here after login."
            )
    elif oauth_error:
        st.info(oauth_error)
        auth_error_shown = True

    if auth_error_shown or st.session_state.get("_last_auth_error"):
        _render_auth_diagnostic()

    return False
