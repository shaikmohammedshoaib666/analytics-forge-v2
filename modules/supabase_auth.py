"""
Supabase Auth integration for Analytics Forge v2.
Falls back to stub auth if SUPABASE_URL/SUPABASE_KEY not set.
"""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_client = None


def _supabase_available() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def init_supabase_client():
    global _client
    if _client is not None:
        return _client
    if not _supabase_available():
        return None
    from supabase import create_client
    _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


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


def get_google_oauth_url() -> Optional[str]:
    client = init_supabase_client()
    if not client:
        return None
    try:
        res = client.auth.sign_in_with_oauth({"provider": "google"})
        return res.url if res else None
    except Exception:
        return None


def render_auth_page() -> bool:
    """Render login/register UI. Returns True if authenticated."""
    if not _supabase_available():
        st.caption("⚠️ Set SUPABASE_URL + SUPABASE_KEY env vars for real auth")
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

    oauth_url = get_google_oauth_url()
    if oauth_url:
        st.markdown(f"[Sign in with Google]({oauth_url})")

    return False
