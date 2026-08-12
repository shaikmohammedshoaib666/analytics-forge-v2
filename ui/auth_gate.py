"""Login / signup gate for Analytics Forge."""
from __future__ import annotations

import streamlit as st

from core import auth
from ui.theme import page_hero


def require_login() -> bool:
    """
    Render auth UI until session has a user.
    Returns True when the user may use the app.
    """
    if st.session_state.get("user"):
        return True

    page_hero(
        "Analytics Forge",
        "Sign in to keep projects, forecasts, and chat history — data stays tied to your email.",
        None,
    )

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@company.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary")
        if submitted:
            result = auth.login_user(email, password)
            if result.get("ok"):
                st.session_state.user = result["user"]
                st.rerun()
            else:
                st.error(result.get("error", "Login failed"))

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            name = st.text_input("Display name", placeholder="Your name")
            email = st.text_input("Email", key="signup_email", placeholder="you@company.com")
            password = st.text_input("Password (min 8 chars)", type="password", key="signup_pw")
            password2 = st.text_input("Confirm password", type="password", key="signup_pw2")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != password2:
                st.error("Passwords do not match.")
            else:
                result = auth.register_user(email, password, display_name=name)
                if result.get("ok"):
                    st.session_state.user = result["user"]
                    st.success("Account created — welcome.")
                    st.rerun()
                else:
                    st.error(result.get("error", "Could not create account"))

    st.caption(
        "Passwords are stored as secure hashes only — never as plain text. "
        "On Oracle Free / any server, your SQLite (or Postgres later) keeps users and project history."
    )
    return False


def render_user_sidebar() -> None:
    user = st.session_state.get("user")
    if not user:
        return
    st.sidebar.markdown(f"**{user.get('display_name') or 'User'}**")
    st.sidebar.caption(user.get("email", ""))
    if st.sidebar.button("Sign out", use_container_width=True):
        st.session_state.user = None
        st.session_state.chat_history = []
        from ui.session import reset_analysis_state

        reset_analysis_state()
        st.rerun()
