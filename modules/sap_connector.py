"""
SAP Connector — config-only, ready for when plant gives OData/RFC access.
Stores credentials per tenant in Supabase (or local fallback).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional

import streamlit as st

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


@dataclass
class SAPConfig:
    sap_host: str = ""
    sap_client: str = ""
    sap_user: str = ""
    sap_password: str = ""
    service_url: str = ""
    odata_path: str = ""
    schedule: str = "disabled"
    enabled: bool = False


def _use_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _get_client():
    from modules.supabase_auth import init_supabase_client
    return init_supabase_client()


def save_sap_config(user_id: str, config: SAPConfig) -> bool:
    data = asdict(config)
    data["user_id"] = user_id
    if _use_supabase():
        client = _get_client()
        client.table("sap_configs").upsert(data, on_conflict="user_id").execute()
        return True
    st.session_state["_sap_config_local"] = data
    return True


def load_sap_config(user_id: str) -> Optional[SAPConfig]:
    if _use_supabase():
        client = _get_client()
        res = client.table("sap_configs").select("*").eq("user_id", user_id).limit(1).execute()
        if res.data:
            row = res.data[0]
            return SAPConfig(
                sap_host=row.get("sap_host", ""),
                sap_client=row.get("sap_client", ""),
                sap_user=row.get("sap_user", ""),
                sap_password=row.get("sap_password", ""),
                service_url=row.get("service_url", ""),
                odata_path=row.get("odata_path", ""),
                schedule=row.get("schedule", "disabled"),
                enabled=row.get("enabled", False),
            )
        return None
    local = st.session_state.get("_sap_config_local")
    if local:
        return SAPConfig(**{k: v for k, v in local.items() if k != "user_id"})
    return None


def pull_sap_data(config: SAPConfig) -> dict:
    """Attempt SAP OData pull. Stub until real access is available."""
    if not config.service_url:
        return {"status": "error", "message": "No OData service URL configured."}

    try:
        import requests
        url = f"{config.service_url.rstrip('/')}/{config.odata_path.lstrip('/')}"
        resp = requests.get(
            url,
            auth=(config.sap_user, config.sap_password) if config.sap_user else None,
            headers={"sap-client": config.sap_client} if config.sap_client else {},
            timeout=15,
        )
        if resp.status_code == 200:
            return {"status": "ok", "data": resp.json()}
        return {"status": "error", "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except ImportError:
        return {"status": "stub", "message": "SAP pull not yet connected. Enter credentials when your plant gives access."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def render_sap_page(user_id: str):
    """Streamlit page for SAP connector config."""
    st.header("🏭 SAP Connect")
    st.caption("Enter SAP OData/RFC details when your plant gives access. Credentials stored per tenant.")

    existing = load_sap_config(user_id) or SAPConfig()

    with st.form("sap_config_form"):
        host = st.text_input("SAP Host", value=existing.sap_host)
        client = st.text_input("SAP Client", value=existing.sap_client)
        user = st.text_input("SAP User", value=existing.sap_user)
        password = st.text_input("SAP Password", type="password", value=existing.sap_password)
        service_url = st.text_input("OData Service URL", value=existing.service_url, placeholder="https://saphost:port/sap/opu/odata/sap/")
        odata_path = st.text_input("OData Path / Entity", value=existing.odata_path, placeholder="ZPLANT_DATA_SRV/DataSet")
        schedule = st.selectbox("Schedule", ["disabled", "daily", "weekly_monday", "weekly_friday"], index=["disabled", "daily", "weekly_monday", "weekly_friday"].index(existing.schedule) if existing.schedule in ["disabled", "daily", "weekly_monday", "weekly_friday"] else 0)

        col1, col2 = st.columns(2)
        save_btn = col1.form_submit_button("💾 Save Config", type="primary")
        pull_btn = col2.form_submit_button("⬇️ Pull Data (Test)")

    if save_btn:
        cfg = SAPConfig(
            sap_host=host, sap_client=client, sap_user=user,
            sap_password=password, service_url=service_url,
            odata_path=odata_path, schedule=schedule, enabled=bool(service_url),
        )
        save_sap_config(user_id, cfg)
        st.success("SAP config saved.")

    if pull_btn:
        cfg = SAPConfig(
            sap_host=host, sap_client=client, sap_user=user,
            sap_password=password, service_url=service_url,
            odata_path=odata_path, schedule=schedule, enabled=bool(service_url),
        )
        result = pull_sap_data(cfg)
        if result["status"] == "ok":
            st.success("Data pulled successfully!")
            st.json(result.get("data", {}))
        elif result["status"] == "stub":
            st.info(result["message"])
        else:
            st.error(result["message"])
