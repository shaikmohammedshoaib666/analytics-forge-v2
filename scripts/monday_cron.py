"""
Monday Cron — sends weekly KPI briefs to users with enabled cron jobs.
Triggered by GitHub Actions every Monday 7:00 UTC.
"""
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")


def _init_client():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None, "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY"
    from modules.supabase_auth import normalize_supabase_url

    normalized_url = normalize_supabase_url(SUPABASE_URL)
    if normalized_url != SUPABASE_URL.strip().rstrip("/"):
        print(
            "[WARN] SUPABASE_URL includes API path suffix; "
            f"using normalized base URL: {normalized_url}"
        )

    try:
        from supabase import create_client
    except Exception as exc:
        return None, f"Supabase import failed: {exc}"
    try:
        return create_client(normalized_url, SUPABASE_SERVICE_KEY), ""
    except Exception as exc:
        message = str(exc)
        if "pgrst125" in message.lower() or "invalid path specified" in message.lower():
            return (
                None,
                "Invalid Supabase URL path detected. "
                f"Resolved base URL: {normalized_url}. Error: {message}",
            )
        return None, f"Supabase client init failed: {message}"


def get_enabled_jobs(client):
    res = client.table("cron_jobs").select("*").eq("enabled", True).eq("job_type", "monday_report").execute()
    return res.data or []


def get_user_latest_session(client, user_id: str):
    res = client.table("user_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
    return res.data[0] if res.data else None


def compose_brief(session_data: dict) -> str:
    meta = session_data.get("metadata", {}) if session_data else {}
    kpis = meta.get("kpis", [])
    actions = meta.get("top_actions", [])
    impact = meta.get("dollar_impact", "N/A")

    lines = [
        "📊 Analytics Forge — Monday Brief",
        f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "🎯 Top 3 Actions:",
    ]
    for i, a in enumerate((actions or ["No actions yet"])[:3], 1):
        lines.append(f"  {i}. {a}")
    lines.append("")
    lines.append(f"💰 $ Impact: {impact}")
    lines.append("")
    lines.append("📈 KPI Summary:")
    for kpi in (kpis or [{"name": "No KPIs", "value": "-"}])[:5]:
        lines.append(f"  • {kpi.get('name', '?')}: {kpi.get('value', '-')}")
    lines.append("")
    lines.append("— Analytics Forge v2")
    return "\n".join(lines)


def send_email(to_email: str, body: str):
    if not EMAIL_USER or not EMAIL_PASSWORD:
        print(f"  [SKIP] No email creds, would send to {to_email}")
        return
    msg = EmailMessage()
    msg["Subject"] = f"Analytics Forge — Monday Brief ({datetime.utcnow().strftime('%b %d')})"
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg.set_content(body)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
    print(f"  [SENT] {to_email}")


def main():
    client, err = _init_client()
    if client is None:
        print(f"[SKIP] Monday cron disabled: {err}")
        return

    jobs = get_enabled_jobs(client)
    print(f"Monday cron: {len(jobs)} enabled job(s)")
    for job in jobs:
        user_id = job["user_id"]
        config = job.get("config", {})
        email = config.get("email", "")
        if not email:
            print(f"  [SKIP] user {user_id}: no email configured")
            continue

        session = get_user_latest_session(client, user_id)
        brief = compose_brief(session)
        send_email(email, brief)

        client.table("cron_jobs").update({"last_run": datetime.utcnow().isoformat()}).eq("id", job["id"]).execute()

    print("Done.")


if __name__ == "__main__":
    main()
