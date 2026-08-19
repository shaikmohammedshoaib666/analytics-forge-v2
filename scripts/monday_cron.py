"""
Monday Cron — sends weekly KPI briefs to users with enabled cron jobs.
Triggered by GitHub Actions every Monday 7:00 UTC.
"""
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage

from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
EMAIL_USER = os.environ.get("EMAIL_USER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")

client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_enabled_jobs():
    res = client.table("cron_jobs").select("*").eq("enabled", True).eq("job_type", "monday_report").execute()
    return res.data or []


def get_user_latest_session(user_id: str):
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
    jobs = get_enabled_jobs()
    print(f"Monday cron: {len(jobs)} enabled job(s)")
    for job in jobs:
        user_id = job["user_id"]
        config = job.get("config", {})
        email = config.get("email", "")
        if not email:
            print(f"  [SKIP] user {user_id}: no email configured")
            continue

        session = get_user_latest_session(user_id)
        brief = compose_brief(session)
        send_email(email, brief)

        client.table("cron_jobs").update({"last_run": datetime.utcnow().isoformat()}).eq("id", job["id"]).execute()

    print("Done.")


if __name__ == "__main__":
    main()
