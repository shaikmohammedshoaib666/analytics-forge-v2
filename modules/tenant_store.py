"""
Tenant-aware data layer: Supabase-backed with local SQLite fallback.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

_LOCAL_DB = Path(__file__).resolve().parent.parent / ".forge_sessions" / "tenant.db"


def _use_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _get_client():
    from modules.supabase_auth import init_supabase_client
    return init_supabase_client()


def _local_db():
    _LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_LOCAL_DB))
    conn.execute("""CREATE TABLE IF NOT EXISTS mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, name TEXT, domain TEXT,
        mapping_json TEXT, source_columns_json TEXT,
        updated_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, session_name TEXT,
        metadata_json TEXT, created_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, filename TEXT, upload_meta TEXT, created_at TEXT
    )""")
    conn.commit()
    return conn


# --- Mappings ---

def save_mapping(user_id: str, name: str, domain: str, mapping: dict, source_cols: list) -> bool:
    if _use_supabase():
        client = _get_client()
        client.table("user_mappings").upsert({
            "user_id": user_id,
            "name": name,
            "domain": domain,
            "mapping_json": mapping,
            "source_columns_json": source_cols,
            "updated_at": datetime.utcnow().isoformat(),
        }, on_conflict="user_id,name").execute()
        return True
    conn = _local_db()
    conn.execute(
        "INSERT INTO mappings (user_id, name, domain, mapping_json, source_columns_json, updated_at) VALUES (?,?,?,?,?,?)",
        (user_id, name, domain, json.dumps(mapping), json.dumps(source_cols), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True


def list_mappings(user_id: str) -> list[dict]:
    if _use_supabase():
        client = _get_client()
        res = client.table("user_mappings").select("*").eq("user_id", user_id).execute()
        return res.data or []
    conn = _local_db()
    rows = conn.execute("SELECT id, name, domain, mapping_json, source_columns_json, updated_at FROM mappings WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "domain": r[2], "mapping_json": json.loads(r[3]), "source_columns_json": json.loads(r[4]), "updated_at": r[5]} for r in rows]


def delete_mapping(user_id: str, mapping_id) -> bool:
    if _use_supabase():
        client = _get_client()
        client.table("user_mappings").delete().eq("user_id", user_id).eq("id", mapping_id).execute()
        return True
    conn = _local_db()
    conn.execute("DELETE FROM mappings WHERE user_id=? AND id=?", (user_id, mapping_id))
    conn.commit()
    conn.close()
    return True


# --- Sessions ---

def save_session(user_id: str, name: str, metadata: dict) -> bool:
    if _use_supabase():
        client = _get_client()
        client.table("user_sessions").insert({
            "user_id": user_id,
            "session_name": name,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    conn = _local_db()
    conn.execute(
        "INSERT INTO sessions (user_id, session_name, metadata_json, created_at) VALUES (?,?,?,?)",
        (user_id, name, json.dumps(metadata), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True


def list_sessions(user_id: str) -> list[dict]:
    if _use_supabase():
        client = _get_client()
        res = client.table("user_sessions").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return res.data or []
    conn = _local_db()
    rows = conn.execute("SELECT id, session_name, metadata_json, created_at FROM sessions WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return [{"id": r[0], "session_name": r[1], "metadata": json.loads(r[2]), "created_at": r[3]} for r in rows]


def delete_session(user_id: str, session_id) -> bool:
    if _use_supabase():
        client = _get_client()
        client.table("user_sessions").delete().eq("user_id", user_id).eq("id", session_id).execute()
        return True
    conn = _local_db()
    conn.execute("DELETE FROM sessions WHERE user_id=? AND id=?", (user_id, session_id))
    conn.commit()
    conn.close()
    return True


# --- Uploads ---

def save_upload_meta(user_id: str, filename: str, meta: dict) -> bool:
    if _use_supabase():
        client = _get_client()
        client.table("user_uploads").insert({
            "user_id": user_id,
            "filename": filename,
            "upload_meta": meta,
            "created_at": datetime.utcnow().isoformat(),
        }).execute()
        return True
    conn = _local_db()
    conn.execute(
        "INSERT INTO uploads (user_id, filename, upload_meta, created_at) VALUES (?,?,?,?)",
        (user_id, filename, json.dumps(meta), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return True
