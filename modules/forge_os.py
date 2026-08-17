"""Forge OS helpers: cloud-safe secrets, SQLite sessions, mappings, $ impact, briefs.

Python 3.9 compatible. Streamlit is imported only inside UI / session helpers.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = ROOT / ".forge_sessions"
DB_PATH = STORE_DIR / "forge_os.db"
ENV_PATH = ROOT / ".env"

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
BROKEN_GEMINI_ALIASES = frozenset({"gemini-flash-latest", "gemini-flash-latest-latest"})

COLUMN_ROLES = (
    "id",
    "date",
    "metric",
    "category",
    "downtime",
    "loss",
    "qty",
    "scrap",
    "asset",
    "availability",
    "performance",
    "quality",
    "unused",
)

OEE_PULSE_GITHUB = "https://github.com/shaikmohammedshoaib666/oee-pulse"
PHASE3_CAPTION = "SaaS login + Monday scheduled email = next phase."
RENDER_KEY_WARNING = "On Render, set GEMINI_API_KEY in Environment. Session key is temporary."

FRAME_KEYS = ("clean_df", "manual_df")


# -----------------------------------------------------------------------------
# Cloud / secrets
# -----------------------------------------------------------------------------

def is_cloud_host() -> bool:
    """True on Render, Streamlit Community Cloud, or similar PaaS."""
    if str(os.getenv("RENDER") or "").strip() in {"1", "true", "True", "yes"}:
        return True
    if os.getenv("RENDER_EXTERNAL_URL") or os.getenv("RENDER_SERVICE_ID"):
        return True
    if str(os.getenv("STREAMLIT_CLOUD") or "").strip() in {"1", "true", "True", "yes"}:
        return True
    host = (os.getenv("HOSTNAME") or os.getenv("RENDER_EXTERNAL_HOSTNAME") or "").lower()
    if "onrender.com" in host or "streamlit.app" in host:
        return True
    return False


def _dotenv_get(key: str) -> str:
    if not ENV_PATH.exists():
        return ""
    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            k, _, val = raw.partition("=")
            if k.strip() == key:
                return val.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _from_secrets(key: str) -> str:
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return ""
        if key in secrets:
            return str(secrets[key] or "").strip()
        for section in ("general", "gemini", "ai"):
            try:
                block = secrets.get(section)  # type: ignore[attr-defined]
            except Exception:
                block = None
            if block is not None and key in block:
                return str(block[key] or "").strip()
    except Exception:
        return ""
    return ""


def _session_override(key: str) -> str:
    try:
        import streamlit as st

        return str(st.session_state.get(key) or "").strip()
    except Exception:
        return ""


def get_gemini_api_key() -> str:
    """session override → os.environ → st.secrets → .env file."""
    session_key = _session_override("gemini_api_key_override")
    if session_key:
        return session_key
    env_key = str(os.environ.get("GEMINI_API_KEY") or "").strip()
    if env_key:
        return env_key
    secret_key = _from_secrets("GEMINI_API_KEY")
    if secret_key:
        return secret_key
    return _dotenv_get("GEMINI_API_KEY")


def get_gemini_model() -> str:
    """Prefer GEMINI_MODEL env/secrets/.env; remap aliases that often fail."""
    raw = (
        str(os.environ.get("GEMINI_MODEL") or "").strip()
        or _from_secrets("GEMINI_MODEL")
        or _dotenv_get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )
    if not raw or raw.lower() in BROKEN_GEMINI_ALIASES:
        return DEFAULT_GEMINI_MODEL
    return raw


def persist_gemini_key(key: str, write_dotenv: bool = True) -> dict[str, Any]:
    """Store Gemini key in session. .env write is best-effort (Render disk is ephemeral)."""
    key = (key or "").strip()
    try:
        import streamlit as st

        st.session_state.gemini_api_key_override = key
    except Exception:
        pass

    dotenv_ok = False
    warning: Optional[str] = None
    if write_dotenv and key:
        try:
            _write_dotenv_key("GEMINI_API_KEY", key)
            model = get_gemini_model()
            if "GEMINI_MODEL=" not in (ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""):
                _write_dotenv_key("GEMINI_MODEL", model)
            dotenv_ok = True
        except OSError:
            warning = RENDER_KEY_WARNING
        except Exception:
            warning = RENDER_KEY_WARNING
    if is_cloud_host() and not warning:
        warning = RENDER_KEY_WARNING
    return {"ok": True, "dotenv": dotenv_ok, "warning": warning, "cloud": is_cloud_host()}


def _write_dotenv_key(name: str, value: str) -> None:
    lines = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    prefix = f"{name}="
    if prefix in lines:
        out = []
        for line in lines.splitlines():
            if line.startswith(prefix):
                out.append(f"{prefix}{value}")
            else:
                out.append(line)
        ENV_PATH.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
        return
    with ENV_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{prefix}{value}\n")


def test_gemini_connection(prompt: str = "Reply with OK") -> dict[str, Any]:
    """Tiny generate_content probe — never swallows the real exception."""
    key = get_gemini_api_key()
    model_name = get_gemini_model()
    if not key:
        return {
            "ok": False,
            "model": model_name,
            "error": "No Gemini API key (session, GEMINI_API_KEY env, secrets, or .env).",
        }
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = (getattr(resp, "text", None) or str(resp) or "").strip()
        if not text:
            return {
                "ok": False,
                "model": model_name,
                "error": "Gemini returned an empty response. Check quota / model name.",
            }
        return {"ok": True, "model": model_name, "text": text[:240]}
    except Exception as exc:
        return {"ok": False, "model": model_name, "error": str(exc)}


def gemini_issue_from_raw(raw: str, *, attempted: bool) -> Optional[str]:
    if not attempted:
        return None
    text = (raw or "").strip()
    if not text:
        return "Gemini returned an empty response. Offline methods continued."
    if text.startswith("[Gemini error]"):
        return text
    return None


def show_gemini_issue(message: Optional[str]) -> None:
    if not message:
        return
    import streamlit as st

    st.error(message)
    st.warning("Offline fallback is in use — Gemini did not contribute to this result.")


def mask_key(key: str) -> str:
    key = (key or "").strip()
    if len(key) > 12:
        return key[:6] + "…" + key[-4:]
    return "set" if key else "missing"


def render_gemini_key_ui(context: str = "upload") -> None:
    import streamlit as st

    st.subheader("Gemini API key")
    current = get_gemini_api_key()
    model_name = get_gemini_model()
    st.caption(
        f"Status: **{mask_key(current)}** · model `{model_name}` · "
        f"used for field auto-detect + Ask/AI ({context})"
    )
    if is_cloud_host():
        st.caption(
            "Cloud host (Render / Streamlit Cloud): set **GEMINI_API_KEY** and "
            f"**GEMINI_MODEL={DEFAULT_GEMINI_MODEL}** in Environment. "
            "Paste-in-UI is session-only and does not survive deploys."
        )
    else:
        st.caption("Local: Save writes `.env` (gitignored). On Render, paste is session-only — use Environment vars.")
    st.caption(PHASE3_CAPTION)

    st.text_input(
        "Paste Gemini API key",
        value="",
        type="password",
        key=f"gemini_key_input_{context}",
        help="Session first. .env write is skipped/warned on Render. Prefer GEMINI_API_KEY in Environment.",
    )
    b1, b2 = st.columns(2)
    with b1:
        save = st.button("Save Gemini key", key=f"save_gemini_{context}")
    with b2:
        test = st.button("Test Gemini", key=f"test_gemini_{context}")

    if save:
        new_key = str(st.session_state.get(f"gemini_key_input_{context}") or "").strip()
        if not new_key:
            st.warning("Paste a non-empty key.")
        else:
            result = persist_gemini_key(new_key, write_dotenv=True)
            if result.get("dotenv"):
                st.success("Gemini key saved for this session and `.env`.")
            else:
                st.success("Gemini key saved for this session.")
            if result.get("warning"):
                st.warning(result["warning"])
            st.rerun()

    if test:
        with st.spinner("Calling Gemini (Reply with OK)..."):
            probe = test_gemini_connection()
        if probe.get("ok"):
            st.success(f"✓ connected · model `{probe.get('model')}`")
            if probe.get("text"):
                st.caption(f"Reply: {probe['text']}")
        else:
            st.error(probe.get("error") or "Gemini test failed.")
            st.caption(f"Model tried: `{probe.get('model')}`")


# -----------------------------------------------------------------------------
# SQLite session store
# -----------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_store() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_name TEXT,
                meta_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_prefs (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS column_mappings (
                name TEXT PRIMARY KEY,
                mapping_json TEXT NOT NULL DEFAULT '{}',
                source_columns_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_store()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def set_pref(key: str, value: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO app_prefs(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_pref(key: str, default: str = "") -> str:
    with connect() as conn:
        row = conn.execute("SELECT value FROM app_prefs WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else default


def list_sessions(limit: int = 12) -> list[dict[str, Any]]:
    try:
        ensure_store()
    except OSError:
        return []
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at, source_name FROM sessions "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def session_exists(session_id: str) -> bool:
    with connect() as conn:
        row = conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone()
    return row is not None


def _session_dir(session_id: str) -> Path:
    path = STORE_DIR / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return {"__dataframe__": True, "rows": int(len(obj)), "cols": int(obj.shape[1])}
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_frames(session_id: str, frames: dict[str, Optional[pd.DataFrame]]) -> dict[str, str]:
    out_dir = _session_dir(session_id)
    saved: dict[str, str] = {}
    for key, df in frames.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        path = out_dir / f"{key}.csv"
        df.to_csv(path, index=False)
        saved[key] = str(path.relative_to(ROOT))
    return saved


def load_frames(session_id: str) -> dict[str, Optional[pd.DataFrame]]:
    out_dir = STORE_DIR / session_id
    result: dict[str, Optional[pd.DataFrame]] = {k: None for k in FRAME_KEYS}
    if not out_dir.exists():
        return result
    for key in FRAME_KEYS:
        path = out_dir / f"{key}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            for col in df.columns:
                cl = str(col).lower()
                if any(x in cl for x in ("timestamp", "datetime", "date", "time")) or cl.endswith("_date"):
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() >= 0.5:
                        df[col] = parsed
            result[key] = df
        except Exception:
            result[key] = None
    return result


def save_session(
    session_id: str,
    *,
    title: Optional[str] = None,
    source_name: str = "",
    frames: Optional[dict[str, Optional[pd.DataFrame]]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> str:
    ensure_store()
    now = _utcnow()
    frames = frames or {}
    meta = meta or {}
    save_frames(session_id, frames)
    with connect() as conn:
        existing = conn.execute(
            "SELECT created_at, title FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        created = existing["created_at"] if existing else now
        final_title = title or (existing["title"] if existing else f"Session {session_id[:8]}")
        payload = json.dumps(_safe_json(meta), ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO sessions(id, title, created_at, updated_at, source_name, meta_json)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              updated_at=excluded.updated_at,
              source_name=excluded.source_name,
              meta_json=excluded.meta_json
            """,
            (session_id, final_title, created, now, source_name, payload),
        )
        conn.commit()
    set_pref("last_session_id", session_id)
    return session_id


def load_session_meta(session_id: str) -> dict[str, Any]:
    with connect() as conn:
        row = conn.execute(
            "SELECT meta_json, title, source_name FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
    if not row:
        return {}
    try:
        meta = json.loads(row["meta_json"] or "{}")
    except json.JSONDecodeError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["_title"] = row["title"]
    meta["_source_name"] = row["source_name"]
    return meta


def collect_streamlit_meta() -> dict[str, Any]:
    try:
        import streamlit as st
    except Exception:
        return {}
    domain_meta = st.session_state.get("domain_meta") or {}
    if isinstance(domain_meta, dict):
        domain_meta = {
            k: v
            for k, v in domain_meta.items()
            if k not in {"scoreboard", "vote_table", "optuna_proba_table"}
        }
    return {
        "domain": st.session_state.get("domain"),
        "domain_meta": domain_meta,
        "manual_name": st.session_state.get("manual_name"),
        "dashboard_insights": st.session_state.get("dashboard_insights") or [],
        "chat_history": st.session_state.get("chat_history") or [],
        "usd_per_hour": st.session_state.get("usd_per_hour") or 0,
        "usd_per_unit": st.session_state.get("usd_per_unit") or 0,
        "column_roles": st.session_state.get("column_roles") or {},
        "clean_engine": st.session_state.get("clean_engine"),
        "prefer_clean_df": bool(st.session_state.get("prefer_clean_df", True)),
        "mode": st.session_state.get("mode"),
    }


def autosave_after_pipeline(title: Optional[str] = None) -> Optional[str]:
    """Best-effort save of cleaned/manual frames after a successful run."""
    try:
        import streamlit as st
    except Exception:
        return None
    clean = st.session_state.get("clean_df")
    manual = st.session_state.get("manual_df")
    has_clean = isinstance(clean, pd.DataFrame) and not clean.empty
    has_manual = isinstance(manual, pd.DataFrame) and not manual.empty
    if not has_clean and not has_manual:
        return None
    sid = str(st.session_state.get("forge_session_id") or "") or new_session_id()
    st.session_state.forge_session_id = sid
    source = str(st.session_state.get("manual_name") or "session")
    final_title = title or st.session_state.get("forge_session_title") or f"{source} · {sid[:8]}"
    st.session_state.forge_session_title = final_title
    try:
        return save_session(
            sid,
            title=final_title,
            source_name=source,
            frames={"clean_df": clean if has_clean else None, "manual_df": manual if has_manual else None},
            meta=collect_streamlit_meta(),
        )
    except OSError:
        return None
    except Exception:
        return None


def restore_session_to_streamlit(session_id: str) -> bool:
    import streamlit as st

    if not session_exists(session_id):
        return False
    frames = load_frames(session_id)
    meta = load_session_meta(session_id)
    clean = frames.get("clean_df")
    manual = frames.get("manual_df")
    if isinstance(clean, pd.DataFrame) and not clean.empty:
        st.session_state.clean_df = clean
        st.session_state.prefer_clean_df = True
    if isinstance(manual, pd.DataFrame) and not manual.empty:
        st.session_state.manual_df = manual
    elif isinstance(clean, pd.DataFrame) and not clean.empty:
        st.session_state.manual_df = clean
    st.session_state.manual_name = meta.get("manual_name") or meta.get("_source_name") or session_id
    st.session_state.domain = meta.get("domain") or st.session_state.get("domain") or "generic"
    if isinstance(meta.get("domain_meta"), dict):
        st.session_state.domain_meta = meta["domain_meta"]
    st.session_state.dashboard_insights = list(meta.get("dashboard_insights") or [])
    st.session_state.chat_history = list(meta.get("chat_history") or [])
    st.session_state.usd_per_hour = float(meta.get("usd_per_hour") or 0)
    st.session_state.usd_per_unit = float(meta.get("usd_per_unit") or 0)
    st.session_state.column_roles = dict(meta.get("column_roles") or {})
    st.session_state.forge_session_id = session_id
    st.session_state.forge_session_title = meta.get("_title") or session_id
    if meta.get("clean_engine"):
        st.session_state.clean_engine = meta["clean_engine"]
    return True


def render_session_sidebar() -> None:
    import streamlit as st

    st.subheader("Recent sessions")
    st.caption(
        "SQLite under `.forge_sessions/` — survives Streamlit reruns. "
        "On Render this disk is wiped on deploy; that is expected."
    )
    try:
        rows = list_sessions(8)
    except OSError:
        st.caption("Session store unavailable (read-only disk).")
        return
    if not rows:
        st.caption("No saved sessions yet — run Clean or Field to snapshot.")
        return
    for rec in rows:
        sid = rec["id"]
        label = rec.get("title") or sid
        stamp = str(rec.get("updated_at") or "")[:16]
        c1, c2 = st.columns([3, 1])
        c1.caption(f"{label} · {stamp}")
        if c2.button("Restore", key=f"forge_restore_{sid}"):
            if restore_session_to_streamlit(sid):
                st.success(f"Restored {label}")
                st.rerun()
            else:
                st.error("Could not restore that session.")


# -----------------------------------------------------------------------------
# Named column mappings
# -----------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def save_named_mapping(
    name: str,
    mapping: dict[str, str],
    source_columns: Optional[list[str]] = None,
) -> None:
    ensure_store()
    label = (name or "").strip() or "default"
    payload = json.dumps({str(k): str(v) for k, v in (mapping or {}).items() if v}, ensure_ascii=False)
    cols = json.dumps(list(source_columns or []), ensure_ascii=False)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO column_mappings(name, mapping_json, source_columns_json, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
              mapping_json=excluded.mapping_json,
              source_columns_json=excluded.source_columns_json,
              updated_at=excluded.updated_at
            """,
            (label, payload, cols, _utcnow()),
        )
        conn.commit()


def load_named_mapping(name: str) -> Optional[dict[str, str]]:
    ensure_store()
    label = (name or "").strip()
    if not label:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT mapping_json FROM column_mappings WHERE name=?", (label,)
        ).fetchone()
    if row is None:
        return None
    try:
        data = json.loads(row["mapping_json"] or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for k, v in data.items() if v}


def list_named_mappings() -> list[dict[str, Any]]:
    try:
        ensure_store()
    except OSError:
        return []
    with connect() as conn:
        rows = conn.execute(
            "SELECT name, mapping_json, source_columns_json, updated_at "
            "FROM column_mappings ORDER BY updated_at DESC"
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            mapping = json.loads(r["mapping_json"] or "{}")
        except json.JSONDecodeError:
            mapping = {}
        try:
            source_columns = json.loads(r["source_columns_json"] or "[]")
        except json.JSONDecodeError:
            source_columns = []
        out.append(
            {
                "name": r["name"],
                "mapping": mapping if isinstance(mapping, dict) else {},
                "source_columns": source_columns if isinstance(source_columns, list) else [],
                "updated_at": r["updated_at"],
            }
        )
    return out


def resolve_mapping_to_frame(
    columns: list[str],
    saved: dict[str, str],
    min_score: float = 0.72,
) -> dict[str, str]:
    """Map saved source_col → role onto a new frame (exact, then similar names)."""
    result: dict[str, str] = {}
    used: set[str] = set()
    col_list = [str(c) for c in columns]
    for src, role in (saved or {}).items():
        if src in col_list:
            result[src] = str(role)
            used.add(src)
    for src, role in (saved or {}).items():
        if src in result:
            continue
        best: Optional[str] = None
        best_score = 0.0
        src_n = _norm_name(src)
        for col in col_list:
            if col in used:
                continue
            score = SequenceMatcher(None, src_n, _norm_name(col)).ratio()
            if src_n and src_n in _norm_name(col):
                score = max(score, 0.88)
            if score > best_score:
                best, best_score = col, score
        if best is not None and best_score >= min_score:
            result[best] = str(role)
            used.add(best)
    return result


def apply_roles_to_frame(df: pd.DataFrame, roles: dict[str, str]) -> pd.DataFrame:
    """Keep original columns; attach roles via attrs. Copy only — never mutate in place."""
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    out = df.copy()
    out.attrs["column_roles"] = dict(roles or {})
    return out


def suggest_roles(columns: list[str]) -> dict[str, str]:
    hints: dict[str, tuple[str, ...]] = {
        "date": ("date", "timestamp", "datetime", "day", "shift_date"),
        "id": ("id", "machine_id", "asset_id", "customer_id", "sku"),
        "downtime": ("downtime", "down_time", "stop_min", "idle_min"),
        "loss": ("loss", "lost", "cost", "waste"),
        "qty": ("qty", "quantity", "units", "count", "produced"),
        "scrap": ("scrap", "reject", "defect"),
        "asset": ("asset", "machine", "equipment", "line"),
        "availability": ("availability", "avail"),
        "performance": ("performance", "perf", "speed_loss"),
        "quality": ("quality", "fpy", "yield"),
        "metric": ("revenue", "sales", "temperature", "vibration", "oee"),
        "category": ("region", "location", "shift", "category", "type"),
    }
    out: dict[str, str] = {}
    used: set[str] = set()
    for role, keys in hints.items():
        for col in columns:
            if col in used:
                continue
            n = _norm_name(col)
            if any(k in n for k in keys):
                out[col] = role
                used.add(col)
                break
    return out


def render_mapping_ui(df: pd.DataFrame, context: str = "upload") -> dict[str, str]:
    import streamlit as st

    st.subheader("Column mapping")
    st.caption(
        "Save a named column → role map and reuse it on the next similar upload. "
        "Stored in `.forge_sessions/` (ephemeral on Render deploys)."
    )
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.caption("Load a file to map columns.")
        return dict(st.session_state.get("column_roles") or {})

    cols = [str(c) for c in df.columns]
    current = dict(st.session_state.get("column_roles") or {})
    if not current:
        current = suggest_roles(cols)

    saved = list_named_mappings()
    names = [r["name"] for r in saved]
    c1, c2, c3 = st.columns([2, 2, 1])
    map_name = c1.text_input("Mapping name", value="default", key=f"map_name_{context}")
    pick = c2.selectbox("Saved mappings", ["(none)"] + names, key=f"map_pick_{context}")
    if c3.button("Load", key=f"map_load_{context}") and pick != "(none)":
        loaded = load_named_mapping(pick)
        if loaded:
            applied = resolve_mapping_to_frame(cols, loaded)
            st.session_state.column_roles = applied
            st.success(f"Applied **{pick}** ({len(applied)} columns).")
            current = applied
        else:
            st.warning("That mapping is empty or missing.")

    with st.expander("Assign roles", expanded=bool(current)):
        new_roles: dict[str, str] = {}
        role_opts = ["(skip)"] + list(COLUMN_ROLES)
        show = cols[:24]
        for col in show:
            default = current.get(col, "(skip)")
            idx = role_opts.index(default) if default in role_opts else 0
            chosen = st.selectbox(
                col,
                role_opts,
                index=idx,
                key=f"role_{context}_{col}",
            )
            if chosen and chosen != "(skip)":
                new_roles[col] = chosen
        if len(cols) > 24:
            st.caption(f"Showing first 24 of {len(cols)} columns.")
        if st.button("Save mapping", key=f"map_save_{context}"):
            if not map_name.strip():
                st.warning("Enter a mapping name.")
            else:
                try:
                    save_named_mapping(map_name.strip(), new_roles, source_columns=cols)
                    st.session_state.column_roles = new_roles
                    st.success(f"Saved mapping **{map_name.strip()}**.")
                except OSError:
                    st.warning("Could not write mapping (cloud disk). Kept in this session only.")
                    st.session_state.column_roles = new_roles
        elif new_roles:
            st.session_state.column_roles = new_roles
    return dict(st.session_state.get("column_roles") or {})


# -----------------------------------------------------------------------------
# $ impact (generic — not OEE-only)
# -----------------------------------------------------------------------------

def _col_from_roles(df: pd.DataFrame, roles: dict[str, str], role: str) -> Optional[str]:
    for col, mapped in (roles or {}).items():
        if mapped == role and col in df.columns:
            return col
    return None


def _find_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        for k, real in lower.items():
            if n.lower() in k:
                return real
    return None


def estimate_dollar_impact(
    df: pd.DataFrame,
    *,
    usd_per_hour: float = 0.0,
    usd_per_unit: float = 0.0,
    roles: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Estimate $ from downtime/loss hours and/or scrap/qty — any domain."""
    empty = {
        "ok": False,
        "reason": "Enter $/hour and/or $/unit, and map downtime / loss / qty columns.",
        "hours": 0.0,
        "qty_loss": 0.0,
        "downtime_usd": 0.0,
        "qty_usd": 0.0,
        "total_usd": 0.0,
        "hours_col": None,
        "qty_col": None,
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty["reason"] = "No data."
        return empty
    roles = roles or {}
    rate_h = float(usd_per_hour or 0.0)
    rate_u = float(usd_per_unit or 0.0)
    if rate_h <= 0 and rate_u <= 0:
        return empty

    hours_col = (
        _col_from_roles(df, roles, "downtime")
        or _col_from_roles(df, roles, "loss")
        or _find_col(df, "downtime_hours", "down_hours", "lost_hours", "idle_hours")
        or _find_col(df, "downtime_minutes", "downtime_min", "down_min", "stop_minutes", "idle_min", "duration_min")
        or _find_col(df, "downtime", "down_time", "idle_time", "lost_time")
    )
    qty_col = (
        _col_from_roles(df, roles, "scrap")
        or _col_from_roles(df, roles, "qty")
        or _find_col(df, "scrap", "reject_count", "rejects", "defect_qty", "loss_qty", "waste_qty")
    )

    hours = 0.0
    minutes_assumed = False
    if hours_col:
        series = pd.to_numeric(df[hours_col], errors="coerce").fillna(0)
        total = float(series.sum())
        name = str(hours_col).lower()
        if "min" in name or total > 24 * max(len(df), 1):
            hours = total / 60.0
            minutes_assumed = "hour" not in name
        else:
            hours = total

    qty_loss = 0.0
    if qty_col:
        qty_loss = float(pd.to_numeric(df[qty_col], errors="coerce").fillna(0).sum())

    downtime_usd = round(max(hours, 0.0) * max(rate_h, 0.0), 2)
    qty_usd = round(max(qty_loss, 0.0) * max(rate_u, 0.0), 2)
    total = round(downtime_usd + qty_usd, 2)
    if total <= 0 and hours <= 0 and qty_loss <= 0:
        empty["hours_col"] = hours_col
        empty["qty_col"] = qty_col
        empty["reason"] = (
            "Found no numeric downtime/loss/qty to price. "
            "Map a downtime or scrap column, or check the file."
        )
        return empty
    return {
        "ok": True,
        "hours": round(hours, 2),
        "qty_loss": round(qty_loss, 2),
        "downtime_usd": downtime_usd,
        "qty_usd": qty_usd,
        "total_usd": total,
        "hours_col": hours_col,
        "qty_col": qty_col,
        "minutes_assumed": minutes_assumed,
        "usd_per_hour": rate_h,
        "usd_per_unit": rate_u,
        "label": "Management estimate",
    }


def render_dollar_impact(df: pd.DataFrame, key_prefix: str = "kpi") -> dict[str, Any]:
    import streamlit as st

    st.subheader("$ impact")
    st.caption(
        "Generic estimate: $/hour × downtime/loss hours + $/unit × scrap/qty. "
        "Not a finance system of record."
    )
    r1, r2 = st.columns(2)
    usd_h = r1.number_input(
        "$ / hour (downtime or lost time)",
        min_value=0.0,
        value=float(st.session_state.get("usd_per_hour") or 0.0),
        step=50.0,
        key=f"{key_prefix}_usd_hour",
    )
    usd_u = r2.number_input(
        "$ / unit (scrap / qty loss)",
        min_value=0.0,
        value=float(st.session_state.get("usd_per_unit") or 0.0),
        step=1.0,
        key=f"{key_prefix}_usd_unit",
    )
    st.session_state.usd_per_hour = float(usd_h)
    st.session_state.usd_per_unit = float(usd_u)
    roles = dict(st.session_state.get("column_roles") or {})
    impact = estimate_dollar_impact(df, usd_per_hour=float(usd_h), usd_per_unit=float(usd_u), roles=roles)
    if impact.get("ok"):
        m1, m2, m3 = st.columns(3)
        m1.metric("Est. lost hours", f"{impact['hours']:.1f}")
        m2.metric("Est. qty loss", f"{impact['qty_loss']:.0f}")
        m3.metric("Est. $ impact", f"${impact['total_usd']:,.0f}")
        bits = []
        if impact.get("hours_col"):
            bits.append(f"time from `{impact['hours_col']}`")
        if impact.get("qty_col"):
            bits.append(f"qty from `{impact['qty_col']}`")
        st.caption(" · ".join(bits) + f" · {impact.get('label')}")
    elif float(usd_h) > 0 or float(usd_u) > 0:
        st.info(impact.get("reason") or "No $ impact yet.")
    return impact


# -----------------------------------------------------------------------------
# Manager brief (Top 3 actions)
# -----------------------------------------------------------------------------

def _quality_fail_lines(quality_checks: Any, limit: int = 2) -> list[str]:
    if quality_checks is None or not isinstance(quality_checks, pd.DataFrame) or quality_checks.empty:
        return []
    work = quality_checks
    status_col = "status" if "status" in work.columns else None
    check_col = "check" if "check" in work.columns else work.columns[0]
    if status_col:
        bad = work[work[status_col].astype(str).str.upper().isin(["FAIL", "WARN"])]
    else:
        bad = work
    lines: list[str] = []
    for _, row in bad.head(limit).iterrows():
        detail = row.get("detail") if "detail" in work.columns else ""
        lines.append(f"Quality {row.get(check_col)}: {detail}".strip(": "))
    return lines


def build_top3_actions(
    *,
    insights: Optional[list[Any]] = None,
    quality_checks: Any = None,
    ml_result: Optional[dict[str, Any]] = None,
    dollar_impact: Optional[dict[str, Any]] = None,
    field_actions: Optional[list[Any]] = None,
    use_gemini: bool = True,
) -> dict[str, Any]:
    """Rule-based Top 3; Gemini polish when the key works."""
    candidates: list[str] = []
    for line in field_actions or []:
        text = str(line).strip()
        if text:
            candidates.append(text)
    for line in insights or []:
        text = str(line).strip()
        if text:
            candidates.append(text)
    if ml_result and isinstance(ml_result, dict):
        brief = str(ml_result.get("manager_briefing") or "").strip()
        if brief:
            candidates.append(brief)
        err = str(ml_result.get("error") or "").strip()
        if err and not ml_result.get("ok"):
            candidates.append(f"ML issue: {err}")
    candidates.extend(_quality_fail_lines(quality_checks))
    if dollar_impact and dollar_impact.get("ok") and float(dollar_impact.get("total_usd") or 0) > 0:
        qty_bit = ""
        qty_loss = dollar_impact.get("qty_loss") or 0
        if qty_loss:
            qty_bit = ", {:.0f} qty".format(float(qty_loss))
        candidates.append(
            "Price the loss: ~${:,.0f} ({:.1f} h downtime/loss{}).".format(
                float(dollar_impact["total_usd"]),
                float(dollar_impact.get("hours") or 0),
                qty_bit,
            )
        )

    seen: set[str] = set()
    unique: list[str] = []
    for line in candidates:
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
        if len(unique) >= 6:
            break
    if not unique:
        unique = [
            "Run Clean + Field so quality checks and a domain model exist.",
            "Pin the weakest KPI on Charts → Dashboard.",
            "Email the pack from the Email page once SMTP is set.",
        ]
    defaults = [
        "Review Auto KPIs and pin the weakest metric on the Dashboard.",
        "Email the pack from the Email page once SMTP is set.",
        "Save a named column mapping so the next upload maps in one click.",
    ]
    for line in defaults:
        if len(unique) >= 3:
            break
        key = re.sub(r"\s+", " ", line).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(line)
    actions = unique[:3]
    gemini_error: Optional[str] = None
    source = "rule"
    if use_gemini and get_gemini_api_key():
        prompt = (
            "You are a concise operations analyst. From these findings, output EXACTLY 3 numbered "
            "manager actions (one line each). No preamble.\n\n" + "\n".join(f"- {a}" for a in unique[:8])
        )
        try:
            import google.generativeai as genai

            genai.configure(api_key=get_gemini_api_key())
            model = genai.GenerativeModel(get_gemini_model())
            resp = model.generate_content(prompt)
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                gemini_error = "Gemini returned an empty response for the manager brief."
            else:
                lines = [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip() for ln in text.splitlines() if ln.strip()]
                lines = [ln for ln in lines if ln and not ln.startswith("#")]
                if lines:
                    actions = lines[:3]
                    source = "gemini"
                else:
                    gemini_error = "Gemini brief had no actionable lines; using rule-based Top 3."
        except Exception as exc:
            gemini_error = f"[Gemini error] {exc}"
    body = "Top 3 actions\n" + "\n".join(f"{i}. {a}" for i, a in enumerate(actions, 1))
    return {"actions": actions, "source": source, "gemini_error": gemini_error, "body": body}


def render_manager_brief(
    *,
    insights: Optional[list[Any]] = None,
    quality_checks: Any = None,
    ml_result: Optional[dict[str, Any]] = None,
    dollar_impact: Optional[dict[str, Any]] = None,
    field_actions: Optional[list[Any]] = None,
    key_prefix: str = "kpi",
) -> dict[str, Any]:
    import streamlit as st

    st.subheader("Top 3 actions")
    st.caption(PHASE3_CAPTION)
    fingerprint = json.dumps(
        {
            "insights": list(insights or [])[:8],
            "usd": (dollar_impact or {}).get("total_usd"),
            "ml": (ml_result or {}).get("manager_briefing") or (ml_result or {}).get("model_id"),
            "field": list(field_actions or [])[:4],
        },
        default=str,
    )
    cache_key = "forge_manager_brief"
    cached = st.session_state.get(cache_key)
    refresh = st.button("Refresh Top 3", key=f"{key_prefix}_refresh_brief")
    if isinstance(cached, dict) and cached.get("fingerprint") == fingerprint and not refresh:
        brief = cached
    else:
        brief = build_top3_actions(
            insights=insights,
            quality_checks=quality_checks,
            ml_result=ml_result,
            dollar_impact=dollar_impact,
            field_actions=field_actions,
            use_gemini=bool(get_gemini_api_key()),
        )
        brief["fingerprint"] = fingerprint
        st.session_state[cache_key] = brief
    if brief.get("gemini_error"):
        show_gemini_issue(str(brief["gemini_error"]))
    st.caption(f"Source: **{brief.get('source')}**")
    for i, line in enumerate(brief.get("actions") or [], 1):
        st.markdown(f"**{i}.** {line}")
    st.session_state[f"{key_prefix}_manager_brief"] = brief
    return brief


# -----------------------------------------------------------------------------
# Industry routing (do not merge OEE Pulse)
# -----------------------------------------------------------------------------

def looks_like_plant_oee(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"match": False, "hits": [], "score": 0}
    toks: set[str] = set()
    for col in df.columns:
        n = _norm_name(col)
        toks.add(n)
        toks.update(p for p in n.split("_") if p)
    strong = {"availability", "oee", "downtime", "scrap", "planned", "performance", "reject"}
    weak = {"quality", "asset", "machine", "runtime", "run", "good", "line", "shift"}
    hits = sorted((toks & strong) | (toks & weak))
    score = 2 * len(toks & strong) + len(toks & weak)
    return {"match": score >= 4, "hits": hits, "score": score}


def render_industry_banner(df: Optional[pd.DataFrame]) -> None:
    import streamlit as st

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return
    meta = looks_like_plant_oee(df)
    if not meta.get("match"):
        return
    hits = ", ".join(f"`{h}`" for h in (meta.get("hits") or [])[:8])
    st.info(
        "This looks like plant/OEE data → use **OEE Pulse** for the weekly plant ritual "
        f"(availability / downtime / scrap). Open [{OEE_PULSE_GITHUB}]({OEE_PULSE_GITHUB}). "
        f"Signals: {hits}."
    )
