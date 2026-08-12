"""
Forge LIVE Gateway — runs near OCP-U (Pi / plant PC) 24/7.

Pipe:  OCP-U (Modbus TCP) → pymodbus → FastAPI → optional data/live.csv
Streamlit (cloud/laptop) can then use connection_type: fastapi or buffer_only.

Run:
  uvicorn gateway:app --host 0.0.0.0 --port 8088
  # or: python gateway.py
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = dict(raw.get("LIVE_MODE") or {})
    # env overrides (Docker / Pi)
    if os.getenv("OCP_U_IP"):
        cfg["ocp_u_ip"] = os.getenv("OCP_U_IP")
    if os.getenv("OCP_U_PORT"):
        cfg["ocp_u_port"] = int(os.getenv("OCP_U_PORT"))
    if os.getenv("LIVE_BUFFER"):
        cfg["buffer_path"] = os.getenv("LIVE_BUFFER")
    cfg.setdefault("ocp_u_ip", "192.168.1.50")
    cfg.setdefault("ocp_u_port", 502)
    cfg.setdefault("unit_id", 1)
    cfg.setdefault("modbus_timeout_s", 2.5)
    cfg.setdefault("buffer_path", "data/live.csv")
    cfg.setdefault("max_buffer_rows", 50000)
    cfg.setdefault(
        "registers",
        {
            "temperature": {"address": 0, "scale": 0.1},
            "vibration": {"address": 1, "scale": 0.001},
            "pressure": {"address": 2, "scale": 0.1},
            "smps_voltage": {"address": 3, "scale": 0.1},
            "smps_current": {"address": 4, "scale": 0.01},
        },
    )
    return cfg


def _read_modbus_row(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError as exc:
        raise RuntimeError("pymodbus not installed") from exc

    host = str(cfg["ocp_u_ip"])
    port = int(cfg["ocp_u_port"])
    unit = int(cfg.get("unit_id") or 1)
    timeout = float(cfg.get("modbus_timeout_s") or 2.5)
    regs_map: dict[str, Any] = cfg.get("registers") or {}
    if not regs_map:
        raise RuntimeError("No registers configured in config.yaml")

    addresses = [int(meta["address"]) for meta in regs_map.values()]
    start = min(addresses)
    count = max(addresses) - start + 1

    client = ModbusTcpClient(host=host, port=port, timeout=timeout)
    try:
        if not client.connect():
            raise RuntimeError(f"Cannot connect Modbus TCP {host}:{port}")
        try:
            result = client.read_holding_registers(address=start, count=count, device_id=unit)
        except TypeError:
            try:
                result = client.read_holding_registers(address=start, count=count, slave=unit)
            except TypeError:
                result = client.read_holding_registers(start, count, unit)
        if result is None or (hasattr(result, "isError") and result.isError()):
            raise RuntimeError(f"Modbus read error: {result}")
        raw = list(result.registers)
        if len(raw) < count:
            raise RuntimeError(f"Expected {count} registers, got {len(raw)}")

        row: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source": "gateway_modbus",
            "ocp_u": f"{host}:{port}",
        }
        for name, meta in regs_map.items():
            addr = int(meta["address"])
            scale = float(meta.get("scale", 1.0))
            idx = addr - start
            val = float(raw[idx]) * scale
            if name == "failure":
                val = 1.0 if val > 0 else 0.0
            row[name] = round(val, 6) if name != "failure" else val
        return row
    finally:
        try:
            client.close()
        except Exception:
            pass


def _append_buffer(row: dict[str, Any], cfg: dict[str, Any]) -> None:
    import pandas as pd

    path = ROOT / str(cfg.get("buffer_path") or "data/live.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
            out = pd.concat([old, frame], ignore_index=True)
        except Exception:
            out = frame
    else:
        out = frame
    max_rows = int(cfg.get("max_buffer_rows") or 50000)
    if len(out) > max_rows:
        out = out.tail(max_rows).reset_index(drop=True)
    out.to_csv(path, index=False)


app = FastAPI(title="Forge LIVE Gateway", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    cfg = load_config()
    return {
        "ok": True,
        "service": "forge-live-gateway",
        "ocp_u": f"{cfg.get('ocp_u_ip')}:{cfg.get('ocp_u_port')}",
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/live")
def get_live(persist: bool = True) -> dict[str, Any]:
    """Read OCP-U via pymodbus; optionally append to data/live.csv."""
    cfg = load_config()
    try:
        row = _read_modbus_row(cfg)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if persist:
        try:
            _append_buffer(row, cfg)
        except Exception as exc:
            row["buffer_warning"] = str(exc)
    return row


@app.get("/config")
def get_config() -> dict[str, Any]:
    cfg = load_config()
    # don't expose secrets — none here
    return {"LIVE_MODE": cfg}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway:app", host="0.0.0.0", port=int(os.getenv("GATEWAY_PORT", "8088")), reload=False)
