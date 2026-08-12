"""Dual-mode support: Manual upload vs Live connector."""
from __future__ import annotations

from enum import Enum


class DataMode(str, Enum):
    MANUAL = "manual"
    LIVE = "live"


def parse_mode(val: str) -> DataMode:
    return DataMode(val.lower().strip())
