"""Load tabular files (CSV / Excel / ZIP / JSON / Parquet) into DataFrames."""
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import pandas as pd


PathLike = Union[str, Path]

# Spreadsheet-style formats managers typically export from Excel / BI tools.
TABULAR_SUFFIXES = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".json",
    ".parquet",
}

# Inside ZIPs, ignore ambiguous .txt (often readmes) — require clear table extensions.
ZIP_TABULAR_SUFFIXES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".json",
    ".parquet",
}

ZIP_EMPTY_MSG = (
    "This ZIP has no tabular files. "
    "Include at least one CSV, TSV, Excel (.xlsx/.xls), JSON, or Parquet file."
)


class ZipNoTabularError(ValueError):
    """Raised when a ZIP archive contains no supported tabular members."""


def _suffix(path: PathLike) -> str:
    return Path(path).suffix.lower()


def is_tabular_name(name: str, *, for_zip: bool = False) -> bool:
    """True if filename looks like a supported table file (ignores macOS junk)."""
    p = Path(name)
    if any(part.startswith("__MACOSX") for part in p.parts):
        return False
    if p.name.startswith("."):
        return False
    allowed = ZIP_TABULAR_SUFFIXES if for_zip else TABULAR_SUFFIXES
    return p.suffix.lower() in allowed


def list_zip_tabular_members(data: bytes) -> list[str]:
    """Return sorted member paths inside a ZIP that look tabular."""
    members: list[str] = []
    with zipfile.ZipFile(BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if is_tabular_name(name, for_zip=True):
                members.append(name)
    return sorted(members)


def _read_json_table(bio: BytesIO, **kwargs) -> pd.DataFrame:
    """Load JSON array-of-objects or line-delimited JSON into a flat table."""
    raw = bio.read()
    bio.seek(0)
    try:
        return pd.read_json(bio, **kwargs)
    except ValueError:
        bio.seek(0)
        try:
            return pd.read_json(bio, lines=True, **kwargs)
        except ValueError:
            pass
    # Last resort: dict-of-lists / records via stdlib
    payload = json.loads(raw.decode("utf-8-sig"))
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        if all(isinstance(v, list) for v in payload.values()):
            return pd.DataFrame(payload)
        return pd.json_normalize(payload)
    raise ValueError(
        "JSON did not look like a table (expected a list of rows or column arrays)."
    )


def _load_from_buffer(bio: BytesIO, suffix: str, **kwargs) -> pd.DataFrame:
    suffix = suffix.lower()
    if suffix in {".csv", ".txt"}:
        sep = kwargs.pop("sep", ",")
        return pd.read_csv(bio, sep=sep, **kwargs)
    if suffix == ".tsv":
        kwargs.pop("sep", None)
        return pd.read_csv(bio, sep="\t", **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(bio, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(bio, **kwargs)
    if suffix == ".json":
        return _read_json_table(bio, **kwargs)
    # Fallback: try CSV
    try:
        bio.seek(0)
        return pd.read_csv(bio, **kwargs)
    except Exception as exc:
        raise ValueError(f"Unsupported file type: {suffix}") from exc


def _load_zip_bytes(
    data: bytes,
    zip_member: Optional[str] = None,
    **kwargs,
) -> tuple[pd.DataFrame, str]:
    """
    Unzip and load the first (or chosen) tabular member.

    Returns (dataframe, member_name_used).
    """
    members = list_zip_tabular_members(data)
    if not members:
        raise ZipNoTabularError(ZIP_EMPTY_MSG)

    if zip_member:
        if zip_member not in members:
            raise ValueError(
                f"ZIP member '{zip_member}' is not a supported tabular file. "
                f"Available: {', '.join(members)}"
            )
        chosen = zip_member
    else:
        chosen = members[0]

    with zipfile.ZipFile(BytesIO(data)) as zf:
        raw = zf.read(chosen)

    bio = BytesIO(raw)
    df = _load_from_buffer(bio, Path(chosen).suffix.lower(), **kwargs)
    return df, chosen


def load_file(
    path: PathLike,
    zip_member: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV, Excel, JSON, Parquet, or ZIP (containing one of those) file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".zip":
        df, _ = _load_zip_bytes(path.read_bytes(), zip_member=zip_member, **kwargs)
        return df

    with open(path, "rb") as f:
        bio = BytesIO(f.read())
    return _load_from_buffer(bio, suffix, **kwargs)


def load_bytes(
    data: bytes,
    filename: str,
    zip_member: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load uploaded bytes by filename extension (ZIP unwraps to a tabular member)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        df, _ = _load_zip_bytes(data, zip_member=zip_member, **kwargs)
        return df
    return _load_from_buffer(BytesIO(data), suffix, **kwargs)


def schema_summary(df: pd.DataFrame) -> dict:
    """Lightweight schema for persistence / AI context."""
    cols = {}
    for c in df.columns:
        s = df[c]
        cols[str(c)] = {
            "dtype": str(s.dtype),
            "nulls": int(s.isna().sum()),
            "nunique": int(s.nunique(dropna=True)),
            "sample": [str(x) for x in s.dropna().head(3).tolist()],
        }
    return {"columns": cols, "n_rows": int(len(df)), "n_cols": int(df.shape[1])}
