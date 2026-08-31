"""Optional DWDM labs on the working dataframe (star, Apriori, K-means, MICE-like)."""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd


def numeric_columns(df: pd.DataFrame) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    return [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 80) -> list[str]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out: list[str] = []
    for c in df.columns:
        nuniq = df[c].nunique(dropna=True)
        if nuniq <= 1 or nuniq > max_unique:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and nuniq > 20:
            continue
        out.append(str(c))
    return out


def build_star_schema(
    df: pd.DataFrame,
    *,
    date_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    fact_cols: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Simple fact + dimension tables at a pandas groupby grain. Not a cube server."""
    empty = {
        "ok": False,
        "error": "Pick an entity dimension and at least one numeric fact.",
        "fact": pd.DataFrame(),
        "dims": {},
        "caption": "",
        "grain": [],
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        empty["error"] = "No working dataframe."
        return empty
    facts = [c for c in (fact_cols or []) if c in df.columns]
    if not entity_col or entity_col not in df.columns or not facts:
        return empty

    work = df.copy()
    grain: list[str] = []
    dims: dict[str, pd.DataFrame] = {}

    if date_col and date_col in work.columns:
        parsed = pd.to_datetime(work[date_col], errors="coerce")
        work["_date_key"] = parsed.dt.normalize()
        date_dim = (
            work.dropna(subset=["_date_key"])
            .drop_duplicates("_date_key")[["_date_key"]]
            .rename(columns={"_date_key": "date_key"})
        )
        dt = pd.to_datetime(date_dim["date_key"])
        date_dim["year"] = dt.dt.year
        date_dim["month"] = dt.dt.month
        date_dim["day"] = dt.dt.day
        date_dim["weekday"] = dt.dt.day_name()
        dims["date_dim"] = date_dim.reset_index(drop=True)
        grain.append("_date_key")

    work["_entity_key"] = work[entity_col].astype(str)
    entity_dim = (
        work.drop_duplicates("_entity_key")[["_entity_key"]]
        .rename(columns={"_entity_key": "entity_key"})
        .reset_index(drop=True)
    )
    entity_dim["entity_role"] = str(entity_col)
    dims["entity_dim"] = entity_dim
    grain.append("_entity_key")

    grouped = work.groupby(grain, dropna=False)
    fact = grouped[facts].sum(min_count=1)
    fact["fact_rows"] = grouped.size()
    fact = fact.reset_index()
    rename = {"_entity_key": "entity_key"}
    if "_date_key" in fact.columns:
        rename["_date_key"] = "date_key"
    fact = fact.rename(columns=rename)

    dim_names = " + ".join(dims.keys())
    grain_parts = (["date"] if "date_dim" in dims else []) + ["entity"]
    grain_label = " × ".join(grain_parts)
    caption = (
        f"Star (lab): fact[{', '.join(facts)}] at grain {grain_label} "
        f"→ {dim_names}. Not an OLAP cube server."
    )
    return {
        "ok": True,
        "fact": fact,
        "dims": dims,
        "caption": caption,
        "grain": list(fact.columns[: len(grain)]),
        "error": None,
    }


def apriori_need_txn_hint() -> str:
    return (
        "Need a transaction id + item column (market-basket). "
        "Student marks / one-row-per-person tables are not baskets — "
        "either map txn+item, or use row-as-basket bins (lab only)."
    )


def baskets_from_txn(df: pd.DataFrame, txn_col: str, item_col: str) -> list[frozenset[str]]:
    if df is None or txn_col not in df.columns or item_col not in df.columns:
        return []
    baskets: list[frozenset[str]] = []
    sub = df[[txn_col, item_col]].dropna()
    if sub.empty:
        return []
    for _, grp in sub.groupby(txn_col, dropna=True):
        items = frozenset(str(x) for x in grp[item_col].tolist() if str(x).strip())
        if items:
            baskets.append(items)
    return baskets


def baskets_row_bins(df: pd.DataFrame, cols: list[str]) -> list[frozenset[str]]:
    """Encode high/low vs median as items per row. Lab proxy, not market-basket."""
    if df is None or df.empty or not cols:
        return []
    work = df[cols].apply(pd.to_numeric, errors="coerce")
    med = work.median(numeric_only=True)
    baskets: list[frozenset[str]] = []
    for _, row in work.iterrows():
        items: set[str] = set()
        for c in cols:
            v = row[c]
            if pd.isna(v) or c not in med.index or pd.isna(med[c]):
                continue
            tag = "HIGH" if float(v) >= float(med[c]) else "LOW"
            items.add(f"{c}={tag}")
        if items:
            baskets.append(frozenset(items))
    return baskets


def mine_apriori(
    baskets: list[frozenset[str]],
    *,
    min_support: float = 0.15,
    min_confidence: float = 0.5,
    max_k: int = 3,
) -> dict[str, Any]:
    """Frequent itemsets of size 2–3 + support / confidence / lift. Small-n lab."""
    n = len(baskets)
    if n < 8:
        return {
            "ok": False,
            "error": "too few baskets",
            "hint": apriori_need_txn_hint(),
            "rules": pd.DataFrame(),
            "n_baskets": n,
        }
    max_k = int(max(2, min(3, max_k)))
    counts: Counter[frozenset[str]] = Counter()
    for basket in baskets:
        items = sorted(basket)
        if not items:
            continue
        upper = min(max_k, len(items))
        for k in range(1, upper + 1):
            for combo in itertools.combinations(items, k):
                counts[frozenset(combo)] += 1

    min_sup = float(min_support)
    freq = {iset: cnt / n for iset, cnt in counts.items() if (cnt / n) >= min_sup}
    rules: list[dict[str, Any]] = []
    for iset, sup in freq.items():
        if len(iset) < 2:
            continue
        members = list(iset)
        for r in range(1, len(members)):
            for ant in itertools.combinations(members, r):
                ant_f = frozenset(ant)
                cons_f = iset - ant_f
                if not cons_f or ant_f not in freq:
                    continue
                conf = sup / max(freq[ant_f], 1e-12)
                if conf < float(min_confidence):
                    continue
                cons_sup = freq.get(cons_f)
                if cons_sup is None:
                    cons_sup = sum(1 for b in baskets if cons_f <= b) / n
                lift = conf / max(cons_sup, 1e-12)
                rules.append(
                    {
                        "antecedent": ", ".join(sorted(ant_f)),
                        "consequent": ", ".join(sorted(cons_f)),
                        "support": round(sup, 3),
                        "confidence": round(conf, 3),
                        "lift": round(float(lift), 3),
                    }
                )
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"], -r["support"]))
    table = pd.DataFrame(rules[:40])
    return {
        "ok": True,
        "error": None,
        "hint": None,
        "rules": table,
        "n_baskets": n,
        "n_rules": len(rules),
    }


def assign_kmeans(
    df: pd.DataFrame,
    cols: list[str],
    *,
    k: int = 3,
    silhouette: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "No data.", "frame": pd.DataFrame()}
    use = [c for c in cols if c in df.columns]
    if len(use) < 1:
        return {"ok": False, "error": "Pick at least one numeric column.", "frame": df.copy()}
    k = int(max(2, min(12, k)))
    X = df[use].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    n_ok = int(mask.sum())
    if n_ok < k:
        return {
            "ok": False,
            "error": f"Need at least k={k} complete rows; have {n_ok}.",
            "frame": df.copy(),
        }
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X.loc[mask].to_numpy())
    out = df.copy()
    out["cluster_id"] = np.nan
    out.loc[mask, "cluster_id"] = labels.astype(int)
    sil: Optional[float] = None
    if silhouette and n_ok > k and len(set(labels)) > 1:
        from sklearn.metrics import silhouette_score

        try:
            sil = round(float(silhouette_score(X.loc[mask].to_numpy(), labels)), 3)
        except Exception:
            sil = None
    return {
        "ok": True,
        "frame": out,
        "k": k,
        "cols": use,
        "silhouette": sil,
        "n_assigned": n_ok,
        "error": None,
    }


def mice_impute(
    df: pd.DataFrame,
    cols: list[str],
    *,
    max_iter: int = 8,
    preview_rows: int = 12,
) -> dict[str, Any]:
    """sklearn IterativeImputer (MICE-like) on numeric columns only. Opt-in."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "error": "No data.", "frame": pd.DataFrame()}
    use = [c for c in cols if c in df.columns]
    if not use:
        return {"ok": False, "error": "Pick numeric columns.", "frame": df.copy()}
    work = df[use].apply(pd.to_numeric, errors="coerce")
    n_miss = int(work.isna().sum().sum())
    if n_miss == 0:
        return {
            "ok": True,
            "frame": df.copy(),
            "preview": work.head(preview_rows),
            "n_imputed": 0,
            "changed": False,
            "warning": None,
            "error": None,
        }
    n_cells = int(work.size)
    warning = None
    if n_cells > 200_000 or len(df) > 20_000:
        warning = "Large frame — IterativeImputer can be slow. This is a lab, not default Clean."
    try:
        from sklearn.impute import IterativeImputer
    except ImportError:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

    imputer = IterativeImputer(max_iter=int(max(2, min(20, max_iter))), random_state=42)
    filled = imputer.fit_transform(work)
    filled_df = pd.DataFrame(filled, columns=use, index=work.index)
    out = df.copy()
    out[use] = filled_df
    head_n = min(int(preview_rows), len(work))
    preview = pd.concat(
        [
            work.head(head_n).add_suffix("_before"),
            filled_df.head(head_n).add_suffix("_after"),
        ],
        axis=1,
    )
    return {
        "ok": True,
        "frame": out,
        "preview": preview,
        "n_imputed": n_miss,
        "changed": True,
        "warning": warning,
        "error": None,
    }
