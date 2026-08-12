from __future__ import annotations

# Recovered from transcript messages 575, 588, 644, 647, 649, 657, and 662.
# This is a splice-ready app.py middle block; it expects the original app imports/globals
# such as pd, np, json, re, Optional, Any, IsolationForest, DBSCAN, StandardScaler,
# KMeans, _gemini_answer, and get_gemini_api_key to exist in the host module.

# DOMAIN CATALOG (8–9 industrial / business fields)
# =============================================================================
DOMAIN_CATALOG: dict[str, dict[str, Any]] = {
    "predictive_maintenance": {
        "label": "Predictive Maintenance / OPC-UA Sensors",
        "keywords": [
            "temperature", "temp", "vibration", "pressure", "rul", "failure", "machine",
            "sensor", "torque", "rpm", "current", "voltage", "opc", "modbus", "asset",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "healthcare": {
        "label": "Healthcare / Hospital",
        "keywords": [
            "patient", "age", "bmi", "bp", "blood", "glucose", "heart", "diagnosis",
            "admit", "ward", "doctor", "hospital", "readmission", "weight", "height",
            "cholesterol", "pulse", "spo2",
        ],
        "dtypes_hint": "mixed_clinical",
    },
    "sales_forecasting": {
        "label": "Sales / Retail / Revenue",
        "keywords": [
            "revenue", "sales", "units", "order", "price", "sku", "customer", "region",
            "channel", "store", "campaign", "discount", "gmv", "asp",
        ],
        "dtypes_hint": "commerce",
    },
    "warehouse_logistics": {
        "label": "Warehouse / Supply Chain",
        "keywords": [
            "warehouse", "sku", "inventory", "stock", "shipment", "delivery", "carrier",
            "aisle", "bin", "lead_time", "defect", "pick", "pack",
        ],
        "dtypes_hint": "ops",
    },
    "energy_utilities": {
        "label": "Energy / Utilities",
        "keywords": [
            "kwh", "mw", "power", "voltage", "current", "grid", "load", "consumption",
            "solar", "wind", "frequency", "pf",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "finance_risk": {
        "label": "Finance / Credit Risk",
        "keywords": [
            "loan", "credit", "score", "default", "interest", "balance", "emi", "income",
            "fraud", "transaction", "amount", "apr",
        ],
        "dtypes_hint": "tabular_finance",
    },
    "telecom_churn": {
        "label": "Telecom / Churn",
        "keywords": [
            "churn", "tenure", "plan", "minutes", "data_usage", "arpu", "subscriber",
            "complaint", "call_drop", "sim",
        ],
        "dtypes_hint": "crm",
    },
    "agriculture_iot": {
        "label": "Agriculture / Agri-IoT",
        "keywords": [
            "soil", "moisture", "humidity", "rainfall", "crop", "yield", "ph", "npk",
            "irrigation", "farm",
        ],
        "dtypes_hint": "numeric_sensors",
    },
    "generic": {
        "label": "Generic Analytics",
        "keywords": [],
        "dtypes_hint": "generic",
    },
}


def suggest_clean_engine(n_rows: int, n_cols: int) -> tuple[str, str]:
    """Suggest engine by size — never force PySpark on tiny files."""
    cells = n_rows * max(1, n_cols)
    if n_rows < 50_000 and cells < 2_000_000:
        return "pandas", f"Suggested: **pandas** ({n_rows:,} rows — small/medium, fastest for interactive UI)."
    if n_rows < 500_000:
        return "polars", f"Suggested: **polars** ({n_rows:,} rows — faster columnar engine for mid-size data)."
    return "pyspark", f"Suggested: **pyspark** ({n_rows:,} rows — big-data scale; slower startup on Mac)."


def list_available_engines() -> list[str]:
    engines = ["pandas"]
    try:
        import polars  # noqa: F401
        engines.append("polars")
    except ImportError:
        pass
    try:
        import pyspark  # noqa: F401
        engines.append("pyspark")
    except ImportError:
        pass
    return engines

def _basic_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Legacy alias — full suite lives in build_quality_report."""
    return build_quality_report(df)["checks"]


def _clean_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    log.append("ETL: strip column names / schema normalize")
    out = out.dropna(how="all")
    out = out.loc[:, ~out.columns.duplicated()]
    before = len(out)
    out = out.drop_duplicates()
    if len(out) != before:
        log.append(f"ETL dedupe {before}->{len(out)}")
    out = out.replace(["", "NA", "N/A", "null", "NULL", "None", "-", "--"], np.nan)
    log.append("ETL null-sentinel fusion")
    for c in list(out.columns):
        if out[c].dtype == object:
            converted = pd.to_numeric(out[c], errors="coerce")
            if out[c].notna().sum() and converted.notna().sum() / max(1, out[c].notna().sum()) >= 0.8:
                out[c] = converted
                log.append(f"schema cast numeric {c}")
    date_hints = ("date", "time", "timestamp", "datetime", "day")
    for c in list(out.columns):
        if any(h in str(c).lower() for h in date_hints) and out[c].dtype == object:
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().sum() > 0:
                out[c] = parsed
                log.append(f"schema cast datetime {c}")
    for c in out.select_dtypes(include=[np.number]).columns:
        if out[c].isna().any():
            s = out[c]
            if s.notna().sum() >= 5:
                idx = np.arange(len(s))
                mask = s.notna().to_numpy()
                coef = np.polyfit(idx[mask], s.to_numpy()[mask], 1)
                pred = np.polyval(coef, idx)
                filled = s.copy()
                filled[s.isna()] = pred[s.isna()]
                out[c] = filled
                log.append(f"DWDM regression imputation {c}")
            else:
                out[c] = s.fillna(s.median())
                log.append(f"median imputation {c}")
    for c in out.select_dtypes(include=["object", "string", "category"]).columns:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill)
            log.append(f"mode imputation {c}")
    for c in list(out.select_dtypes(include=[np.number]).columns)[:6]:
        try:
            out[f"{c}_bin"] = pd.qcut(out[c], q=min(5, max(2, out[c].nunique())), duplicates="drop").astype(str)
            log.append(f"DWDM binning {c}")
        except Exception:
            pass
    for c in list(out.select_dtypes(include=[np.number]).columns):
        cl = str(c).lower()
        if any(h in cl for h in ("temp", "vib", "pressure", "current", "voltage", "speed")) and not cl.endswith("_smooth") and not cl.endswith("_bin"):
            out[f"{c}_smooth"] = out[c].rolling(window=min(5, max(2, len(out) // 10)), min_periods=1).mean()
            log.append(f"DWDM smoothing {c}")
    return out.reset_index(drop=True), log


def _clean_polars(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import polars as pl

    log = ["engine=polars"]
    pldf = pl.from_pandas(df)
    before = pldf.height
    null_cols = [c for c in pldf.columns if pldf[c].null_count() == pldf.height]
    if null_cols:
        pldf = pldf.drop(null_cols)
        log.append(f"drop null cols {null_cols}")
    pldf = pldf.unique()
    log.append(f"unique {before}->{pldf.height}")
    for c in pldf.columns:
        dtype = pldf[c].dtype
        if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            if pldf[c].null_count() > 0:
                med = pldf[c].median()
                pldf = pldf.with_columns(pl.col(c).fill_null(med))
                log.append(f"polars fill_median {c}")
        elif str(dtype) in ("Utf8", "String") or dtype in (getattr(pl, "Utf8", None), getattr(pl, "String", None)):
            if pldf[c].null_count() > 0:
                pldf = pldf.with_columns(pl.col(c).fill_null("Unknown"))
    pdf = pldf.to_pandas()
    pdf2, log2 = _clean_pandas(pdf)
    return pdf2, log + log2


def _clean_pyspark(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("forge_v2_clean")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    log = ["engine=pyspark"]
    try:
        sdf = spark.createDataFrame(df.astype(str))
        before = sdf.count()
        sdf = sdf.dropDuplicates()
        after = sdf.count()
        log.append(f"spark dropDuplicates {before}->{after}")
        pdf = sdf.toPandas()
        cleaned, plog = _clean_pandas(pdf)
        log.extend(plog)
        return cleaned, log
    finally:
        spark.stop()


def _engine_clean(df: pd.DataFrame, engine: str) -> tuple[pd.DataFrame, list[str]]:
    engine = (engine or "pandas").lower()
    if engine == "polars":
        try:
            return _clean_polars(df)
        except Exception as exc:
            out, log = _clean_pandas(df)
            return out, [f"polars failed ({exc}) → pandas"] + log
    if engine == "pyspark":
        try:
            return _clean_pyspark(df)
        except Exception as exc:
            out, log = _clean_pandas(df)
            return out, [f"pyspark failed ({exc}) → pandas"] + log
    return _clean_pandas(df)


def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    for n in names:
        for k, real in lower.items():
            if n.lower() in k:
                return real
    return None


def _zscore_iqr_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number])
    z_hits, iqr_hits = 0, 0
    details = []
    for c in num.columns:
        s = num[c].dropna()
        if len(s) < 5:
            continue
        z = (s - s.mean()) / (s.std() + 1e-9)
        zc = int((z.abs() > 3).sum())
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        ic = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        z_hits += zc
        iqr_hits += ic
        if zc or ic:
            details.append(f"{c}:z={zc},iqr={ic}")
    return {"z_hits": z_hits, "iqr_hits": iqr_hits, "details": details[:8]}


def _isolation_forest_flags(df: pd.DataFrame) -> dict[str, Any]:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 15:
        return {"ok": False, "reason": "need >=2 numeric cols & 15 rows"}
    iso = IsolationForest(contamination=0.08, random_state=42)
    labels = iso.fit_predict(num.values)
    n = int((labels == -1).sum())
    return {"ok": True, "anomalies": n, "rate_pct": round(100.0 * n / len(num), 2)}


def _dbscan_noise(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False, "reason": "need more numeric rows"}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    noise = int((labels == -1).sum())
    return {"ok": True, "noise_points": noise, "clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}


def _kmeans_clean_proxy(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 20:
        return {"ok": False}
    X = StandardScaler().fit_transform(num.values[: min(2000, len(num))])
    km = KMeans(n_clusters=min(3, max(2, len(X) // 5)), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    dists = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)
    far = int((dists > dists.mean() + 2 * dists.std()).sum())
    return {"ok": True, "far_from_cluster": far}


def _rolling_impossible_jumps(df: pd.DataFrame) -> dict[str, Any]:
    flags = []
    total = 0
    for c in df.select_dtypes(include=[np.number]).columns:
        cl = str(c).lower()
        if not any(h in cl for h in ("temp", "vib", "pressure", "speed", "current")):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        delta = s.diff().abs()
        thr = 50 if "temp" in cl else (5 if "vib" in cl else (30 if "pressure" in cl else float(s.std() or 1) * 4))
        n = int((delta > thr).fillna(False).sum())
        if n:
            flags.append(f"{c}:{n} jumps>{thr}")
            total += n
    return {"ok": True, "impossible_jumps": flags[:10], "count": total}


def _lag_correlation_break(df: pd.DataFrame) -> dict[str, Any]:
    t = _col(df, "temperature", "temp")
    p = _col(df, "pressure")
    v = _col(df, "vibration", "vib")
    pairs = []
    for a, b in [(t, p), (t, v), (p, v)]:
        if a and b:
            corr = pd.to_numeric(df[a], errors="coerce").corr(pd.to_numeric(df[b], errors="coerce"))
            pairs.append({"pair": f"{a}|{b}", "corr": None if pd.isna(corr) else round(float(corr), 3)})
    broken = [x for x in pairs if x["corr"] is not None and abs(x["corr"]) < 0.05]
    return {"pairs": pairs, "dead_correlations": broken}


def _domain_opc_rules(df: pd.DataFrame) -> list[str]:
    flags = []
    t = _col(df, "temperature", "temp")
    v = _col(df, "vibration", "vib")
    p = _col(df, "pressure")
    r = _col(df, "rul")
    fcol = _col(df, "failure", "fault")
    if t and v:
        tt = pd.to_numeric(df[t], errors="coerce")
        vv = pd.to_numeric(df[v], errors="coerce")
        stuck = int(((tt > 150) & (vv < 0.1)).fillna(False).sum())
        if stuck:
            flags.append(f"Sensor stuck pattern: {stuck} rows (temp>150 & vib<0.1)")
    if p:
        speed = _col(df, "speed", "flow", "load")
        if speed:
            pp = pd.to_numeric(df[p], errors="coerce")
            ss = pd.to_numeric(df[speed], errors="coerce")
            n = int(((pp.diff() < -10) & (ss.diff().abs() < 0.5)).fillna(False).sum())
            if n:
                flags.append(f"Leak/sensor fault suspect: {n} rows")
    if r:
        rr = pd.to_numeric(df[r], errors="coerce")
        d = rr.diff().dropna()
        if len(d) and d.gt(0).mean() > 0.6:
            flags.append("RUL calculation broken: RUL increases over time")
    if fcol and v:
        ff = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        vv = pd.to_numeric(df[v], errors="coerce")
        missed = int(((ff == 0) & (vv > vv.mean() + 3 * (vv.std() or 1))).fillna(False).sum())
        if missed:
            flags.append(f"Possible missed failures: {missed} rows")
    return flags


def _run_great_expectations(df: pd.DataFrame) -> dict[str, Any]:
    results = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean())
        results.append({
            "expectation": "expect_column_values_to_not_be_null",
            "column": col,
            "success": null_pct < 0.2,
            "detail": f"null_pct={null_pct:.3f}",
        })
    t = _col(df, "temperature", "temp")
    if t:
        s = pd.to_numeric(df[t], errors="coerce")
        ok = bool(((s.dropna() >= 0) & (s.dropna() <= 200)).all()) if s.notna().any() else False
        results.append({"expectation": "expect_column_values_to_be_between", "column": t, "success": ok, "detail": "temp in [0,200]"})
    r = _col(df, "rul")
    if r:
        s = pd.to_numeric(df[r], errors="coerce").dropna()
        success = bool(s.diff().dropna().le(0).mean() >= 0.5) if len(s) > 3 else True
        results.append({"expectation": "expect_column_pair_values_A_to_be_greater_than_B", "column": r, "success": success, "detail": "RUL mostly non-increasing"})
    ts = _col(df, "timestamp", "time", "datetime", "date")
    mid = _col(df, "machine_id", "machine", "asset_id")
    if ts and mid:
        dup = int(df.duplicated([ts, mid]).sum())
        results.append({"expectation": "expect_compound_columns_to_be_unique", "column": f"{ts}+{mid}", "success": dup == 0, "detail": f"dup_keys={dup}"})
    results.append({"expectation": "expect_table_row_count_to_be_between", "column": "*", "success": 1 <= len(df) <= 5_000_000, "detail": f"rows={len(df)}"})
    ge_available = False
    try:
        import great_expectations as gx  # noqa: F401
        ge_available = True
    except Exception:
        ge_available = False
    passed = sum(1 for r in results if r["success"])
    return {"engine": "great_expectations", "available": ge_available, "passed": passed, "total": len(results), "results": results[:40], "ok": True}


def _run_ydata(df: pd.DataFrame) -> dict[str, Any]:
    high_card = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun > max(50, int(0.5 * len(df))):
            high_card.append(c)
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(df.head(min(400, len(df))), minimal=True, progress_bar=False)
        desc = profile.get_description()
        return {"engine": "ydata-profiling", "ok": True, "variables": len(desc.get("variables", {})), "alerts": len(desc.get("alerts", [])), "high_cardinality": high_card[:8]}
    except Exception as exc:
        return {"engine": "ydata-profiling", "ok": False, "error": str(exc), "high_cardinality": high_card[:8]}


def _run_cleanlab(df: pd.DataFrame) -> dict[str, Any]:
    fcol = _col(df, "failure", "fault", "label", "churn", "default")
    num = df.select_dtypes(include=[np.number])
    out: dict[str, Any] = {"engine": "cleanlab"}
    try:
        from cleanlab import Datalab
        work = num.dropna()
        if work.shape[1] >= 2 and len(work) >= 15:
            lab = Datalab(data=work.reset_index(drop=True))
            lab.find_issues(features=work.values)
            issues = lab.get_issues()
            n_out = int(issues["is_outlier_issue"].sum()) if "is_outlier_issue" in issues.columns else 0
            out.update({"ok": True, "outlier_issues": n_out})
        else:
            out.update({"ok": True, "skipped": "numeric too small"})
    except Exception as exc:
        out.update({"ok": False, "error": str(exc)})
    dirty = []
    if fcol and _col(df, "vibration", "vib"):
        v = pd.to_numeric(df[_col(df, "vibration", "vib")], errors="coerce")
        f = pd.to_numeric(df[fcol], errors="coerce").fillna(0)
        if v.notna().any():
            false_pos = int(((f == 1) & (v < v.quantile(0.2))).sum())
            false_neg = int(((f == 0) & (v > v.quantile(0.95))).sum())
            if false_pos:
                dirty.append(f"{false_pos} rows failure=1 but low vibration")
            if false_neg:
                dirty.append(f"{false_neg} rows failure=0 but extreme vibration")
    out["dirty_label_flags"] = dirty
    return out


def _pca_drift(df: pd.DataFrame) -> dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or len(num) < 30:
        return {"ok": False}
    half = len(num) // 2
    X1 = StandardScaler().fit_transform(num.iloc[:half])
    X2 = StandardScaler().fit_transform(num.iloc[half:])
    ncomp = min(3, X1.shape[1])
    r1 = float(PCA(n_components=ncomp).fit(X1).explained_variance_ratio_.sum())
    r2 = float(PCA(n_components=min(3, X2.shape[1])).fit(X2).explained_variance_ratio_.sum())
    drift = abs(r1 - r2)
    return {"ok": True, "pca_var_early": round(r1, 3), "pca_var_late": round(r2, 3), "drift_score": round(drift, 3), "concept_drift": drift > 0.15}


def _association_rules_proxy(df: pd.DataFrame) -> dict[str, Any]:
    """
    Lightweight DWDM association-rule mining (Apriori-style) on binarized numeric highs
    + low-cardinality categoricals. Flags co-occurring anomaly baskets.
    """
    try:
        items: list[set[str]] = []
        cats = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique(dropna=True) <= 12][:4]
        nums = list(df.select_dtypes(include=[np.number]).columns)[:6]
        sample = df.tail(min(800, len(df)))
        for _, row in sample.iterrows():
            basket: set[str] = set()
            for c in cats:
                val = row[c]
                if pd.notna(val):
                    basket.add(f"{c}={val}")
            for c in nums:
                s = pd.to_numeric(sample[c], errors="coerce")
                thr = s.quantile(0.9)
                v = pd.to_numeric(row[c], errors="coerce")
                if pd.notna(v) and pd.notna(thr) and v >= thr:
                    basket.add(f"{c}=HIGH")
            if len(basket) >= 2:
                items.append(basket)
        if len(items) < 20:
            return {"ok": True, "skipped": "too few baskets", "suspicious_rules": []}
        from collections import Counter
        pair_counts: Counter[tuple[str, str]] = Counter()
        item_counts: Counter[str] = Counter()
        for basket in items:
            for a in basket:
                item_counts[a] += 1
            bl = sorted(basket)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    pair_counts[(bl[i], bl[j])] += 1
        n = len(items)
        rules = []
        for (a, b), cnt in pair_counts.most_common(30):
            support = cnt / n
            conf_ab = cnt / max(1, item_counts[a])
            conf_ba = cnt / max(1, item_counts[b])
            if support >= 0.05 and max(conf_ab, conf_ba) >= 0.55:
                rules.append({"rule": f"{a} => {b}", "support": round(support, 3), "confidence": round(max(conf_ab, conf_ba), 3)})
        suspicious = [r for r in rules if "HIGH" in r["rule"] and r["confidence"] >= 0.7][:8]
        return {"ok": True, "rules_found": len(rules), "top_rules": rules[:5], "suspicious_rules": suspicious}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "suspicious_rules": []}


def build_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    n, m = df.shape
    miss = float(df.isna().sum().sum() / max(1, df.size))
    checks.append({"check": "NULLS / Missing%", "status": "FAIL" if miss > 0.2 else ("WARN" if miss > 0.05 else "PASS"), "detail": f"{miss*100:.2f}% missing"})
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append({"check": "CONSTANT", "status": "FAIL" if const_cols else "PASS", "detail": str(const_cols[:6]) if const_cols else "none"})
    num = df.select_dtypes(include=[np.number])
    zero_ratio = float((num == 0).sum().sum() / max(1, num.size)) if num.size else 0
    checks.append({"check": "ZEROS", "status": "WARN" if zero_ratio > 0.3 else "PASS", "detail": f"{zero_ratio*100:.1f}% zeros"})
    dups = int(df.duplicated().sum())
    checks.append({"check": "DUPLICATES", "status": "WARN" if dups else "PASS", "detail": f"{dups} dup rows"})
    zi = _zscore_iqr_flags(df)
    checks.append({"check": "Z-SCORE (>3σ)", "status": "WARN" if zi["z_hits"] else "PASS", "detail": f"{zi['z_hits']} hits; {zi['details'][:3]}"})
    checks.append({"check": "IQR OUTLIER", "status": "WARN" if zi["iqr_hits"] else "PASS", "detail": f"{zi['iqr_hits']} hits"})
    iso = _isolation_forest_flags(df)
    checks.append({"check": "ISOLATION FOREST", "status": "WARN" if iso.get("anomalies", 0) else ("PASS" if iso.get("ok") else "INFO"), "detail": json.dumps({k: iso[k] for k in iso if k != "ok"})[:160]})
    db = _dbscan_noise(df)
    checks.append({"check": "DBSCAN NOISE", "status": "WARN" if db.get("noise_points", 0) else ("PASS" if db.get("ok") else "INFO"), "detail": json.dumps(db)[:160]})
    km = _kmeans_clean_proxy(df)
    checks.append({"check": "KMEANS DISTANCE", "status": "WARN" if km.get("far_from_cluster", 0) else ("PASS" if km.get("ok") else "INFO"), "detail": json.dumps(km)[:160]})
    jumps = _rolling_impossible_jumps(df)
    checks.append({"check": "ROLLING IMPOSSIBLE JUMP", "status": "FAIL" if jumps.get("count", 0) else "PASS", "detail": str(jumps.get("impossible_jumps") or "none")[:160]})
    lag = _lag_correlation_break(df)
    checks.append({"check": "LAG / SENSOR CORRELATION", "status": "WARN" if lag.get("dead_correlations") else "PASS", "detail": json.dumps(lag)[:160]})
    ge = _run_great_expectations(df)
    checks.append({"check": "GE EXPECTATIONS", "status": "PASS" if ge["passed"] == ge["total"] else "WARN", "detail": f"{ge['passed']}/{ge['total']} passed; available={ge['available']}"})
    yd = _run_ydata(df)
    checks.append({"check": "YDATA CARDINALITY", "status": "WARN" if yd.get("high_cardinality") else ("PASS" if yd.get("ok") else "INFO"), "detail": json.dumps(yd.get("high_cardinality") or yd)[:160]})
    cl = _run_cleanlab(df)
    checks.append({"check": "CLEANLAB / DIRTY LABELS", "status": "WARN" if cl.get("dirty_label_flags") else ("PASS" if cl.get("ok") else "INFO"), "detail": str(cl.get("dirty_label_flags") or cl)[:160]})
    pca = _pca_drift(df)
    checks.append({"check": "PCA / CONCEPT DRIFT", "status": "WARN" if pca.get("concept_drift") else ("PASS" if pca.get("ok") else "INFO"), "detail": json.dumps(pca)[:160]})
    domain_flags = _domain_opc_rules(df)
    checks.append({"check": "DOMAIN OPC / PHYSICS RULES", "status": "FAIL" if domain_flags else "PASS", "detail": "; ".join(domain_flags) if domain_flags else "no domain violations"})
    assoc = _association_rules_proxy(df)
    checks.append({"check": "ASSOCIATION RULE MINING", "status": "WARN" if assoc.get("suspicious_rules") else ("PASS" if assoc.get("ok") else "INFO"), "detail": json.dumps(assoc)[:160]})
    checks.append({"check": "SCHEMA / ROWCOUNT", "status": "PASS" if n > 0 else "FAIL", "detail": f"{n} rows × {m} cols"})
    checks.append({"check": "TIMESTAMP PRESENT", "status": "PASS" if _col(df, "timestamp", "date", "time", "datetime") else "WARN", "detail": str(_col(df, "timestamp", "date", "time", "datetime") or "missing")})
    return {"checks": checks, "ge": ge, "ydata": yd, "cleanlab": cl, "pca": pca, "domain_flags": domain_flags, "association": assoc}


def clean_data(df: pd.DataFrame, engine: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = engine or st.session_state.get("clean_engine") or "pandas"
    clean_df, engine_logs = _engine_clean(df, engine)
    report = build_quality_report(clean_df)
    table = pd.DataFrame(report["checks"])
    st.session_state.clean_df = clean_df
    st.session_state.clean_checks = table
    st.session_state.clean_report = {**report, "engine_logs": engine_logs, "engine": engine}
    return clean_df, table


def get_kpis(df: pd.DataFrame) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    miss = round(float(df.isna().sum().sum() / max(1, df.size) * 100), 2)
    domain = st.session_state.get("domain") or "generic"
    tcol = _col(df, "temperature", "temp")
    vcol = _col(df, "vibration", "vib")
    pcol = _col(df, "pressure")
    rcol = _col(df, "rul", "remaining_useful_life")
    fcol = _col(df, "failure", "fault", "alarm")

    def mean_of(col: Optional[str]) -> Any:
        if not col:
            return "—"
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() == 0:
            return "—"
        return round(float(s.mean()), 3)

    kpis = {
        "Rows": int(n_rows),
        "Cols": int(n_cols),
        "Missing%": miss,
        "Mean_temp": mean_of(tcol),
        "Mean_vib": mean_of(vcol),
        "Mean_pressure": mean_of(pcol),
        "Mean_RUL": mean_of(rcol),
        "Failure_Count": int(pd.to_numeric(df[fcol], errors="coerce").fillna(0).sum()) if fcol else 0,
        "Min_RUL": (
            round(float(pd.to_numeric(df[rcol], errors="coerce").min()), 2)
            if rcol and pd.to_numeric(df[rcol], errors="coerce").notna().any()
            else "—"
        ),
    }
    if domain == "healthcare":
        w, h = _col(df, "weight"), _col(df, "height")
        if w and h:
            ww = pd.to_numeric(df[w], errors="coerce")
            hh = pd.to_numeric(df[h], errors="coerce") / 100.0
            bmi = ww / (hh.replace(0, np.nan) ** 2)
            kpis["Mean_BMI"] = round(float(bmi.mean()), 2) if bmi.notna().any() else "—"
    if domain == "sales_forecasting":
        rev = _col(df, "revenue", "sales", "gmv", "amount")
        kpis["Total_Revenue"] = round(float(pd.to_numeric(df[rev], errors="coerce").sum()), 2) if rev else "—"
    return kpis


def apply_domain_feature_engineering(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    out = df.copy()
    if domain == "healthcare":
        w, h = _col(out, "weight"), _col(out, "height")
        if w and h:
            ww = pd.to_numeric(out[w], errors="coerce")
            hh = pd.to_numeric(out[h], errors="coerce")
            hh_m = np.where(hh > 3, hh / 100.0, hh)
            out["BMI"] = ww / np.square(np.where(hh_m == 0, np.nan, hh_m))
            bp = _col(out, "bp", "blood_pressure", "systolic")
            glu = _col(out, "glucose", "blood_sugar")
            age = _col(out, "age")
            score = pd.Series(0.0, index=out.index)
            if bp is not None:
                score += pd.to_numeric(out[bp], errors="coerce").fillna(120) / 120.0
            if glu is not None:
                score += pd.to_numeric(out[glu], errors="coerce").fillna(100) / 100.0
            if age is not None:
                score += pd.to_numeric(out[age], errors="coerce").fillna(40) / 100.0
            out["BRISK_SCORE"] = score
    elif domain == "predictive_maintenance":
        t = _col(out, "temperature", "temp")
        v = _col(out, "vibration", "vib")
        p = _col(out, "pressure")
        flow = _col(out, "flow", "flow_rate")
        if t:
            out["temp_gradient"] = pd.to_numeric(out[t], errors="coerce").diff()
        if v:
            out["vib_rolling_std"] = pd.to_numeric(out[v], errors="coerce").rolling(5, min_periods=1).std()
        if t and v:
            out["thermal_mech_index"] = pd.to_numeric(out[t], errors="coerce").fillna(0) / 100.0 + pd.to_numeric(out[v], errors="coerce").fillna(0)
        if p and flow:
            out["pressure_flow_ratio"] = pd.to_numeric(out[p], errors="coerce") / pd.to_numeric(out[flow], errors="coerce").replace(0, np.nan)
    elif domain == "sales_forecasting":
        rev = _col(out, "revenue", "sales", "amount")
        units = _col(out, "units", "qty", "quantity")
        if rev and units:
            out["ASP"] = pd.to_numeric(out[rev], errors="coerce") / pd.to_numeric(out[units], errors="coerce").replace(0, np.nan)
        if rev:
            out["revenue_ma7"] = pd.to_numeric(out[rev], errors="coerce").rolling(7, min_periods=1).mean()
            out["revenue_pct_change"] = pd.to_numeric(out[rev], errors="coerce").pct_change()
    elif domain == "finance_risk":
        bal = _col(out, "balance", "amount", "loan_amount")
        inc = _col(out, "income", "annual_income")
        if bal and inc:
            out["DTI_proxy"] = pd.to_numeric(out[bal], errors="coerce") / pd.to_numeric(out[inc], errors="coerce").replace(0, np.nan)
        score = _col(out, "credit_score", "score")
        if score:
            out["credit_score_z"] = (
                pd.to_numeric(out[score], errors="coerce") - pd.to_numeric(out[score], errors="coerce").mean()
            ) / (pd.to_numeric(out[score], errors="coerce").std() or 1)
    elif domain == "warehouse_logistics":
        stock = _col(out, "inventory", "stock", "on_hand")
        demand = _col(out, "demand", "orders", "shipments")
        lead = _col(out, "lead_time", "leadtime")
        if stock and demand:
            out["days_of_cover"] = pd.to_numeric(out[stock], errors="coerce") / pd.to_numeric(out[demand], errors="coerce").replace(0, np.nan)
        if lead:
            out["lead_time_ma"] = pd.to_numeric(out[lead], errors="coerce").rolling(5, min_periods=1).mean()
    elif domain == "energy_utilities":
        load = _col(out, "load", "consumption", "kwh", "mw", "power")
        volt = _col(out, "voltage")
        curr = _col(out, "current")
        if load:
            out["load_rolling_std"] = pd.to_numeric(out[load], errors="coerce").rolling(6, min_periods=1).std()
            out["load_gradient"] = pd.to_numeric(out[load], errors="coerce").diff()
        if volt and curr:
            out["apparent_power_proxy"] = pd.to_numeric(out[volt], errors="coerce") * pd.to_numeric(out[curr], errors="coerce")
    elif domain == "telecom_churn":
        tenure = _col(out, "tenure")
        arpu = _col(out, "arpu", "revenue")
        usage = _col(out, "data_usage", "minutes", "usage")
        if tenure and arpu:
            out["lifetime_value_proxy"] = pd.to_numeric(out[tenure], errors="coerce") * pd.to_numeric(out[arpu], errors="coerce")
        if usage:
            out["usage_z"] = (
                pd.to_numeric(out[usage], errors="coerce") - pd.to_numeric(out[usage], errors="coerce").mean()
            ) / (pd.to_numeric(out[usage], errors="coerce").std() or 1)
    elif domain == "agriculture_iot":
        moist = _col(out, "moisture", "soil_moisture")
        rain = _col(out, "rainfall", "rain")
        ph = _col(out, "ph")
        if moist:
            out["moisture_gradient"] = pd.to_numeric(out[moist], errors="coerce").diff()
        if moist and rain:
            out["irrigation_stress"] = pd.to_numeric(out[moist], errors="coerce") / (
                pd.to_numeric(out[rain], errors="coerce").fillna(0) + 1.0
            )
        if ph:
            out["ph_dev_neutral"] = (pd.to_numeric(out[ph], errors="coerce") - 7.0).abs()
    return out


def detect_field(df: pd.DataFrame, use_gemini: bool = True) -> dict[str, Any]:
    cols = [str(c).lower() for c in df.columns]
    col_join = " ".join(cols)
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}
    for dom, meta in DOMAIN_CATALOG.items():
        if dom == "generic":
            continue
        hit = []
        sc = 0.0
        for kw in meta["keywords"]:
            if kw in col_join:
                sc += 1.0
                hit.append(kw)
        num_ratio = df.select_dtypes(include=[np.number]).shape[1] / max(1, df.shape[1])
        if meta["dtypes_hint"] == "numeric_sensors" and num_ratio > 0.6:
            sc += 1.5
            hit.append("numeric_sensor_schema")
        if meta["dtypes_hint"] == "commerce" and any(k in col_join for k in ("revenue", "sales", "order")):
            sc += 1.0
        scores[dom] = sc
        reasons[dom] = hit[:12]
    if scores:
        heur = max(scores, key=scores.get)
        heur_conf = scores[heur] / max(1.0, max(scores.values()))
    else:
        heur, heur_conf = "generic", 0.2

    gemini_domain = None
    gemini_raw = ""
    if use_gemini and get_gemini_api_key():
        schema = [{"column": str(c), "dtype": str(df[c].dtype), "sample": [str(x) for x in df[c].dropna().head(3).tolist()]} for c in df.columns[:40]]
        prompt = (
            "Classify this industrial/business dataset into ONE domain key from: "
            + ", ".join(DOMAIN_CATALOG.keys())
            + ".\nReturn JSON only: {\"domain\": \"...\", \"confidence\": 0-1, \"why\": \"...\"}\n"
            f"Columns/dtypes/samples: {json.dumps(schema)[:4000]}"
        )
        gemini_raw = _gemini_answer(prompt)
        try:
            start = gemini_raw.find("{")
            end = gemini_raw.rfind("}") + 1
            if start >= 0 and end > start:
                payload = json.loads(gemini_raw[start:end])
                gd = str(payload.get("domain", "")).strip()
                if gd in DOMAIN_CATALOG:
                    gemini_domain = gd
                    gconf = float(payload.get("confidence", 0.9))
                    final = gemini_domain
                    conf = min(0.98, 0.55 * heur_conf + 0.45 * gconf + (0.2 if gemini_domain == heur else 0))
                    return {
                        "domain": final,
                        "label": DOMAIN_CATALOG[final]["label"],
                        "confidence": round(conf, 3),
                        "heuristic": heur,
                        "heuristic_scores": scores,
                        "reasons": reasons.get(final) or reasons.get(heur) or [],
                        "gemini_domain": gemini_domain,
                        "gemini_why": payload.get("why", ""),
                        "gemini_raw": gemini_raw[:500],
                    }
        except Exception:
            pass

    final = heur if scores.get(heur, 0) > 0 else "generic"
    return {
        "domain": final,
        "label": DOMAIN_CATALOG[final]["label"],
        "confidence": round(float(min(0.92, max(0.25, heur_conf))), 3),
        "heuristic": heur,
        "heuristic_scores": scores,
        "reasons": reasons.get(final, []),
        "gemini_domain": gemini_domain,
        "gemini_why": "",
        "gemini_raw": gemini_raw[:500],
    }


def _numeric_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, Optional[str]]:
    work = df.copy()
    fail_col = _col(work, "failure", "fault", "alarm", "label", "churn", "default", "readmission")
    num_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in num_cols if c != fail_col][:12]
    if not feats:
        raise RuntimeError("Need numeric columns for field prediction.")
    X = work[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, fail_col


def field_predict(df: pd.DataFrame) -> float:
    return float(field_risk_explain(df)["risk_pct"])


def field_risk_explain(df: pd.DataFrame) -> dict[str, Any]:
    domain = st.session_state.get("domain") or detect_field(df, use_gemini=False)["domain"]
    work = apply_domain_feature_engineering(df, domain)
    X, label_col = _numeric_xy(work)
    explanations: list[str] = []
    risk = 15.0
    t = _col(work, "temperature", "temp")
    v = _col(work, "vibration", "vib")
    g = "temp_gradient" if "temp_gradient" in work.columns else None
    if domain in ("predictive_maintenance", "energy_utilities", "agriculture_iot"):
        if v is not None:
            vv = float(pd.to_numeric(work[v], errors="coerce").iloc[-1])
            explanations.append(f"vibration={vv:.3f}")
            mean_v = float(pd.to_numeric(work[v], errors="coerce").mean())
            std_v = float(pd.to_numeric(work[v], errors="coerce").std() or 1)
            if vv > mean_v + 2 * std_v:
                risk += 25
        if t is not None:
            tt = float(pd.to_numeric(work[t], errors="coerce").iloc[-1])
            explanations.append(f"temperature={tt:.2f}")
            if tt > 90:
                risk += 15
        if g is not None and pd.notna(work[g].iloc[-1]):
            gg = float(work[g].iloc[-1])
            explanations.append(f"temp_gradient={gg:.2f}")
            if abs(gg) > 10:
                risk += 20
    if domain == "healthcare":
        if "BMI" in work.columns and pd.notna(work["BMI"].iloc[-1]):
            bmi = float(work["BMI"].iloc[-1])
            explanations.append(f"BMI={bmi:.1f}")
            if bmi >= 30:
                risk += 20
        if "BRISK_SCORE" in work.columns and pd.notna(work["BRISK_SCORE"].iloc[-1]):
            brisk = float(work["BRISK_SCORE"].iloc[-1])
            explanations.append(f"BRISK_SCORE={brisk:.2f}")
            if brisk > 3.5:
                risk += 18
    if domain == "sales_forecasting":
        rev = _col(work, "revenue", "sales")
        if rev:
            r = pd.to_numeric(work[rev], errors="coerce")
            if len(r) > 5 and r.iloc[-1] < r.mean() * 0.7:
                risk += 25
                explanations.append(f"revenue_drop latest={r.iloc[-1]:.1f} vs mean={r.mean():.1f}")
        if "ASP" in work.columns and pd.notna(work["ASP"].iloc[-1]):
            explanations.append(f"ASP={float(work['ASP'].iloc[-1]):.2f}")
    if domain == "finance_risk":
        if "DTI_proxy" in work.columns and pd.notna(work["DTI_proxy"].iloc[-1]):
            dti = float(work["DTI_proxy"].iloc[-1])
            explanations.append(f"DTI_proxy={dti:.2f}")
            if dti > 0.45:
                risk += 28
    if domain == "warehouse_logistics" and "days_of_cover" in work.columns:
        doc = pd.to_numeric(work["days_of_cover"], errors="coerce")
        if doc.notna().any():
            latest = float(doc.iloc[-1])
            explanations.append(f"days_of_cover={latest:.1f}")
            if latest < 3:
                risk += 30
    if domain == "energy_utilities":
        load = _col(work, "load", "consumption", "kwh", "mw", "power")
        if load:
            lv = float(pd.to_numeric(work[load], errors="coerce").iloc[-1])
            explanations.append(f"load={lv:.2f}")
            if "load_gradient" in work.columns and abs(float(work["load_gradient"].iloc[-1] or 0)) > abs(lv) * 0.2:
                risk += 22
                explanations.append(f"load_gradient={float(work['load_gradient'].iloc[-1]):.2f}")
    if domain == "telecom_churn":
        churn = _col(work, "churn")
        if churn is not None:
            rate = float(pd.to_numeric(work[churn], errors="coerce").fillna(0).mean())
            risk = max(risk, rate * 100)
            explanations.append(f"churn_rate={rate*100:.1f}%")
        if "lifetime_value_proxy" in work.columns and pd.notna(work["lifetime_value_proxy"].iloc[-1]):
            explanations.append(f"LTV_proxy={float(work['lifetime_value_proxy'].iloc[-1]):.1f}")
    if domain == "agriculture_iot":
        if "irrigation_stress" in work.columns and pd.notna(work["irrigation_stress"].iloc[-1]):
            stress = float(work["irrigation_stress"].iloc[-1])
            explanations.append(f"irrigation_stress={stress:.2f}")
            if stress > 20:
                risk += 25
        if "ph_dev_neutral" in work.columns and pd.notna(work["ph_dev_neutral"].iloc[-1]):
            phd = float(work["ph_dev_neutral"].iloc[-1])
            explanations.append(f"ph_dev_neutral={phd:.2f}")
            if phd > 1.5:
                risk += 15
    try:
        if len(X) >= 15:
            iso = IsolationForest(contamination=0.1, random_state=42)
            iso_rate = float((iso.fit_predict(X) == -1).mean())
            risk = 0.5 * risk + 0.5 * (iso_rate * 100)
            explanations.append(f"isolation_forest_anomaly_rate={iso_rate*100:.1f}%")
            if label_col and work_has_binary(work, label_col):
                y = (pd.to_numeric(work[label_col], errors="coerce").fillna(0) > 0).astype(int)
                if y.nunique() > 1:
                    try:
                        import optuna
                        optuna.logging.set_verbosity(optuna.logging.WARNING)

                        def obj(trial):
                            clf = RandomForestClassifier(
                                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                                max_depth=trial.suggest_int("max_depth", 2, 12),
                                random_state=42,
                                class_weight="balanced",
                            )
                            return float(cross_val_score(clf, X, y, cv=3, scoring="f1").mean())

                        study = optuna.create_study(direction="maximize")
                        study.optimize(obj, n_trials=15, show_progress_bar=False)
                        best = study.best_params
                        clf = RandomForestClassifier(**best, random_state=42, class_weight="balanced")
                        clf.fit(X, y)
                        proba = float(clf.predict_proba(X.tail(min(24, len(X))))[:, 1].mean())
                        risk = 0.4 * risk + 0.6 * (proba * 100)
                        imps = sorted(zip(X.columns, clf.feature_importances_), key=lambda z: -z[1])
                        explanations.append("top_features=" + ", ".join(f"{a}:{b:.2f}" for a, b in imps[:4]))
                        explanations.append(f"optuna_best={best}")
                    except Exception as exc:
                        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
                        gb = GradientBoostingClassifier(random_state=42)
                        rf.fit(X, y)
                        gb.fit(X, y)
                        latest = X.tail(min(24, len(X)))
                        proba = 0.5 * rf.predict_proba(latest)[:, 1].mean() + 0.5 * gb.predict_proba(latest)[:, 1].mean()
                        risk = 0.4 * risk + 0.6 * (proba * 100)
                        explanations.append(f"ensemble_proba={proba:.3f}; optuna_skip={exc}")
    except Exception as exc:
        explanations.append(f"ml_note={exc}")
    risk = float(min(99.5, max(0.5, risk)))
    because = ", ".join(explanations[:6]) if explanations else "insufficient history"
    text = f"{DOMAIN_CATALOG.get(domain, {}).get('label', domain)} risk {risk:.1f}% because {because}"
    return {"risk_pct": round(risk, 1), "domain": domain, "explanation": text, "factors": explanations}


def work_has_binary(df: pd.DataFrame, col: str) -> bool:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return False
    return s.nunique() <= 5

