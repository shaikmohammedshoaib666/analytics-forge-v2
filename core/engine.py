"""Clean engine selector — pandas (default), Polars, or PySpark with graceful fallback."""
from __future__ import annotations

from typing import Any

import pandas as pd

from core.clean import clean_dataframe


def available_engines() -> list[str]:
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


def clean_with_engine(df: pd.DataFrame, engine: str = "pandas") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Route cleaning through selected engine. Falls back to pandas if engine unavailable."""
    if engine == "polars":
        try:
            return _clean_polars(df)
        except Exception:
            pass
    elif engine == "pyspark":
        try:
            return _clean_pyspark(df)
        except Exception:
            pass
    return clean_dataframe(df)


def _clean_polars(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    import polars as pl

    log: list[dict[str, Any]] = []
    pldf = pl.from_pandas(df)
    rows_before = pldf.height

    # Drop fully-null columns
    null_cols = [c for c in pldf.columns if pldf[c].null_count() == pldf.height]
    if null_cols:
        pldf = pldf.drop(null_cols)
        log.append({"operation": "polars_drop_null_cols", "detail": str(null_cols), "rows_before": rows_before, "rows_after": pldf.height})

    # Drop duplicates
    before = pldf.height
    pldf = pldf.unique()
    if pldf.height != before:
        log.append({"operation": "polars_unique", "detail": "pl.unique()", "rows_before": before, "rows_after": pldf.height})

    # Fill nulls: numeric with median, string with "Unknown"
    for c in pldf.columns:
        dtype = pldf[c].dtype
        if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
            if pldf[c].null_count() > 0:
                med = pldf[c].median()
                pldf = pldf.with_columns(pl.col(c).fill_null(med))
                log.append({"operation": "polars_fill_median", "detail": f"{c} -> {med}", "rows_before": pldf.height, "rows_after": pldf.height})
        elif dtype == pl.Utf8 or dtype == pl.String:
            if pldf[c].null_count() > 0:
                pldf = pldf.with_columns(pl.col(c).fill_null("Unknown"))

    log.append({"operation": "polars_done", "detail": "Polars native clean complete", "rows_before": rows_before, "rows_after": pldf.height})
    return pldf.to_pandas(), log


def _clean_pyspark(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """PySpark clean path — requires pyspark installed and a local SparkSession."""
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, median as spark_median

    spark = SparkSession.builder.master("local[*]").appName("forge_clean").getOrCreate()
    sdf = spark.createDataFrame(df)
    log: list[dict[str, Any]] = []
    rows_before = sdf.count()

    sdf = sdf.dropDuplicates()
    after = sdf.count()
    if after != rows_before:
        log.append({"operation": "spark_dedup", "detail": "dropDuplicates()", "rows_before": rows_before, "rows_after": after})

    log.append({"operation": "spark_done", "detail": "PySpark clean complete", "rows_before": rows_before, "rows_after": after})
    result = sdf.toPandas()
    spark.stop()
    return result, log
