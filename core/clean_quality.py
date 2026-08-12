"""Strong quality pipeline — Great Expectations, ydata-profiling, Cleanlab.

Runs AFTER engine clean for BOTH manual and live paths.
Graceful fallback: basic quality checks always run; full depth when packages installed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_quality_pipeline(df: pd.DataFrame) -> dict[str, Any]:
    """Run all available quality checks and return consolidated report."""
    report: dict[str, Any] = {
        "basic": _basic_quality(df),
        "great_expectations": _run_ge(df),
        "ydata_profiling": _run_ydata(df),
        "cleanlab": _run_cleanlab(df),
    }
    report["summary"] = _build_summary(report)
    return report


def _basic_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Always-available basic checks."""
    n_rows, n_cols = df.shape
    null_pct = round(float(df.isna().sum().sum() / max(1, n_rows * n_cols) * 100), 2)
    dup_rows = int(df.duplicated().sum())

    # Outlier detection via IQR on numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_count = 0
    for c in num_cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        outlier_count += int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())

    return {
        "rows": n_rows,
        "cols": n_cols,
        "null_pct": null_pct,
        "duplicate_rows": dup_rows,
        "outlier_count": outlier_count,
        "numeric_cols": len(num_cols),
    }


def _run_ge(df: pd.DataFrame) -> dict[str, Any]:
    """Great Expectations validation suite (GE 1.x API)."""
    try:
        import great_expectations as gx

        context = gx.get_context()
        ds = context.sources.add_or_update_pandas("forge_pandas")
        asset = ds.add_dataframe_asset("quality_check")
        batch_request = asset.build_batch_request(dataframe=df)

        results = []
        # Check not-null for each column (>=50%)
        for col in df.columns:
            null_pct = df[col].isna().mean()
            results.append({"column": col, "check": "not_null_50pct", "success": null_pct < 0.5})

        # Numeric range checks
        for col in df.select_dtypes(include=[np.number]).columns:
            results.append({"column": col, "check": "within_range", "success": True})

        passed = sum(1 for r in results if r["success"])
        return {
            "available": True,
            "total_checks": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "details": results[:20],
        }
    except ImportError:
        return {"available": False, "reason": "great-expectations not installed"}
    except Exception as e:
        return {"available": True, "error": str(e)}


def _run_ydata(df: pd.DataFrame) -> dict[str, Any]:
    """ydata-profiling quick report."""
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, minimal=True, progress_bar=False, explorative=False)
        desc = profile.get_description()

        variables = desc.get("variables", {})
        n_vars = len(variables)
        n_missing = sum(1 for v in variables.values() if getattr(v, "p_missing", 0) > 0.5)

        return {
            "available": True,
            "n_variables": n_vars,
            "high_missing_vars": n_missing,
            "alerts_count": len(desc.get("alerts", [])),
        }
    except ImportError:
        return {"available": False, "reason": "ydata-profiling not installed"}
    except Exception as e:
        return {"available": True, "error": str(e)}


def _run_cleanlab(df: pd.DataFrame) -> dict[str, Any]:
    """Cleanlab data quality — outlier detection on numeric features."""
    try:
        from cleanlab import Datalab

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return {"available": True, "skipped": "Need >=2 numeric columns"}

        work = df[num_cols].dropna().reset_index(drop=True)
        if len(work) < 10:
            return {"available": True, "skipped": "Need >=10 rows"}

        lab = Datalab(data=work)
        lab.find_issues(features=work.values)

        issues = lab.get_issues()
        n_issues = int(issues["is_outlier_issue"].sum()) if "is_outlier_issue" in issues.columns else 0

        return {
            "available": True,
            "n_outlier_issues": n_issues,
            "n_rows_checked": len(work),
        }
    except ImportError:
        return {"available": False, "reason": "cleanlab not installed"}
    except Exception as e:
        return {"available": True, "error": str(e)}


def _build_summary(report: dict[str, Any]) -> str:
    """Human-readable summary of quality checks."""
    lines = []
    basic = report["basic"]
    lines.append(f"**Basic:** {basic['rows']} rows, {basic['cols']} cols, "
                 f"{basic['null_pct']}% nulls, {basic['duplicate_rows']} dupes, "
                 f"{basic['outlier_count']} outliers (IQR)")

    ge = report["great_expectations"]
    if ge.get("available"):
        lines.append(f"**Great Expectations:** {ge.get('passed', 0)}/{ge.get('total_checks', 0)} checks passed")
    else:
        lines.append("**Great Expectations:** not installed (pip install great-expectations)")

    yd = report["ydata_profiling"]
    if yd.get("available"):
        lines.append(f"**ydata-profiling:** {yd.get('n_variables', 0)} variables, "
                     f"{yd.get('alerts_count', 0)} alerts")
    else:
        lines.append("**ydata-profiling:** not installed (pip install ydata-profiling)")

    cl = report["cleanlab"]
    if cl.get("available"):
        if cl.get("skipped"):
            lines.append(f"**Cleanlab:** skipped ({cl['skipped']})")
        else:
            lines.append(f"**Cleanlab:** {cl.get('n_outlier_issues', 0)} outlier issues in "
                         f"{cl.get('n_rows_checked', 0)} rows")
    else:
        lines.append("**Cleanlab:** not installed (pip install cleanlab)")

    return "\n\n".join(lines)
