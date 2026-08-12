"""Optuna hyperparameter tuning — picks stronger params and returns industry business insight."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from core.templates import insight_text


def optuna_available() -> bool:
    try:
        import optuna  # noqa: F401
        return True
    except ImportError:
        return False


def tune_model(
    df: pd.DataFrame,
    model_id: str,
    target: str,
    features: list[str],
    n_trials: int = 20,
    task: str = "regression",
    domain: str = "generic",
) -> dict[str, Any]:
    """Run Optuna tuning and return best params + industry-language insight."""
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    except ImportError:
        return {"ok": False, "error": "Optuna not installed. pip install optuna"}

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0)
    scoring = "r2" if task == "regression" else "accuracy"

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 20, 200)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        if task == "classification":
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, random_state=42,
            )
        else:
            clf = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_split=min_samples_split, random_state=42,
            )
        scores = cross_val_score(clf, X, y, cv=3, scoring=scoring)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Business-facing insight from tuned score + domain templates
    score = float(study.best_value)
    if task == "classification":
        kind = "failure" if score >= 0.7 else "forecast_down"
        business = insight_text(domain, kind) or (
            "Model found a usable risk pattern — review high-risk rows."
            if score >= 0.7
            else "Signal is weak — collect more labeled outcomes."
        )
    else:
        # Compare recent vs earlier mean of target as simple direction cue
        half = max(5, len(y) // 2)
        recent = float(y.iloc[-half:].mean()) if len(y) else 0
        earlier = float(y.iloc[:half].mean()) if len(y) else 0
        if earlier != 0 and (recent - earlier) / abs(earlier) > 0.05:
            business = insight_text(domain, "forecast_up") or "The target metric is trending up."
        elif earlier != 0 and (recent - earlier) / abs(earlier) < -0.05:
            business = insight_text(domain, "forecast_down") or "The target metric is trending down."
        else:
            business = insight_text(domain, "forecast_up") or "Trend is relatively stable."

    return {
        "ok": True,
        "best_params": study.best_params,
        "best_score": round(score, 4),
        "n_trials": n_trials,
        "scoring": scoring,
        "business_insight": business,
        "domain": domain,
        "recommended_model": model_id,
    }
