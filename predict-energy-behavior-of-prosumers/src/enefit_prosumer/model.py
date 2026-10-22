from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from xgboost import XGBRegressor


DEFAULT_XGB_CONFIG = {
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "enable_categorical": True,
    "n_estimators": 900,
    "learning_rate": 0.04,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
}


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Mapping[str, object] | None = None,
    n_jobs: int = -1,
) -> XGBRegressor:
    """Train the XGBoost MAE model used by the competition pipeline."""

    params = DEFAULT_XGB_CONFIG | {"n_jobs": n_jobs, "base_score": float(y.mean())}
    if config:
        params |= dict(config)
    return XGBRegressor(**params).fit(X, y)
