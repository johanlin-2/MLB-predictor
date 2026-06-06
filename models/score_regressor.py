"""Score regressor: predicts home_runs and away_runs.

Primary: Poisson GLM (statsmodels). Secondary: XGBRegressor with Tweedie loss
(close cousin of Poisson, better tail behaviour for high-scoring games).

Predictions:
 * predicted_total  = home + away
 * predicted_spread = home − away
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

import config
from models._common import save_model, safe_xy

logger = logging.getLogger(__name__)

HOME_LABEL = "home_score"
AWAY_LABEL = "visitor_score"


@dataclass
class ScoreFoldMetrics:
    fold: int
    val: int
    mae_total: float
    rmse_total: float
    mae_spread: float
    rmse_spread: float


def _fit_poisson(X: pd.DataFrame, y: pd.Series):
    """statsmodels Poisson GLM. Returns a fitted result object."""
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.GLM(y, X_const, family=sm.families.Poisson())
    return model.fit(maxiter=100, disp=False)


def _fit_xgb(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        objective="reg:tweedie", tweedie_variance_power=1.3,
        tree_method="hist", n_jobs=4, random_state=42,
    )
    model.fit(X, y, verbose=False)
    return model


def walk_forward_cv(df: pd.DataFrame) -> list[ScoreFoldMetrics]:
    metrics: list[ScoreFoldMetrics] = []
    for i, fold in enumerate(config.WALK_FORWARD_FOLDS):
        tr_mask = ((df["season"] >= fold["train_start"]) &
                   (df["season"] <= fold["train_end"]) &
                   (df["season"] != 2020))
        train, val = df[tr_mask], df[df["season"] == fold["val"]]
        if train.empty or val.empty:
            continue

        X_tr, _ = safe_xy(train, HOME_LABEL)
        X_va, _ = safe_xy(val, HOME_LABEL)
        common = X_tr.columns.intersection(X_va.columns)
        X_tr, X_va = X_tr[common], X_va[common]
        y_tr_home = train[HOME_LABEL]; y_tr_away = train[AWAY_LABEL]

        home_xgb = _fit_xgb(X_tr, y_tr_home)
        away_xgb = _fit_xgb(X_tr, y_tr_away)
        p_home = home_xgb.predict(X_va)
        p_away = away_xgb.predict(X_va)
        p_total = p_home + p_away
        p_spread = p_home - p_away

        y_total = val[HOME_LABEL] + val[AWAY_LABEL]
        y_spread = val[HOME_LABEL] - val[AWAY_LABEL]

        m = ScoreFoldMetrics(
            fold=i, val=fold["val"],
            mae_total=float(mean_absolute_error(y_total, p_total)),
            rmse_total=float(np.sqrt(mean_squared_error(y_total, p_total))),
            mae_spread=float(mean_absolute_error(y_spread, p_spread)),
            rmse_spread=float(np.sqrt(mean_squared_error(y_spread, p_spread))),
        )
        logger.info("fold %d: MAE total=%.2f spread=%.2f", i, m.mae_total, m.mae_spread)
        metrics.append(m)
    return metrics


def fit_final(df: pd.DataFrame):
    sub = df[df["season"].isin(config.TRAIN_YEARS + config.VAL_YEARS)]
    X, _ = safe_xy(sub, HOME_LABEL)
    home_xgb = _fit_xgb(X, sub[HOME_LABEL])
    away_xgb = _fit_xgb(X, sub[AWAY_LABEL])
    # Poisson GLMs alongside, for the EV-on-totals comparison
    try:
        home_poisson = _fit_poisson(X, sub[HOME_LABEL]).params.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("home Poisson failed: %s", exc)
        home_poisson = None
    try:
        away_poisson = _fit_poisson(X, sub[AWAY_LABEL]).params.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("away Poisson failed: %s", exc)
        away_poisson = None
    save_model({"home": home_xgb, "away": away_xgb,
                "home_poisson_params": home_poisson,
                "away_poisson_params": away_poisson,
                "features": list(X.columns)},
               "score_regressor")
    return {"home": home_xgb, "away": away_xgb, "features": list(X.columns)}


def run(df: pd.DataFrame) -> dict:
    folds = walk_forward_cv(df)
    out = [asdict(m) for m in folds]
    means = pd.DataFrame(out).mean(numeric_only=True).to_dict() if out else {}
    with open(config.MODEL_ARTIFACTS_DIR / "score_regressor_cv.json", "w") as f:
        json.dump({"folds": out, "means": means}, f, indent=2)
    final = fit_final(df)
    return {"cv": out, "means": means, "final": final}
