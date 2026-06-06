"""Home win/loss classifier. XGBoost primary, logistic-regression baseline.

Walk-forward CV per config.WALK_FORWARD_FOLDS. Outputs:
 * `home_win_xgb.joblib` — trained XGBoost classifier
 * `home_win_lr.joblib`  — logistic-regression baseline
 * `home_win_cv.json`    — per-fold metrics (AUC, Brier)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config
from models._common import feature_columns, save_model, safe_xy

logger = logging.getLogger(__name__)

LABEL = "home_win"
MODEL_NAME = "home_win"


@dataclass
class FoldMetrics:
    fold: int
    train_start: int
    train_end: int
    val: int
    auc_xgb: float
    brier_xgb: float
    auc_lr: float
    brier_lr: float


def _train_xgb(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    pos_rate = float(y.mean())
    scale_pos = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    model = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=3,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    model.fit(X, y, verbose=False)
    return model


def _train_lr(X: pd.DataFrame, y: pd.Series) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xs, y)
    return scaler, lr


def walk_forward_cv(df: pd.DataFrame) -> list[FoldMetrics]:
    metrics: list[FoldMetrics] = []
    for i, fold in enumerate(config.WALK_FORWARD_FOLDS):
        train_mask = (df["season"] >= fold["train_start"]) & (df["season"] <= fold["train_end"])
        train_mask &= df["season"] != 2020
        train = df[train_mask]
        val = df[df["season"] == fold["val"]]
        if train.empty or val.empty:
            logger.warning("fold %d skipped — empty train or val split", i)
            continue
        X_tr, y_tr = safe_xy(train, LABEL)
        X_va, y_va = safe_xy(val, LABEL)
        # Align columns (val may have fewer features if some are all-NaN)
        common = X_tr.columns.intersection(X_va.columns)
        X_tr, X_va = X_tr[common], X_va[common]

        xgb = _train_xgb(X_tr, y_tr)
        p_xgb = xgb.predict_proba(X_va)[:, 1]
        scaler, lr = _train_lr(X_tr, y_tr)
        p_lr = lr.predict_proba(scaler.transform(X_va))[:, 1]

        m = FoldMetrics(
            fold=i, train_start=fold["train_start"], train_end=fold["train_end"],
            val=fold["val"],
            auc_xgb=float(roc_auc_score(y_va, p_xgb)),
            brier_xgb=float(brier_score_loss(y_va, p_xgb)),
            auc_lr=float(roc_auc_score(y_va, p_lr)),
            brier_lr=float(brier_score_loss(y_va, p_lr)),
        )
        logger.info("fold %d: AUC xgb=%.3f lr=%.3f | Brier xgb=%.3f lr=%.3f",
                    i, m.auc_xgb, m.auc_lr, m.brier_xgb, m.brier_lr)
        metrics.append(m)
    return metrics


def fit_final(df: pd.DataFrame, train_years: Iterable[int] = config.TRAIN_YEARS):
    """Refit on train+val once CV is satisfactory."""
    sub = df[df["season"].isin(train_years)]
    X, y = safe_xy(sub, LABEL)
    xgb = _train_xgb(X, y)
    scaler, lr = _train_lr(X, y)
    save_model({"model": xgb, "features": list(X.columns)}, f"{MODEL_NAME}_xgb")
    save_model({"scaler": scaler, "model": lr, "features": list(X.columns)}, f"{MODEL_NAME}_lr")
    return {"xgb": xgb, "lr": (scaler, lr), "features": list(X.columns)}


def run(df: pd.DataFrame) -> dict:
    folds = walk_forward_cv(df)
    out_metrics = [asdict(m) for m in folds]
    means = pd.DataFrame(out_metrics).mean(numeric_only=True).to_dict() if out_metrics else {}
    with open(config.MODEL_ARTIFACTS_DIR / f"{MODEL_NAME}_cv.json", "w") as f:
        json.dump({"folds": out_metrics, "means": means}, f, indent=2)
    final = fit_final(df)
    return {"cv": out_metrics, "means": means, "final": final}
