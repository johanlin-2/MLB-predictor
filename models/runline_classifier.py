"""Runline cover classifier — does the home team win by 2+?

Target: `home_covered` ∈ {0, 1}. Algorithm: XGBoost binary logistic.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import config
from models._common import save_model, safe_xy

logger = logging.getLogger(__name__)

LABEL = "home_covered"
MODEL_NAME = "runline"


@dataclass
class FoldMetrics:
    fold: int
    val: int
    auc: float
    brier: float
    accuracy: float


def _train(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=4, random_state=42,
    )
    model.fit(X, y, verbose=False)
    return model


def walk_forward_cv(df: pd.DataFrame) -> list[FoldMetrics]:
    metrics: list[FoldMetrics] = []
    for i, fold in enumerate(config.WALK_FORWARD_FOLDS):
        tr_mask = ((df["season"] >= fold["train_start"]) &
                   (df["season"] <= fold["train_end"]) &
                   (df["season"] != 2020))
        train, val = df[tr_mask], df[df["season"] == fold["val"]]
        if train.empty or val.empty:
            continue
        X_tr, y_tr = safe_xy(train, LABEL)
        X_va, y_va = safe_xy(val, LABEL)
        common = X_tr.columns.intersection(X_va.columns)
        X_tr, X_va = X_tr[common], X_va[common]
        model = _train(X_tr, y_tr)
        p = model.predict_proba(X_va)[:, 1]
        m = FoldMetrics(
            fold=i, val=fold["val"],
            auc=float(roc_auc_score(y_va, p)),
            brier=float(brier_score_loss(y_va, p)),
            accuracy=float(accuracy_score(y_va, p >= 0.5)),
        )
        logger.info("runline fold %d: AUC=%.3f Brier=%.3f", i, m.auc, m.brier)
        metrics.append(m)
    return metrics


def fit_final(df: pd.DataFrame):
    sub = df[df["season"].isin(config.TRAIN_YEARS)]
    X, y = safe_xy(sub, LABEL)
    model = _train(X, y)
    save_model({"model": model, "features": list(X.columns)}, MODEL_NAME)
    return {"model": model, "features": list(X.columns)}


def run(df: pd.DataFrame) -> dict:
    folds = walk_forward_cv(df)
    out = [asdict(m) for m in folds]
    means = pd.DataFrame(out).mean(numeric_only=True).to_dict() if out else {}
    with open(config.MODEL_ARTIFACTS_DIR / f"{MODEL_NAME}_cv.json", "w") as f:
        json.dump({"folds": out, "means": means}, f, indent=2)
    final = fit_final(df)
    return {"cv": out, "means": means, "final": final}
