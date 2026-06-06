"""Shared helpers across all three model files."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric, non-leaky feature columns.

    Excludes labels, identifiers, dates, and anything containing odds-related
    substrings. This is the same blocklist used by build_dataset's no-odds check.
    """
    forbidden_substrings = ("odds", "price", "moneyline", "spread_line",
                            "total_line", "h2h", "spreads", "totals", "book")
    exclude_exact = {
        "game_id", "date", "season", "home_team", "visiting_team", "day_of_week",
        "park_id", "day_night", "completion_info", "forfeit_info",
        "home_starting_pitcher_id", "visitor_starting_pitcher_id",
        "home_starting_pitcher_name", "visitor_starting_pitcher_name",
        # labels
        "home_win", "home_covered", "total_runs", "run_diff_home",
        "home_score", "visitor_score",
        # season-level fangraphs identifiers carried over from merges
        "team",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(sub in c.lower() for sub in forbidden_substrings):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def save_model(model, name: str) -> Path:
    path = config.MODEL_ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    return joblib.load(config.MODEL_ARTIFACTS_DIR / f"{name}.joblib")


def safe_xy(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    feats = feature_columns(df)
    X = df[feats].astype(np.float32)
    # Drop columns that are entirely NaN in this split (can't impute a median).
    X = X.loc[:, X.notna().any()]
    medians = X.median()
    X = X.fillna(medians)
    y = df[label]
    return X, y
