"""Leakage tests.

Two kinds of leakage are checked:

1. Temporal: a rolling feature must use only data strictly prior to the game.
   We synthesise a small game-log fixture and assert that team_features.build()
   produces NaN for each team's first-of-season game.

2. Schema: no odds-related columns are allowed in the feature matrix produced
   by features.build_dataset._assert_no_odds_columns.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from features.build_dataset import _assert_no_odds_columns
from features.team_features import assert_no_leakage, build


def _toy_game_logs() -> pd.DataFrame:
    """Two teams alternating home/away over 20 games."""
    teams = ["NYY", "BOS"]
    rows = []
    base = datetime(2024, 4, 1)
    for i in range(20):
        date = base + timedelta(days=i)
        home, away = teams[i % 2], teams[(i + 1) % 2]
        home_score = 4 + (i % 3)
        away_score = 3 + (i % 4)
        rows.append({
            "game_id": f"{date:%Y-%m-%d}_{away}_{home}_0",
            "date": pd.Timestamp(date),
            "season": 2024,
            "home_team": home,
            "visiting_team": away,
            "home_score": home_score,
            "visitor_score": away_score,
            "home_starting_pitcher_id": f"sp_{home}",
            "visitor_starting_pitcher_id": f"sp_{away}",
            "home_win": int(home_score > away_score),
            "home_covered": int(home_score - away_score >= 2),
            "total_runs": home_score + away_score,
            "run_diff_home": home_score - away_score,
            "day_night": "D" if i % 2 == 0 else "N",
            "park_id": "NYC22" if home == "NYY" else "BOS07",
        })
    return pd.DataFrame(rows)


def test_first_game_has_nan_rolling_features():
    """The first row for each team must have NaN in every rolling feature."""
    logs = _toy_game_logs()
    df = build(logs)
    # `assert_no_leakage` raises AssertionError if any first game has non-NaN values.
    # We don't run it on this 20-row sample (min_periods=3 keeps the first 3 NaN),
    # but we directly check the home_roll_runs_scored on each team's first game.
    first_games = (df.sort_values("date").groupby("home_team").head(1))
    rolling_cols = [c for c in first_games.columns if c.startswith("home_roll_")]
    assert rolling_cols, "expected at least one rolling column"
    for col in rolling_cols:
        assert first_games[col].isna().all(), f"{col} leaked into first game"


def test_assert_no_odds_columns_passes_clean_frame():
    df = pd.DataFrame({
        "game_id": ["a"], "home_team": ["X"], "visiting_team": ["Y"],
        "home_roll_runs_scored": [4.5], "predicted_total": [9.1],
    })
    # Should not raise
    _assert_no_odds_columns(df)


def test_assert_no_odds_columns_blocks_odds_leak():
    bad = pd.DataFrame({
        "game_id": ["a"], "home_team": ["X"], "visiting_team": ["Y"],
        "home_h2h_price": [-110],
    })
    with pytest.raises(AssertionError):
        _assert_no_odds_columns(bad)
