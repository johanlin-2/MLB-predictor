"""Merge every feature source into data/processed/games.parquet.

Drops the 2020 season entirely (signed-off ambiguity 3 → option b).
Drops postseason games (ambiguity 8). Asserts no odds columns are present
on the feature matrix.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config
from features import context_features, pitcher_features, team_features

logger = logging.getLogger(__name__)

OUT_PATH: Path = config.PROCESSED_DIR / "games.parquet"

EXCLUDED_SEASONS = {2020}


def build(force: bool = False, with_weather: bool = True) -> pd.DataFrame:
    # Lazy-import pybaseball-backed ingestion so unit tests don't need it.
    from ingestion import fangraphs, retrosheet, statcast

    if OUT_PATH.exists() and not force:
        logger.info("loading cached %s", OUT_PATH)
        return pd.read_parquet(OUT_PATH)

    game_logs = retrosheet.load()
    game_logs = game_logs[~game_logs["season"].isin(EXCLUDED_SEASONS)].copy()
    logger.info("loaded %d games (post-2020 exclusion)", len(game_logs))

    team_bat = fangraphs.fetch_team_batting()
    team_pit = fangraphs.fetch_team_pitching()
    fg_pitcher = fangraphs.fetch_pitcher_season_stats()

    df = team_features.build(game_logs, team_bat, team_pit)
    team_features.assert_no_leakage(df)

    # Statcast — only load months we have on disk; safe no-op if empty.
    try:
        statcast_pitch = statcast.load_range(
            f"{config.ALL_YEARS[0]}-03-01",
            f"{config.ALL_YEARS[-1]}-11-30",
        )
        sc_summary = pitcher_features.aggregate_statcast(statcast_pitch)
    except Exception as exc:  # noqa: BLE001
        logger.warning("statcast aggregation failed (%s) — continuing without it", exc)
        sc_summary = None

    pitcher_df = pitcher_features.build(game_logs, sc_summary, fg_pitcher)
    df = df.merge(pitcher_df, on="game_id", how="left")
    df = context_features.build(df, fetch_weather=with_weather)

    # Consolidate duplicate home_win columns produced by the team_features merge.
    if "home_win_x" in df.columns:
        df = df.rename(columns={"home_win_x": "home_win"}).drop(
            columns=["home_win_y"], errors="ignore"
        )

    _assert_no_odds_columns(df)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("wrote %d processed rows (%d cols) to %s",
                len(df), df.shape[1], OUT_PATH)
    return df


def _assert_no_odds_columns(df: pd.DataFrame) -> None:
    forbidden_substrings = ("odds", "price", "moneyline", "spread_line", "total_line",
                            "h2h", "spreads", "totals", "book")
    bad = [c for c in df.columns if any(s in c.lower() for s in forbidden_substrings)]
    if bad:
        raise AssertionError(f"odds columns leaked into feature matrix: {bad}")
    logger.info("no-odds check passed (%d feature columns)", df.shape[1])


def split(df: pd.DataFrame, train_years=config.TRAIN_YEARS,
          val_years=config.VAL_YEARS, test_years=config.TEST_YEARS):
    train = df[df["season"].isin(train_years)]
    val = df[df["season"].isin(val_years)]
    test = df[df["season"].isin(test_years)]
    return train, val, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build()
