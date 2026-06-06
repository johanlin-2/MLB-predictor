"""Rolling team offensive / defensive / bullpen features.

Leakage discipline: every rolling stat is built as
    series.groupby(team).shift(1).rolling(WINDOW).mean()
so the current game is never used to compute its own features.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

WINDOW = config.TEAM_ROLLING_WINDOW
WINDOW_SHORT = config.TEAM_ROLLING_WINDOW_SHORT


def _build_team_game_records(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Explode each game into two rows (one per team) so we can groupby team."""
    home = game_logs.rename(columns={
        "home_team": "team", "visiting_team": "opponent",
        "home_score": "runs_scored", "visitor_score": "runs_allowed",
        "home_starting_pitcher_id": "sp_id",
        "home_pitchers_used": "pitchers_used",
    }).assign(is_home=1)
    away = game_logs.rename(columns={
        "visiting_team": "team", "home_team": "opponent",
        "visitor_score": "runs_scored", "home_score": "runs_allowed",
        "visitor_starting_pitcher_id": "sp_id",
        "visitor_pitchers_used": "pitchers_used",
    }).assign(is_home=0)
    cols = ["game_id", "date", "team", "opponent", "runs_scored",
            "runs_allowed", "is_home", "sp_id", "season", "pitchers_used"]
    home = home[[c for c in cols if c in home.columns]]
    away = away[[c for c in cols if c in away.columns]]
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    long["win"] = (long["runs_scored"] > long["runs_allowed"]).astype(int)
    long["run_diff"] = long["runs_scored"] - long["runs_allowed"]
    return long


def _rolling(series: pd.Series, window: int = WINDOW) -> pd.Series:
    """`.shift(1).rolling(window).mean()` — strictly leakage-safe."""
    return series.shift(1).rolling(window, min_periods=3).mean()


def _rolling_short(series: pd.Series) -> pd.Series:
    """7-game leakage-safe rolling mean — captures recent streaks."""
    return series.shift(1).rolling(WINDOW_SHORT, min_periods=2).mean()


def _add_team_rolling(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    g = long.groupby("team", group_keys=False)

    # 15-game (baseline trend)
    long["roll_runs_scored"] = g["runs_scored"].apply(_rolling)
    long["roll_runs_allowed"] = g["runs_allowed"].apply(_rolling)
    long["roll_win_pct"] = g["win"].apply(_rolling)
    long["roll_run_diff"] = g["run_diff"].apply(_rolling)

    # 7-game (recent form / streak)
    long["roll7_runs_scored"] = g["runs_scored"].apply(_rolling_short)
    long["roll7_runs_allowed"] = g["runs_allowed"].apply(_rolling_short)
    long["roll7_win_pct"] = g["win"].apply(_rolling_short)
    long["roll7_run_diff"] = g["run_diff"].apply(_rolling_short)

    # Bullpen workload proxy: rolling mean of pitchers used per game.
    # High values = taxed bullpen entering this game.
    if "pitchers_used" in long.columns:
        long["pitchers_used"] = pd.to_numeric(long["pitchers_used"], errors="coerce")
        long["roll7_pitchers_used"] = g["pitchers_used"].apply(_rolling_short)

    # Home / away splits (rolling per team within the home/away subset only)
    for is_home_flag, suffix in ((1, "home"), (0, "away")):
        mask = long["is_home"] == is_home_flag
        sub = long[mask].copy()
        g_sub = sub.groupby("team", group_keys=False)
        long.loc[mask, f"roll_runs_scored_{suffix}"] = g_sub["runs_scored"].apply(_rolling).values
        long.loc[mask, f"roll_runs_allowed_{suffix}"] = g_sub["runs_allowed"].apply(_rolling).values

    return long


def _merge_back_to_game(long: pd.DataFrame, game_logs: pd.DataFrame) -> pd.DataFrame:
    """Pivot the team-long table back into a per-game row with home_* / away_* columns."""
    feature_cols = [c for c in long.columns if c.startswith("roll_") or c.startswith("roll7_")]
    home_feats = long[long["is_home"] == 1][["game_id"] + feature_cols].add_prefix("home_")
    home_feats = home_feats.rename(columns={"home_game_id": "game_id"})
    away_feats = long[long["is_home"] == 0][["game_id"] + feature_cols].add_prefix("away_")
    away_feats = away_feats.rename(columns={"away_game_id": "game_id"})
    out = game_logs.merge(home_feats, on="game_id", how="left").merge(
        away_feats, on="game_id", how="left")
    return out


def _add_fangraphs_priors(game_df: pd.DataFrame,
                          team_batting: pd.DataFrame,
                          team_pitching: pd.DataFrame) -> pd.DataFrame:
    """Attach prior-season FanGraphs team rate stats (wOBA, FIP, etc.).

    Using prior-season values avoids in-season leakage. Free at the start of
    the year and approximately right by mid-season since rolling-15 features
    dominate by then.
    """
    cols_bat = [c for c in ["Team", "Season", "wOBA", "wRC+", "OBP", "SLG", "ISO", "K%", "BB%", "BABIP"]
                if c in team_batting.columns]
    cols_pit = [c for c in ["Team", "Season", "ERA", "FIP", "xFIP", "WHIP", "HR/9", "K/9", "BB/9"]
                if c in team_pitching.columns]
    bat = team_batting[cols_bat].rename(columns={"Team": "team", "Season": "season"})
    pit = team_pitching[cols_pit].rename(columns={"Team": "team", "Season": "season"})
    bat["season"] = bat["season"].astype(int) + 1   # use prior season
    pit["season"] = pit["season"].astype(int) + 1

    for side, team_col in (("home", "home_team"), ("away", "visiting_team")):
        if team_col not in game_df.columns:
            continue
        renamed_bat = bat.rename(columns={c: f"{side}_prior_bat_{c}" for c in bat.columns
                                          if c not in ("team", "season")})
        renamed_pit = pit.rename(columns={c: f"{side}_prior_pit_{c}" for c in pit.columns
                                          if c not in ("team", "season")})
        game_df = game_df.merge(renamed_bat, left_on=[team_col, "season"],
                                right_on=["team", "season"], how="left").drop(columns=["team"])
        game_df = game_df.merge(renamed_pit, left_on=[team_col, "season"],
                                right_on=["team", "season"], how="left").drop(columns=["team"])
    return game_df


def build(game_logs: pd.DataFrame,
          team_batting: pd.DataFrame | None = None,
          team_pitching: pd.DataFrame | None = None) -> pd.DataFrame:
    """Top-level entry. Returns per-game DataFrame with all team features attached."""
    logger.info("building team rolling features over %d games", len(game_logs))
    long = _build_team_game_records(game_logs)
    long = _add_team_rolling(long)
    game_df = _merge_back_to_game(long, game_logs)
    if team_batting is not None and team_pitching is not None:
        game_df = _add_fangraphs_priors(game_df, team_batting, team_pitching)
    return game_df


def assert_no_leakage(game_df: pd.DataFrame) -> None:
    """Sanity: rolling features must be NaN on a team's very first game ever.

    A team's first *home* game may legitimately have non-NaN rolling features
    computed from prior *away* games — that is correct leakage-safe behavior.
    We only flag rows where a team's first home game is also their first game
    overall (i.e. they had zero prior appearances of any kind).
    """
    home_cols = [c for c in game_df.columns if c.startswith("home_roll_")]
    if not home_cols:
        return

    sorted_df = game_df.sort_values("date")

    # Earliest date each team appears (home OR away).
    earliest = pd.concat([
        sorted_df[["home_team", "date"]].rename(columns={"home_team": "team"}),
        sorted_df[["visiting_team", "date"]].rename(columns={"visiting_team": "team"}),
    ]).groupby("team")["date"].min()

    first_home = sorted_df.groupby("home_team").head(1)

    # Only flag if this home game IS the team's first appearance overall.
    truly_first = first_home[
        first_home.apply(
            lambda r: r["date"] == earliest.get(r["home_team"], r["date"]), axis=1
        )
    ]

    leak = truly_first[home_cols].notna().any(axis=1).sum()
    if leak:
        raise AssertionError(f"team rolling leakage detected on {leak} first-ever home games")
    logger.info(
        "team feature leakage check passed (%d truly-first home games clean)", len(truly_first)
    )
