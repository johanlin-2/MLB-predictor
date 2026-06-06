"""Starting pitcher features.

For the announced starter on each side we compute:
 * Season-to-date xFIP, FIP, WHIP, K/9, BB/9, HR/9 — using only games BEFORE the
   target game (computed by `.shift(1).expanding().mean()` per pitcher).
 * Last-3-starts ERA, K%, opponent wOBA against — rolling-3.
 * Days rest since previous start.
 * Statcast pitch-level aggregates from the prior start (avg EV against,
   whiff rate, called-strike rate).

If the starter is not announced at predict time (option `b` was rejected;
option `c` is in effect — predictions re-run hourly), the row is left with
NaN pitcher features and the dataset builder marks `pitcher_confirmed=False`
so the picks pipeline can choose to skip that game.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _build_starter_records(game_logs: pd.DataFrame) -> pd.DataFrame:
    """One row per (pitcher, game) with their team's runs allowed as a proxy
    for ERA when we don't have boxscore-level pitcher lines."""
    rows = []
    for _, g in game_logs.iterrows():
        if pd.notna(g.get("home_starting_pitcher_id")):
            rows.append({
                "game_id": g["game_id"], "date": g["date"], "season": g.get("season"),
                "pitcher_id": g["home_starting_pitcher_id"],
                "pitcher_name": g.get("home_starting_pitcher_name"),
                "team": g["home_team"], "opponent": g["visiting_team"],
                "is_home": 1, "runs_allowed": g["visitor_score"],
            })
        if pd.notna(g.get("visitor_starting_pitcher_id")):
            rows.append({
                "game_id": g["game_id"], "date": g["date"], "season": g.get("season"),
                "pitcher_id": g["visitor_starting_pitcher_id"],
                "pitcher_name": g.get("visitor_starting_pitcher_name"),
                "team": g["visiting_team"], "opponent": g["home_team"],
                "is_home": 0, "runs_allowed": g["home_score"],
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["pitcher_id", "date"]).reset_index(drop=True)


def _add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby("pitcher_id", group_keys=False)
    w = config.PITCHER_RECENT_STARTS

    df["sp_season_runs_allowed_pg"] = g["runs_allowed"].apply(
        lambda s: s.shift(1).expanding().mean())
    df["sp_last3_runs_allowed_pg"] = g["runs_allowed"].apply(
        lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    # Fraction of last-3 starts allowing ≤3 runs (quality-start proxy)
    df["sp_last3_quality_pct"] = g["runs_allowed"].apply(
        lambda s: (s <= 3).shift(1).rolling(w, min_periods=1).mean())

    df["sp_days_rest"] = g["date"].apply(lambda s: (s - s.shift(1)).dt.days)
    return df


def _attach_fangraphs_priors(starters: pd.DataFrame,
                             fg_pitcher: pd.DataFrame | None) -> pd.DataFrame:
    """Attach prior-season FanGraphs rate stats (xFIP, FIP, K/9, etc.).

    Join on (pitcher_name, prior_season) — both sources use "First Last" format.
    Using season N-1 stats avoids any in-season leakage.
    """
    if fg_pitcher is None or fg_pitcher.empty:
        for col in ["sp_prior_xFIP", "sp_prior_FIP", "sp_prior_WHIP",
                    "sp_prior_K9", "sp_prior_BB9", "sp_prior_HR9",
                    "sp_prior_Kpct", "sp_prior_BBpct", "sp_throws_right"]:
            starters[col] = np.nan
        return starters

    # Build lookup: prior-season stats keyed on (PlayerName, current_season)
    throws_cols = ["Throws"] if "Throws" in fg_pitcher.columns else []
    fg = fg_pitcher[["PlayerName", "Season", "xMLBAMID",
                      "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%",
                      "Team"] + throws_cols].copy()
    # For traded players FanGraphs has both per-team rows and a combined "2 Tms" row.
    # Keep "2 Tms" when available, otherwise keep the last team's row.
    fg["_sort"] = (fg["Team"] == "2 Tms").astype(int)
    fg = (fg.sort_values(["PlayerName", "Season", "_sort"], ascending=[True, True, False])
            .drop_duplicates(subset=["PlayerName", "Season"], keep="first")
            .drop(columns=["_sort", "Team"]))
    fg = fg.rename(columns={
        "PlayerName": "pitcher_name", "Season": "prior_season",
        "K/9": "sp_prior_K9", "BB/9": "sp_prior_BB9", "HR/9": "sp_prior_HR9",
        "K%": "sp_prior_Kpct", "BB%": "sp_prior_BBpct",
        "FIP": "sp_prior_FIP", "xFIP": "sp_prior_xFIP", "WHIP": "sp_prior_WHIP",
    })
    fg["season"] = fg["prior_season"] + 1   # prior_season data used for season+1 games

    # Encode handedness: 1=RHP, 0=LHP, NaN=unknown
    if "Throws" in fg.columns:
        fg["sp_throws_right"] = fg["Throws"].map({"R": 1, "L": 0})
        fg = fg.drop(columns=["Throws"])

    prior_cols = ["sp_prior_FIP", "sp_prior_xFIP", "sp_prior_WHIP",
                  "sp_prior_K9", "sp_prior_BB9", "sp_prior_HR9",
                  "sp_prior_Kpct", "sp_prior_BBpct", "xMLBAMID"]
    if "sp_throws_right" in fg.columns:
        prior_cols.append("sp_throws_right")
    starters = starters.merge(
        fg[["pitcher_name", "season"] + prior_cols],
        on=["pitcher_name", "season"],
        how="left",
    )
    return starters


def _attach_statcast_priors(starters: pd.DataFrame,
                            statcast_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Attach Statcast aggregates from the pitcher's prior start.

    `statcast_summary` should already be aggregated to (pitcher_id, game_date)
    granularity to keep this merge cheap. Computed in build_dataset.
    """
    if statcast_summary is None or statcast_summary.empty:
        starters["sp_prior_avg_ev"] = np.nan
        starters["sp_prior_whiff_rate"] = np.nan
        starters["sp_prior_called_strike_rate"] = np.nan
        return starters

    # Statcast uses MLBAM pitcher IDs; starters has xMLBAMID from FanGraphs join.
    # If xMLBAMID wasn't attached (no FanGraphs data), fall back gracefully.
    if "xMLBAMID" not in starters.columns:
        starters["sp_prior_avg_ev"] = np.nan
        starters["sp_prior_whiff_rate"] = np.nan
        starters["sp_prior_called_strike_rate"] = np.nan
        return starters

    starters = starters.merge(
        statcast_summary.rename(columns={"pitcher_id": "xMLBAMID",
                                         "game_date": "sc_game_date"}),
        left_on=["xMLBAMID", "date"],
        right_on=["xMLBAMID", "sc_game_date"],
        how="left",
    ).drop(columns=["sc_game_date"], errors="ignore")

    # Shift one start back per pitcher so features are from the *prior* start.
    starters = starters.sort_values(["pitcher_id", "date"])
    for col in ["avg_ev", "whiff_rate", "called_strike_rate"]:
        if col in starters.columns:
            starters[f"sp_prior_{col}"] = starters.groupby("pitcher_id")[col].shift(1)
            starters = starters.drop(columns=[col])
    return starters


def build(game_logs: pd.DataFrame,
          statcast_summary: pd.DataFrame | None = None,
          fg_pitcher: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a per-game DataFrame with home_sp_* and away_sp_* columns."""
    starters = _build_starter_records(game_logs)
    starters = _add_rolling(starters)
    starters = _attach_fangraphs_priors(starters, fg_pitcher)
    starters = _attach_statcast_priors(starters, statcast_summary)

    feature_cols = [c for c in starters.columns
                    if c.startswith("sp_") and c not in {"sp_id"}]
    home = (starters[starters["is_home"] == 1][["game_id"] + feature_cols]
            .rename(columns={c: f"home_{c}" for c in feature_cols}))
    away = (starters[starters["is_home"] == 0][["game_id"] + feature_cols]
            .rename(columns={c: f"away_{c}" for c in feature_cols}))

    out = game_logs[["game_id"]].drop_duplicates().merge(home, on="game_id", how="left") \
                                                 .merge(away, on="game_id", how="left")
    out["pitcher_confirmed"] = out[[c for c in out.columns if c.endswith("days_rest")]] \
        .notna().any(axis=1)
    return out


def aggregate_statcast(statcast_pitch: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast to (pitcher_id, game_date)."""
    if statcast_pitch is None or statcast_pitch.empty:
        return pd.DataFrame(columns=["pitcher_id", "game_date", "avg_ev", "whiff_rate",
                                     "called_strike_rate"])
    df = statcast_pitch.copy()
    df = df.rename(columns={"pitcher": "pitcher_id"})
    df["game_date"] = pd.to_datetime(df["game_date"])
    grouped = df.groupby(["pitcher_id", "game_date"])
    out = grouped.agg(
        avg_ev=("launch_speed", "mean"),
        swings=("description", lambda s: s.isin(
            {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
             "hit_into_play"}).sum()),
        whiffs=("description", lambda s: s.isin(
            {"swinging_strike", "swinging_strike_blocked"}).sum()),
        called_strikes=("description", lambda s: (s == "called_strike").sum()),
        pitches=("description", "count"),
    ).reset_index()
    out["whiff_rate"] = out["whiffs"] / out["swings"].replace(0, np.nan)
    out["called_strike_rate"] = out["called_strikes"] / out["pitches"].replace(0, np.nan)
    return out[["pitcher_id", "game_date", "avg_ev", "whiff_rate", "called_strike_rate"]]
