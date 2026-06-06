"""FanGraphs team-level pitching and batting stats via direct API.

Uses the FanGraphs major-league leaderboards JSON API (not pybaseball, which
hits the legacy HTML endpoint that now returns 403). Data is pulled once per
season at the team level and cached to data/raw/ as Parquet.

We do NOT use season totals as model features directly (season-to-date stats
accumulate → leakage risk). They are used only as prior-season priors in
team_features._add_fangraphs_priors and as per-pitcher season aggregates in
pitcher_features.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

TEAM_PITCHING_PATH: Path = config.RAW_DIR / "fangraphs_team_pitching.parquet"
TEAM_BATTING_PATH: Path = config.RAW_DIR / "fangraphs_team_batting.parquet"

_BASE = "https://www.fangraphs.com/api/leaders/major-league/data"

# FanGraphs → Retrosheet 3-letter team code mapping.
FG_TO_RETRO: dict[str, str] = {
    "LAA": "ANA",
    "CHW": "CHA",
    "CHC": "CHN",
    "KCR": "KCA",
    "LAD": "LAN",
    "NYY": "NYA",
    "NYM": "NYN",
    "SDP": "SDN",
    "SFG": "SFN",
    "STL": "SLN",
    "TBR": "TBA",
    "WSN": "WAS",
}

# Columns we need from team batting (used as prior-season features).
# TeamNameAbb is the clean abbreviation; Team contains an HTML anchor.
_BAT_KEEP = ["TeamNameAbb", "Season", "wOBA", "wRC+", "OBP", "SLG", "ISO", "K%", "BB%", "BABIP"]
# Columns we need from team pitching.
_PIT_KEEP = ["TeamNameAbb", "Season", "ERA", "FIP", "xFIP", "WHIP", "HR/9", "K/9", "BB/9", "K%", "BB%"]
# Pitcher-level columns used by pitcher_features.py
_SP_KEEP = [
    "Name", "PlayerName", "TeamNameAbb", "Season", "playerid", "xMLBAMID",
    "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%",
    "IP", "GS", "G", "Throws",
]


def _get(params: dict, retries: int = 4) -> list[dict]:
    """GET the FanGraphs leaderboards API with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(_BASE, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()["data"]
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("FanGraphs request failed (%s) — retrying in %ds", exc, wait)
            time.sleep(wait)
    return []


def _fetch_season(year: int, stats: str) -> pd.DataFrame:
    """Fetch one season's team-level leaderboard (stats='bat' or 'pit')."""
    params = {
        "age": 0,
        "pos": "all",
        "stats": stats,
        "lg": "all",
        "qual": 0,
        "season": year,
        "season1": year,
        "startdate": "",
        "enddate": "",
        "month": 0,
        "hand": "",
        "team": "0,ts",   # team totals
        "pageitems": 2000,
        "pagenum": 1,
        "ind": 0,
        "rost": "",
        "players": "",
        "type": 8,
        "postseason": "",
        "sortdir": "default",
        "sortstat": "WAR",
    }
    rows = _get(params)
    df = pd.DataFrame(rows)
    df["Season"] = year
    return df


def _fetch_season_individual(year: int, stats: str) -> pd.DataFrame:
    """Fetch individual pitcher/batter stats for one season (team=0 for all players)."""
    params = {
        "age": 0,
        "pos": "all",
        "stats": stats,
        "lg": "all",
        "qual": 0,
        "season": year,
        "season1": year,
        "startdate": "",
        "enddate": "",
        "month": 0,
        "hand": "",
        "team": 0,
        "pageitems": 2000,
        "pagenum": 1,
        "ind": 0,
        "rost": "",
        "players": "",
        "type": 8,
        "postseason": "",
        "sortdir": "default",
        "sortstat": "WAR",
    }
    rows = _get(params)
    df = pd.DataFrame(rows)
    df["Season"] = year
    return df


def fetch_team_pitching(seasons: Iterable[int] = config.ALL_YEARS, force: bool = False) -> pd.DataFrame:
    """Pull FanGraphs team pitching for each season. Cached to TEAM_PITCHING_PATH."""
    if TEAM_PITCHING_PATH.exists() and not force:
        logger.info("loading cached team pitching from %s", TEAM_PITCHING_PATH)
        return pd.read_parquet(TEAM_PITCHING_PATH)

    frames = []
    for year in seasons:
        logger.info("fetching FanGraphs team pitching for %d", year)
        try:
            df = _fetch_season(year, "pit")
        except Exception as exc:  # noqa: BLE001
            logger.warning("team_pitching failed for %d: %s", year, exc)
            continue
        keep = [c for c in _PIT_KEEP if c in df.columns]
        frames.append(df[keep])

    if not frames:
        raise RuntimeError("no FanGraphs pitching data fetched")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"TeamNameAbb": "Team"})
    out["Team"] = out["Team"].replace(FG_TO_RETRO)
    out.to_parquet(TEAM_PITCHING_PATH, index=False)
    logger.info("wrote %d rows to %s", len(out), TEAM_PITCHING_PATH)
    return out


def fetch_team_batting(seasons: Iterable[int] = config.ALL_YEARS, force: bool = False) -> pd.DataFrame:
    """Pull FanGraphs team batting for each season. Cached to TEAM_BATTING_PATH."""
    if TEAM_BATTING_PATH.exists() and not force:
        logger.info("loading cached team batting from %s", TEAM_BATTING_PATH)
        return pd.read_parquet(TEAM_BATTING_PATH)

    frames = []
    for year in seasons:
        logger.info("fetching FanGraphs team batting for %d", year)
        try:
            df = _fetch_season(year, "bat")
        except Exception as exc:  # noqa: BLE001
            logger.warning("team_batting failed for %d: %s", year, exc)
            continue
        keep = [c for c in _BAT_KEEP if c in df.columns]
        frames.append(df[keep])

    if not frames:
        raise RuntimeError("no FanGraphs batting data fetched")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"TeamNameAbb": "Team"})
    out["Team"] = out["Team"].replace(FG_TO_RETRO)
    out.to_parquet(TEAM_BATTING_PATH, index=False)
    logger.info("wrote %d rows to %s", len(out), TEAM_BATTING_PATH)
    return out


def fetch_pitcher_season_stats(seasons: Iterable[int] = config.ALL_YEARS, force: bool = False) -> pd.DataFrame:
    """Per-pitcher season aggregates. Used as priors in pitcher_features."""
    path = config.RAW_DIR / "fangraphs_pitcher_season.parquet"
    if path.exists() and not force:
        logger.info("loading cached pitcher season stats from %s", path)
        return pd.read_parquet(path)

    frames = []
    for year in seasons:
        logger.info("fetching FanGraphs pitcher season stats for %d", year)
        try:
            df = _fetch_season_individual(year, "pit")
        except Exception as exc:  # noqa: BLE001
            logger.warning("pitcher_season_stats failed for %d: %s", year, exc)
            continue
        keep = [c for c in _SP_KEEP if c in df.columns]
        frames.append(df[keep])

    if not frames:
        raise RuntimeError("no FanGraphs pitcher season data fetched")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"TeamNameAbb": "Team"})
    out["Team"] = out["Team"].replace(FG_TO_RETRO)
    out.to_parquet(path, index=False)
    logger.info("wrote %d rows to %s", len(out), path)
    return out


def fetch_batter_season_stats(seasons: Iterable[int] = config.ALL_YEARS, force: bool = False) -> pd.DataFrame:
    """Per-batter season aggregates."""
    path = config.RAW_DIR / "fangraphs_batter_season.parquet"
    if path.exists() and not force:
        logger.info("loading cached batter season stats from %s", path)
        return pd.read_parquet(path)

    frames = []
    for year in seasons:
        logger.info("fetching FanGraphs batter season stats for %d", year)
        try:
            df = _fetch_season_individual(year, "bat")
        except Exception as exc:  # noqa: BLE001
            logger.warning("batter_season_stats failed for %d: %s", year, exc)
            continue
        bat_keep = [c for c in ["Name", "TeamNameAbb", "Season", "playerid", "xMLBAMID",
                                 "wOBA", "wRC+", "OBP", "SLG", "ISO", "K%", "BB%",
                                 "BABIP", "PA", "G"] if c in df.columns]
        frames.append(df[bat_keep])

    if not frames:
        raise RuntimeError("no FanGraphs batter season data fetched")

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"TeamNameAbb": "Team"})
    out["Team"] = out["Team"].replace(FG_TO_RETRO)
    out.to_parquet(path, index=False)
    logger.info("wrote %d rows to %s", len(out), path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_team_pitching()
    fetch_team_batting()
    fetch_pitcher_season_stats()
    fetch_batter_season_stats()
