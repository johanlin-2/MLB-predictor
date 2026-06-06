"""Retrosheet game logs 2015-2024 via direct Chadwick Bureau raw CSV download.

Establishes the canonical `game_id` schema used by every downstream module:
    game_id = f"{date_iso}_{away}_{home}_{double_header_index}"

Only the regular season is included for training (ambiguity 8). Suspended
games are dropped (ambiguity 9). 2020 is excluded by the dataset builder, not
here — we keep the raw data complete.

Fetches directly from raw.githubusercontent.com to avoid the GH_TOKEN
requirement in pybaseball.retrosheet.season_game_logs.
"""
from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

RAW_PATH: Path = config.RAW_DIR / "game_logs.parquet"

# Chadwick Bureau hosts Retrosheet gamelogs on GitHub.
_GAMELOG_URL = (
    "https://raw.githubusercontent.com/chadwickbureau/retrosheet/master/seasons/{year}/GL{year}.TXT"
)

# Full column list from pybaseball.retrosheet.gamelog_columns (161 columns).
_GAMELOG_COLUMNS = [
    "date", "game_num", "day_of_week", "visiting_team",
    "visiting_team_league", "visiting_team_game_num", "home_team",
    "home_team_league", "home_team_game_num", "visiting_score",
    "home_score", "num_outs", "day_night", "completion_info",
    "forfeit_info", "protest_info", "park_id", "attendance",
    "time_of_game_minutes", "visiting_line_score",
    "home_line_score", "visiting_abs", "visiting_hits",
    "visiting_doubles", "visiting_triples", "visiting_homeruns",
    "visiting_rbi", "visiting_sac_hits", "visiting_sac_flies",
    "visiting_hbp", "visiting_bb", "visiting_iw", "visiting_k",
    "visiting_sb", "visiting_cs", "visiting_gdp", "visiting_ci",
    "visiting_lob", "visiting_pitchers_used",
    "visiting_individual_er", "visiting_er", "visiting_wp",
    "visiting_balks", "visiting_po", "visiting_assists",
    "visiting_errors", "visiting_pb", "visiting_dp",
    "visiting_tp", "home_abs", "home_hits", "home_doubles",
    "home_triples", "home_homeruns", "home_rbi",
    "home_sac_hits", "home_sac_flies", "home_hbp", "home_bb",
    "home_iw", "home_k", "home_sb", "home_cs", "home_gdp",
    "home_ci", "home_lob", "home_pitchers_used",
    "home_individual_er", "home_er", "home_wp", "home_balks",
    "home_po", "home_assists", "home_errors", "home_pb",
    "home_dp", "home_tp", "ump_home_id", "ump_home_name",
    "ump_first_id", "ump_first_name", "ump_second_id",
    "ump_second_name", "ump_third_id", "ump_third_name",
    "ump_lf_id", "ump_lf_name", "ump_rf_id", "ump_rf_name",
    "visiting_manager_id", "visiting_manager_name",
    "home_manager_id", "home_manager_name",
    "winning_pitcher_id", "winning_pitcher_name",
    "losing_pitcher_id", "losing_pitcher_name",
    "save_pitcher_id", "save_pitcher_name",
    "game_winning_rbi_id", "game_winning_rbi_name",
    "visiting_starting_pitcher_id",
    "visiting_starting_pitcher_name",
    "home_starting_pitcher_id", "home_starting_pitcher_name",
    "visiting_1_id", "visiting_1_name", "visiting_1_pos",
    "visiting_2_id", "visiting_2_name", "visiting_2_pos",
    "visiting_2_id.1", "visiting_3_name", "visiting_3_pos",
    "visiting_4_id", "visiting_4_name", "visiting_4_pos",
    "visiting_5_id", "visiting_5_name", "visiting_5_pos",
    "visiting_6_id", "visiting_6_name", "visiting_6_pos",
    "visiting_7_id", "visiting_7_name", "visiting_7_pos",
    "visiting_8_id", "visiting_8_name", "visiting_8_pos",
    "visiting_9_id", "visiting_9_name", "visiting_9_pos",
    "home_1_id", "home_1_name", "home_1_pos", "home_2_id",
    "home_2_name", "home_2_pos", "home_3_id", "home_3_name",
    "home_3_pos", "home_4_id", "home_4_name", "home_4_pos",
    "home_5_id", "home_5_name", "home_5_pos", "home_6_id",
    "home_6_name", "home_6_pos", "home_7_id", "home_7_name",
    "home_7_pos", "home_8_id", "home_8_name", "home_8_pos",
    "home_9_id", "home_9_name", "home_9_pos", "misc",
    "acquisition_info",
]

# Columns we actually need downstream (mapped to our naming convention).
# visiting_score → visitor_score for compatibility with team_features.py.
KEEP_COLS = [
    "date",
    "game_num",         # used as double_header proxy
    "visiting_team",
    "home_team",
    "visiting_score",   # renamed → visitor_score after load
    "home_score",
    "day_of_week",
    "park_id",
    "attendance",
    "time_of_game_minutes",
    "day_night",
    "completion_info",
    "forfeit_info",
    "visiting_starting_pitcher_id",
    "visiting_starting_pitcher_name",
    "home_starting_pitcher_id",
    "home_starting_pitcher_name",
    "visiting_pitchers_used",
    "home_pitchers_used",
]


def _fetch_year(year: int) -> pd.DataFrame:
    """Download one year's gamelog TXT and return a raw DataFrame."""
    url = _GAMELOG_URL.format(year=year)
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 3:
                raise
            wait = 2 ** attempt
            logger.warning("attempt %d failed for %d (%s) — retrying in %ds", attempt + 1, year, exc, wait)
            time.sleep(wait)

    # The TXT has no header; assign column names. Trim to actual column count
    # in case a year has fewer fields (older seasons).
    df = pd.read_csv(
        StringIO(resp.text),
        header=None,
        low_memory=False,
    )
    n_cols = min(len(df.columns), len(_GAMELOG_COLUMNS))
    df.columns = _GAMELOG_COLUMNS[:n_cols]
    return df


def _build_game_id(row: pd.Series) -> str:
    """game_id = date_AWAY_HOME_dh-index."""
    date_iso = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
    dh = int(row.get("double_header", row.get("game_num", 0)) or 0)
    return f"{date_iso}_{row['visiting_team']}_{row['home_team']}_{dh}"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Retrosheet dates are YYYYMMDD integers; handle both int and already-parsed forms.
    df["date"] = pd.to_datetime(df["date"].astype(str), format="mixed", errors="coerce")
    df = df.dropna(subset=["date"])

    # Rename for downstream compatibility.
    df = df.rename(columns={
        "visiting_score": "visitor_score",
        "time_of_game_minutes": "game_duration",
        "visiting_starting_pitcher_id": "visitor_starting_pitcher_id",
        "visiting_starting_pitcher_name": "visitor_starting_pitcher_name",
        "visiting_pitchers_used": "visitor_pitchers_used",
        "game_num": "double_header",
    })
    # double_header: 0 = single game, 1 = first of DH, 2 = second of DH.
    # Keep as-is; game_id uses it as the suffix index.
    df["double_header"] = pd.to_numeric(df["double_header"], errors="coerce").fillna(0).astype(int)

    df["game_id"] = df.apply(_build_game_id, axis=1)
    df["home_win"] = (df["home_score"] > df["visitor_score"]).astype(int)
    df["home_covered"] = (df["home_score"] - df["visitor_score"] >= 2).astype(int)
    df["total_runs"] = df["home_score"] + df["visitor_score"]
    df["run_diff_home"] = df["home_score"] - df["visitor_score"]
    return df


def _drop_suspended(df: pd.DataFrame) -> pd.DataFrame:
    """Drop games that were suspended (ambiguity 9)."""
    if "completion_info" in df.columns:
        mask = df["completion_info"].fillna("").str.len() == 0
        dropped = (~mask).sum()
        if dropped:
            logger.info("dropping %d suspended/forfeited game(s)", dropped)
        df = df[mask]
    if "forfeit_info" in df.columns:
        df = df[df["forfeit_info"].fillna("").str.len() == 0]
    return df


def fetch_game_logs(seasons: Iterable[int] = range(2015, 2025), force: bool = False) -> pd.DataFrame:
    """Pull game logs for the given seasons. Cached to RAW_PATH.

    Args:
        seasons: iterable of years to pull.
        force: re-download even if the parquet cache exists.

    Returns:
        DataFrame keyed on ``game_id``.
    """
    if RAW_PATH.exists() and not force:
        logger.info("loading cached game logs from %s", RAW_PATH)
        return pd.read_parquet(RAW_PATH)

    frames: list[pd.DataFrame] = []
    for year in seasons:
        logger.info("fetching retrosheet game logs for %d", year)
        try:
            raw = _fetch_year(year)
        except Exception as exc:  # noqa: BLE001
            logger.warning("retrosheet pull failed for %d: %s", year, exc)
            continue
        keep = [c for c in KEEP_COLS if c in raw.columns]
        df = raw[keep].copy()
        df["season"] = year
        frames.append(df)

    if not frames:
        raise RuntimeError("no game logs fetched — check network or season range")

    out = pd.concat(frames, ignore_index=True)
    out = _normalise(out)
    out = _drop_suspended(out)
    out = out.drop_duplicates(subset=["game_id"]).sort_values("date").reset_index(drop=True)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_PATH, index=False)
    logger.info("wrote %d games to %s", len(out), RAW_PATH)
    return out


def load() -> pd.DataFrame:
    """Convenience: load the cached parquet, fetching if absent."""
    if not RAW_PATH.exists():
        return fetch_game_logs()
    return pd.read_parquet(RAW_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_game_logs()
