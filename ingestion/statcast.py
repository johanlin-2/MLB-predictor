"""Pitch-level Statcast 2015-present via pybaseball.

Pulled in monthly chunks to bound memory. Partitioned on disk as
    data/raw/statcast/YYYY/MM.parquet

so downstream code can load just the months it needs without ever
materialising the full ~7 M-row corpus.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
from pybaseball import statcast

import config

logger = logging.getLogger(__name__)

STATCAST_DIR: Path = config.RAW_DIR / "statcast"


def _month_path(year: int, month: int) -> Path:
    return STATCAST_DIR / f"{year:04d}" / f"{month:02d}.parquet"


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last = next_first.toordinal() - 1
    end = date.fromordinal(last)
    return (date(year, month, 1).isoformat(), end.isoformat())


# Regular-season months only (March-October). Statcast has no postseason data
# we need for training fits.
SEASON_MONTHS = (3, 4, 5, 6, 7, 8, 9, 10)


def fetch_statcast(years: Iterable[int] = config.ALL_YEARS, force: bool = False) -> None:
    """Pull Statcast pitch-level data for each (year, month) chunk.

    Idempotent: skips chunks whose parquet already exists unless force=True.
    """
    for year in years:
        for month in SEASON_MONTHS:
            path = _month_path(year, month)
            if path.exists() and not force:
                continue
            start, end = _month_bounds(year, month)
            logger.info("statcast %s..%s", start, end)
            try:
                df = statcast(start_dt=start, end_dt=end)
            except Exception as exc:  # noqa: BLE001
                logger.warning("statcast pull failed %s..%s: %s", start, end, exc)
                continue
            if df is None or df.empty:
                logger.info("no statcast rows for %s..%s — skipping", start, end)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=False)
            logger.info("wrote %d statcast rows to %s", len(df), path)


def load_range(start: str, end: str) -> pd.DataFrame:
    """Load all cached Statcast rows in [start, end] inclusive."""
    start_d = pd.Timestamp(start)
    end_d = pd.Timestamp(end)
    frames: list[pd.DataFrame] = []
    for year_dir in sorted(STATCAST_DIR.glob("*")):
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            df = pd.read_parquet(path, columns=None)
            if "game_date" in df.columns:
                df["game_date"] = pd.to_datetime(df["game_date"])
                df = df[(df["game_date"] >= start_d) & (df["game_date"] <= end_d)]
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_statcast()
