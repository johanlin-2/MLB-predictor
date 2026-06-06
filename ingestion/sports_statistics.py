"""sports-statistics.com historical odds CSV loader for 2015-2021.

The site publishes per-season CSVs with opening + closing moneyline, runline,
and totals. We use Pinnacle's closing line as the gold-standard no-vig
benchmark — they keep the tightest market in baseball.

Drop the raw CSVs into data/raw/sports_statistics/mlb_YYYY.csv. The loader is
idempotent and produces a single tidy parquet shaped to match odds_api output
so downstream code can union the two.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)

CSV_DIR: Path = config.RAW_DIR / "sports_statistics"
OUT_PATH: Path = config.RAW_DIR / "odds_historical_pre2022.parquet"


def _normalise(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Reshape a sports-statistics CSV into the canonical schema."""
    # The CSVs publish each game across two rows (one per team) with shared
    # rotation numbers. Re-pair them.
    df = df.rename(columns={c: c.strip().lower().replace(" ", "_") for c in df.columns})
    if "date" not in df.columns or "team" not in df.columns:
        raise ValueError(f"unexpected schema in {year} csv: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    games: list[dict] = []
    for _, group in df.groupby(["date", "rot"] if "rot" in df.columns else ["date"]):
        if len(group) != 2:
            continue
        away_row, home_row = group.iloc[0], group.iloc[1]
        if "vh" in group.columns:
            away_row = group[group["vh"].str.upper() == "V"].iloc[0]
            home_row = group[group["vh"].str.upper() == "H"].iloc[0]
        commence = pd.Timestamp(away_row["date"]).isoformat()
        snapshot = commence  # closing line snapshot ≈ game date
        base = {
            "snapshot_ts": snapshot,
            "event_id": f"ss_{year}_{away_row.get('rot', '')}",
            "commence_time": commence,
            "home_team": home_row["team"],
            "away_team": away_row["team"],
            "book": "pinnacle",
        }
        # h2h
        if "close" in away_row.index:
            games.append({**base, "market": "h2h", "outcome_name": away_row["team"],
                          "price": _to_int(away_row.get("close")), "point": None,
                          "last_update": snapshot})
            games.append({**base, "market": "h2h", "outcome_name": home_row["team"],
                          "price": _to_int(home_row.get("close")), "point": None,
                          "last_update": snapshot})
        # runline (uses -1.5/+1.5 convention)
        if "run_line" in away_row.index:
            games.append({**base, "market": "spreads", "outcome_name": away_row["team"],
                          "price": _to_int(_split_runline(away_row.get("run_line"))[1]),
                          "point": _to_float(_split_runline(away_row.get("run_line"))[0]),
                          "last_update": snapshot})
            games.append({**base, "market": "spreads", "outcome_name": home_row["team"],
                          "price": _to_int(_split_runline(home_row.get("run_line"))[1]),
                          "point": _to_float(_split_runline(home_row.get("run_line"))[0]),
                          "last_update": snapshot})
        # total
        if "total" in away_row.index:
            total_pt = _to_float(_split_total(away_row.get("total"))[0])
            games.append({**base, "market": "totals", "outcome_name": "Over",
                          "price": _to_int(_split_total(away_row.get("total"))[1]),
                          "point": total_pt, "last_update": snapshot})
            games.append({**base, "market": "totals", "outcome_name": "Under",
                          "price": _to_int(_split_total(home_row.get("total"))[1]),
                          "point": total_pt, "last_update": snapshot})
    return pd.DataFrame(games)


def _to_int(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_runline(value):
    """Parse strings like '-1.5 +160'."""
    if value is None or pd.isna(value):
        return (None, None)
    parts = str(value).split()
    if len(parts) != 2:
        return (None, None)
    return parts[0], parts[1]


def _split_total(value):
    """Parse strings like '8.5 -110'."""
    return _split_runline(value)


def build_historical(force: bool = False) -> pd.DataFrame:
    if OUT_PATH.exists() and not force:
        return pd.read_parquet(OUT_PATH)
    frames: list[pd.DataFrame] = []
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for year in range(config.SPORTS_STATISTICS_RANGE[0], config.SPORTS_STATISTICS_RANGE[1] + 1):
        path = CSV_DIR / f"mlb_{year}.csv"
        if not path.exists():
            logger.warning("missing %s — download from sports-statistics.com and rerun", path)
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read %s: %s", path, exc)
            continue
        frames.append(_normalise(df, year))
    if not frames:
        logger.warning("no sports-statistics CSVs found in %s", CSV_DIR)
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("wrote %d historical (pre-2022) odds rows to %s", len(out), OUT_PATH)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_historical()
