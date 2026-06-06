"""The Odds API client — live + historical lines.

Approved plan: historical pulls cover 2022+ only. 2015-2021 closing lines come
from sports-statistics.com (see sports_statistics.py).

The API returns `x-requests-remaining` / `x-requests-used` in response headers;
we log both on every call so the quota burn is visible.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, retry_if_exception

import config

logger = logging.getLogger(__name__)


class OddsApiError(RuntimeError):
    pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, OddsApiError) and str(exc).startswith("401"):
        return False   # auth errors are permanent — don't retry
    return isinstance(exc, (requests.RequestException, OddsApiError))


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _get(path: str, params: dict[str, Any]) -> tuple[Any, dict[str, str]]:
    if not config.ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set in .env")
    url = f"{config.ODDS_API_BASE}{path}"
    params = {**params, "apiKey": config.ODDS_API_KEY}
    resp = requests.get(url, params=params, timeout=30)
    quota_remaining = resp.headers.get("x-requests-remaining", "?")
    quota_used = resp.headers.get("x-requests-used", "?")
    logger.info("odds-api %s status=%d remaining=%s used=%s",
                path, resp.status_code, quota_remaining, quota_used)
    if resp.status_code == 401:
        # Auth / plan restriction — do not retry, surface immediately
        raise OddsApiError(f"401: {resp.text[:300]}")
    if resp.status_code == 429:
        raise OddsApiError("rate limited")
    if not resp.ok:
        raise OddsApiError(f"{resp.status_code}: {resp.text[:200]}")
    return resp.json(), dict(resp.headers)


def _flatten_event(event: dict[str, Any], snapshot_ts: str) -> list[dict[str, Any]]:
    """Flatten a single event into per-book/per-market rows."""
    rows: list[dict[str, Any]] = []
    base = {
        "snapshot_ts": snapshot_ts,
        "event_id": event.get("id"),
        "commence_time": event.get("commence_time"),
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
    }
    for book in event.get("bookmakers", []):
        book_key = book.get("key")
        if book_key not in config.BOOKMAKERS:
            continue
        last_update = book.get("last_update")
        for market in book.get("markets", []):
            m_key = market.get("key")
            for outcome in market.get("outcomes", []):
                rows.append({
                    **base,
                    "book": book_key,
                    "market": m_key,
                    "last_update": last_update,
                    "outcome_name": outcome.get("name"),
                    "price": outcome.get("price"),
                    "point": outcome.get("point"),
                })
    return rows


def fetch_live() -> pd.DataFrame:
    """Today's slate, all configured books and markets. Returns a DataFrame.

    Cached to data/raw/odds_live.parquet so predict.py can re-run hourly off a
    single fresh pull.
    """
    data, _ = _get(
        "/sports/baseball_mlb/odds",
        {
            "regions": "us",
            "markets": ",".join(config.MARKETS),
            "oddsFormat": "american",
            "bookmakers": ",".join(config.BOOKMAKERS),
        },
    )
    snapshot_ts = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for event in data:
        rows.extend(_flatten_event(event, snapshot_ts))
    df = pd.DataFrame(rows)
    out = config.RAW_DIR / "odds_live.parquet"
    df.to_parquet(out, index=False)
    logger.info("wrote %d live odds rows to %s", len(df), out)
    return df


def _closing_iso(commence_time: str, lookback_min: int = 60) -> str:
    """Snapshot timestamp `lookback_min` before first pitch."""
    t = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
    return (t - timedelta(minutes=lookback_min)).isoformat()


def fetch_historical(years: Iterable[int]) -> pd.DataFrame:
    """Pull closing lines (~1h before first pitch) for each game in the given years.

    Cached year-by-year under data/raw/odds_YYYY.parquet. Idempotent.
    """
    frames: list[pd.DataFrame] = []
    for year in years:
        if year < config.ODDS_API_HISTORICAL_START_YEAR:
            logger.warning("year %d below ODDS_API_HISTORICAL_START_YEAR; use sports_statistics.py", year)
            continue
        out_path = config.RAW_DIR / f"odds_{year}.parquet"
        if out_path.exists():
            logger.info("loading cached %s", out_path)
            frames.append(pd.read_parquet(out_path))
            continue
        year_rows = _pull_year_historical(year)
        df = pd.DataFrame(year_rows)
        df.to_parquet(out_path, index=False)
        logger.info("wrote %d historical odds rows for %d to %s", len(df), year, out_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pull_year_historical(year: int) -> list[dict[str, Any]]:
    """For each day in the regular-season window, fetch the closing snapshot."""
    rows: list[dict[str, Any]] = []
    season_start = datetime(year, 3, 20, tzinfo=timezone.utc)
    season_end = datetime(year, 10, 5, tzinfo=timezone.utc)
    cursor = season_start
    while cursor <= season_end:
        ts_iso = cursor.isoformat()
        try:
            data, _ = _get(
                "/historical/sports/baseball_mlb/odds",
                {
                    "regions": "us",
                    "markets": ",".join(config.MARKETS),
                    "oddsFormat": "american",
                    "bookmakers": ",".join(config.BOOKMAKERS),
                    "date": ts_iso,
                },
            )
        except OddsApiError as exc:
            logger.warning("historical pull failed for %s: %s", ts_iso, exc)
            cursor += timedelta(days=1)
            continue
        for event in data.get("data", []):
            rows.extend(_flatten_event(event, snapshot_ts=ts_iso))
        cursor += timedelta(days=1)
        time.sleep(0.2)                  # gentle pacing
    return rows


def fetch_historical_dates(dates: list[date_type], cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch closing-line snapshots for specific calendar dates.

    Uses two snapshots per day (16:00 UTC ≈ noon ET and 23:00 UTC ≈ 7pm ET)
    so both afternoon and evening first pitches are covered.  Keeps the
    latest snapshot available for each event (closest to first pitch).

    Credit cost: 2 per date (14 total for a 7-day window).
    """
    from datetime import date as date_type_cls
    if cache_dir is None:
        cache_dir = config.RAW_DIR / "odds_historical_dates"
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for d in dates:
        d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
        cache_path = cache_dir / f"odds_{d_str}.parquet"
        if cache_path.exists():
            logger.info("loading cached historical odds for %s", d_str)
            all_rows.extend(pd.read_parquet(cache_path).to_dict("records"))
            continue

        day_rows: list[dict[str, Any]] = []
        # Two snapshots: 16:00 UTC (noon ET) and 23:00 UTC (7pm ET)
        for hour_utc in (16, 23):
            ts = f"{d_str}T{hour_utc:02d}:00:00Z"
            try:
                data, _ = _get(
                    "/historical/sports/baseball_mlb/odds",
                    {
                        "regions": "us",
                        "markets": "h2h",
                        "oddsFormat": "american",
                        "bookmakers": ",".join(config.BOOKMAKERS),
                        "date": ts,
                    },
                )
                for event in (data or {}).get("data", []):
                    day_rows.extend(_flatten_event(event, snapshot_ts=ts))
                logger.info("fetched %d rows for %s @%02d:00Z", len(day_rows), d_str, hour_utc)
            except OddsApiError as exc:
                logger.warning("historical pull failed for %s @%02dZ: %s", d_str, hour_utc, exc)
                if "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN" in str(exc):
                    logger.warning("Historical odds require a paid Odds API plan. Skipping remaining dates.")
                    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

        if day_rows:
            df_day = pd.DataFrame(day_rows)
            # Keep latest snapshot per (event_id, book, market, outcome_name)
            df_day = (df_day.sort_values("snapshot_ts")
                            .drop_duplicates(subset=["event_id", "book", "market", "outcome_name"],
                                             keep="last"))
            df_day.to_parquet(cache_path, index=False)
            all_rows.extend(df_day.to_dict("records"))

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", action="store_true",
                        help="Pull 2022..2024 closing lines instead of today's slate.")
    args = parser.parse_args()
    if args.historical:
        fetch_historical(years=range(config.ODDS_API_HISTORICAL_START_YEAR, 2025))
    else:
        fetch_live()
