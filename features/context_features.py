"""Context features: park factor, rest days, season progress, weather.

Weather endpoint selection:
 * Historical games  → archive-api.open-meteo.com/v1/archive  (ERA5 reanalysis)
 * Upcoming games    → api.open-meteo.com/v1/forecast

The two endpoints share the same parameter naming.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import config

logger = logging.getLogger(__name__)

PARK_FACTORS_PATH: Path = Path(__file__).parent / "park_factors.csv"


def load_park_factors() -> pd.DataFrame:
    return pd.read_csv(PARK_FACTORS_PATH)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
    reraise=True,
)
def _weather_at(lat: float, lon: float, when: datetime) -> tuple[float | None, float | None]:
    """Return (temperature_c, windspeed_kmh) for the given lat/lon/datetime.

    Picks the forecast endpoint if `when` is in the future, archive endpoint
    otherwise.
    """
    now = datetime.utcnow()
    historical = when < now
    base = config.OPEN_METEO_ARCHIVE if historical else config.OPEN_METEO_FORECAST
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m",
        "start_date": when.strftime("%Y-%m-%d"),
        "end_date": when.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    resp = requests.get(base, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("hourly", {})
    hours = data.get("time", [])
    if not hours:
        return (None, None)
    target_hour = when.strftime("%Y-%m-%dT%H:00")
    try:
        idx = hours.index(target_hour)
    except ValueError:
        idx = min(range(len(hours)), key=lambda i: abs(
            datetime.fromisoformat(hours[i]).timestamp() - when.timestamp()))
    return (
        data.get("temperature_2m", [None])[idx],
        data.get("windspeed_10m", [None])[idx],
    )


def _rest_days(game_df: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous game (regardless of home/away)."""
    long = pd.concat([
        game_df[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"}),
        game_df[["game_id", "date", "visiting_team"]].rename(columns={"visiting_team": "team"}),
    ])
    long = long.sort_values(["team", "date"])
    long["rest_days"] = long.groupby("team")["date"].diff().dt.days
    long["rest_days"] = long["rest_days"].fillna(4)  # season opener default
    out = game_df.merge(
        long.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})[
            ["game_id", "home_team", "home_rest_days"]],
        on=["game_id", "home_team"], how="left")
    out = out.merge(
        long.rename(columns={"team": "visiting_team", "rest_days": "away_rest_days"})[
            ["game_id", "visiting_team", "away_rest_days"]],
        on=["game_id", "visiting_team"], how="left")
    return out


def _park_factor(game_df: pd.DataFrame) -> pd.DataFrame:
    parks = load_park_factors().rename(columns={"team": "home_team"})
    return game_df.merge(parks[["home_team", "run_factor", "lat", "lon"]],
                         on="home_team", how="left")


def _season_progress(game_df: pd.DataFrame) -> pd.DataFrame:
    # cumcount() gives 0-based index of each home game within the season.
    # We want games played *before* this game (leakage-safe), so no +1.
    game_df = game_df.copy()
    game_df["season_game_num"] = (game_df.sort_values("date")
                                  .groupby(["home_team", "season"])
                                  .cumcount())
    game_df["season_progress"] = game_df["season_game_num"] / 162.0
    return game_df


def build(game_df: pd.DataFrame, *, fetch_weather: bool = False) -> pd.DataFrame:
    """Attach context features. Weather is opt-in because it's slow."""
    out = _park_factor(game_df)
    out = _rest_days(out)
    out = _season_progress(out)
    out["is_day_game"] = (out.get("day_night", "").astype(str).str.upper() == "D").astype(int)
    if fetch_weather:
        temps, winds = [], []
        for _, row in out.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                temps.append(None); winds.append(None); continue
            when = pd.Timestamp(row["date"]).to_pydatetime()
            try:
                t, w = _weather_at(row["lat"], row["lon"], when)
            except Exception as exc:  # noqa: BLE001
                logger.warning("weather lookup failed for %s: %s", row["game_id"], exc)
                t, w = (None, None)
            temps.append(t); winds.append(w)
        out["temperature_c"] = temps
        out["windspeed_kmh"] = winds
    return out
