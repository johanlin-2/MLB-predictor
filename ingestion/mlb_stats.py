"""Live 2026 team and pitcher stats from the MLB Stats API.

Used by pipeline.predict to replace stale 2023 rolling features with
current-season actuals when building live feature rows for today's slate.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 15

_MLB_TO_RETRO: dict[str, str] = {
    "ATH": "OAK", "AZ": "ARI", "CHC": "CHN", "CWS": "CHA",
    "KC": "KCA", "LAA": "ANA", "LAD": "LAN", "NYM": "NYN",
    "NYY": "NYA", "SD": "SDN", "SF": "SFN", "STL": "SLN",
    "TB": "TBA", "WSH": "WAS",
}


def fetch_team_season_stats(season: int = 2026) -> dict[str, dict]:
    """Return per-team season stats keyed by Retrosheet 3-letter code.

    Pulls from the standings endpoint which includes runsScored, runsAllowed,
    wins, losses, and gamesPlayed — enough to compute all rolling proxies.

    Returns:
        {
          "NYA": {
            "win_pct": 0.597,
            "runs_per_game": 5.23,
            "runs_allowed_per_game": 4.48,
            "run_diff_per_game": 0.75,
            "games_played": 62,
          }, ...
        }
    """
    try:
        resp = requests.get(
            f"{_BASE}/standings",
            params={"leagueId": "103,104", "season": season,
                    "standingsType": "regularSeason", "hydrate": "team"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("MLB standings fetch failed: %s", exc)
        return {}

    stats: dict[str, dict] = {}
    for division in resp.json().get("records", []):
        for rec in division.get("teamRecords", []):
            abbr = rec["team"]["abbreviation"]
            retro = _MLB_TO_RETRO.get(abbr, abbr)
            gp = max(rec.get("gamesPlayed", 1), 1)
            rs = rec.get("runsScored", 0) or 0
            ra = rec.get("runsAllowed", 0) or 0
            w = rec.get("wins", 0) or 0
            l = rec.get("losses", 0) or 0
            total = max(w + l, 1)
            stats[retro] = {
                "win_pct": w / total,
                "runs_per_game": rs / gp,
                "runs_allowed_per_game": ra / gp,
                "run_diff_per_game": (rs - ra) / gp,
                "games_played": gp,
            }

    logger.info("fetched 2026 team stats for %d teams", len(stats))
    return stats


def fetch_pitcher_season_stats(player_id: str | int, season: int = 2026) -> dict:
    """Return current-season pitching stats for a single player.

    Returns a flat dict with keys matching the feature column names used by
    pitcher_features.py (era, whip, k9, bb9, runs_per_start).
    Returns {} on failure.
    """
    if not player_id:
        return {}
    try:
        resp = requests.get(
            f"{_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": "pitching", "season": season},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception as exc:
        logger.debug("pitcher stats fetch failed for %s: %s", player_id, exc)
        return {}

    if not splits:
        return {}

    s = splits[0]["stat"]
    gs = max(float(s.get("gamesStarted") or 1), 1)
    runs = float(s.get("runs") or 0)
    try:
        era = float(s.get("era") or 0)
        whip = float(s.get("whip") or 0)
        k9 = float(s.get("strikeoutsPer9Inn") or 0)
        bb9 = float(s.get("walksPer9Inn") or 0)
    except (TypeError, ValueError):
        return {}

    return {
        "era": era,
        "whip": whip,
        "k9": k9,
        "bb9": bb9,
        "runs_per_start": runs / gs,
    }
