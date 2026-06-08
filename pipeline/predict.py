"""Daily prediction pipeline.

Re-runs hourly until first pitch (option `c`). Each invocation:
 1. Pulls live odds (Odds API)
 2. Pulls live Polymarket + Kalshi markets
 3. Builds feature rows for today's slate using the same pipeline that built
    the training set — same leakage discipline applies.
 4. Predicts win/loss, score (total + spread), runline cover.
 5. Computes edge vs. each book, consensus, Polymarket, Kalshi.
 6. Applies the per-model calibration gate.
 7. Writes output/picks/picks_YYYY-MM-DD.csv and prints flagged rows.

Schedule via cron:
    0 9-23/1 * * *  cd /path/to/mlb-predictor && python -m pipeline.predict --date today
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import requests

import config
from edge import edge_calculator, kelly
from edge.kelly import american_to_decimal
from features import build_dataset, context_features
from ingestion import mlb_stats, odds_api, prediction_markets
from models import calibration
from models._common import feature_columns, load_model
from pipeline import notify

# MLB Stats API abbreviation → Retrosheet 3-letter code.
_MLB_TO_RETRO: dict[str, str] = {
    "ATH": "OAK", "AZ": "ARI",  "CHC": "CHN", "CWS": "CHA",
    "KC":  "KCA", "LAA": "ANA", "LAD": "LAN", "NYM": "NYN",
    "NYY": "NYA", "SD":  "SDN", "SF":  "SFN", "STL": "SLN",
    "TB":  "TBA", "WSH": "WAS",
}

# Odds API full team name → Retrosheet 3-letter code.
_FULL_TO_RETRO: dict[str, str] = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHN", "Chicago White Sox": "CHA",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET",
    "Houston Astros": "HOU", "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA", "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYN",
    "New York Yankees": "NYA", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN", "San Francisco Giants": "SFN",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WAS",
    "Athletics": "OAK",
}

_MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"

logger = logging.getLogger(__name__)


def _ensure_str_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    if d.lower() == "today":
        return date.today()
    return datetime.fromisoformat(d).date()


def _fetch_mlb_schedule(target: date) -> list[dict]:
    """Pull today's MLB schedule with probable pitchers from the Stats API."""
    try:
        resp = requests.get(
            _MLB_SCHEDULE,
            params={"sportId": 1, "date": target.isoformat(),
                    "gameType": "R", "hydrate": "probablePitcher,team,venue"},
            timeout=15,
        )
        resp.raise_for_status()
        dates = resp.json().get("dates", [])
        return dates[0].get("games", []) if dates else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLB schedule fetch failed: %s", exc)
        return []


def _apply_live_team_stats(row: dict, home: str, away: str,
                           team_stats: dict[str, dict]) -> dict:
    """Override stale 2023 rolling features with current-season actuals."""
    for side, team in (("home", home), ("away", away)):
        ts = team_stats.get(team)
        if not ts:
            continue
        row[f"{side}_roll_win_pct"]          = ts["win_pct"]
        row[f"{side}_roll7_win_pct"]         = ts["win_pct"]
        row[f"{side}_roll_runs_scored"]      = ts["runs_per_game"]
        row[f"{side}_roll7_runs_scored"]     = ts["runs_per_game"]
        row[f"{side}_roll_runs_allowed"]     = ts["runs_allowed_per_game"]
        row[f"{side}_roll7_runs_allowed"]    = ts["runs_allowed_per_game"]
        row[f"{side}_roll_run_diff"]         = ts["run_diff_per_game"]
        row[f"{side}_roll7_run_diff"]        = ts["run_diff_per_game"]
    return row


def _apply_live_pitcher_stats(row: dict, home_sp_id: str, away_sp_id: str,
                               season: int) -> dict:
    """Override stale SP features with current-season actuals per pitcher."""
    for prefix, pid in (("home_sp_", home_sp_id), ("away_sp_", away_sp_id)):
        if not pid:
            continue
        ps = mlb_stats.fetch_pitcher_season_stats(pid, season)
        if not ps:
            continue
        row[f"{prefix}season_runs_allowed_pg"] = ps["runs_per_start"]
        row[f"{prefix}last3_runs_allowed_pg"]  = ps["runs_per_start"]
        row[f"{prefix}prior_ERA"]              = ps["era"]
        row[f"{prefix}prior_WHIP"]             = ps["whip"]
        row[f"{prefix}prior_K9"]               = ps["k9"]
        row[f"{prefix}prior_BB9"]              = ps["bb9"]
        row[f"{prefix}prior_FIP"]              = ps["era"]   # FIP proxy
        row[f"{prefix}prior_xFIP"]             = ps["era"]
    return row


def _build_today_features(target: date) -> pd.DataFrame:
    """Build the feature matrix for the target date's slate.

    If the date exists in the processed parquet (historical replay), return
    those rows directly. Otherwise pull today's schedule from the MLB Stats API,
    look up the most-recent rolling state per team from the processed dataset,
    and assemble synthetic feature rows for prediction.
    """
    df = build_dataset.build()
    df["date"] = pd.to_datetime(df["date"])
    today_games = df[df["date"].dt.date == target].copy()
    if not today_games.empty:
        return today_games

    logger.warning("no rows in processed dataset for %s — building live feature rows", target)

    games = _fetch_mlb_schedule(target)
    if not games:
        logger.error("MLB schedule returned no games for %s", target)
        return pd.DataFrame()

    # Most-recent known rolling state per team (last row each team appeared as home).
    latest_home = (df.sort_values("date")
                   .groupby("home_team", as_index=False)
                   .last())
    latest_away = (df.sort_values("date")
                   .groupby("visiting_team", as_index=False)
                   .last())

    # Park factors + lat/lon lookup.
    parks = context_features.load_park_factors().set_index("team")

    # Fetch current-season team stats once for the whole slate.
    live_team_stats = mlb_stats.fetch_team_season_stats(target.year)

    rows = []
    seen_pks: set[int] = set()
    for g in games:
        pk = g.get("gamePk")
        if pk in seen_pks:
            continue
        seen_pks.add(pk)

        ht_mlb = g["teams"]["home"]["team"]["abbreviation"]
        at_mlb = g["teams"]["away"]["team"]["abbreviation"]
        home = _MLB_TO_RETRO.get(ht_mlb, ht_mlb)
        away = _MLB_TO_RETRO.get(at_mlb, at_mlb)

        home_sp = g["teams"]["home"].get("probablePitcher", {})
        away_sp = g["teams"]["away"].get("probablePitcher", {})

        game_dt = pd.Timestamp(g.get("gameDate", target.isoformat()))
        game_id = f"{target.isoformat()}_{away}_{home}_0"

        row: dict = {
            "game_id": game_id,
            "date": pd.Timestamp(target),
            "season": target.year,
            "home_team": home,
            "visiting_team": away,
            "home_starting_pitcher_name": home_sp.get("fullName"),
            "visitor_starting_pitcher_name": away_sp.get("fullName"),
            "home_starting_pitcher_id": str(home_sp.get("id", "")),
            "visitor_starting_pitcher_id": str(away_sp.get("id", "")),
            "pitcher_confirmed": bool(home_sp and away_sp),
            "day_night": "N" if game_dt.hour >= 17 else "D",
            "is_day_game": int(game_dt.hour < 17),
        }

        # Rolling team features — carry forward most recent values.
        roll_home_cols = [c for c in df.columns if c.startswith("home_roll")
                          or c.startswith("home_prior_")]
        roll_away_cols = [c for c in df.columns if c.startswith("away_roll")
                          or c.startswith("away_prior_")]

        h_state = latest_home[latest_home["home_team"] == home]
        if not h_state.empty:
            for c in roll_home_cols:
                row[c] = h_state.iloc[-1].get(c)

        a_state = latest_away[latest_away["visiting_team"] == away]
        if not a_state.empty:
            for c in roll_away_cols:
                row[c] = a_state.iloc[-1].get(c)

        # SP features — carry forward last known values for this pitcher name.
        for prefix, sp_name in [("home_sp_", home_sp.get("fullName")),
                                 ("away_sp_", away_sp.get("fullName"))]:
            if not sp_name:
                continue
            sp_col = "home_starting_pitcher_name" if prefix == "home_sp_" else "visitor_starting_pitcher_name"
            sp_rows = df[df[sp_col] == sp_name].sort_values("date")
            if not sp_rows.empty:
                sp_feats = [c for c in df.columns if c.startswith(prefix)]
                for c in sp_feats:
                    row[c] = sp_rows.iloc[-1].get(c)

        # Context features.
        park = parks.loc[home] if home in parks.index else pd.Series(dtype=float)
        row["run_factor"] = park.get("run_factor")
        row["lat"] = park.get("lat")
        row["lon"] = park.get("lon")

        # Rest days: days since last game for each team.
        h_dates = df[
            (df["home_team"] == home) | (df["visiting_team"] == home)
        ]["date"].sort_values()
        a_dates = df[
            (df["home_team"] == away) | (df["visiting_team"] == away)
        ]["date"].sort_values()
        row["home_rest_days"] = (
            (pd.Timestamp(target) - h_dates.iloc[-1]).days if not h_dates.empty else 4
        )
        row["away_rest_days"] = (
            (pd.Timestamp(target) - a_dates.iloc[-1]).days if not a_dates.empty else 4
        )
        row["season_game_num"] = 0
        row["season_progress"] = 0.5   # mid-season default for live games

        # Override stale 2023 rolling features with current-season actuals.
        row = _apply_live_team_stats(row, home, away, live_team_stats)
        row = _apply_live_pitcher_stats(
            row,
            row.get("home_starting_pitcher_id", ""),
            row.get("visitor_starting_pitcher_id", ""),
            target.year,
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    logger.info("built %d live feature rows for %s", len(out), target)
    return out


def _gate_passed(model_name: str) -> bool:
    """Look up the calibration summary to decide if `model_name` is gated out."""
    path = config.CALIBRATION_DIR / "summary.json"
    if not path.exists():
        return True
    summary = json.loads(path.read_text())
    for item in summary:
        if item["model_name"] == model_name:
            return bool(item["passed_gate"])
    return True


def _coerce_X(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Reindex to model feature columns and return a clean float32 matrix."""
    X = df.reindex(columns=feats, fill_value=np.nan)
    X = X.apply(lambda c: pd.to_numeric(c, errors="coerce")).astype(np.float32)
    X = X.fillna(X.median()).fillna(0)  # fallback 0 for all-NaN columns
    return X


_HOME_WIN_FALLBACK = 0.54  # historical MLB home win rate

def _predict_win(today: pd.DataFrame) -> pd.DataFrame:
    xgb_model = load_model("home_win_xgb")
    feats = xgb_model["features"]
    X = _coerce_X(today, feats)
    p_xgb = xgb_model["model"].predict_proba(X)[:, 1]
    try:
        p_xgb = calibration.apply_calibration("home_win_xgb", p_xgb)
    except FileNotFoundError:
        pass

    p_home = p_xgb

    # Rows where every feature was NaN produce degenerate 0.0 output; fall back
    # to the historical home win rate so downstream EV math stays sensible.
    all_nan_rows = X.isna().all(axis=1).values
    p_home = np.where(all_nan_rows | (p_home == 0.0), _HOME_WIN_FALLBACK, p_home)
    today["model_win_prob"] = p_home
    today["model_win_prob_away"] = 1.0 - p_home
    return today


def _predict_runline(today: pd.DataFrame) -> pd.DataFrame:
    rl = load_model("runline")
    feats = rl["features"]
    X = _coerce_X(today, feats)
    p = rl["model"].predict_proba(X)[:, 1]
    try:
        p = calibration.apply_calibration("runline", p)
    except FileNotFoundError:
        pass
    today["model_cover_prob"] = p
    return today


def _predict_score(today: pd.DataFrame) -> pd.DataFrame:
    score = load_model("score_regressor")
    feats = score["features"]
    X = _coerce_X(today, feats)
    today["predicted_home_runs"] = score["home"].predict(X)
    today["predicted_away_runs"] = score["away"].predict(X)
    today["predicted_total"] = today["predicted_home_runs"] + today["predicted_away_runs"]
    today["predicted_spread"] = today["predicted_home_runs"] - today["predicted_away_runs"]
    return today


def _wire_odds(today: pd.DataFrame, odds_long: pd.DataFrame,
               polymarket: pd.DataFrame, kalshi: pd.DataFrame) -> pd.DataFrame:
    """Join model probabilities to market lines and compute edge."""
    if odds_long.empty:
        return today
    odds_h2h = odds_long[odds_long["market"] == "h2h"].copy()

    # Odds API uses full team names; normalize to Retrosheet 3-letter codes so
    # the join against today (which uses Retrosheet codes) actually matches.
    odds_h2h["home_retro"] = odds_h2h["home_team"].map(_FULL_TO_RETRO).fillna(odds_h2h["home_team"])
    odds_h2h["away_retro"] = odds_h2h["away_team"].map(_FULL_TO_RETRO).fillna(odds_h2h["away_team"])
    # Also normalize outcome_name (the team that wins the bet) to Retrosheet codes.
    odds_h2h["outcome_name"] = odds_h2h["outcome_name"].map(_FULL_TO_RETRO).fillna(odds_h2h["outcome_name"])

    # Attach game_id from today using the normalized codes.
    odds_h2h = odds_h2h.merge(
        today[["game_id", "home_team", "visiting_team"]],
        left_on=["home_retro", "away_retro"],
        right_on=["home_team", "visiting_team"],
        how="left",
    ).drop_duplicates(subset=["event_id", "book", "market", "outcome_name"])

    matched = odds_h2h["game_id"].notna().sum()
    logger.info("odds join: %d / %d h2h rows matched to today's games", matched, len(odds_h2h))

    # Build model_long: one row per (game, outcome) with event_id attached.
    model_long = pd.concat([
        today[["game_id", "home_team", "model_win_prob"]].rename(
            columns={"home_team": "outcome_name", "model_win_prob": "model_prob"}),
        today[["game_id", "visiting_team", "model_win_prob_away"]].rename(
            columns={"visiting_team": "outcome_name", "model_win_prob_away": "model_prob"}),
    ])
    model_long["market"] = "h2h"
    model_long = model_long.merge(
        odds_h2h[["event_id", "game_id"]].drop_duplicates(), on="game_id", how="left")

    edge_df = edge_calculator.calculate(model_long, odds_h2h, polymarket, kalshi)

    # edge_df has two rows per game (home + away outcome). Keep only the home
    # team outcome so merging back onto today (one row/game) doesn't double rows.
    home_edge = edge_df.merge(
        today[["game_id", "home_team"]].rename(columns={"home_team": "outcome_name"}),
        on=["game_id", "outcome_name"],
        how="inner",
    ).drop_duplicates(subset=["game_id"])

    today = today.merge(
        home_edge.drop(columns=["outcome_name", "market", "model_prob", "event_id",
                                 "point"], errors="ignore"),
        on="game_id", how="left", suffixes=("", "_edge"),
    )
    return today


def _make_picks_sheet(today: pd.DataFrame) -> pd.DataFrame:
    runline_gate_ok = _gate_passed("runline")
    win_gate_ok = _gate_passed("home_win_xgb")
    df = today.copy()
    df["ev_at_best"] = df.apply(
        lambda r: kelly.expected_value(r["model_win_prob"], int(r["best_price"]))
        if pd.notna(r.get("best_price")) and pd.notna(r.get("model_win_prob"))
        else np.nan,
        axis=1,
    )
    df["kelly_fraction"] = df.apply(
        lambda r: kelly.fractional_kelly(r.get("edge_vs_best", 0.0),
                                         american_to_decimal(int(r["best_price"])))
        if pd.notna(r.get("best_price")) else 0.0,
        axis=1,
    )
    df["recommended_stake_usd"] = df["kelly_fraction"] * config.BANKROLL_USD
    df["recommended_stake_usd"] = df["recommended_stake_usd"].round(2)

    runline_confirms = (
        df.get("model_cover_prob", pd.Series(0.0, index=df.index)) >= config.MIN_COVER_PROB
    )
    df["flag_bet"] = (
        (df["edge_vs_best"] >= config.EDGE_THRESHOLD) &
        (df["ev_at_best"] > 0) &
        (df["model_win_prob"] >= config.MIN_MODEL_PROB) &
        runline_confirms &
        win_gate_ok
    )

    cols = [
        "game_id", "date", "home_team", "visiting_team",
        "model_win_prob", "model_cover_prob", "best_book", "best_price", "best_no_vig_prob",
        "edge_vs_best", "consensus_no_vig_prob", "edge_vs_consensus", "ev_at_best",
        "predicted_home_runs", "predicted_away_runs", "predicted_total",
        "polymarket_prob", "edge_vs_polymarket",
        "kalshi_prob", "edge_vs_kalshi",
        "kelly_fraction", "recommended_stake_usd", "flag_bet",
        "pitcher_confirmed",
    ]
    return df[[c for c in cols if c in df.columns]]


def run(target: date) -> Path:
    odds_long = odds_api.fetch_live()
    poly = prediction_markets.fetch_polymarket()
    ksh = prediction_markets.fetch_kalshi()

    today = _build_today_features(target)
    if today.empty:
        logger.error("no games to predict for %s", target)
        out = config.PICKS_DIR / f"picks_{target.isoformat()}.csv"
        pd.DataFrame().to_csv(out, index=False)
        return out

    # Score must run first: its outputs become stacking features (stack_pred_*)
    # consumed by the win and runline classifiers.
    today = _predict_score(today)
    today["stack_pred_total"] = today["predicted_total"]
    today["stack_pred_spread"] = today["predicted_spread"]
    today = _predict_win(today)
    today = _predict_runline(today)
    today = _wire_odds(today, odds_long, poly, ksh)
    sheet = _make_picks_sheet(today)

    sheet = sheet.drop_duplicates(subset=["game_id"])
    out = config.PICKS_DIR / f"picks_{target.isoformat()}.csv"
    sheet.to_csv(out, index=False)
    flagged = sheet[sheet.get("flag_bet", False) == True]   # noqa: E712
    if not flagged.empty:
        print("\n=== Flagged bets ===")
        print(flagged.to_string(index=False))
    logger.info("wrote %d rows (%d flagged) to %s",
                len(sheet), len(flagged), out)
    notify.send_daily_summary(out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="today", help="ISO date or 'today'")
    args = parser.parse_args()
    run(_ensure_str_date(args.date))


if __name__ == "__main__":
    main()
