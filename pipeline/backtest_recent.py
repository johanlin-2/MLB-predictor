"""Backtest the current model over a recent date range using saved picks CSVs.

For each date that has a picks_YYYY-MM-DD.csv file in output/picks/, this script:
  1. Re-runs model predictions using the features built at prediction time
  2. Reuses the market odds already saved in the picks CSV (real prices from that day)
  3. Fetches actual game outcomes from the MLB Stats API
  4. Computes per-game accuracy and P&L on flagged bets

Usage:
    python -m pipeline.backtest_recent                      # last 30 days
    python -m pipeline.backtest_recent --start 2026-05-01  # from a specific date
    python -m pipeline.backtest_recent --start 2026-05-01 --end 2026-05-31
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config
from edge.kelly import american_to_decimal
from models._common import load_model
from models import calibration
from pipeline.predict import _build_today_features, _coerce_X, _gate_passed

logger = logging.getLogger(__name__)

_MLB_SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
_MLB_TO_RETRO: dict[str, str] = {
    "ATH": "OAK", "AZ": "ARI", "CHC": "CHN", "CWS": "CHA",
    "KC": "KCA", "LAA": "ANA", "LAD": "LAN", "NYM": "NYN",
    "NYY": "NYA", "SD": "SDN", "SF": "SFN", "STL": "SLN",
    "TB": "TBA", "WSH": "WAS",
}


def _fetch_results(target: date) -> dict[str, dict]:
    """Returns {game_id: {home_score, away_score, home_win}} for final games."""
    try:
        resp = requests.get(
            _MLB_SCHEDULE,
            params={"sportId": 1, "date": target.isoformat(),
                    "gameType": "R", "hydrate": "linescore,team"},
            timeout=15,
        )
        resp.raise_for_status()
        games = resp.json().get("dates", [{}])[0].get("games", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLB schedule fetch failed for %s: %s", target, exc)
        return {}

    results = {}
    seen: set[int] = set()
    for g in games:
        if g.get("status", {}).get("abstractGameState") != "Final":
            continue
        pk = g.get("gamePk")
        if pk in seen:
            continue
        seen.add(pk)
        ht = _MLB_TO_RETRO.get(g["teams"]["home"]["team"]["abbreviation"],
                               g["teams"]["home"]["team"]["abbreviation"])
        at = _MLB_TO_RETRO.get(g["teams"]["away"]["team"]["abbreviation"],
                               g["teams"]["away"]["team"]["abbreviation"])
        game_id = f"{target.isoformat()}_{at}_{ht}_0"
        home_score = g["teams"]["home"].get("score", 0) or 0
        away_score = g["teams"]["away"].get("score", 0) or 0
        results[game_id] = {
            "home_score": home_score,
            "away_score": away_score,
            "home_win": int(home_score > away_score),
        }
    return results


def _predict_proba(today: pd.DataFrame) -> np.ndarray:
    xgb = load_model("home_win_xgb")
    X = _coerce_X(today, xgb["features"])
    p_xgb = xgb["model"].predict_proba(X)[:, 1]
    try:
        p_xgb = calibration.apply_calibration("home_win_xgb", p_xgb)
    except FileNotFoundError:
        pass

    return p_xgb


def backtest_date(target: date) -> pd.DataFrame | None:
    picks_path = config.PICKS_DIR / f"picks_{target.isoformat()}.csv"
    if not picks_path.exists():
        return None

    try:
        saved = pd.read_csv(picks_path)
    except Exception:
        return None
    if saved.empty or "best_price" not in saved.columns:
        return None
    # Only keep rows that have odds (no odds = can't evaluate)
    saved = saved.dropna(subset=["best_price"])
    if saved.empty:
        return None

    # Re-run model for this date with current artifacts
    features = _build_today_features(target)
    if features.empty:
        return None

    p_home = _predict_proba(features)
    prob_map = dict(zip(features["game_id"], p_home))

    # Fetch actual results
    results = _fetch_results(target)

    rows = []
    win_gate_ok = _gate_passed("home_win_xgb")
    for _, s in saved.iterrows():
        gid = s["game_id"]
        model_prob = prob_map.get(gid)
        if model_prob is None:
            continue
        result = results.get(gid)

        best_price = s["best_price"]
        best_no_vig = s.get("best_no_vig_prob")
        if pd.isna(best_price) or pd.isna(best_no_vig):
            continue

        edge = float(model_prob) - float(best_no_vig)
        decimal = american_to_decimal(int(best_price))
        ev = model_prob * (decimal - 1.0) - (1.0 - model_prob)
        flag = (edge >= config.EDGE_THRESHOLD and ev > 0
                and float(model_prob) >= config.MIN_MODEL_PROB and win_gate_ok)

        row = {
            "date": target.isoformat(),
            "game_id": gid,
            "home_team": s["home_team"],
            "away_team": s["visiting_team"],
            "model_prob": round(float(model_prob), 4),
            "best_no_vig_prob": round(float(best_no_vig), 4),
            "edge": round(edge, 4),
            "best_book": s.get("best_book"),
            "best_price": int(best_price),
            "ev": round(float(ev), 4),
            "flag_bet": flag,
            "home_win_actual": result["home_win"] if result else None,
            "home_score": result["home_score"] if result else None,
            "away_score": result["away_score"] if result else None,
        }
        if result:
            # P&L: bet $1 on home team if flagged (model is pricing home win)
            row["model_correct"] = int(result["home_win"] == int(model_prob >= 0.5))
            if flag:
                payout = decimal - 1.0
                row["pnl"] = payout if result["home_win"] else -1.0
            else:
                row["pnl"] = None
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def run(start: date, end: date) -> pd.DataFrame:
    all_rows = []
    d = start
    while d <= end:
        df = backtest_date(d)
        if df is not None and not df.empty:
            all_rows.append(df)
            logger.info("%s: %d games, %d flagged", d, len(df), df["flag_bet"].sum())
        d += timedelta(days=1)

    if not all_rows:
        logger.error("no data found for %s to %s", start, end)
        return pd.DataFrame()

    results = pd.concat(all_rows, ignore_index=True)
    _print_summary(results)

    out = config.OUTPUT_DIR / f"backtest_{start.isoformat()}_to_{end.isoformat()}.csv"
    results.to_csv(out, index=False)
    logger.info("wrote %s", out)
    return results


def _print_summary(df: pd.DataFrame) -> None:
    flagged = df[df["flag_bet"] & df["pnl"].notna()]
    all_with_result = df[df["home_win_actual"].notna()]

    print(f"\n{'='*55}")
    print(f"  BACKTEST SUMMARY  ({df['date'].min()} → {df['date'].max()})")
    print(f"{'='*55}")
    print(f"  Total games evaluated : {len(all_with_result)}")
    print(f"  Model correct (all)   : {all_with_result['model_correct'].mean():.1%}")
    print(f"  Flagged bets          : {len(flagged)}")

    if not flagged.empty:
        wins = (flagged["pnl"] > 0).sum()
        print(f"  Win rate (flagged)    : {wins}/{len(flagged)} = {wins/len(flagged):.1%}")
        print(f"  Total P&L (flagged)   : {flagged['pnl'].sum():+.2f} units")
        print(f"  ROI (flagged)         : {flagged['pnl'].mean():+.3f} units/bet")
        print(f"\n  By edge tier:")
        bins = [0.055, 0.07, 0.09, 0.12, np.inf]
        labels = ["5.5-7%", "7-9%", "9-12%", "12%+"]
        flagged = flagged.copy()
        flagged["tier"] = pd.cut(flagged["edge"], bins=bins, labels=labels)
        tier_summary = flagged.groupby("tier", observed=True).agg(
            bets=("pnl", "size"), wins=("pnl", lambda x: (x > 0).sum()),
            roi=("pnl", "mean")
        )
        for tier, row in tier_summary.iterrows():
            print(f"    {tier:8s}: {int(row['bets'])} bets, "
                  f"{int(row['wins'])}/{int(row['bets'])} wins, "
                  f"ROI {row['roi']:+.3f}")
    else:
        print("  No flagged bets with results found.")
    print(f"{'='*55}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="Start date ISO (default: 30 days ago)")
    parser.add_argument("--end", default=None, help="End date ISO (default: yesterday)")
    args = parser.parse_args()

    today = date.today()
    end = date.fromisoformat(args.end) if args.end else today - timedelta(days=1)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=29)
    run(start, end)


if __name__ == "__main__":
    main()
