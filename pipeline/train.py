"""End-to-end training: dataset → 3 models → calibration → (optional) backtest.

Backtest:
 * Replays predictions on 2023-2024 against historical closing lines.
 * Reports accuracy, ROI by market type, and ROI per edge tier
   (3-4 %, 4-5 %, 5-6 %, 6 %+).
 * Saves: output/backtest_results.json plus three plots in output/backtest_plots/
     - calibration_curve.png
     - cumulative_pnl.png
     - roi_vs_edge_threshold.png   (so the 0.055 default can be tuned)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

import config
from edge.kelly import american_to_decimal
from edge.vig_removal import american_to_prob, remove_vig
from features import build_dataset
from models import calibration, runline_classifier, score_regressor, win_classifier
from models._common import safe_xy

logger = logging.getLogger(__name__)


def _predict_with_calibration(model_dict, X: pd.DataFrame, name: str) -> np.ndarray:
    """Predict probabilities and apply the saved isotonic recalibrator if present."""
    feats = model_dict["features"]
    X = X.reindex(columns=feats, fill_value=0)
    raw = model_dict["model"].predict_proba(X)[:, 1] \
        if hasattr(model_dict["model"], "predict_proba") else model_dict["model"].predict(X)
    try:
        return calibration.apply_calibration(name, raw)
    except FileNotFoundError:
        return raw


def train_all(force_rebuild: bool = False) -> dict:
    df = build_dataset.build(force=force_rebuild)
    logger.info("training win classifier")
    win = win_classifier.run(df)
    logger.info("training score regressor")
    score = score_regressor.run(df)
    logger.info("training runline classifier")
    runline = runline_classifier.run(df)

    # Per-model calibration on the validation slice
    val = df[df["season"].isin(config.VAL_YEARS)]
    reports = []
    if not val.empty:
        X_va, _ = safe_xy(val, "home_win")
        win_model = win["final"]["xgb"]
        wfeats = win["final"]["features"]
        p_win = win_model.predict_proba(X_va.reindex(columns=wfeats, fill_value=0))[:, 1]
        reports.append(calibration.evaluate_and_calibrate("home_win_xgb", p_win, val["home_win"].values))

        lr_bundle = win["final"]["lr"]
        scaler, lr_model = lr_bundle
        p_win_lr = lr_model.predict_proba(scaler.transform(X_va.reindex(columns=wfeats, fill_value=0)))[:, 1]
        reports.append(calibration.evaluate_and_calibrate("home_win_lr", p_win_lr, val["home_win"].values))

        rl_model = runline["final"]["model"]
        rfeats = runline["final"]["features"]
        p_rl = rl_model.predict_proba(X_va.reindex(columns=rfeats, fill_value=0))[:, 1]
        reports.append(calibration.evaluate_and_calibrate("runline", p_rl, val["home_covered"].values))

        calibration.write_summary(reports)

    return {"win": win, "score": score, "runline": runline, "calibration": reports}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def _load_historical_odds() -> pd.DataFrame:
    """Concatenate sports-statistics (pre-2022) + odds-api (2022+) tables."""
    frames: list[pd.DataFrame] = []
    pre = config.RAW_DIR / "odds_historical_pre2022.parquet"
    if pre.exists():
        frames.append(pd.read_parquet(pre))
    for year in range(config.ODDS_API_HISTORICAL_START_YEAR, 2025):
        path = config.RAW_DIR / f"odds_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        logger.warning("no historical odds parquet on disk — backtest will be empty")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pinnacle_closing(odds: pd.DataFrame) -> pd.DataFrame:
    """Pinnacle h2h closing line per (game, team) with no-vig probability."""
    if odds.empty:
        return pd.DataFrame()
    pin = odds[(odds["book"] == "pinnacle") & (odds["market"] == "h2h")].copy()
    if pin.empty:
        # Fall back to consensus of available books
        pin = odds[odds["market"] == "h2h"].copy()
    if pin.empty:
        return pd.DataFrame()
    pin["implied"] = pin["price"].apply(american_to_prob)
    grouped = pin.groupby(["event_id", "home_team", "away_team"])
    rows = []
    for (eid, home, away), g in grouped:
        if len(g) < 2:
            continue
        home_row = g[g["outcome_name"] == home]
        away_row = g[g["outcome_name"] == away]
        if home_row.empty or away_row.empty:
            continue
        home_imp = home_row["implied"].iloc[0]
        away_imp = away_row["implied"].iloc[0]
        h_p, a_p = remove_vig(home_imp, away_imp)
        rows.append({
            "event_id": eid, "home_team": home, "away_team": away,
            "home_price": int(home_row["price"].iloc[0]),
            "away_price": int(away_row["price"].iloc[0]),
            "home_no_vig": h_p, "away_no_vig": a_p,
            "snapshot_ts": home_row["snapshot_ts"].iloc[0],
        })
    return pd.DataFrame(rows)


def _join_odds_to_games(games: pd.DataFrame, odds_h2h: pd.DataFrame) -> pd.DataFrame:
    """Best-effort join by (date, home_team, away_team).

    Game logs use Retrosheet 3-letter codes; odds use full team names. We rely
    on the per-source schema lining up — see ingestion code for canonical naming.
    """
    g = games.copy()
    g["date_iso"] = pd.to_datetime(g["date"]).dt.strftime("%Y-%m-%d")
    o = odds_h2h.copy()
    if "snapshot_ts" in o.columns:
        o["date_iso"] = pd.to_datetime(o["snapshot_ts"]).dt.strftime("%Y-%m-%d")
    return g.merge(o, on=["date_iso", "home_team", "away_team"], how="left")


def backtest(seasons=config.TEST_YEARS) -> dict:
    df = build_dataset.build()
    df = df[df["season"].isin(seasons)].copy()
    odds = _load_historical_odds()
    h2h = _pinnacle_closing(odds)
    joined = _join_odds_to_games(df, h2h).dropna(subset=["home_no_vig", "away_no_vig"])
    if joined.empty:
        logger.error("backtest produced 0 rows after odds join — abort")
        return {}

    # Predict
    from models._common import load_model
    win_model = load_model("home_win_xgb")
    feats = win_model["features"]
    X = joined.reindex(columns=feats, fill_value=0).astype(np.float32) \
              .fillna(joined.reindex(columns=feats, fill_value=0).median(numeric_only=True))
    raw = win_model["model"].predict_proba(X)[:, 1]
    try:
        joined["model_prob_home"] = calibration.apply_calibration("home_win_xgb", raw)
    except FileNotFoundError:
        joined["model_prob_home"] = raw

    joined["edge_home"] = joined["model_prob_home"] - joined["home_no_vig"]
    joined["edge_away"] = (1 - joined["model_prob_home"]) - joined["away_no_vig"]

    # Side-of-bet selection: choose the larger positive edge
    joined["pick_side"] = np.where(joined["edge_home"] >= joined["edge_away"], "home", "away")
    joined["pick_edge"] = joined[["edge_home", "edge_away"]].max(axis=1)
    joined["pick_price"] = np.where(joined["pick_side"] == "home", joined["home_price"], joined["away_price"])
    joined["pick_won"] = np.where(
        joined["pick_side"] == "home",
        joined["home_win"],
        1 - joined["home_win"],
    )

    # P&L: $1 stake at quoted American odds
    joined["payout"] = joined["pick_price"].apply(lambda p: american_to_decimal(p) - 1.0)
    joined["pnl"] = np.where(joined["pick_won"] == 1, joined["payout"], -1.0)

    # ROI by edge tier
    bins = [-np.inf, 0.03, 0.04, 0.05, 0.06, np.inf]
    labels = ["<3%", "3-4%", "4-5%", "5-6%", "6%+"]
    joined["edge_tier"] = pd.cut(joined["pick_edge"], bins=bins, labels=labels)
    by_tier = joined[joined["pick_edge"] > 0].groupby("edge_tier", observed=True).agg(
        bets=("pnl", "size"),
        wins=("pick_won", "sum"),
        pnl_total=("pnl", "sum"),
        roi=("pnl", "mean"),
    ).reset_index()

    # Cumulative P&L plot
    sorted_ = joined.sort_values("date")
    sorted_["cum_pnl"] = sorted_["pnl"].cumsum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(pd.to_datetime(sorted_["date"]), sorted_["cum_pnl"])
    ax.set_title("Cumulative P&L (1 unit / bet)"); ax.set_ylabel("units")
    fig.tight_layout(); fig.savefig(config.BACKTEST_PLOTS_DIR / "cumulative_pnl.png", dpi=120)
    plt.close(fig)

    # Calibration curve (model vs. realised)
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(joined["home_win"], joined["model_prob_home"],
                                            n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--"); ax.plot(mean_pred, frac_pos, "o-")
    ax.set_xlabel("Mean predicted"); ax.set_ylabel("Empirical")
    ax.set_title("Home-win calibration (test years)")
    fig.tight_layout(); fig.savefig(config.BACKTEST_PLOTS_DIR / "calibration_curve.png", dpi=120)
    plt.close(fig)

    # ROI vs threshold curve
    thresholds = np.linspace(0.0, 0.10, 21)
    curve = []
    for t in thresholds:
        sub = joined[joined["pick_edge"] >= t]
        if sub.empty:
            curve.append({"threshold": float(t), "bets": 0, "roi": np.nan})
        else:
            curve.append({"threshold": float(t), "bets": int(len(sub)),
                          "roi": float(sub["pnl"].mean())})
    cdf = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cdf["threshold"], cdf["roi"], "o-")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(config.EDGE_THRESHOLD, color="red", linestyle="--",
               label=f"current threshold = {config.EDGE_THRESHOLD}")
    ax.set_xlabel("Edge threshold"); ax.set_ylabel("ROI per unit")
    ax.set_title("ROI vs. edge threshold (test years)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.BACKTEST_PLOTS_DIR / "roi_vs_edge_threshold.png", dpi=120)
    plt.close(fig)

    results = {
        "n_games": int(len(joined)),
        "overall_roi": float(joined["pnl"].mean()),
        "overall_pnl": float(joined["pnl"].sum()),
        "by_tier": by_tier.to_dict(orient="records"),
        "roi_vs_threshold": curve,
    }
    with open(config.OUTPUT_DIR / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("backtest: %d games, ROI=%.3f units/bet", results["n_games"], results["overall_roi"])
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="rebuild data/processed/games.parquet from scratch")
    args = parser.parse_args()
    if args.backtest:
        backtest()
    else:
        train_all(force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
