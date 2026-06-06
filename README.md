# MLB Predictor

End-to-end ML pipeline that predicts MLB game outcomes, prices each game
against real sportsbook lines and prediction-market quotes, and outputs a
daily picks sheet flagging positive-EV bets.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys + bankroll

# 1. Pull historical data (multi-hour first run; cached as Parquet)
python -m ingestion.retrosheet
python -m ingestion.fangraphs
python -m ingestion.statcast
python -m ingestion.sports_statistics   # 2015-2021 closing lines
python -m ingestion.odds_api --historical  # 2022+

# 2. Build features + train
python -m features.build_dataset
python -m pipeline.train

# 3. Daily picks (re-runs hourly until first pitch)
python -m pipeline.predict --date today
```

## Data decisions (signed off in PLAN.md)

* **2020 season excluded entirely.** Shortened schedule, universal DH, 7-inning
  doubleheaders all break feature distributions.
* **Historical odds split:** sports-statistics.com CSV for 2015–2021 closing
  lines, The Odds API for 2022+. Saves ~95 % of the API's historical credits.
* **William Hill US removed** — folded into Caesars in 2021.
* **Polymarket + Kalshi are live-only.** Their MLB markets are too sparse before
  2024 to use in backtest.
* **Predictions re-run hourly** until first pitch (starting pitcher
  announcements move).
* **Per-model Brier gate** at 0.25. A bad runline model never blocks the
  moneyline slate.

## Project layout

```
mlb-predictor/
├── config.py            shared paths, keys, thresholds (loads .env)
├── ingestion/           one file per data source, all cached to data/raw/
├── features/            leakage-safe rolling features → data/processed/games.parquet
├── models/              win classifier, score regressor, runline classifier, calibration
├── edge/                vig removal, edge math, fractional Kelly
├── pipeline/            train.py (with --backtest) and predict.py (daily)
├── tests/               vig, edge, leakage
└── output/
    ├── picks/           picks_YYYY-MM-DD.csv
    ├── calibration/     reliability diagrams
    └── backtest_plots/  ROI vs edge, cumulative P&L
```

## Picks sheet

`output/picks/picks_YYYY-MM-DD.csv` is the daily deliverable. Columns include
`flag_bet` (True when edge ≥ 5.5 % at the best available book, EV > 0, and the
model passed the calibration gate), `kelly_fraction`, and
`recommended_stake_usd` (`kelly_fraction * BANKROLL_USD`, capped at 5 % of
bankroll per game as a defensive limit).

Update your bankroll between sessions:
```python
from config import update_bankroll
update_bankroll(1750.00)
```

## Backtest

`python -m pipeline.train --backtest` replays 2023–2024 against historical
closing lines. Outputs `output/backtest_results.json` plus three plots:
calibration curve, cumulative P&L, and ROI vs. edge-threshold curve (so the
0.055 default can be tuned).
