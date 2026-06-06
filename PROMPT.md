# MLB Predictor — Claude Code Prompt

Paste this entire file as your first message in a new Claude Code session.

---

You are building a production-grade MLB game prediction system in Python. Before writing any code, read this entire prompt, then output a detailed implementation plan to PLAN.md and wait for approval before proceeding.

---

## PROJECT GOAL

Build an end-to-end ML pipeline that:
1. Ingests historical and live MLB data from multiple sources
2. Engineers game-level features (no data leakage)
3. Trains three models: win/loss classifier, score regressor, runline cover classifier
4. Pulls real sportsbook lines (DraftKings, FanDuel, BetMGM, Caesars) and prediction market prices (Polymarket, Kalshi)
5. Calculates per-game edge against each book and market
6. Outputs a daily picks sheet flagging positive-EV bets above a calibrated threshold

---

## PROJECT STRUCTURE

Create the following layout:

```
mlb-predictor/
├── CLAUDE.md
├── PLAN.md
├── README.md
├── requirements.txt
├── .env.example
├── config.py                  # API keys, thresholds, paths — loaded from .env
├── data/
│   ├── raw/                   # Parquet files, one per source
│   └── processed/             # Merged game-level dataframe
├── ingestion/
│   ├── __init__.py
│   ├── retrosheet.py          # Game logs 2010–2024 via pybaseball
│   ├── statcast.py            # Pitch-level Statcast via pybaseball
│   ├── fangraphs.py           # Team batting/pitching advanced stats
│   ├── odds_api.py            # The Odds API — live + historical lines
│   └── prediction_markets.py  # Polymarket + Kalshi API
├── features/
│   ├── __init__.py
│   ├── team_features.py       # Rolling team offensive/defensive metrics
│   ├── pitcher_features.py    # Starting pitcher + bullpen features
│   ├── context_features.py    # Park factor, rest days, weather
│   └── build_dataset.py       # Merge all features into game-level df
├── models/
│   ├── __init__.py
│   ├── win_classifier.py      # XGBoost binary: home win vs. loss
│   ├── score_regressor.py     # Poisson regression: runs per team
│   ├── runline_classifier.py  # XGBoost binary: cover ±1.5
│   └── calibration.py        # Reliability diagrams, Brier score, isotonic recal
├── edge/
│   ├── __init__.py
│   ├── vig_removal.py         # Convert American odds → no-vig true probability
│   ├── edge_calculator.py     # Model prob − market true prob per game/market
│   └── kelly.py               # Fractional Kelly sizing (default: 25% Kelly)
├── pipeline/
│   ├── train.py               # Full training run with walk-forward CV
│   └── predict.py             # Daily prediction run → picks sheet
├── tests/
│   ├── test_vig_removal.py
│   ├── test_edge_calculator.py
│   └── test_leakage.py
└── output/
    └── picks/                 # Daily CSV picks sheets
```

---

## DATA SOURCES & INGESTION DETAILS

### 1. pybaseball (free, Python)
- `retrosheet.season_game_logs(season)` — game-level logs 2010–2024, outcome labels
- `pitching_stats(start, end)` — FanGraphs team pitching (ERA, FIP, xFIP, WHIP, K%, BB%)
- `batting_stats(start, end)` — FanGraphs team batting (wOBA, wRC+, ISO, OBP, SLG)
- `statcast(start_dt, end_dt)` — pitch-level Statcast 2015–present
- Cache all calls to data/raw/ as Parquet. Never re-fetch if file exists.

### 2. The Odds API (requires ODDS_API_KEY in .env)
- Endpoint: `GET https://api.the-odds-api.com/v4/sports/baseball_mlb/odds`
- Parameters: `regions=us`, `markets=h2h,spreads,totals`, `oddsFormat=american`
- Bookmakers: `draftkings,fanduel,betmgm,caesars,williamhill_us,bovada`
- Historical: `GET https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds`
  - Pull closing line (~1hr before first pitch) for each training game
- Store per-game, per-book, per-market in raw/odds_YYYY.parquet

### 3. sports-statistics.com CSV (free, historical 2010–2021)
- URL: https://sports-statistics.com/sports-data/mlb-historical-odds-scores-datasets/
- Contains opening + closing moneyline, runline, totals for Pinnacle and other books
- Use Pinnacle closing line as gold-standard no-vig benchmark for training

### 4. Polymarket API (no key required for public markets)
- REST: `GET https://clob.polymarket.com/markets` — filter by tag "MLB"
- Store implied probability (price field, 0–1 scale) — already no-vig

### 5. Kalshi API (requires KALSHI_API_KEY in .env)
- REST: `GET https://trading-api.kalshi.com/trade-api/v2/events?series_ticker=KXMLB`
- Store yes_ask price as implied probability

---

## FEATURE ENGINEERING — CRITICAL RULES

**No data leakage.** Every feature must be computed using only data available BEFORE the
game being predicted. Specifically:
- Use rolling windows computed with `.shift(1)` BEFORE `.rolling()` — this ensures the
  current game is excluded from its own features
- Never include season-to-date stats that accumulate past the game date
- Starting pitcher features must be from the pitcher's LAST start, not the current game
- Betting odds (h2h, spreads, totals) must NEVER be used as model input features
- Train/val/test split: train 2015–2021, validate 2022, test 2023–2024

### Team features (team_features.py)
Compute per team, rolling 15-game window, home and away separately:
- Offensive: wOBA, wRC+, OBP, SLG, ISO, K%, BB%, BABIP
- Defensive/pitching: team ERA, FIP, xFIP, WHIP, HR/9, K/9, BB/9
- Bullpen: bullpen ERA, leverage index, recent workload (innings last 3 days)
- Win%, run differential per game, runs scored per game, runs allowed per game

### Starting pitcher features (pitcher_features.py)
For the announced starter on each side:
- Season xFIP, FIP, WHIP, K/9, BB/9, HR/9
- Last-3-starts ERA, K%, opponent wOBA against
- Days rest since last start
- Statcast: avg exit velocity against, whiff rate, called strike rate

### Context features (context_features.py)
- Park factor (runs) — static lookup table by ballpark
- Home/away indicator
- Days rest for each team
- Time of season (game number / 162)
- Temperature and wind speed via Open-Meteo:
  `GET https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,windspeed_10m`

---

## MODEL SPECIFICATIONS

### Model 1: Win/loss classifier (win_classifier.py)
- Target: `home_win` (1/0)
- Algorithm: XGBoost with `objective='binary:logistic'`
- Also train logistic regression baseline for calibration comparison
- Output: probability of home team winning (float 0–1)
- Evaluation: AUC-ROC, Brier score, calibration curve vs. closing line implied prob

### Model 2: Score regressor (score_regressor.py)
- Target: home_runs and away_runs (two separate models)
- Algorithm: Poisson GLM as primary; XGBoost regressor as secondary
- predicted_total = home_runs + away_runs
- predicted_spread = home_runs − away_runs
- Evaluation: MAE, RMSE on runs; compare predicted total vs. actual O/U line

### Model 3: Runline cover classifier (runline_classifier.py)
- Target: `home_covered` (1 if home won by 2+, 0 otherwise)
- Algorithm: XGBoost with `objective='binary:logistic'`
- Evaluation: AUC-ROC, accuracy vs. closing runline price

### Walk-forward cross-validation
Do NOT use random k-fold. Use walk-forward:
- Fold 1: train 2015–2019 / val 2020
- Fold 2: train 2015–2020 / val 2021
- Fold 3: train 2015–2021 / val 2022
- Report mean ± std of all evaluation metrics across folds

### Calibration (calibration.py)
After training, calibrate each model using isotonic regression on the validation set.
Generate and save reliability diagrams for each model to output/calibration/.
Calibration gate: if Brier score > 0.25 on holdout, raise a warning and skip picks.

---

## EDGE CALCULATION

### vig_removal.py
```python
def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    return prob_a / total, prob_b / total
```

### edge_calculator.py
For each game, for each market (moneyline, runline, total):
- `model_prob`: output from the relevant model
- `no_vig_prob`: vig-removed implied probability from each book
- `edge_vs_book = model_prob - no_vig_prob`
- `edge_vs_polymarket = model_prob - polymarket_price`
- `edge_vs_kalshi = model_prob - kalshi_price`
- `best_book_edge`: max edge across all available books
- `consensus_edge`: edge vs. average no-vig prob across all books

### kelly.py
```python
def fractional_kelly(edge: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Returns fraction of bankroll to bet. Default: 25% Kelly."""
    full_kelly = edge / (decimal_odds - 1)
    return max(0.0, full_kelly * fraction)

def american_to_decimal(american_odds: int) -> float:
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1

def expected_value(model_prob: float, american_odds: int) -> float:
    decimal = american_to_decimal(american_odds)
    return (model_prob * (decimal - 1)) - (1 - model_prob)
```

### Bet flagging logic
Flag `flag_bet = True` if ALL conditions met:
1. `best_book_edge >= 0.055` (5.5% — covers typical 4–5% vig with buffer)
2. `ev_at_best > 0` (positive EV at best available line)
3. Model calibration Brier score < 0.25 on holdout (calibration gate)

---

## DAILY PICKS SHEET

File: `output/picks/picks_YYYY-MM-DD.csv`

Columns:
```
game_id, date, home_team, away_team,
model_win_prob, best_book, best_book_line, best_book_true_prob, edge_vs_best,
consensus_true_prob, edge_vs_consensus, ev_at_best,
predicted_home_runs, predicted_away_runs, predicted_total,
book_total_line, edge_vs_total, total_ev,
polymarket_prob, edge_vs_polymarket,
kalshi_prob, edge_vs_kalshi,
kelly_fraction, flag_bet
```

Also print a clean summary table to stdout showing only `flag_bet = True` games.

---

## BACKTESTING

In `pipeline/train.py --backtest`, replay the model on 2023–2024 seasons:
- Use closing lines from The Odds API historical endpoint
- Compute: accuracy, ROI by market type, ROI by edge tier (3–4%, 4–5%, 5–6%, 6%+)
- Generate plots: ROI vs. edge threshold curve, calibration curve, cumulative P&L
- Save to output/backtest_results.json and output/backtest_plots/

---

## UNIT TESTS

### tests/test_vig_removal.py
- `american_to_prob(-110)` ≈ 0.5238
- `american_to_prob(100)` == 0.5
- `remove_vig(0.5238, 0.5238)` → (0.5, 0.5) within tolerance
- `remove_vig(0.574, 0.488)` → sums to 1.0

### tests/test_edge_calculator.py
- Edge = model_prob − no_vig_prob (direction check)
- EV positive when model_prob > implied prob
- Kelly fraction = 0 when edge <= 0

### tests/test_leakage.py
- For every row in processed/games.parquet, assert all feature computation
  dates < game_date
- Assert no odds columns present in feature matrix

---

## IMPLEMENTATION REQUIREMENTS

- Python 3.11+
- Requirements: pybaseball, xgboost, scikit-learn, pandas, numpy, requests,
  python-dotenv, pyarrow, matplotlib, scipy, statsmodels
- All API keys in .env, loaded via python-dotenv
- Parquet for all intermediate storage
- All rolling features use `.shift(1)` before `.rolling()`
- Log API calls with timestamps and remaining quota (Odds API returns quota in headers)
- Exponential backoff on all external API calls
- Type hints throughout
- Docstrings on all public functions

---

## EXECUTION ORDER

Build and test in this order to manage dependencies:
1. config.py + .env.example
2. ingestion/retrosheet.py — establishes game_id schema
3. ingestion/fangraphs.py — team stats backbone
4. features/team_features.py + features/build_dataset.py (stub)
5. models/win_classifier.py — first working model
6. edge/vig_removal.py + edge/edge_calculator.py — core math
7. pipeline/predict.py — wires everything together
8. ingestion/odds_api.py — add live lines
9. ingestion/prediction_markets.py — Polymarket + Kalshi
10. models/score_regressor.py + models/runline_classifier.py
11. models/calibration.py
12. edge/kelly.py
13. pipeline/train.py --backtest
14. tests/

---

## FIRST STEP

Output a complete PLAN.md covering:
1. File creation order and dependencies
2. Expected row counts and file sizes for each data artifact
3. Any ambiguities or decisions you need me to resolve before starting
4. Estimated total lines of code

Wait for my approval of PLAN.md before writing any code.
