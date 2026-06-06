"""Central configuration for the MLB predictor.

All paths, API keys, and tunable parameters live here. Loaded once at import
time via python-dotenv. Importing this module performs no network I/O.

Approved decisions baked in (see PLAN.md section 3):
 * 2020 season excluded entirely.
 * `williamhill_us` dropped from the bookmaker list.
 * Historical odds: sports-statistics.com CSV for 2015-2021, Odds API for 2022+.
 * Calibration gate applied per model, not globally.
 * Predictions re-run hourly until first pitch (option c).
 * Polymarket / Kalshi used in live picks only — not in backtest.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
PICKS_DIR: Path = OUTPUT_DIR / "picks"
CALIBRATION_DIR: Path = OUTPUT_DIR / "calibration"
BACKTEST_PLOTS_DIR: Path = OUTPUT_DIR / "backtest_plots"
MODEL_ARTIFACTS_DIR: Path = PROJECT_ROOT / "models" / "_artifacts"

for _d in (
    RAW_DIR,
    PROCESSED_DIR,
    PICKS_DIR,
    CALIBRATION_DIR,
    BACKTEST_PLOTS_DIR,
    MODEL_ARTIFACTS_DIR,
    RAW_DIR / "statcast",
):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load .env (silently no-op if not present so unit tests run cleanly)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API keys (None ⇒ unavailable; ingestion modules raise on first use)
# ---------------------------------------------------------------------------
ODDS_API_KEY: str | None = os.getenv("ODDS_API_KEY")
KALSHI_API_KEY: str | None = os.getenv("KALSHI_API_KEY")
KALSHI_API_SECRET: str | None = os.getenv("KALSHI_API_SECRET")

# ---------------------------------------------------------------------------
# Bankroll and risk
# ---------------------------------------------------------------------------
BANKROLL_USD: float = float(os.getenv("BANKROLL_USD", "1000"))
KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))
EDGE_THRESHOLD: float = float(os.getenv("EDGE_THRESHOLD", "0.08"))
MIN_MODEL_PROB: float = float(os.getenv("MIN_MODEL_PROB", "0.58"))
BRIER_GATE: float = float(os.getenv("BRIER_GATE", "0.25"))

# ---------------------------------------------------------------------------
# Train / val / test cutoffs (PROMPT.md). 2020 dropped.
# ---------------------------------------------------------------------------
TRAIN_YEARS: Tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2021)
VAL_YEARS: Tuple[int, ...] = (2022,)
TEST_YEARS: Tuple[int, ...] = (2023, 2024)
ALL_YEARS: Tuple[int, ...] = TRAIN_YEARS + VAL_YEARS + TEST_YEARS

# Walk-forward folds. Each fold trains on a closed year range and validates on
# a single subsequent year. 2020 is excluded from train ranges automatically by
# the dataset builder.
WALK_FORWARD_FOLDS = (
    {"train_start": 2015, "train_end": 2019, "val": 2021},
    {"train_start": 2015, "train_end": 2021, "val": 2022},
    {"train_start": 2015, "train_end": 2022, "val": 2023},
)

# ---------------------------------------------------------------------------
# Bookmakers (williamhill_us removed)
# ---------------------------------------------------------------------------
BOOKMAKERS: Tuple[str, ...] = (
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "bovada",
)
MARKETS: Tuple[str, ...] = ("h2h", "spreads", "totals")

# ---------------------------------------------------------------------------
# Rolling-window sizes
# ---------------------------------------------------------------------------
TEAM_ROLLING_WINDOW: int = 15
TEAM_ROLLING_WINDOW_SHORT: int = 7   # captures hot/cold streaks
PITCHER_RECENT_STARTS: int = 3

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------
ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
ODDS_API_HISTORICAL_START_YEAR: int = 2022          # sports-statistics covers earlier
SPORTS_STATISTICS_RANGE: Tuple[int, int] = (2015, 2021)

POLYMARKET_BASE: str = "https://clob.polymarket.com"
KALSHI_BASE: str = "https://trading-api.kalshi.com/trade-api/v2"

OPEN_METEO_FORECAST: str = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE: str = "https://archive-api.open-meteo.com/v1/archive"


def update_bankroll(new_bankroll: float) -> float:
    """Update the in-process bankroll and persist it to .env.

    Returns the new value. Use this from a Python REPL or a notebook between
    sessions; pipeline scripts always pick up the latest value at import time.
    """
    global BANKROLL_USD
    BANKROLL_USD = float(new_bankroll)
    env_path = PROJECT_ROOT / ".env"
    lines: list[str] = []
    found = False
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("BANKROLL_USD="):
                lines.append(f"BANKROLL_USD={new_bankroll}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"BANKROLL_USD={new_bankroll}")
    env_path.write_text("\n".join(lines).rstrip() + "\n")
    return BANKROLL_USD
