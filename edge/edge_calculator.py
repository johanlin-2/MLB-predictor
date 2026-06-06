"""Per-game edge calculation against books and prediction markets.

Given:
  * `model_probs`  — DataFrame[game_id, market, outcome_name, model_prob]
  * `odds_long`    — DataFrame[game_id, book, market, outcome_name, price, point]
                     (output of ingestion.odds_api.fetch_live)
  * `polymarket`   — optional DataFrame[game_id, outcome, price]   (live only)
  * `kalshi`       — optional DataFrame[game_id, outcome, yes_ask] (live only)

We compute:
  * book-level no-vig probability via vig_removal.market_no_vig
  * `edge_vs_book` = model_prob − no_vig_prob per (game, book, market, outcome)
  * `best_book_edge` = max over books for the same (game, market, outcome)
  * `consensus_edge` = model_prob − mean(no_vig_prob across books)
  * `edge_vs_polymarket`, `edge_vs_kalshi` when those frames are supplied
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from edge.vig_removal import american_to_prob, remove_vig

logger = logging.getLogger(__name__)


def _no_vig_per_book(odds: pd.DataFrame) -> pd.DataFrame:
    """For each (game, book, market), produce a tidy no-vig prob per outcome.

    The two-way market (h2h, spreads, totals) is normalised within each
    book/game/market group.
    """
    odds = odds.copy()
    odds["raw_prob"] = odds["price"].apply(american_to_prob)
    keys = ["event_id", "book", "market"]
    if "point" in odds.columns:
        # Spreads and totals share keys across both sides through point sign.
        # Keep the point on the row for downstream joining; do NOT group by it.
        pass

    # Vectorised vig-removal per (event_id, book, market) — no DataFrame.apply,
    # no FutureWarning, and much faster on large slates.
    group_totals = odds.groupby(keys)["raw_prob"].transform("sum")
    odds["no_vig_prob"] = np.where(group_totals > 0, odds["raw_prob"] / group_totals, np.nan)
    return odds


def _best_and_consensus(no_vig: pd.DataFrame) -> pd.DataFrame:
    """Reduce per-book rows to per-(game, market, outcome) with best + consensus."""
    keys = ["event_id", "market", "outcome_name"]
    consensus = (no_vig.groupby(keys)["no_vig_prob"]
                 .mean().rename("consensus_no_vig_prob").reset_index())
    best_price = no_vig.loc[no_vig.groupby(keys)["price"].idxmax()][
        ["event_id", "market", "outcome_name", "book", "price", "no_vig_prob", "point"]
    ].rename(columns={"book": "best_book", "price": "best_price",
                      "no_vig_prob": "best_no_vig_prob"})
    out = best_price.merge(consensus, on=keys, how="left")
    return out


def calculate(model_probs: pd.DataFrame,
              odds_long: pd.DataFrame,
              polymarket: Optional[pd.DataFrame] = None,
              kalshi: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Top-level entry. Returns a wide DataFrame ready for the picks sheet."""
    if odds_long.empty:
        logger.warning("no odds rows; returning empty edge frame")
        return pd.DataFrame()

    no_vig = _no_vig_per_book(odds_long)
    reduced = _best_and_consensus(no_vig)

    merged = model_probs.merge(reduced, on=["event_id", "market", "outcome_name"], how="left")
    merged["edge_vs_best"] = merged["model_prob"] - merged["best_no_vig_prob"]
    merged["edge_vs_consensus"] = merged["model_prob"] - merged["consensus_no_vig_prob"]

    if polymarket is not None and not polymarket.empty:
        merged = merged.merge(
            polymarket.rename(columns={"price": "polymarket_prob",
                                       "outcome": "outcome_name"}),
            on=["event_id", "outcome_name"], how="left",
        )
        merged["edge_vs_polymarket"] = merged["model_prob"] - merged["polymarket_prob"]
    else:
        merged["polymarket_prob"] = np.nan
        merged["edge_vs_polymarket"] = np.nan

    if kalshi is not None and not kalshi.empty:
        merged = merged.merge(
            kalshi.rename(columns={"yes_ask": "kalshi_prob",
                                   "ticker": "outcome_name"}),
            on=["event_id", "outcome_name"], how="left",
        )
        merged["edge_vs_kalshi"] = merged["model_prob"] - merged["kalshi_prob"]
    else:
        merged["kalshi_prob"] = np.nan
        merged["edge_vs_kalshi"] = np.nan

    return merged
