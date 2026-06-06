"""Tests for edge.kelly and edge.edge_calculator math."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from edge.edge_calculator import calculate
from edge.kelly import (
    expected_value,
    fractional_kelly,
    recommended_stake_usd,
    MAX_FRACTION_PER_GAME,
)
from edge.vig_removal import american_to_decimal


def test_kelly_zero_when_edge_nonpositive():
    assert fractional_kelly(0.0, 2.0) == 0.0
    assert fractional_kelly(-0.05, 2.0) == 0.0


def test_kelly_clamped_to_max_fraction():
    # Huge edge would request > MAX_FRACTION; ensure cap holds.
    f = fractional_kelly(edge=0.5, decimal_odds=2.0, fraction=1.0)
    assert f <= MAX_FRACTION_PER_GAME + 1e-9


def test_kelly_quarter_fraction():
    # Edge = 5 %, decimal = 2.0 (even money) ⇒ full Kelly = 0.05; 25 % Kelly = 0.0125
    f = fractional_kelly(edge=0.05, decimal_odds=2.0, fraction=0.25)
    assert math.isclose(f, 0.0125, abs_tol=1e-6)


def test_expected_value_positive_when_model_beats_market():
    # Model says 55 % to win at -110 (implied 52.38 %) ⇒ EV > 0
    ev = expected_value(0.55, -110)
    assert ev > 0


def test_expected_value_negative_when_model_below_market():
    ev = expected_value(0.40, -110)
    assert ev < 0


def test_recommended_stake_uses_bankroll():
    # Override bankroll for determinism in test
    stake = recommended_stake_usd(edge=0.05, american_odds=100, bankroll=1000, fraction=0.25)
    # decimal odds = 2.0, fraction = 0.05 / 1 * 0.25 = 0.0125 → $12.50
    assert math.isclose(stake, 12.5, abs_tol=0.01)


def test_recommended_stake_zero_when_edge_nonpositive():
    assert recommended_stake_usd(edge=-0.01, american_odds=-110, bankroll=1000) == 0.0


# ---------------------------------------------------------------------------
# edge_calculator integration: edge direction + columns
# ---------------------------------------------------------------------------
def _toy_odds():
    """Two-game slate, two books each. Symmetric -110/-110 ⇒ no_vig 0.5/0.5."""
    return pd.DataFrame([
        {"event_id": "g1", "book": "draftkings", "market": "h2h",
         "outcome_name": "HOME", "price": -110, "point": None,
         "snapshot_ts": "2026-05-14T00:00:00", "home_team": "HOME", "away_team": "AWAY"},
        {"event_id": "g1", "book": "draftkings", "market": "h2h",
         "outcome_name": "AWAY", "price": -110, "point": None,
         "snapshot_ts": "2026-05-14T00:00:00", "home_team": "HOME", "away_team": "AWAY"},
        {"event_id": "g1", "book": "fanduel", "market": "h2h",
         "outcome_name": "HOME", "price": -120, "point": None,
         "snapshot_ts": "2026-05-14T00:00:00", "home_team": "HOME", "away_team": "AWAY"},
        {"event_id": "g1", "book": "fanduel", "market": "h2h",
         "outcome_name": "AWAY", "price": +100, "point": None,
         "snapshot_ts": "2026-05-14T00:00:00", "home_team": "HOME", "away_team": "AWAY"},
    ])


def test_edge_direction():
    """If model says HOME 0.60 vs no-vig 0.50, edge_vs_best should be ~+0.10."""
    model_probs = pd.DataFrame([
        {"event_id": "g1", "market": "h2h", "outcome_name": "HOME", "model_prob": 0.60},
        {"event_id": "g1", "market": "h2h", "outcome_name": "AWAY", "model_prob": 0.40},
    ])
    out = calculate(model_probs, _toy_odds())
    home = out[out["outcome_name"] == "HOME"].iloc[0]
    assert home["edge_vs_best"] > 0
    assert home["edge_vs_consensus"] > 0
    assert home["best_book"] in {"draftkings", "fanduel"}


def test_no_edge_when_model_matches_market():
    model_probs = pd.DataFrame([
        {"event_id": "g1", "market": "h2h", "outcome_name": "HOME", "model_prob": 0.5},
        {"event_id": "g1", "market": "h2h", "outcome_name": "AWAY", "model_prob": 0.5},
    ])
    out = calculate(model_probs, _toy_odds())
    home = out[out["outcome_name"] == "HOME"].iloc[0]
    # Expect small (~0) edge after vig removal
    assert abs(home["edge_vs_consensus"]) < 0.05
