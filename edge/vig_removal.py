"""American-odds → no-vig probability conversion.

Pure functions, no I/O. Every other module that touches market prices flows
through here.
"""
from __future__ import annotations


def american_to_prob(odds: int | float) -> float:
    """Implied probability of an American moneyline (with vig)."""
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def american_to_decimal(odds: int | float) -> float:
    """Convert American odds → decimal odds (e.g. -110 → 1.909, +150 → 2.5)."""
    o = float(odds)
    if o > 0:
        return (o / 100.0) + 1.0
    return (100.0 / abs(o)) + 1.0


def remove_vig(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Normalise a two-way market so probabilities sum to 1.

    Uses the simple proportional method (a.k.a. "multiplicative vig removal").
    For typical MLB moneylines the overround is small (4-5 %) and proportional
    removal matches the Shin model within a few bps.
    """
    total = prob_a + prob_b
    if total <= 0:
        raise ValueError(f"non-positive total probability: {total}")
    return prob_a / total, prob_b / total


def remove_vig_three_way(prob_a: float, prob_b: float, prob_c: float) -> tuple[float, float, float]:
    """Same idea, three-way market. Not used for MLB regular season but ready
    for runline pushes and similar edge cases."""
    total = prob_a + prob_b + prob_c
    if total <= 0:
        raise ValueError(f"non-positive total probability: {total}")
    return prob_a / total, prob_b / total, prob_c / total


def market_no_vig(home_odds: int | float, away_odds: int | float) -> tuple[float, float]:
    """Convenience: American odds → vig-removed (home_prob, away_prob)."""
    return remove_vig(american_to_prob(home_odds), american_to_prob(away_odds))
