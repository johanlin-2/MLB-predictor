"""Fractional Kelly stake sizing with a configurable bankroll.

`recommended_stake_usd` = `fractional_kelly(edge, decimal_odds) * BANKROLL_USD`
clamped to a defensive maximum of 5 % of bankroll per game.

The bankroll can be updated between sessions via `config.update_bankroll(new)`.
"""
from __future__ import annotations

import config
from edge.vig_removal import american_to_decimal


# Hard cap to prevent runaway sizing when the edge estimate is wrong.
MAX_FRACTION_PER_GAME = 0.05


def fractional_kelly(edge: float, decimal_odds: float,
                     fraction: float = config.KELLY_FRACTION) -> float:
    """Fraction of bankroll to bet. Defaults to 25 % Kelly.

    edge : model_prob − market_no_vig_prob (already on the 0-1 scale)
    decimal_odds : payout multiplier (1.91 for -110, 2.5 for +150)
    """
    if edge <= 0:
        return 0.0
    payoff = decimal_odds - 1.0
    if payoff <= 0:
        return 0.0
    full_kelly = edge / payoff
    return max(0.0, min(MAX_FRACTION_PER_GAME, full_kelly * fraction))


def expected_value(model_prob: float, american_odds: int) -> float:
    """Per-unit EV at the quoted American line."""
    decimal = american_to_decimal(american_odds)
    return (model_prob * (decimal - 1.0)) - (1.0 - model_prob)


def recommended_stake_usd(edge: float, american_odds: int,
                          bankroll: float | None = None,
                          fraction: float = config.KELLY_FRACTION) -> float:
    """Dollar stake using current bankroll (or override)."""
    bankroll = bankroll if bankroll is not None else config.BANKROLL_USD
    decimal = american_to_decimal(american_odds)
    f = fractional_kelly(edge, decimal, fraction)
    return round(f * bankroll, 2)
