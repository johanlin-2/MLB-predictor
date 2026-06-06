"""Tests for edge.vig_removal."""
from __future__ import annotations

import math

import pytest

from edge.vig_removal import (
    american_to_decimal,
    american_to_prob,
    market_no_vig,
    remove_vig,
    remove_vig_three_way,
)


def test_american_to_prob_negative():
    # -110 ⇒ 110 / 210 = 0.52380...
    assert math.isclose(american_to_prob(-110), 0.5238, abs_tol=1e-3)


def test_american_to_prob_pickem():
    assert american_to_prob(100) == 0.5


def test_american_to_prob_positive():
    # +150 ⇒ 100 / 250 = 0.40
    assert math.isclose(american_to_prob(150), 0.40, abs_tol=1e-6)


def test_american_to_decimal_round_trip():
    for o in (-200, -110, 100, 120, 250):
        d = american_to_decimal(o)
        # implied prob via decimal odds equals direct calc within tolerance
        assert math.isclose(1.0 / d, american_to_prob(o), abs_tol=1e-9)


def test_remove_vig_symmetric_market():
    a, b = remove_vig(0.5238, 0.5238)
    assert math.isclose(a, 0.5, abs_tol=1e-6)
    assert math.isclose(b, 0.5, abs_tol=1e-6)


def test_remove_vig_asymmetric_market_sums_to_one():
    a, b = remove_vig(0.574, 0.488)
    assert math.isclose(a + b, 1.0, abs_tol=1e-12)
    assert 0 < a < 1
    assert 0 < b < 1


def test_remove_vig_three_way_sums_to_one():
    a, b, c = remove_vig_three_way(0.4, 0.4, 0.3)
    assert math.isclose(a + b + c, 1.0, abs_tol=1e-12)


def test_remove_vig_rejects_zero_total():
    with pytest.raises(ValueError):
        remove_vig(0.0, 0.0)


def test_market_no_vig_full_path():
    # Standard -110/-110 line should give 50/50 after vig removal.
    h, a = market_no_vig(-110, -110)
    assert math.isclose(h, 0.5, abs_tol=1e-6)
    assert math.isclose(a, 0.5, abs_tol=1e-6)
