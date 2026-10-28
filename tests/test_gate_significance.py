"""
scripts/gate_significance.py -- exact Poisson-binomial test of gate
accuracies against per-question chance (D1, D5).

    python -m pytest tests/test_gate_significance.py
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))

import gate_significance as gs  # noqa: E402


def test_equal_probabilities_reduce_to_the_binomial():
    dist = gs.poisson_binomial([0.25] * 10)
    expected = [math.comb(10, k) * 0.25 ** k * 0.75 ** (10 - k) for k in range(11)]
    assert dist == pytest.approx(expected) and sum(dist) == pytest.approx(1.0)


def test_mixed_probabilities():
    assert gs.poisson_binomial([0.5, 1 / 3]) == pytest.approx([1 / 3, 1 / 2, 1 / 6])


def test_significance_at_chance_and_below():
    ps = [0.25] * 400
    at = gs.significance(0.25, ps)
    below = gs.significance(0.18, ps)

    assert (at["hits"], at["z"]) == (100, 0.0) and at["p"] == pytest.approx(1.0, abs=0.06)
    assert below["hits"] == 72 and below["z"] < -3 and below["p"] < 0.002
    assert below["se"] == pytest.approx(math.sqrt(0.18 * 0.82 / 400))


def test_test_split_is_refused():
    with pytest.raises(ValueError, match="never gated"):
        gs.option_counts("ARC-Easy", "test")
