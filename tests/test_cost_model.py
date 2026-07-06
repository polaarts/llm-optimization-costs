"""Smoke tests for the cost model."""
from __future__ import annotations

import pytest

from src.cost_model import aggregate, usd_from_tokens


def test_usd_from_tokens_basic():
    breakdown = usd_from_tokens("MiniMax-M2.5-highspeed", tokens_in=1_000_000, tokens_out=0)
    assert breakdown.cost_in == pytest.approx(0.20, rel=1e-3)
    assert breakdown.cost_out == 0.0
    assert breakdown.cost_total == pytest.approx(0.20, rel=1e-3)


def test_usd_from_tokens_mixed():
    breakdown = usd_from_tokens("MiniMax-M2.5-highspeed", tokens_in=500_000, tokens_out=100_000)
    # 0.5 * 0.20 + 0.1 * 1.20 = 0.10 + 0.12 = 0.22
    assert breakdown.cost_in == pytest.approx(0.10, rel=1e-3)
    assert breakdown.cost_out == pytest.approx(0.12, rel=1e-3)
    assert breakdown.cost_total == pytest.approx(0.22, rel=1e-3)


def test_aggregate_sums_correctly():
    a = usd_from_tokens("MiniMax-M2.5-highspeed", 1_000_000, 100_000)
    b = usd_from_tokens("MiniMax-M2.5-highspeed", 500_000, 50_000)
    total = aggregate([a, b])
    assert total.cost_in == pytest.approx(a.cost_in + b.cost_in, rel=1e-6)
    assert total.cost_out == pytest.approx(a.cost_out + b.cost_out, rel=1e-6)


def test_unknown_model_falls_back_to_conservative_pricing():
    breakdown = usd_from_tokens("never-heard-of", tokens_in=1_000_000, tokens_out=0)
    # Conservative fallback must produce a positive, non-zero cost.
    assert breakdown.cost_in > 0
