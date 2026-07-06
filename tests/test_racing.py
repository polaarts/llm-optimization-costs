"""Smoke tests for the racing evaluator.

These tests are critical: they prove (a) the Holm-Bonferroni step-down is
correctly implemented and (b) the racing loop is conservative enough to
keep the best candidate when there is no significant difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.racing import Candidate, RacingEvaluator, holm_bonferroni, make_candidates


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------
def test_holm_all_null_keeps_everything():
    # p-values all way above threshold → every hypothesis is rejected
    # (in Holm sense: "the first failure rejects all the rest").
    pvals = [0.9, 0.95, 0.99]
    out = holm_bonferroni(pvals, alpha=0.05)
    # First p > threshold → rejects; so all should be rejected
    assert out == [True, True, True]


def test_holm_all_significant_passes():
    pvals = [0.001, 0.002, 0.003]
    out = holm_bonferroni(pvals, alpha=0.05)
    # All p-values are well below alpha/k for any k, so all survive.
    assert out == [False, False, False]


def test_holm_mixed():
    pvals = [0.01, 0.04, 0.5]
    out = holm_bonferroni(pvals, alpha=0.05)
    # Sorted: 0.01 vs 0.05/3 = 0.0167 (survive), 0.04 vs 0.05/2 = 0.025 (rejected
    # because 0.04 > 0.025), 0.5 (rejected). Once we hit a rejection all
    # following are also rejected.
    assert out[0] is False
    assert out[1] is True
    assert out[2] is True


# ---------------------------------------------------------------------------
# Racing invariant: no false elimination of the best
# ---------------------------------------------------------------------------
def _fake_evaluate_factory(targets: dict[str, float]):
    """Return an evaluate_fn that yields the per-item score set in `targets`."""

    def evaluate(cand: Candidate, batch: list[dict]) -> list[float]:
        n = len(batch)
        # Deterministic noise pattern so the test is reproducible.
        base = targets[cand.id]
        # Inject small per-item variability from a hash of (cand, item)
        scores: list[float] = []
        for j in range(n):
            jitter = ((hash(cand.id) ^ j) % 7) / 100.0  # ±0.07
            scores.append(min(1.0, max(0.0, base + jitter - 0.03)))
        return scores

    return evaluate


def test_racing_keeps_best_when_all_equal():
    """If every candidate has the same underlying score, the best survives."""
    candidates = make_candidates(["a", "b", "c", "d"])
    targets = {c.id: 0.7 for c in candidates}
    dataset = [{"x": i} for i in range(20)]
    racing = RacingEvaluator(block_size=5, alpha=0.05, n_survive=1, z_max=4, pairwise_test="ttest")
    rng = np.random.default_rng(123)
    result = racing.run(candidates, dataset, _fake_evaluate_factory(targets), rng=rng)
    # The racing evaluator must not eliminate the *unique* best because there
    # isn't one: any candidate is acceptable as a survivor. We just check the
    # survivor count is exactly n_survive.
    assert len(result.survivors) == 1


def test_racing_eliminates_clear_loser():
    candidates = make_candidates(["good", "good", "good", "bad"])
    targets = {"c00": 0.9, "c01": 0.9, "c02": 0.9, "c03": 0.4}
    dataset = [{"x": i} for i in range(30)]
    racing = RacingEvaluator(block_size=5, alpha=0.05, n_survive=2, z_max=6, pairwise_test="ttest")
    rng = np.random.default_rng(0)
    result = racing.run(candidates, dataset, _fake_evaluate_factory(targets), rng=rng)
    # The clear loser should be eliminated, and the survivors should all
    # have an underlying score of 0.9.
    survivor_ids = {s.id for s in result.survivors}
    assert "c03" not in survivor_ids
    for s in result.survivors:
        assert s.mean >= 0.7
