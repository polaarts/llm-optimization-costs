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


# ---------------------------------------------------------------------------
# Iteration 2: Wilcoxon pairwise test + correction ablation
# ---------------------------------------------------------------------------
def test_racing_wilcoxon_pairwise_keeps_best():
    """Wilcoxon should be at least as conservative as t-test on ties."""
    candidates = make_candidates(["a", "b", "c", "d"])
    targets = {c.id: 0.7 for c in candidates}
    dataset = [{"x": i} for i in range(20)]
    racing = RacingEvaluator(
        block_size=5, alpha=0.05, n_survive=1, z_max=4,
        pairwise_test="wilcoxon", correction="holm",
    )
    rng = np.random.default_rng(123)
    result = racing.run(candidates, dataset, _fake_evaluate_factory(targets), rng=rng)
    assert len(result.survivors) == 1


def test_racing_ttest_no_correction_keeps_best():
    """CAPO paper behaviour: paired t-test without multiple-testing correction.

    On a tied population the racing loop must still keep exactly one survivor
    (i.e. it should not over-eliminate). The test also asserts that with a
    clear loser and ``correction='none'`` the loser is still eliminated.
    """
    candidates = make_candidates(["a", "b", "c", "d"])
    targets = {c.id: 0.7 for c in candidates}
    dataset = [{"x": i} for i in range(20)]
    racing = RacingEvaluator(
        block_size=5, alpha=0.05, n_survive=1, z_max=4,
        pairwise_test="ttest", correction="none",
    )
    rng = np.random.default_rng(123)
    result = racing.run(candidates, dataset, _fake_evaluate_factory(targets), rng=rng)
    assert len(result.survivors) == 1

    # With a clear loser and no correction, the loser must still go.
    candidates2 = make_candidates(["good", "good", "good", "bad"])
    targets2 = {"c00": 0.9, "c01": 0.9, "c02": 0.9, "c03": 0.4}
    racing2 = RacingEvaluator(
        block_size=5, alpha=0.05, n_survive=2, z_max=6,
        pairwise_test="ttest", correction="none",
    )
    rng2 = np.random.default_rng(0)
    result2 = racing2.run(candidates2, dataset, _fake_evaluate_factory(targets2), rng=rng2)
    assert "c03" not in {s.id for s in result2.survivors}


def test_racing_correction_bonferroni_runs():
    """Bonferroni is the strictest of the three — racing must still finish."""
    candidates = make_candidates(["a", "b", "c", "d"])
    targets = {c.id: 0.7 for c in candidates}
    dataset = [{"x": i} for i in range(20)]
    racing = RacingEvaluator(
        block_size=5, alpha=0.05, n_survive=1, z_max=4,
        pairwise_test="ttest", correction="bonferroni",
    )
    rng = np.random.default_rng(123)
    result = racing.run(candidates, dataset, _fake_evaluate_factory(targets), rng=rng)
    # On a tied population at least one candidate must survive.
    assert len(result.survivors) >= 1


def test_racing_rejects_invalid_correction():
    import pytest
    with pytest.raises(ValueError):
        RacingEvaluator(correction="bh")  # not in the allow-list
