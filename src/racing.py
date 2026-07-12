"""RacingEvaluator with Holm-Bonferroni early elimination.

The CAPO paper uses a paired t-test **without** correction for multiple
testing (cf. Algorithm 2 in Zehle et al. 2025). For the prototype we follow
the spec given to us and apply Holm-Bonferroni on top: this is more
conservative and provides a cleaner statistical guarantee.

The evaluator operates on *blocks* of fixed size `b`. After each block we
compute, for every surviving candidate, the set of p-values against every
other candidate, sort them, and apply Holm's step-down procedure at level
`alpha`. Any candidate that fails the threshold against `n_survive` or more
other candidates is eliminated.

This conservative behaviour is exactly what the smoke tests check: if every
candidate is genuinely equally good, the racing evaluator must keep all of
them and the best one must survive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    """A single candidate prompt with its running score buffer."""
    id: str
    prompt: str
    block_scores: list[float] = field(default_factory=list)  # 0..1 per item in block
    eliminated_at_block: Optional[int] = None
    elimination_reason: Optional[str] = None

    @property
    def n(self) -> int:
        return len(self.block_scores)

    @property
    def mean(self) -> float:
        return float(np.mean(self.block_scores)) if self.block_scores else 0.0


@dataclass
class RacingResult:
    survivors: list[Candidate]
    eliminated: list[Candidate]
    blocks_used: int
    per_block_summary: list[dict]


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Return `True` for each p-value that *passes* the Holm step-down test.

    The classic Holm procedure: sort p-values ascending, then for each k
    compare p_(k) to alpha / (m - k + 1). The first rejection stops the
    chain: all subsequent hypotheses are also rejected.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    sorted_p = np.asarray(pvals)[order]
    reject = np.zeros(m, dtype=bool)
    stopped = False
    for k in range(m):
        threshold = alpha / (m - k)
        if sorted_p[k] <= threshold and not stopped:
            reject[order[k]] = False  # this hypothesis survives
        else:
            # First failure (or any later one) — every remaining is rejected
            # for "being at least as extreme" in the Holm sense.
            stopped = True
            reject[order[k]] = True
    return [bool(x) for x in reject]


# ---------------------------------------------------------------------------
# RacingEvaluator
# ---------------------------------------------------------------------------
class RacingEvaluator:
    """Population-based racing with Holm-Bonferroni early elimination.

    Parameters
    ----------
    block_size:
        Number of items per mini-batch.
    alpha:
        Family-wise significance level for Holm-Bonferroni.
    n_survive:
        Target number of survivors. The loop runs until either this is
        reached or `z_max` blocks have been consumed.
    z_max:
        Hard cap on the number of blocks evaluated. Mirrors CAPO's `z_max`.
    pairwise_test:
        `ttest` (paired t-test, the CAPO default) or `wilcoxon` (paired
        Wilcoxon, more robust on tiny blocks).
    correction:
        Multiple-testing correction applied on top of the pairwise test:

        * ``"holm"`` (default) — Holm-Bonferroni step-down, the original
          prototype behaviour.
        * ``"none"`` — paired t-test without correction, matching the CAPO
          paper's Algorithm 2 and used for the ablation in
          ``reports/informe.md`` §8.2.
        * ``"bonferroni"`` — classical Bonferroni (single-step) for a
          middle-ground reference.

        Ignored when ``pairwise_test == "wilcoxon"`` because the Wilcoxon
        test is already non-parametric and the original implementation only
        runs the correction step on the t-test branch.
    """

    def __init__(
        self,
        *,
        block_size: int = 5,
        alpha: float = 0.05,
        n_survive: int = 1,
        z_max: int = 10,
        pairwise_test: str = "ttest",
        correction: str = "holm",
    ) -> None:
        if pairwise_test not in {"ttest", "wilcoxon"}:
            raise ValueError("pairwise_test must be 'ttest' or 'wilcoxon'")
        if correction not in {"holm", "none", "bonferroni"}:
            raise ValueError("correction must be 'holm', 'none', or 'bonferroni'")
        self.block_size = block_size
        self.alpha = alpha
        self.n_survive = n_survive
        self.z_max = z_max
        self.pairwise_test = pairwise_test
        self.correction = correction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _pairwise_pvalue(self, a: np.ndarray, b: np.ndarray) -> float:
        """Paired test; return 1.0 when there is no signal (avoids NaNs)."""
        if len(a) < 2 or len(b) < 2:
            return 1.0
        if np.allclose(a, b):
            return 1.0
        try:
            if self.pairwise_test == "ttest":
                _, p = sp_stats.ttest_rel(a, b)
            else:
                # Wilcoxon: zero_method='wilcox' to handle ties gracefully.
                diffs = a - b
                if np.all(diffs == 0):
                    return 1.0
                _, p = sp_stats.wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            return 1.0
        if math.isnan(p):
            return 1.0
        return float(p)

    def _evaluate_block(
        self,
        survivors: list[Candidate],
        batch_items: Sequence[dict],
        evaluate_fn: Callable[[Candidate, Sequence[dict]], list[float]],
    ) -> None:
        """Call `evaluate_fn` for each surviving candidate on the same block."""
        for cand in survivors:
            scores = evaluate_fn(cand, batch_items)
            cand.block_scores.extend(scores)

    def _eliminate(
        self,
        candidates: list[Candidate],
        *,
        block_index: int,
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Apply the configured correction to the current score buffer.

        A candidate is eliminated iff, for at least `n_survive` opponents,
        the corrected procedure rejects the null "this candidate is no worse
        than the opponent". Equivalently: there are at least `n_survive`
        opponents that the test declares *significantly* better than this
        candidate.

        Behaviour by ``correction`` value:

        * ``"holm"`` — Holm-Bonferroni step-down (default, original
          prototype behaviour).
        * ``"none"`` — single-step comparison at level ``alpha`` (matches
          the CAPO paper, no correction for multiple testing).
        * ``"bonferroni"`` — classical Bonferroni (single-step) at level
          ``alpha / m``.
        """
        active = [c for c in candidates if c.eliminated_at_block is None]
        if len(active) <= self.n_survive:
            return active, [c for c in candidates if c.eliminated_at_block is not None]

        eliminated_now: list[Candidate] = []
        for cand in active:
            pvals: list[float] = []
            for other in active:
                if other is cand:
                    continue
                a = np.asarray(cand.block_scores, dtype=float)
                b = np.asarray(other.block_scores, dtype=float)
                n = min(len(a), len(b))
                pvals.append(self._pairwise_pvalue(a[:n], b[:n]))
            opponents_better = self._count_opponents_better(pvals)
            if opponents_better >= max(1, self.n_survive):
                cand.eliminated_at_block = block_index
                cand.elimination_reason = (
                    f"{self.correction}: {opponents_better} opponents significantly better"
                )
                eliminated_now.append(cand)

        new_active = [c for c in active if c.eliminated_at_block is None]
        already_out = [c for c in candidates if c.eliminated_at_block is not None and c not in eliminated_now]
        return new_active, eliminated_now + already_out

    def _count_opponents_better(self, pvals: list[float]) -> int:
        """Return how many opponents are *significantly* better than the candidate.

        Dispatches to the configured multiple-testing correction:

        * ``holm``        — Holm-Bonferroni step-down (the original prototype
          logic). ``opponents_better`` counts opponents whose Holm step
          rejects the null at level ``alpha``.
        * ``bonferroni``  — single-step Bonferroni at ``alpha / m``.
        * ``none``        — raw ``p <= alpha`` per opponent, matching the
          CAPO paper Algorithm 2.
        """
        m = len(pvals)
        if m == 0:
            return 0
        if self.correction == "holm":
            survive_flags = holm_bonferroni(pvals, alpha=self.alpha)
            # `survive_flags[i] == False` ⇒ Holm rejected the null ⇒
            # opponent i is significantly better than `cand`.
            return sum(1 for f in survive_flags if not f)
        if self.correction == "bonferroni":
            threshold = self.alpha / m
            return sum(1 for p in pvals if p <= threshold)
        # "none": CAPO paper behaviour — raw p <= alpha per opponent.
        return sum(1 for p in pvals if p <= self.alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        candidates: list[Candidate],
        dataset: Sequence[dict],
        evaluate_fn: Callable[[Candidate, Sequence[dict]], list[float]],
        *,
        shuffle_blocks: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> RacingResult:
        """Run the racing loop and return a `RacingResult`."""
        if rng is None:
            rng = np.random.default_rng(0)
        n = len(dataset)
        b = self.block_size
        if n < b:
            raise ValueError(
                f"Dataset has {n} items but block_size={b}; provide at least b items."
            )
        z = min(self.z_max, n // b)
        if z == 0:
            z = 1
        # Split the dataset into z contiguous blocks.
        blocks = [dataset[i * b : (i + 1) * b] for i in range(z)]
        if shuffle_blocks:
            order = list(range(z))
            rng.shuffle(order)
            blocks = [blocks[i] for i in order]

        eliminated_total: list[Candidate] = []
        per_block_summary: list[dict] = []
        for j, block in enumerate(blocks, start=1):
            active = [c for c in candidates if c.eliminated_at_block is None]
            self._evaluate_block(active, block, evaluate_fn)
            new_active, eliminated = self._eliminate(candidates, block_index=j)
            eliminated_total = [c for c in eliminated if c not in eliminated_total]
            per_block_summary.append(
                {
                    "block": j,
                    "n_active_before": len(active),
                    "n_active_after": len(new_active),
                    "n_eliminated_this_block": len(eliminated)
                    - len([c for c in eliminated if c.eliminated_at_block is not None and c.eliminated_at_block < j]),
                }
            )
            if len(new_active) <= self.n_survive:
                break

        survivors = [c for c in candidates if c.eliminated_at_block is None]
        if len(survivors) > self.n_survive:
            survivors.sort(key=lambda c: c.mean, reverse=True)
            tail = survivors[self.n_survive :]
            for c in tail:
                c.eliminated_at_block = -1
                c.elimination_reason = "truncated to n_survive"
                eliminated_total.append(c)
            survivors = survivors[: self.n_survive]
        return RacingResult(
            survivors=survivors,
            eliminated=eliminated_total,
            blocks_used=j,
            per_block_summary=per_block_summary,
        )


# ---------------------------------------------------------------------------
# Convenience: build Candidate objects from raw prompts
# ---------------------------------------------------------------------------
def make_candidates(prompts: list[str]) -> list[Candidate]:
    return [Candidate(id=f"c{i:02d}", prompt=p) for i, p in enumerate(prompts)]
