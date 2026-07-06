"""Lightweight statistical helpers used in the analysis stage."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import stats as sp_stats


def paired_wilcoxon(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test (two-sided). Returns p-value + effect."""
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if len(a_arr) != len(b_arr) or len(a_arr) < 2:
        return {"statistic": float("nan"), "pvalue": float("nan"), "n": int(len(a_arr))}
    # If every diff is zero, the test is undefined; report nans explicitly.
    diffs = a_arr - b_arr
    if np.all(diffs == 0):
        return {"statistic": 0.0, "pvalue": 1.0, "n": int(len(a_arr))}
    result = sp_stats.wilcoxon(a_arr, b_arr, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue), "n": int(len(a_arr))}


def bootstrap_ci(
    values: Iterable[float],
    *,
    statistic=np.mean,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for any scalar statistic."""
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    point = float(statistic(arr))
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = statistic(arr[idx], axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return {"point": point, "low": float(low), "high": float(high), "n": int(n)}


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    """Cohen's d for two independent samples (Hedges' g style; small-sample ok)."""
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if len(a_arr) < 2 or len(b_arr) < 2:
        return float("nan")
    n_a, n_b = len(a_arr), len(b_arr)
    var_a = a_arr.var(ddof=1)
    var_b = b_arr.var(ddof=1)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / np.sqrt(pooled))
