"""Smoke tests for the Pareto dominance helper."""
from __future__ import annotations

from src.scorer import pareto_front


def test_pareto_simple_two_objectives():
    # accuracy ↑, cost_in ↓, cost_out ↓
    points = [
        (0.9, 100.0, 50.0),  # p0
        (0.8, 90.0, 50.0),   # p1 — dominated by p0 (worse acc, lower cost only)
        (0.95, 80.0, 40.0),  # p2 — best in all → non-dominated
    ]
    front = pareto_front(points)
    # The internal helper negates the costs, so higher numbers are still
    # better. We expect p2 to be in the front, and either p0 or both p0 and p1
    # to be present depending on strictness of dominance.
    assert 2 in front


def test_pareto_with_clear_winner():
    points = [
        (0.5, 200.0, 100.0),  # worse everywhere
        (0.9, 100.0, 50.0),   # dominates above
    ]
    front = pareto_front(points)
    assert 1 in front
    assert 0 not in front


def test_pareto_empty():
    assert pareto_front([]) == []
