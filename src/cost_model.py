"""Centralised cost model.

Two responsibilities:
  1. Convert raw `tokens_in` / `tokens_out` numbers into USD.
  2. Provide `cost_in`, `cost_out`, `cost_total` for any candidate prompt and
     a history of LLM responses.

We never rely on `tiktoken` for the canonical numbers because the MiniMax
tokeniser is not publicly documented. The pipeline always trusts the
`usage` field returned by the API and only consults `tiktoken` for early
sanity checks or offline previews.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import SETTINGS, PricingTier


@dataclass(frozen=True)
class CostBreakdown:
    cost_in: float
    cost_out: float
    cost_total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cost_in": self.cost_in,
            "cost_out": self.cost_out,
            "cost_total": self.cost_total,
        }


def price_for(model: str) -> PricingTier:
    tier = SETTINGS.pricing.get(model)
    if tier is None:
        # Conservative fallback so a missing tier doesn't make every cost 0.
        return PricingTier(input_per_million=0.5, output_per_million=1.5)
    return tier


def usd_from_tokens(model: str, tokens_in: int, tokens_out: int) -> CostBreakdown:
    """Convert token counts to USD using the per-model pricing table."""
    tier = price_for(model)
    cost_in = (tokens_in / 1_000_000.0) * tier.input_per_million
    cost_out = (tokens_out / 1_000_000.0) * tier.output_per_million
    return CostBreakdown(cost_in=cost_in, cost_out=cost_out, cost_total=cost_in + cost_out)


def aggregate(records: Iterable[CostBreakdown]) -> CostBreakdown:
    """Sum a sequence of `CostBreakdown` records."""
    total_in = 0.0
    total_out = 0.0
    for r in records:
        total_in += r.cost_in
        total_out += r.cost_out
    return CostBreakdown(cost_in=total_in, cost_out=total_out, cost_total=total_in + total_out)
