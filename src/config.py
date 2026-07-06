"""Project configuration loader.

Reads environment variables from `.env` (if present) and exposes a single
`Settings` instance with typed access to every knob the rest of the code
needs. Keeping the configuration surface in one place makes the rest of the
codebase easier to test and avoids scattered `os.environ.get(...)` calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from repo root regardless of CWD so experiments and tests work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class PricingTier:
    """Cost per 1M tokens (USD) for a model."""

    input_per_million: float
    output_per_million: float


@dataclass(frozen=True)
class Settings:
    """Read-only container for all project settings."""

    api_key: str
    api_base: str
    model: str
    repo_root: Path
    data_path: Path
    raw_dir: Path
    tables_dir: Path
    figures_dir: Path
    pricing: dict[str, PricingTier] = field(default_factory=dict)

    # Default hyperparameters (mostly follow CAPO paper, Appendix C.4).
    alpha: float = 0.2  # significance level for the racing statistical test
    block_size: int = 30
    z_max: int = 10
    k_max: int = 3
    mu: int = 6
    crossovers_per_iter: int = 3
    gamma: float = 0.05  # length penalty (CAPO scalarisation)
    max_budget_usd: float = 5.0


def _build_default_pricing() -> dict[str, PricingTier]:
    """Return a per-model pricing table.

    Prices are USD per 1M tokens, taken from the official pricing sheet
    (consulted 2026-07-03). M3 is intentionally priced at the "before
    discount" tier ($0.60 input / $2.40 output for ≤ 512k input) because
    no discount tier is available without an explicit commercial contract.
    For prompts > 512k input tokens the cost doubles; this prototype stays
    well under that threshold (largest observed prompt ≈ 4k tokens).

    Prompt caching rates (read / write per 1M tokens) are recorded for
    reference but the prototype does not yet hit cached prompts because
    every call ships a fresh JSONL trace.
    """
    return {
        # M2.5 highspeed — same performance as M2.5 but faster (~100 tps).
        # Kept as a fallback / historical reference for the runs documented
        # in reports/informe.md §7.3.
        "MiniMax-M2.5-highspeed": PricingTier(
            input_per_million=0.20, output_per_million=1.20
        ),
        # M2.5 standard — slower, same price as the highspeed variant.
        "MiniMax-M2.5": PricingTier(
            input_per_million=0.20, output_per_million=1.20
        ),
        # M2.7 — standard tier. Input $0.30, output $1.20, caching read
        # $0.06, caching write $0.375.
        "MiniMax-M2.7": PricingTier(
            input_per_million=0.30, output_per_million=1.20
        ),
        # M2.7-highspeed — 2× the standard M2.7 price. Input $0.60, output
        # $2.40, caching read $0.06, caching write $0.375.
        "MiniMax-M2.7-highspeed": PricingTier(
            input_per_million=0.60, output_per_million=2.40
        ),
        # M3 — top-of-the-line, 1M context, multimodal. Default target for
        # CAPO × CROP. Prices quoted at the "before discount" tier because
        # the prototype has no commercial contract: input $0.60 / output
        # $2.40 for ≤ 512k input tokens. Beyond 512k input the price doubles
        # ($1.20 / $4.80); see reports/informe.md §8.1.
        "MiniMax-M3": PricingTier(
            input_per_million=0.60, output_per_million=2.40
        ),
    }


def load_settings() -> Settings:
    """Read environment, validate required keys, return a `Settings` object."""
    api_key = os.getenv("API_KEY", "").strip()
    api_base = os.getenv("URL_API_BASE", "https://api.minimax.io/v1").strip()
    model = os.getenv("MODEL", "MiniMax-M2.5-highspeed").strip()

    # We do NOT raise on a missing API key during pure code generation /
    # test runs: callers that actually want to hit the API should fail loud
    # and clear at the boundary (see `LLMClient._require_key`).
    return Settings(
        api_key=api_key,
        api_base=api_base,
        model=model,
        repo_root=_REPO_ROOT,
        data_path=_REPO_ROOT / "data" / "toy_qa.jsonl",
        raw_dir=_REPO_ROOT / "results" / "raw",
        tables_dir=_REPO_ROOT / "results" / "tables",
        figures_dir=_REPO_ROOT / "results" / "figures",
        pricing=_build_default_pricing(),
    )


# Module-level singleton — convenient for callers that don't want to thread
# settings through every function. Tests can monkeypatch the env vars and
# call `load_settings()` again to get a fresh instance.
SETTINGS: Settings = load_settings()
