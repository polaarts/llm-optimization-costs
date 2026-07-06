"""Random-seed utilities.

The LLM API itself is non-deterministic at temperature > 0, but every other
source of randomness in the project (initial pool sampling, block shuffling,
critic invocations) must be reproducible per seed. This module centralises
that.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

# torch is optional — most code paths don't need it but the racing
# evaluator and the LLM client will call `set_seed` uniformly.
try:  # pragma: no cover - import-time fallback
    import torch  # type: ignore

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """Seed every RNG we can get our hands on.

    Parameters
    ----------
    seed:
        The integer seed to use everywhere.
    deterministic_torch:
        If True and torch is installed, enable cuDNN deterministic mode. This
        is slow and only useful for unit tests of the racing evaluator.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # LiteLLM exposes a `seed` argument; we just record the env so callers
    # can read it. LLMClient pulls it back out via `os.environ["LITELLM_SEED"]`.
    os.environ["LITELLM_SEED"] = str(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_seed() -> Optional[int]:
    """Return the seed previously set via `set_seed`, or None."""
    raw = os.environ.get("LITELLM_SEED")
    return int(raw) if raw is not None else None
