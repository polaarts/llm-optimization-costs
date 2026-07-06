"""BrevityFeedbackGenerator — the CROP "Critic LM".

CROP's core idea: when the candidate's output is too long, ask a Critic LM to
suggest a shorter rewrite. The Critic also produces a `brevity_score` in
[0, 1] which the multi-objective scorer can use.

Following the project's spec, we **only** invoke the Critic on candidates
whose `cost_out` is above the 70th percentile of the current pool. This
keeps the optimisation cheap — the Critic itself costs tokens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .llm_client import LLMClient
from .utils.logging import get_logger

# ---------------------------------------------------------------------------
# Critic prompt
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = (
    "Eres un crítico de brevedad. Tu única tarea es sugerir una versión más "
    "corta del texto que recibirás, sin perder información esencial. Devuelve "
    "únicamente un JSON con la forma "
    '{"rewritten": "<texto>", "feedback": "<sugerencia breve>", '
    '"brevity_score": <0..1>}.'
)
CRITIC_USER_TEMPLATE = (
    "Texto original (longitud actual: {length} caracteres, "
    "objetivo: <= {target} caracteres):\n\n{output}\n\n"
    "Devuelve el JSON con la versión más corta y un puntaje de brevedad "
    "(1 = muy conciso, 0 = muy extenso)."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _safe_parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    match = _JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Result and Critic
# ---------------------------------------------------------------------------
@dataclass
class CriticResult:
    rewritten: str
    feedback: str
    brevity_score: float
    original_length: int
    new_length: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "rewritten": self.rewritten,
            "feedback": self.feedback,
            "brevity_score": self.brevity_score,
            "original_length": self.original_length,
            "new_length": self.new_length,
        }


class BrevityFeedbackGenerator:
    """Wrapper around the Critic LM."""

    def __init__(
        self,
        *,
        target_length: int = 80,
        max_output_tokens: int = 256,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.target_length = target_length
        self.max_output_tokens = max_output_tokens
        self.client = client or LLMClient()

    def should_invoke(self, candidate_cost_out: float, pool_cost_out: list[float]) -> bool:
        """Only call the Critic if the candidate sits above the 70th percentile.

        Returns False on an empty pool (no signal yet) so the first generation
        is never charged for a Critic call.
        """
        if not pool_cost_out:
            return False
        threshold = sorted(pool_cost_out)[int(0.7 * len(pool_cost_out))]
        return candidate_cost_out > threshold

    def critique(
        self,
        *,
        prompt: str,
        output: str,
        seed: Optional[int] = None,
    ) -> Optional[CriticResult]:
        """Return a `CriticResult`, or `None` if the Critic failed to produce JSON."""
        length = len(output)
        try:
            resp = self.client.complete(
                prompt=CRITIC_USER_TEMPLATE.format(
                    length=length, target=self.target_length, output=output
                ),
                system=CRITIC_SYSTEM,
                temperature=0.0,
                max_tokens=self.max_output_tokens,
                seed=seed,
                role="critic",
            )
        except Exception as exc:  # pragma: no cover - API failure
            get_logger().log(event="critic_error", error=str(exc))
            return None

        parsed = _safe_parse_json(resp.text)
        if not parsed:
            get_logger().log(event="critic_parse_error", raw=resp.text[:200])
            return None
        rewritten = str(parsed.get("rewritten", "")).strip() or output
        feedback = str(parsed.get("feedback", "")).strip()
        score = float(parsed.get("brevity_score", 0.5) or 0.5)
        score = max(0.0, min(1.0, score))
        return CriticResult(
            rewritten=rewritten,
            feedback=feedback,
            brevity_score=score,
            original_length=length,
            new_length=len(rewritten),
        )
