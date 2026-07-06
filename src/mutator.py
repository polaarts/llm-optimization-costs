"""Prompt mutators.

Two operators are required by the spec (`paraphrase`, `add_constraint`) plus
an optional `swap_fewshot` for when there is time. The class is intentionally
light-weight: every operator returns a new prompt string and a structured
trace that the pipeline can log.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from .llm_client import LLMClient
from .utils.logging import get_logger

# ---------------------------------------------------------------------------
# Operator prompts
# ---------------------------------------------------------------------------
PARAPHRASE_SYSTEM = (
    "Eres un editor de prompts. Tu única tarea es reformular el prompt "
    "suministrado preservando su intención, tono y formato. Responde con el "
    "prompt reformulado directamente, sin prefijos, sin explicaciones y sin "
    "envolverlo en etiquetas."
)
PARAPHRASE_USER_TEMPLATE = (
    "Reformula el siguiente prompt, manteniendo su significado pero variando "
    "el estilo lingüístico (vocabulario, estructura de las oraciones, registro). "
    "No añadas instrucciones nuevas.\n\n"
    "Prompt original:\n{prompt}\n\n"
    "Devuelve únicamente el prompt reformulado, sin comillas ni etiquetas "
    "que lo envuelvan."
)

ADD_CONSTRAINT_TEMPLATE = (
    "{base}\n\nRestricción adicional: {constraint}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_PROMPT_TAG_RE = re.compile(r"<prompt>(.*?)</prompt>", re.IGNORECASE | re.DOTALL)

# Prefixes the model sometimes prepends when it wants to "explain" before the
# prompt. We strip these to avoid contaminating the candidate prompt with the
# model's commentary.
_LEADING_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"aquí\s+(?:va|está|tienes)\s+(?:el|la|los|las)?\s*(?:prompt)?\s*[:.\-]?\s*"
    r"|el\s+prompt\s+reformulado\s+es\s*[:.\-]?\s*"
    r"|prompt\s+reformulado\s*[:.\-]?\s*"
    r"|reformulaci[óo]n\s*[:.\-]?\s*"
    r"|nuevo\s+prompt\s*[:.\-]?\s*"
    r")",
    re.IGNORECASE,
)


def _strip_prompt_tag(text: str) -> str:
    """Extract the candidate prompt from the model's raw output.

    Handles three failure modes observed in production with M2.5/M3:

    1. Model wraps correctly in <prompt>...</prompt> → extract contents.
    2. Model emits unclosed ``<prompt>`` (token limit hit mid-response) → take
       everything after the opening tag.
    3. Model skips tags entirely and emits a leading phrase like
       "Aquí va el prompt: ..." → strip the prefix and return the remainder.
    """
    if not text:
        return ""
    text = text.strip()

    # Case 1: closed tag
    match = _PROMPT_TAG_RE.search(text)
    if match:
        return match.group(1).strip()

    # Case 2: unclosed opening tag (model hit max_tokens mid-tag)
    open_idx = text.lower().find("<prompt>")
    if open_idx != -1:
        after = text[open_idx + len("<prompt>") :].lstrip(" :\n-")
        if after:
            return after.strip()

    # Case 3: no tags at all, but a leading explanatory phrase
    return _LEADING_NOISE_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
@dataclass
class MutationTrace:
    operator: str
    parent_prompt: str
    child_prompt: str
    constraint: Optional[str] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------
class PromptMutator:
    """Apply genetic operators to a list of parent prompts.

    Parameters
    ----------
    paraphrase_pool:
        Number of paraphrases to produce per parent. Default 1 (one new
        candidate per parent). The racing evaluator will discard the bad
        ones downstream.
    constraint_candidates:
        A list of candidate constraints the `add_constraint` operator can
        attach. Each call picks one at random.
    swap_fewshot_prob:
        Probability of running the `swap_fewshot` operator (only used if
        examples are present in the prompt).
    client:
        Optional LLMClient. Defaults to the global one.
    """

    DEFAULT_CONSTRAINTS: list[str] = [
        "Responde en máximo 30 palabras.",
        "Responde de forma concisa, sin repetir la pregunta.",
        "Estructura la respuesta en un solo párrafo.",
        "Responde con la respuesta final primero, luego una breve justificación.",
    ]

    def __init__(
        self,
        *,
        paraphrase_pool: int = 1,
        constraint_candidates: Optional[list[str]] = None,
        swap_fewshot_prob: float = 0.0,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.paraphrase_pool = paraphrase_pool
        self.constraint_candidates = constraint_candidates or list(self.DEFAULT_CONSTRAINTS)
        self.swap_fewshot_prob = swap_fewshot_prob
        self.client = client or LLMClient()

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------
    def paraphrase(self, prompt: str, *, seed: Optional[int] = None) -> MutationTrace:
        """LLM-based paraphrase of the prompt instruction."""
        try:
            resp = self.client.complete(
                prompt=PARAPHRASE_USER_TEMPLATE.format(prompt=prompt),
                system=PARAPHRASE_SYSTEM,
                temperature=0.7,
                seed=seed,
                role="mutator_paraphrase",
            )
            new_prompt = _strip_prompt_tag(resp.text) or prompt
        except Exception as exc:  # pragma: no cover - API failure fallback
            get_logger().log(event="mutator_error", op="paraphrase", error=str(exc))
            new_prompt = prompt
        get_logger().log(
            event="mutation",
            operator="paraphrase",
            parent_len=len(prompt),
            child_len=len(new_prompt),
        )
        return MutationTrace(
            operator="paraphrase",
            parent_prompt=prompt,
            child_prompt=new_prompt,
            seed=seed,
        )

    def add_constraint(self, prompt: str, *, seed: Optional[int] = None) -> MutationTrace:
        """Append a length / style constraint to the prompt."""
        rng = random.Random(seed)
        constraint = rng.choice(self.constraint_candidates)
        new_prompt = ADD_CONSTRAINT_TEMPLATE.format(base=prompt.rstrip(), constraint=constraint)
        get_logger().log(
            event="mutation",
            operator="add_constraint",
            parent_len=len(prompt),
            child_len=len(new_prompt),
            constraint=constraint,
        )
        return MutationTrace(
            operator="add_constraint",
            parent_prompt=prompt,
            child_prompt=new_prompt,
            constraint=constraint,
            seed=seed,
        )

    def swap_fewshot(self, prompt: str, examples: list[str], *, seed: Optional[int] = None) -> MutationTrace:
        """Shuffle the few-shot examples block, if any is present."""
        if not examples:
            return MutationTrace(
                operator="swap_fewshot",
                parent_prompt=prompt,
                child_prompt=prompt,
            )
        rng = random.Random(seed)
        order = list(range(len(examples)))
        rng.shuffle(order)
        shuffled = [examples[i] for i in order]
        # Try to swap an "Examples:" block, otherwise just return as-is.
        new_prompt = prompt
        block_re = re.compile(r"(Ejemplos:?\s*[\s\S]*?)(?=\n[A-Z]|\Z)", re.MULTILINE)
        match = block_re.search(prompt)
        if match:
            joined = "\n".join(f"- {e}" for e in shuffled)
            new_prompt = prompt[: match.start()] + f"Ejemplos:\n{joined}\n" + prompt[match.end():]
        get_logger().log(event="mutation", operator="swap_fewshot")
        return MutationTrace(
            operator="swap_fewshot",
            parent_prompt=prompt,
            child_prompt=new_prompt,
        )

    # ------------------------------------------------------------------
    # Batch entry point
    # ------------------------------------------------------------------
    def mutate_pool(
        self,
        parents: list[str],
        *,
        seed: Optional[int] = None,
    ) -> list[MutationTrace]:
        """Apply a randomly-chosen operator to each parent.

        The distribution is uniform across the operators we have, so the
        pipeline gets a mix of paraphrases and constraint additions in each
        generation.
        """
        rng = random.Random(seed)
        traces: list[MutationTrace] = []
        for i, p in enumerate(parents):
            child_seed = (seed or 0) * 1000 + i
            op_choice = rng.random()
            if op_choice < 0.5:
                traces.append(self.paraphrase(p, seed=child_seed))
            elif op_choice < 0.9:
                traces.append(self.add_constraint(p, seed=child_seed))
            else:
                traces.append(self.swap_fewshot(p, [], seed=child_seed))
        return traces
