"""Scoring utilities.

`MultiObjectiveScorer` combines three signals into a single scalar that the
optimisers can rank candidates by:

  * accuracy  – fraction of correctly answered questions
  * cost_in   – mean input tokens per question
  * cost_out  – mean output tokens per question

For `expected_short` rows we use **exact match** (normalised, case-insensitive,
accent-insensitive). For `expected_long` rows we use **LLM-as-judge** with a
fixed rubric, which is more robust than simple overlap. The judge itself is
budgeted: the scorer falls back to a length-aware heuristic if the judge is
disabled.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .llm_client import LLMClient
from .utils.logging import get_logger

try:
    # rapidfuzz provides fast, C-backed Levenshtein ratios. Required by the
    # iteration-2 scorer to recognise short answers that the model wraps in a
    # verbose explanation (e.g. "El cuerpo humano adulto tiene 206 huesos" for
    # the expected answer "206").
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except ImportError:  # pragma: no cover - exercised only when missing dep
    _rf_fuzz = None

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_FINAL_ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: str) -> str:
    """Return the text inside <final_answer>...</final_answer>, or the raw text."""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------
def _normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace — for exact-match comparison."""
    if text is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents.lower()).strip()


def _fuzzy_short_score(prediction: str, expected: str) -> float:
    """Levenshtein-style ratio between normalised prediction and expected.

    Uses ``rapidfuzz.fuzz.token_set_ratio`` because short answers often appear
    inside verbose model outputs (e.g. expected ``"206"`` appears verbatim
    inside ``"El cuerpo humano adulto tiene 206 huesos"``). Character-level
    ratios penalise the extra words heavily; token-set ratio compares the
    intersection of token sets and is robust to extra surrounding tokens.

    Returns a score in [0, 1]. Falls back to ``difflib.SequenceMatcher.ratio``
    when rapidfuzz is not installed so the scorer stays usable in minimal
    environments (the stdlib path is slower but functionally equivalent on
    small strings).
    """
    a = _normalise(prediction)
    b = _normalise(expected)
    if not a or not b:
        return 0.0
    if _rf_fuzz is not None:
        return float(_rf_fuzz.token_set_ratio(a, b)) / 100.0
    # Stdlib fallback: SequenceMatcher.ratio returns the same [0, 1] range.
    # It is character-based and therefore stricter than token_set_ratio, but
    # it provides a usable answer when rapidfuzz is unavailable.
    import difflib
    return float(difflib.SequenceMatcher(None, a, b).ratio())


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class EvaluationRow:
    """Result of evaluating a single question."""

    id: str
    question: str
    expected_short: str
    expected_long: str
    response_text: str
    extracted: str
    correct: bool
    judge_score: Optional[float] = None  # 0..1, only populated for long rows
    fuzzy_score: Optional[float] = None  # 0..1, only populated for short rows
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class CandidateScore:
    """Aggregate metrics for a candidate prompt over a batch of rows."""

    accuracy: float
    accuracy_short: float
    accuracy_long: float
    fuzzy_short_accuracy: float  # mean fuzzy_score over short rows (0..1)
    judge_score_long: float
    n: int
    n_short: int
    n_long: int
    tokens_in_total: int
    tokens_out_total: int
    tokens_in_mean: float
    tokens_out_mean: float
    score: float  # cost-aware scalar used by the racing evaluator
    rows: list[EvaluationRow] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "accuracy_short": self.accuracy_short,
            "accuracy_long": self.accuracy_long,
            "fuzzy_short_accuracy": self.fuzzy_short_accuracy,
            "judge_score_long": self.judge_score_long,
            "n": self.n,
            "n_short": self.n_short,
            "n_long": self.n_long,
            "tokens_in_total": self.tokens_in_total,
            "tokens_out_total": self.tokens_out_total,
            "tokens_in_mean": self.tokens_in_mean,
            "tokens_out_mean": self.tokens_out_mean,
            "score": self.score,
        }


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "Eres un juez de respuestas. Tu única tarea es asignar una puntuación de "
    "0 a 1 que mida si la respuesta del modelo contiene la información clave "
    "de la respuesta de referencia. Responde SOLO con un JSON con la forma "
    '{"score": <número entre 0 y 1>, "reason": "<justificación corta>"}.'
)
JUDGE_USER_TEMPLATE = (
    "Pregunta:\n{question}\n\n"
    "Respuesta de referencia:\n{reference}\n\n"
    "Respuesta del modelo:\n{response}\n\n"
    "Devuelve únicamente el JSON pedido."
)


def _parse_judge_json(text: str) -> Optional[float]:
    """Extract a 0..1 score from the judge's response, robust to formatting noise."""
    if not text:
        return None
    text = text.strip()
    # Find the first {...} block in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # No JSON braces; try to parse the text itself as a number.
        try:
            return float(text)
        except ValueError:
            return None
    block = text[start : end + 1]
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        # Sometimes the model returns a number directly; try that.
        try:
            return float(block.strip())
        except ValueError:
            return None
    score = obj.get("score")
    if isinstance(score, (int, float)):
        return max(0.0, min(1.0, float(score)))
    return None


# ---------------------------------------------------------------------------
# MultiObjectiveScorer
# ---------------------------------------------------------------------------
class MultiObjectiveScorer:
    """Evaluate a candidate prompt and produce a `CandidateScore`.

    Parameters
    ----------
    alpha:
        Trade-off between accuracy and *normalised* input cost. The final
        scalar is `accuracy - alpha * cost_in_norm` where `cost_in_norm` is
        `tokens_in_mean / max_input_tokens_baseline`. Setting alpha=0 disables
        the cost penalty (pure accuracy).
    beta:
        Trade-off between accuracy and normalised output cost (CROP-style).
        Setting beta=0 disables it.
    max_input_tokens_baseline:
        Token budget used to normalise input cost. Calibrated against the
        longest prompt observed in the seed population.
    max_output_tokens_baseline:
        Same idea for output cost.
    use_judge:
        If True, long-answer rows are graded with an LLM judge. If False the
        scorer falls back to a token-overlap heuristic (cheap and good enough
        for offline tests).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.0,
        beta: float = 0.0,
        max_input_tokens_baseline: float = 600.0,
        max_output_tokens_baseline: float = 200.0,
        use_judge: bool = True,
        fuzzy_threshold: float = 0.85,
        client: Optional[LLMClient] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.max_input_tokens_baseline = max_input_tokens_baseline
        self.max_output_tokens_baseline = max_output_tokens_baseline
        self.use_judge = use_judge
        self.fuzzy_threshold = fuzzy_threshold
        self.client = client or LLMClient()

    # ------------------------------------------------------------------
    # Single-row evaluation
    # ------------------------------------------------------------------
    def _is_short_correct(self, prediction: str, expected: str) -> tuple[bool, float]:
        """Decide whether a short answer is correct using the fuzzy scorer.

        Returns ``(correct, fuzzy_score)`` where ``fuzzy_score`` is the raw
        continuous similarity in [0, 1] (always populated for short rows). The
        binary ``correct`` flag is derived from the configured threshold so the
        caller can keep using boolean correctness in the racing loop while
        still surfacing the continuous score in the analysis layer.
        """
        score = _fuzzy_short_score(prediction, expected)
        return score >= self.fuzzy_threshold, score

    def _judge_long(self, question: str, response: str, reference: str) -> float:
        if not self.use_judge:
            # Cheap fallback: token overlap (Jaccard) — bounded between 0 and 1.
            a = set(_normalise(response).split())
            b = set(_normalise(reference).split())
            if not a or not b:
                return 0.0
            return float(len(a & b) / len(a | b))
        try:
            resp = self.client.complete(
                prompt=JUDGE_USER_TEMPLATE.format(
                    question=question, reference=reference, response=response
                ),
                system=JUDGE_SYSTEM,
                temperature=0.0,
                role="judge",
            )
        except Exception as exc:  # pragma: no cover - API failure fallback
            get_logger().log(event="judge_error", error=str(exc))
            return 0.0
        score = _parse_judge_json(resp.text)
        return score if score is not None else 0.0

    def _evaluate_row(self, row: dict, response_text: str, tokens_in: int, tokens_out: int) -> EvaluationRow:
        extracted = extract_final_answer(response_text)
        # Decide whether the row is "short" or "long" based on the expected answer.
        is_short = len(_normalise(row["expected_short"]).split()) <= 5
        if is_short:
            correct, fuzzy_score = self._is_short_correct(extracted, row["expected_short"])
            return EvaluationRow(
                id=row["id"],
                question=row["question"],
                expected_short=row["expected_short"],
                expected_long=row["expected_long"],
                response_text=response_text,
                extracted=extracted,
                correct=correct,
                judge_score=None,
                fuzzy_score=fuzzy_score,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        judge = self._judge_long(row["question"], response_text, row["expected_long"])
        # Treat judge >= 0.5 as correct for the accuracy binary, but keep the
        # continuous judge score for finer analysis.
        return EvaluationRow(
            id=row["id"],
            question=row["question"],
            expected_short=row["expected_short"],
            expected_long=row["expected_long"],
            response_text=response_text,
            extracted=extracted,
            correct=judge >= 0.5,
            judge_score=judge,
            fuzzy_score=None,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    # ------------------------------------------------------------------
    # Aggregate scoring
    # ------------------------------------------------------------------
    def aggregate(self, rows: Sequence[EvaluationRow]) -> CandidateScore:
        n = len(rows)
        n_short = sum(1 for r in rows if r.judge_score is None)
        n_long = n - n_short
        if n == 0:
            return CandidateScore(
                accuracy=0.0, accuracy_short=0.0, accuracy_long=0.0,
                fuzzy_short_accuracy=0.0, judge_score_long=0.0,
                n=0, n_short=0, n_long=0,
                tokens_in_total=0, tokens_out_total=0,
                tokens_in_mean=0.0, tokens_out_mean=0.0, score=0.0,
                rows=[],
            )
        accuracy = sum(1 for r in rows if r.correct) / n
        acc_short = (
            sum(1 for r in rows if r.judge_score is None and r.correct) / n_short
            if n_short else 0.0
        )
        acc_long = (
            sum(1 for r in rows if r.judge_score is not None and r.correct) / n_long
            if n_long else 0.0
        )
        fuzzy_short = (
            sum(r.fuzzy_score or 0.0 for r in rows if r.judge_score is None) / n_short
            if n_short else 0.0
        )
        judge_long = (
            sum(r.judge_score or 0.0 for r in rows if r.judge_score is not None) / n_long
            if n_long else 0.0
        )
        tin = sum(r.tokens_in for r in rows)
        tout = sum(r.tokens_out for r in rows)
        tin_mean = tin / n
        tout_mean = tout / n
        cost_in_norm = tin_mean / max(self.max_input_tokens_baseline, 1.0)
        cost_out_norm = tout_mean / max(self.max_output_tokens_baseline, 1.0)
        score = accuracy - self.alpha * cost_in_norm - self.beta * cost_out_norm
        return CandidateScore(
            accuracy=accuracy,
            accuracy_short=acc_short,
            accuracy_long=acc_long,
            fuzzy_short_accuracy=fuzzy_short,
            judge_score_long=judge_long,
            n=n,
            n_short=n_short,
            n_long=n_long,
            tokens_in_total=tin,
            tokens_out_total=tout,
            tokens_in_mean=tin_mean,
            tokens_out_mean=tout_mean,
            score=score,
            rows=list(rows),
        )

    # ------------------------------------------------------------------
    # Convenience: run an entire batch end-to-end
    # ------------------------------------------------------------------
    def score_prompt(
        self,
        prompt: str,
        rows: Sequence[dict],
        *,
        # 1024 leaves room for ~250 reasoning tokens + a complete <final_answer>
        # block on M2.5-highspeed. With the previous 256 the model frequently
        # hit the ceiling mid-reasoning and produced no visible answer, which
        # collapsed accuracy to 0 (see run capos with seed 0 on 2026-07-03).
        max_tokens: int = 1024,
    ) -> CandidateScore:
        evaluations: list[EvaluationRow] = []
        for row in rows:
            full_prompt = (
                f"{prompt}\n\n---\n\nPregunta: {row['question']}\n"
                f"Responde dentro de <final_answer>...</final_answer>."
            )
            try:
                resp = self.client.complete(
                    full_prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    role="evaluation",
                )
            except Exception as exc:  # pragma: no cover - API failure fallback
                get_logger().log(event="eval_error", id=row["id"], error=str(exc))
                evaluations.append(
                    EvaluationRow(
                        id=row["id"],
                        question=row["question"],
                        expected_short=row["expected_short"],
                        expected_long=row["expected_long"],
                        response_text="",
                        extracted="",
                        correct=False,
                        judge_score=None,
                        tokens_in=0,
                        tokens_out=0,
                    )
                )
                continue
            evaluations.append(
                self._evaluate_row(row, resp.text, resp.tokens_in, resp.tokens_out)
            )
        return self.aggregate(evaluations)


# ---------------------------------------------------------------------------
# Pareto helper
# ---------------------------------------------------------------------------
def pareto_front(points: Iterable[tuple[float, float, float]]) -> list[int]:
    """Return indices of non-dominated points in (accuracy, -cost_in, -cost_out).

    A point `p` is dominated by `q` if `q` is at least as good in all
    objectives and strictly better in at least one. We maximise accuracy and
    minimise both cost_in and cost_out — the helper negates the costs so the
    caller can keep the natural interpretation of "higher is better".
    """
    pts = [(acc, -ci, -co, i) for i, (acc, ci, co) in enumerate(points)]
    front: list[int] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if q[0] >= p[0] and q[1] >= p[1] and q[2] >= p[2] and (
                q[0] > p[0] or q[1] > p[1] or q[2] > p[2]
            ):
                dominated = True
                break
        if not dominated:
            front.append(pts[i][3])
    return sorted(front)
