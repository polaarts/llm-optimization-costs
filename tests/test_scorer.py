"""Smoke tests for the scorer: answer extraction, exact-match, judge parsing."""
from __future__ import annotations

import json

from src.scorer import (
    CandidateScore,
    EvaluationRow,
    MultiObjectiveScorer,
    _fuzzy_short_score,
    _normalise,
    _parse_judge_json,
    extract_final_answer,
)


def test_extract_final_answer_with_tag():
    text = "Lo siento, no estoy seguro.\n<final_answer>París</final_answer> y nada más."
    assert extract_final_answer(text) == "París"


def test_extract_final_answer_without_tag():
    text = "  La capital es Madrid  "
    assert extract_final_answer(text) == "La capital es Madrid"


def test_normalise_lowercase_no_accents():
    assert _normalise("  París  ") == "paris"
    assert _normalise("Árbol MÁGICO") == "arbol magico"


def test_parse_judge_json_clean():
    assert _parse_judge_json(json.dumps({"score": 0.7, "reason": "ok"})) == 0.7


def test_parse_judge_json_extra_text():
    txt = "Aquí está mi veredicto:\n" + json.dumps({"score": 0.42}) + "\nFin."
    assert _parse_judge_json(txt) == 0.42


def test_parse_judge_json_bare_number():
    assert _parse_judge_json("0.5") == 0.5


def test_parse_judge_json_garbage():
    assert _parse_judge_json("no json here") is None


def test_candidate_score_aggregate_basic():
    rows = [
        EvaluationRow(
            id="a", question="q1", expected_short="x", expected_long="long",
            response_text="<final_answer>x</final_answer>", extracted="x", correct=True,
            judge_score=None, tokens_in=10, tokens_out=5,
        ),
        EvaluationRow(
            id="b", question="q2", expected_short="y", expected_long="long",
            response_text="<final_answer>z</final_answer>", extracted="z", correct=False,
            judge_score=None, tokens_in=10, tokens_out=5,
        ),
    ]
    # Build a candidate score manually so we don't hit the LLM.
    from src.scorer import MultiObjectiveScorer
    scorer = MultiObjectiveScorer(use_judge=False, alpha=0.0)
    cs = scorer.aggregate(rows)
    assert cs.accuracy == 0.5
    assert cs.accuracy_short == 0.5
    assert cs.accuracy_long == 0.0
    assert cs.tokens_in_total == 20
    assert cs.tokens_out_total == 10


# ---------------------------------------------------------------------------
# Fuzzy match (iteration 2 — reports/informe.md §8.2)
# ---------------------------------------------------------------------------
def test_fuzzy_short_match_close_enough():
    """Verbose prediction that contains the expected token verbatim."""
    score = _fuzzy_short_score("El cuerpo humano adulto tiene 206 huesos", "206")
    assert score >= 0.5  # token_set_ratio rewards inclusion


def test_fuzzy_short_match_exact():
    assert _fuzzy_short_score("París", "París") == 1.0
    # Accent stripping is part of `_normalise` so unaccented input matches.
    assert _fuzzy_short_score("Paris", "París") >= 0.5


def test_fuzzy_short_match_threshold():
    s = MultiObjectiveScorer(use_judge=False, alpha=0.0, fuzzy_threshold=0.85)
    # Above threshold → correct.
    assert s._is_short_correct("París", "París")[0] is True
    # Below threshold → not correct (returns the continuous score for analysis).
    correct, score = s._is_short_correct("alguna respuesta sin relacion", "París")
    assert correct is False
    assert 0.0 <= score < 0.85


def test_fuzzy_short_accuracy_in_aggregate():
    """`CandidateScore.fuzzy_short_accuracy` is the mean fuzzy_score over short rows."""
    rows = [
        EvaluationRow(
            id="s1", question="q", expected_short="206", expected_long="x",
            response_text="<final_answer>El cuerpo tiene 206 huesos</final_answer>",
            extracted="El cuerpo tiene 206 huesos", correct=True,
            judge_score=None, fuzzy_score=1.0, tokens_in=5, tokens_out=5,
        ),
        EvaluationRow(
            id="s2", question="q", expected_short="7", expected_long="x",
            response_text="<final_answer>siete</final_answer>", extracted="siete",
            correct=False, judge_score=None, fuzzy_score=0.0, tokens_in=5, tokens_out=5,
        ),
    ]
    scorer = MultiObjectiveScorer(use_judge=False, alpha=0.0)
    cs = scorer.aggregate(rows)
    assert cs.fuzzy_short_accuracy == 0.5
    assert "fuzzy_short_accuracy" in cs.as_dict()
