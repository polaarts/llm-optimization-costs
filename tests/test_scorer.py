"""Smoke tests for the scorer: answer extraction, exact-match, judge parsing."""
from __future__ import annotations

import json

from src.scorer import (
    CandidateScore,
    EvaluationRow,
    extract_final_answer,
    _normalise,
    _parse_judge_json,
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
