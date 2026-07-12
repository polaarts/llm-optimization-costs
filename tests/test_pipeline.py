"""Integration tests using the mock LLM client.

These run the full pipeline against canned responses and assert that the
racing + mutator + scorer glue behaves as expected without any network
calls. They are the best end-to-end safety net we have for a 1-week project.

Iteration 2: mock response counts are sized for ``population_size=8`` and
``n_generations=4`` (the new defaults). With ~223 dataset rows and the 60/20/20
split, ``run_capo`` makes ~4 × 223 ≈ 900 LLM calls per seed at the smallest
configuration, so the cap is raised to 3000 to leave headroom.
"""
from __future__ import annotations

from src.llm_client import _enable_mock
from src.data_gen import load_dataset
from src.config import SETTINGS
from src.pipeline import run_baseline, run_capo, run_crop, run_unified
from src.utils.logging import JSONLLogger
import tempfile
from pathlib import Path


# A minimal canned response list — must cover the number of LLM calls each
# pipeline makes for a tiny run.
def _mock_responses(n: int = 200) -> list[str]:
    return [
        "<final_answer>París</final_answer>" if i % 2 == 0
        else "Reasoning... <final_answer>Madrid</final_answer>"
        for i in range(n)
    ]


def _setup_mock(n: int = 3000):
    _enable_mock(_mock_responses(n))


def test_run_baseline_smoke(tmp_path: Path):
    _setup_mock(500)
    logger = JSONLLogger(tmp_path / "b.jsonl", run_id="b")
    res = run_baseline(seed=0, logger=logger, max_budget_usd=1.0)
    assert res.condition == "baseline"
    assert 0.0 <= res.accuracy <= 1.0
    # Iteration 2: the scorer now exposes a fuzzy_short_accuracy field.
    assert hasattr(res, "fuzzy_short_accuracy")
    assert 0.0 <= res.fuzzy_short_accuracy <= 1.0


def test_run_capo_smoke(tmp_path: Path):
    _setup_mock(3000)
    logger = JSONLLogger(tmp_path / "c.jsonl", run_id="c")
    res = run_capo(
        seed=0, n_generations=1, population_size=2,
        logger=logger, max_budget_usd=1.0,
    )
    assert res.condition == "capo"
    assert res.final_prompt
    # Iteration 2: notes carries the pairwise test + correction label so the
    # ablation matrix is queryable from the analysis layer.
    assert res.notes["pairwise_test"] == "wilcoxon"
    assert res.notes["correction"] == "holm"


def test_run_capo_notes_reflect_args(tmp_path: Path):
    """Custom pairwise_test / correction kwargs propagate to ``RunResult.notes``."""
    _setup_mock(3000)
    logger = JSONLLogger(tmp_path / "c2.jsonl", run_id="c2")
    res = run_capo(
        seed=1, n_generations=1, population_size=2,
        pairwise_test="wilcoxon", correction="none",
        logger=logger, max_budget_usd=1.0,
    )
    assert res.notes["pairwise_test"] == "wilcoxon"
    assert res.notes["correction"] == "none"


def test_run_crop_smoke(tmp_path: Path):
    _enable_mock(_mock_responses(500) + [
        json_dummy_critic(),
    ])
    logger = JSONLLogger(tmp_path / "p.jsonl", run_id="p")
    res = run_crop(seed=0, n_iterations=1, logger=logger, max_budget_usd=1.0)
    assert res.condition == "crop"
    assert res.final_prompt


def test_run_unified_smoke(tmp_path: Path):
    _enable_mock(_mock_responses(3000) + [json_dummy_critic()])
    logger = JSONLLogger(tmp_path / "u.jsonl", run_id="u")
    res = run_unified(
        seed=0, n_generations=1, population_size=2,
        logger=logger, max_budget_usd=1.0,
    )
    assert res.condition == "unified"
    assert res.final_prompt
    assert res.notes["pairwise_test"] == "wilcoxon"


def json_dummy_critic() -> str:
    import json
    return json.dumps({
        "rewritten": "Respuesta concisa.",
        "feedback": "Reduce a una frase.",
        "brevity_score": 0.8,
    })
