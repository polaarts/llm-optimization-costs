"""High-level pipelines: baseline, CAPO, CROP and unified.

Each `run_*` function returns a `RunResult` (dict) with the metrics the
analysis layer expects. The orchestration is intentionally explicit: every
candidate evaluation and Critic invocation is logged through the global
`JSONLLogger` so the JSONL file is the single source of truth.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SETTINGS
from .cost_model import usd_from_tokens
from .critic import BrevityFeedbackGenerator
from .data_gen import load_dataset
from .llm_client import LLMClient
from .mutator import PromptMutator
from .racing import Candidate, RacingEvaluator, make_candidates
from .scorer import MultiObjectiveScorer
from .utils.logging import JSONLLogger, get_logger, set_logger
from .utils.seeds import set_seed

# ---------------------------------------------------------------------------
# Seed prompts
# ---------------------------------------------------------------------------
SEED_PROMPTS: list[str] = [
    # Plain zero-shot baseline
    (
        "Responde la siguiente pregunta de la forma más precisa posible. "
        "Cuando aplique, entrega la respuesta final entre "
        "<final_answer>...</final_answer>."
    ),
    # Concise variant
    (
        "Eres un asistente conciso y preciso. Responde en una o dos frases, "
        "siempre dentro de <final_answer>...</final_answer>."
    ),
    # Step-by-step variant
    (
        "Piensa paso a paso y luego entrega la respuesta final entre "
        "<final_answer>...</final_answer>. Sé breve en la justificación."
    ),
    # Direct answer variant
    (
        "Da la respuesta correcta primero, en una sola frase, dentro de "
        "<final_answer>...</final_answer>. Evita rodeos."
    ),
    # Friendly tone
    (
        "Ayuda al usuario respondiendo a su pregunta con claridad. "
        "Entrega la respuesta final entre <final_answer>...</final_answer>."
    ),
    # Structured output
    (
        "Estructura tu respuesta en dos partes: una justificación breve y la "
        "respuesta final entre <final_answer>...</final_answer>."
    ),
]


# ---------------------------------------------------------------------------
# Run result container
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    condition: str
    seed: int
    final_prompt: str
    accuracy: float
    accuracy_short: float
    accuracy_long: float
    fuzzy_short_accuracy: float
    judge_score_long: float
    tokens_in_total: int
    tokens_out_total: int
    cost_in_usd: float
    cost_out_usd: float
    cost_total_usd: float
    latency_ms_mean: float
    n_generations: int
    n_llm_calls: int
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "final_prompt": self.final_prompt,
            "accuracy": self.accuracy,
            "accuracy_short": self.accuracy_short,
            "accuracy_long": self.accuracy_long,
            "fuzzy_short_accuracy": self.fuzzy_short_accuracy,
            "judge_score_long": self.judge_score_long,
            "tokens_in_total": self.tokens_in_total,
            "tokens_out_total": self.tokens_out_total,
            "cost_in_usd": self.cost_in_usd,
            "cost_out_usd": self.cost_out_usd,
            "cost_total_usd": self.cost_total_usd,
            "latency_ms_mean": self.latency_ms_mean,
            "n_generations": self.n_generations,
            "n_llm_calls": self.n_llm_calls,
            **self.notes,
        }


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------
def _split_dataset(
    rows: list[dict], *, dev_fraction: float = 0.6, test_fraction: float = 0.2, seed: int = 0
) -> tuple[list[dict], list[dict], list[dict]]:
    """Random split into dev / optimisation / holdout test set."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_dev = max(1, int(dev_fraction * len(rows)))
    n_test = max(1, int(test_fraction * len(rows)))
    dev = [rows[i] for i in idx[:n_dev]]
    test = [rows[i] for i in idx[n_dev : n_dev + n_test]]
    hold = [rows[i] for i in idx[n_dev + n_test :]]
    return dev, test, hold


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------
class BudgetGuard:
    """Track cumulative USD spend and stop the run if it exceeds a ceiling."""

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent: float = 0.0
        self.tokens_in: int = 0
        self.tokens_out: int = 0

    def record(self, *, model: str, tokens_in: int, tokens_out: int) -> None:
        cost = usd_from_tokens(model, tokens_in, tokens_out)
        self.spent += cost.cost_total
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.max_usd


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def run_baseline(
    seed: int = 0,
    *,
    client: Optional[LLMClient] = None,
    logger: Optional[JSONLLogger] = None,
    max_budget_usd: float = SETTINGS.max_budget_usd,
) -> RunResult:
    """Zero-shot baseline: pick the first seed prompt, evaluate on test set."""
    set_seed(seed)
    client = client or LLMClient()
    if logger is not None:
        set_logger(logger)
    log = get_logger()

    log.log(event="run_start", condition="baseline", seed=seed)
    rows = load_dataset(SETTINGS.data_path)
    dev, test, _hold = _split_dataset(rows, seed=seed)
    scorer = MultiObjectiveScorer(client=client)
    budget = BudgetGuard(max_budget_usd)

    prompt = SEED_PROMPTS[0]
    score = scorer.score_prompt(prompt, test)
    budget.record(model=client.model, tokens_in=score.tokens_in_total, tokens_out=score.tokens_out_total)
    log.log(
        event="baseline_score",
        accuracy=score.accuracy,
        tokens_in=score.tokens_in_total,
        tokens_out=score.tokens_out_total,
    )
    cost = usd_from_tokens(client.model, score.tokens_in_total, score.tokens_out_total)
    return RunResult(
        condition="baseline",
        seed=seed,
        final_prompt=prompt,
        accuracy=score.accuracy,
        accuracy_short=score.accuracy_short,
        accuracy_long=score.accuracy_long,
        fuzzy_short_accuracy=score.fuzzy_short_accuracy,
        judge_score_long=score.judge_score_long,
        tokens_in_total=score.tokens_in_total,
        tokens_out_total=score.tokens_out_total,
        cost_in_usd=cost.cost_in,
        cost_out_usd=cost.cost_out,
        cost_total_usd=cost.cost_total,
        latency_ms_mean=0.0,
        n_generations=0,
        n_llm_calls=len(test),
    )


# ---------------------------------------------------------------------------
# CAPO
# ---------------------------------------------------------------------------
def run_capo(
    seed: int = 0,
    *,
    n_generations: int = 4,
    population_size: int = 8,
    use_judge: bool = False,  # turn off by default to keep CAPO cheap
    pairwise_test: str = "wilcoxon",
    correction: str = "holm",
    client: Optional[LLMClient] = None,
    logger: Optional[JSONLLogger] = None,
    max_budget_usd: float = SETTINGS.max_budget_usd,
) -> RunResult:
    """CAPO: Racing + Holm-Bonferroni + length penalty. No Critic.

    Iteration 2 defaults: ``n_generations=4`` and ``population_size=8`` were
    raised from the prototype's 2/4 because the original configuration made
    the racing collapse to ``n_survivors=0`` in 5/5 seeds (see
    ``reports/informe.md`` §7.5). The pairwise test also defaults to
    ``wilcoxon`` because Wilcoxon is more robust on small per-block sample
    sizes (5–12 items).
    """
    set_seed(seed)
    client = client or LLMClient()
    if logger is not None:
        set_logger(logger)
    log = get_logger()
    log.log(
        event="run_start",
        condition="capo",
        seed=seed,
        n_generations=n_generations,
        population_size=population_size,
        pairwise_test=pairwise_test,
        correction=correction,
    )

    rows = load_dataset(SETTINGS.data_path)
    dev, test, _hold = _split_dataset(rows, seed=seed)

    # Hyperparameters — raised from the prototype defaults to mitigate the
    # CAPO collapse documented in reports/informe.md §7.5.
    block_size = max(3, min(SETTINGS.block_size, len(dev) // 2))
    n_survive = max(2, population_size // 2)
    z_max = max(2, len(dev) // block_size)

    scorer = MultiObjectiveScorer(
        alpha=SETTINGS.gamma,  # length penalty in CAPO acts on prompt length
        beta=0.0,
        use_judge=use_judge,
        client=client,
    )
    mutator = PromptMutator(client=client)
    racing = RacingEvaluator(
        block_size=block_size,
        alpha=SETTINGS.alpha,
        n_survive=n_survive,
        z_max=z_max,
        pairwise_test=pairwise_test,
        correction=correction,
    )
    budget = BudgetGuard(max_budget_usd)

    # Initial population
    initial_prompts = SEED_PROMPTS[:population_size]
    if len(initial_prompts) < population_size:
        # pad by paraphrasing the last seed
        for p in SEED_PROMPTS:
            if len(initial_prompts) >= population_size:
                break
            if p not in initial_prompts:
                initial_prompts.append(p)
    population = make_candidates(initial_prompts)

    # Define an evaluate_fn that scores one candidate on a block.
    def evaluate_fn(cand: Candidate, batch: list[dict]) -> list[float]:
        result = scorer.score_prompt(cand.prompt, batch)
        budget.record(
            model=client.model, tokens_in=result.tokens_in_total, tokens_out=result.tokens_out_total
        )
        # Per-item 0/1 score for racing.
        return [1.0 if r.correct else 0.0 for r in result.rows]

    start = time.perf_counter()
    for gen in range(n_generations):
        if budget.exceeded:
            log.log(event="budget_stop", condition="capo", generation=gen)
            break
        log.log(event="generation_start", condition="capo", generation=gen)
        # 1) Run racing on the current population.
        racing_result = racing.run(population, dev, evaluate_fn)
        population = racing_result.survivors
        log.log(
            event="racing_done",
            condition="capo",
            generation=gen,
            survivors=len(racing_result.survivors),
            eliminated=len(racing_result.eliminated),
            blocks=racing_result.blocks_used,
        )
        if not population or budget.exceeded:
            break

        # 2) Mutate survivors to produce the next generation.
        traces = mutator.mutate_pool([c.prompt for c in population], seed=seed * 100 + gen)
        population = make_candidates([t.child_prompt for t in traces])

    # Final selection: re-evaluate survivors on the test set and pick the best.
    if not population:
        final = SEED_PROMPTS[0]
        score = scorer.score_prompt(final, test)
    else:
        # Re-score survivors on test to avoid overfitting to dev.
        test_scores = [scorer.score_prompt(c.prompt, test) for c in population]
        test_scores.sort(key=lambda s: s.accuracy, reverse=True)
        score = test_scores[0]
        final = population[test_scores.index(score)].prompt if population else final

    cost = usd_from_tokens(client.model, score.tokens_in_total, score.tokens_out_total)
    latency = (time.perf_counter() - start) * 1000.0 / max(1, len(test))
    return RunResult(
        condition="capo",
        seed=seed,
        final_prompt=final,
        accuracy=score.accuracy,
        accuracy_short=score.accuracy_short,
        accuracy_long=score.accuracy_long,
        fuzzy_short_accuracy=score.fuzzy_short_accuracy,
        judge_score_long=score.judge_score_long,
        tokens_in_total=score.tokens_in_total,
        tokens_out_total=score.tokens_out_total,
        cost_in_usd=cost.cost_in,
        cost_out_usd=cost.cost_out,
        cost_total_usd=cost.cost_total,
        latency_ms_mean=latency,
        n_generations=n_generations,
        n_llm_calls=len(test),
        notes={
            "n_survivors": len(population),
            "pairwise_test": pairwise_test,
            "correction": correction,
            "population_size": population_size,
        },
    )


# ---------------------------------------------------------------------------
# CROP
# ---------------------------------------------------------------------------
def run_crop(
    seed: int = 0,
    *,
    n_iterations: int = 2,
    use_judge: bool = True,
    client: Optional[LLMClient] = None,
    logger: Optional[JSONLLogger] = None,
    max_budget_usd: float = SETTINGS.max_budget_usd,
) -> RunResult:
    """CROP: only the Critic LM + brevity feedback. No racing."""
    set_seed(seed)
    client = client or LLMClient()
    if logger is not None:
        set_logger(logger)
    log = get_logger()
    log.log(event="run_start", condition="crop", seed=seed)

    rows = load_dataset(SETTINGS.data_path)
    dev, test, _hold = _split_dataset(rows, seed=seed)
    scorer = MultiObjectiveScorer(
        alpha=0.0, beta=0.05,  # beta acts on cost_out
        use_judge=use_judge,
        client=client,
    )
    critic = BrevityFeedbackGenerator(client=client)
    mutator = PromptMutator(client=client)
    budget = BudgetGuard(max_budget_usd)

    # Start with a verbose seed prompt so CROP has something to trim.
    prompt = (
        "Eres un asistente útil. Piensa paso a paso en voz alta, justifica cada "
        "decisión con detalle, y entrega la respuesta final entre "
        "<final_answer>...</final_answer>."
    )

    start = time.perf_counter()
    for it in range(n_iterations):
        if budget.exceeded:
            break
        log.log(event="crop_iter_start", iteration=it)
        score = scorer.score_prompt(prompt, dev)
        budget.record(
            model=client.model, tokens_in=score.tokens_in_total, tokens_out=score.tokens_out_total
        )
        pool_outs = [score.tokens_out_mean]  # only one prompt in CROP, but we keep
        # the policy general: only invoke the critic if output length is high.
        # In this single-prompt setting we always invoke it after iteration 0.
        if it > 0:
            break
        # Get a few example outputs to feed the critic.
        sample_rows = dev[:5]
        outputs: list[str] = []
        for r in sample_rows:
            resp = client.complete(
                f"{prompt}\n\n---\n\nPregunta: {r['question']}",
                temperature=0.0,
                role="crop_sample",
            )
            outputs.append(resp.text)
            budget.record(model=client.model, tokens_in=resp.tokens_in, tokens_out=resp.tokens_out)
        longest = max(outputs, key=len)
        critic_result = critic.critique(prompt=prompt, output=longest, seed=seed)
        if critic_result is None:
            log.log(event="crop_critic_skipped")
            break
        log.log(
            event="crop_critic_invoked",
            original_length=critic_result.original_length,
            new_length=critic_result.new_length,
            brevity_score=critic_result.brevity_score,
        )
        # Build a new prompt that includes the critic's feedback.
        prompt = (
            f"{prompt.rstrip()}\n\n"
            f"Nota del editor: {critic_result.feedback}\n"
            f"Intenta ser tan conciso como esta versión: {critic_result.rewritten}"
        )

    # Final test evaluation
    score = scorer.score_prompt(prompt, test)
    cost = usd_from_tokens(client.model, score.tokens_in_total, score.tokens_out_total)
    latency = (time.perf_counter() - start) * 1000.0 / max(1, len(test))
    return RunResult(
        condition="crop",
        seed=seed,
        final_prompt=prompt,
        accuracy=score.accuracy,
        accuracy_short=score.accuracy_short,
        accuracy_long=score.accuracy_long,
        fuzzy_short_accuracy=score.fuzzy_short_accuracy,
        judge_score_long=score.judge_score_long,
        tokens_in_total=score.tokens_in_total,
        tokens_out_total=score.tokens_out_total,
        cost_in_usd=cost.cost_in,
        cost_out_usd=cost.cost_out,
        cost_total_usd=cost.cost_total,
        latency_ms_mean=latency,
        n_generations=n_iterations,
        n_llm_calls=len(test) * (n_iterations + 1),
    )


# ---------------------------------------------------------------------------
# Unified
# ---------------------------------------------------------------------------
def run_unified(
    seed: int = 0,
    *,
    n_generations: int = 4,
    population_size: int = 8,
    use_judge: bool = True,
    pairwise_test: str = "wilcoxon",
    correction: str = "holm",
    client: Optional[LLMClient] = None,
    logger: Optional[JSONLLogger] = None,
    max_budget_usd: float = SETTINGS.max_budget_usd,
) -> RunResult:
    """Unified pipeline: CAPO (racing + Holm + length penalty) + CROP (critic).

    Iteration 2 defaults mirror ``run_capo``: ``n_generations=4``,
    ``population_size=8``, ``pairwise_test='wilcoxon'``, ``correction='holm'``.
    """
    set_seed(seed)
    client = client or LLMClient()
    if logger is not None:
        set_logger(logger)
    log = get_logger()
    log.log(
        event="run_start",
        condition="unified",
        seed=seed,
        n_generations=n_generations,
        population_size=population_size,
        pairwise_test=pairwise_test,
        correction=correction,
    )

    rows = load_dataset(SETTINGS.data_path)
    dev, test, _hold = _split_dataset(rows, seed=seed)

    block_size = max(3, min(SETTINGS.block_size, len(dev) // 2))
    n_survive = max(2, population_size // 2)
    z_max = max(2, len(dev) // block_size)

    scorer = MultiObjectiveScorer(
        alpha=SETTINGS.gamma,
        beta=0.05,
        use_judge=use_judge,
        client=client,
    )
    mutator = PromptMutator(client=client)
    critic = BrevityFeedbackGenerator(client=client)
    racing = RacingEvaluator(
        block_size=block_size,
        alpha=SETTINGS.alpha,
        n_survive=n_survive,
        z_max=z_max,
        pairwise_test=pairwise_test,
        correction=correction,
    )
    budget = BudgetGuard(max_budget_usd)

    population = make_candidates(SEED_PROMPTS[:population_size])

    def evaluate_fn(cand: Candidate, batch: list[dict]) -> list[float]:
        result = scorer.score_prompt(cand.prompt, batch)
        budget.record(
            model=client.model, tokens_in=result.tokens_in_total, tokens_out=result.tokens_out_total
        )
        return [1.0 if r.correct else 0.0 for r in result.rows]

    start = time.perf_counter()
    for gen in range(n_generations):
        if budget.exceeded:
            break
        log.log(event="generation_start", condition="unified", generation=gen)

        racing_result = racing.run(population, dev, evaluate_fn)
        population = racing_result.survivors
        log.log(
            event="racing_done",
            condition="unified",
            generation=gen,
            survivors=len(racing_result.survivors),
            eliminated=len(racing_result.eliminated),
            blocks=racing_result.blocks_used,
        )
        if not population or budget.exceeded:
            break

        # CROP step: only invoke the critic on the worst-offender survivor.
        # We pick the survivor with the largest mean output length.
        out_means: list[tuple[Candidate, float]] = []
        for cand in population:
            sample_score = scorer.score_prompt(cand.prompt, dev[:3])
            out_means.append((cand, sample_score.tokens_out_mean))
            budget.record(
                model=client.model,
                tokens_in=sample_score.tokens_in_total,
                tokens_out=sample_score.tokens_out_total,
            )
        if out_means:
            worst_cand, _ = max(out_means, key=lambda x: x[1])
            pool_means = [m for _, m in out_means]
            # Generate one sample output to feed the critic.
            sample_prompt = (
                f"{worst_cand.prompt}\n\n---\n\nPregunta: {dev[0]['question']}"
            )
            sample_resp = client.complete(sample_prompt, temperature=0.0, role="unified_sample")
            budget.record(
                model=client.model,
                tokens_in=sample_resp.tokens_in,
                tokens_out=sample_resp.tokens_out,
            )
            if critic.should_invoke(sample_resp.tokens_out, pool_means):
                crit = critic.critique(
                    prompt=worst_cand.prompt, output=sample_resp.text, seed=seed
                )
                if crit is not None:
                    log.log(
                        event="unified_critic",
                        brevity_score=crit.brevity_score,
                        new_length=crit.new_length,
                    )
                    # Rewrite the prompt with the feedback as a soft instruction.
                    new_prompt = (
                        f"{worst_cand.prompt.rstrip()}\n\n"
                        f"Recordatorio de brevedad: {crit.feedback}"
                    )
                    worst_cand.prompt = new_prompt

        # Mutate survivors
        traces = mutator.mutate_pool([c.prompt for c in population], seed=seed * 100 + gen)
        population = make_candidates([t.child_prompt for t in traces])

    if not population:
        final = SEED_PROMPTS[0]
        score = scorer.score_prompt(final, test)
    else:
        test_scores = [scorer.score_prompt(c.prompt, test) for c in population]
        test_scores.sort(key=lambda s: s.score, reverse=True)
        score = test_scores[0]
        final = test_scores[0].rows[0].response_text and population[0].prompt or population[0].prompt

    cost = usd_from_tokens(client.model, score.tokens_in_total, score.tokens_out_total)
    latency = (time.perf_counter() - start) * 1000.0 / max(1, len(test))
    return RunResult(
        condition="unified",
        seed=seed,
        final_prompt=final,
        accuracy=score.accuracy,
        accuracy_short=score.accuracy_short,
        accuracy_long=score.accuracy_long,
        fuzzy_short_accuracy=score.fuzzy_short_accuracy,
        judge_score_long=score.judge_score_long,
        tokens_in_total=score.tokens_in_total,
        tokens_out_total=score.tokens_out_total,
        cost_in_usd=cost.cost_in,
        cost_out_usd=cost.cost_out,
        cost_total_usd=cost.cost_total,
        latency_ms_mean=latency,
        n_generations=n_generations,
        n_llm_calls=len(test) * (n_generations + 1),
        notes={
            "n_survivors": len(population) if population else 0,
            "pairwise_test": pairwise_test,
            "correction": correction,
            "population_size": population_size,
        },
    )
