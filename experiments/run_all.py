"""Run all four conditions × N seeds.

Usage:
    python -m experiments.run_all --seeds 3 --budget 5

The orchestrator fails soft: if one condition crashes it logs the error and
moves to the next, so we never lose the entire matrix because of a transient
API issue.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from src.config import SETTINGS
from src.pipeline import run_baseline, run_capo, run_crop, run_unified
from src.utils.logging import JSONLLogger, set_logger


CONDITION_FNS = {
    "baseline": run_baseline,
    "capo": run_capo,
    "crop": run_crop,
    "unified": run_unified,
}


def _save_result(condition: str, seed: int, result_dict: dict) -> None:
    out_dir = SETTINGS.raw_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed{seed}.json").write_text(
        json.dumps(result_dict, indent=2, ensure_ascii=False)
    )


def _run_condition(
    condition: str,
    seed: int,
    budget: float,
    generations: int,
    population: int,
) -> None:
    fn = CONDITION_FNS[condition]
    out_dir = SETTINGS.raw_dir / condition
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(out_dir / f"seed{seed}.jsonl", run_id=f"{condition}-s{seed}")
    set_logger(logger)

    start = time.perf_counter()
    try:
        if condition == "baseline":
            res = fn(seed=seed, logger=logger, max_budget_usd=budget)
        elif condition == "capo":
            res = fn(
                seed=seed, n_generations=generations, population_size=population,
                logger=logger, max_budget_usd=budget,
            )
        elif condition == "crop":
            res = fn(
                seed=seed, n_iterations=generations, logger=logger,
                max_budget_usd=budget,
            )
        else:  # unified
            res = fn(
                seed=seed, n_generations=generations, population_size=population,
                logger=logger, max_budget_usd=budget,
            )
        elapsed = time.perf_counter() - start
        result_dict = res.as_dict()
        result_dict["elapsed_seconds"] = elapsed
        _save_result(condition, seed, result_dict)
        print(f"[{condition} seed={seed}] OK  acc={result_dict['accuracy']:.3f}  "
              f"cost_usd={result_dict['cost_total_usd']:.4f}  ({elapsed:.1f}s)")
    except Exception as exc:  # pragma: no cover - top-level guard
        elapsed = time.perf_counter() - start
        err_path = out_dir / f"seed{seed}.error.json"
        err_path.write_text(json.dumps({
            "condition": condition,
            "seed": seed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": elapsed,
        }, indent=2, ensure_ascii=False))
        print(f"[{condition} seed={seed}] ERROR ({elapsed:.1f}s): {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=float, default=SETTINGS.max_budget_usd)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument(
        "--condition",
        choices=["all", "baseline", "capo", "crop", "unified"],
        default="all",
    )
    args = parser.parse_args()

    conditions = (
        list(CONDITION_FNS.keys()) if args.condition == "all" else [args.condition]
    )
    print(f"Running {len(conditions)} conditions x {len(args.seeds)} seeds ...")
    for cond in conditions:
        for seed in args.seeds:
            _run_condition(cond, seed, args.budget, args.generations, args.population)
    print("Done.")


if __name__ == "__main__":
    main()
