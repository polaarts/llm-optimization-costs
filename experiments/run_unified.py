"""Run a single unified-pipeline seed from the CLI.

Usage:
    python -m experiments.run_unified --seed 0 --budget 5 --generations 2
"""
from __future__ import annotations

import argparse
import json

from src.config import SETTINGS
from src.pipeline import run_unified
from src.utils.logging import JSONLLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=SETTINGS.max_budget_usd)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=4)
    args = parser.parse_args()

    out_dir = SETTINGS.raw_dir / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(out_dir / f"seed{args.seed}.jsonl", run_id=f"unified-s{args.seed}")

    result = run_unified(
        seed=args.seed,
        n_generations=args.generations,
        population_size=args.population,
        logger=logger,
        max_budget_usd=args.budget,
    )
    (out_dir / f"seed{args.seed}.json").write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False)
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
