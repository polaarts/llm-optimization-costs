"""Run a single CROP seed from the CLI.

Usage:
    python -m experiments.run_crop --seed 0 --budget 5 --iterations 2
"""
from __future__ import annotations

import argparse
import json

from src.config import SETTINGS
from src.pipeline import run_crop
from src.utils.logging import JSONLLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=SETTINGS.max_budget_usd)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()

    out_dir = SETTINGS.raw_dir / "crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(out_dir / f"seed{args.seed}.jsonl", run_id=f"crop-s{args.seed}")

    result = run_crop(
        seed=args.seed,
        n_iterations=args.iterations,
        logger=logger,
        max_budget_usd=args.budget,
    )
    (out_dir / f"seed{args.seed}.json").write_text(
        json.dumps(result.as_dict(), indent=2, ensure_ascii=False)
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
