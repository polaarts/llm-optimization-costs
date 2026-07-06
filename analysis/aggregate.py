"""Aggregate the per-seed JSON outputs into a single CSV per condition + a
combined summary CSV used by the report and figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import SETTINGS


CONDITIONS = ["baseline", "capo", "crop", "unified"]


def _read_results() -> pd.DataFrame:
    """Walk `results/raw/<condition>/seed*.json` and load every successful run."""
    rows: list[dict] = []
    for cond in CONDITIONS:
        cond_dir = SETTINGS.raw_dir / cond
        if not cond_dir.exists():
            continue
        for path in sorted(cond_dir.glob("seed*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            payload["_path"] = str(path)
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per condition over seeds."""
    if df.empty:
        return df
    metrics = [
        "accuracy",
        "accuracy_short",
        "accuracy_long",
        "judge_score_long",
        "tokens_in_total",
        "tokens_out_total",
        "cost_in_usd",
        "cost_out_usd",
        "cost_total_usd",
        "latency_ms_mean",
    ]
    metrics = [m for m in metrics if m in df.columns]
    grouped = df.groupby("condition")[metrics].agg(["mean", "std", "count"])
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=SETTINGS.tables_dir / "summary.csv"
    )
    parser.add_argument(
        "--long", type=Path, default=SETTINGS.tables_dir / "long.csv"
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = _read_results()
    if df.empty:
        print("No results found in results/raw/. Run `python -m experiments.run_all` first.")
        return
    summary = _summary(df)
    summary.to_csv(args.out)
    df.to_csv(args.long, index=False)
    print(f"Wrote {args.out} and {args.long} ({len(df)} rows).")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
