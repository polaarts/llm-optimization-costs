"""Statistical comparisons between conditions.

Writes a CSV with paired Wilcoxon p-values (unified vs each other) plus
bootstrap confidence intervals for the key metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SETTINGS
from src.utils.stats import bootstrap_ci, cohens_d, paired_wilcoxon


CONDITIONS = ["baseline", "capo", "crop", "unified"]
METRICS = [
    "accuracy",
    "fuzzy_short_accuracy",
    "tokens_in_total",
    "tokens_out_total",
    "cost_total_usd",
]


def _load_long() -> pd.DataFrame:
    path = SETTINGS.tables_dir / "long.csv"
    if not path.exists():
        # Fallback: try to read raw files.
        from .aggregate import _read_results
        return _read_results()
    return pd.read_csv(path)


def _pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.pivot_table(
        index="seed", columns="condition", values=metric, aggfunc="mean"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=SETTINGS.tables_dir / "stats.csv"
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = _load_long()
    if df.empty:
        print("No data found. Run experiments first.")
        return

    rows = []
    pivot_acc = _pivot(df, "accuracy")
    if {"unified"}.issubset(pivot_acc.columns):
        for other in ["baseline", "capo", "crop"]:
            if other not in pivot_acc.columns:
                continue
            pair = pivot_acc.dropna(subset=["unified", other])
            u = pair["unified"].to_numpy()
            o = pair[other].to_numpy()
            test = paired_wilcoxon(u, o)
            rows.append({
                "metric": "accuracy",
                "comparison": f"unified vs {other}",
                "n_seeds": int(test["n"]),
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_pvalue": test["pvalue"],
                "cohens_d": cohens_d(u, o),
                "unified_mean": float(u.mean()) if len(u) else float("nan"),
                "other_mean": float(o.mean()) if len(o) else float("nan"),
            })

    # Bootstrap CIs for the unified condition alone, per metric.
    unified = df[df["condition"] == "unified"]
    for metric in METRICS:
        if metric not in unified.columns or unified[metric].empty:
            continue
        ci = bootstrap_ci(unified[metric].dropna().to_numpy(), seed=0)
        rows.append({
            "metric": metric,
            "comparison": f"unified (bootstrap CI)",
            "n_seeds": ci["n"],
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
            "cohens_d": float("nan"),
            "unified_mean": ci["point"],
            "other_mean": ci["low"],
            "ci_high": ci["high"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
