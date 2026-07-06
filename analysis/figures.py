"""Generate the three required figures from the per-seed JSON results.

Usage:
    python -m analysis.figures

Outputs to `results/figures/figure1_*.png`, `figure2_*.png`, `figure3_*.png`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import SETTINGS


CONDITION_COLORS = {
    "baseline": "#4C72B0",
    "capo": "#55A868",
    "crop": "#C44E52",
    "unified": "#8172B3",
}
CONDITION_ORDER = ["baseline", "capo", "crop", "unified"]
METRICS = [
    ("accuracy", "Accuracy"),
    ("tokens_in_total", "Tokens de entrada (total)"),
    ("tokens_out_total", "Tokens de salida (total)"),
    ("cost_total_usd", "Costo total (USD)"),
]


def _load_long() -> pd.DataFrame:
    path = SETTINGS.tables_dir / "long.csv"
    if path.exists():
        return pd.read_csv(path)
    # Fall back to reading raw JSON.
    rows: list[dict] = []
    for cond in CONDITION_ORDER:
        cond_dir = SETTINGS.raw_dir / cond
        if not cond_dir.exists():
            continue
        for f in sorted(cond_dir.glob("seed*.json")):
            try:
                payload = json.loads(f.read_text(encoding="utf-8"))
                payload["_path"] = str(f)
                rows.append(payload)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: grouped bar chart of mean ± std per condition
# ---------------------------------------------------------------------------
def figure1_bars(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.0 * len(METRICS), 4.0))
    for ax, (metric, title) in zip(axes, METRICS):
        if metric not in df.columns:
            ax.set_title(f"{title}\n(sin datos)")
            ax.axis("off")
            continue
        means = []
        stds = []
        labels = []
        for cond in CONDITION_ORDER:
            sub = df[df["condition"] == cond][metric].dropna()
            if sub.empty:
                continue
            means.append(sub.mean())
            stds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
            labels.append(cond)
        if not means:
            ax.set_title(f"{title}\n(sin datos)")
            ax.axis("off")
            continue
        x = np.arange(len(labels))
        colors = [CONDITION_COLORS[c] for c in labels]
        ax.bar(x, means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Comparación de condiciones (media ± std sobre seeds)")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: scatter (tokens_out, accuracy) coloured by condition
# ---------------------------------------------------------------------------
def figure2_pareto(df: pd.DataFrame, out: Path) -> None:
    if df.empty or "tokens_out_total" not in df.columns or "accuracy" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        if sub.empty:
            continue
        ax.scatter(
            sub["tokens_out_total"],
            sub["accuracy"],
            s=80,
            alpha=0.75,
            color=CONDITION_COLORS[cond],
            label=cond,
            edgecolor="white",
        )
    ax.set_xlabel("Tokens de salida (total)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Frente Pareto: (tokens_out ↓, accuracy ↑)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: racing convergence per generation
# ---------------------------------------------------------------------------
def figure3_convergence(out: Path) -> None:
    """Read the JSONL logs and plot mean population accuracy per generation."""
    records: list[dict] = []
    for cond in ["capo", "unified"]:
        for f in (SETTINGS.raw_dir / cond).glob("seed*.jsonl"):
            try:
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("event") == "racing_done":
                            records.append({
                                "condition": cond,
                                "seed": rec.get("run_id"),
                                "generation": rec.get("generation"),
                                "survivors": rec.get("survivors"),
                                "eliminated": rec.get("eliminated"),
                                "blocks": rec.get("blocks"),
                            })
            except OSError:
                continue
    if not records:
        return
    rdf = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, color in CONDITION_COLORS.items():
        if cond not in ("capo", "unified"):
            continue
        sub = rdf[rdf["condition"] == cond]
        if sub.empty:
            continue
        # The "racing_done" event does not store accuracy directly, but
        # `survivors / (survivors + eliminated)` is a reasonable proxy for
        # how selective the racing was. We plot survivors to keep the
        # visual honest.
        pivot = sub.groupby("generation")["survivors"].mean()
        ax.plot(pivot.index, pivot.values, marker="o", color=color, label=cond)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Supervivientes medios")
    ax.set_title("Convergencia del Racing a lo largo de las generaciones")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    SETTINGS.figures_dir.mkdir(parents=True, exist_ok=True)
    df = _load_long()
    figure1_bars(df, SETTINGS.figures_dir / "figure1_bars.png")
    figure2_pareto(df, SETTINGS.figures_dir / "figure2_pareto.png")
    figure3_convergence(SETTINGS.figures_dir / "figure3_convergence.png")
    print(f"Wrote figures to {SETTINGS.figures_dir}")


if __name__ == "__main__":
    main()
