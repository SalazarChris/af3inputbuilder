"""Structural-distance bar charts (RMSD +/- CI, TM-score)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

from . import style


def plot_distances(df: pd.DataFrame, baseline_name: str, plots_dir: Path) -> List[Path]:
    """
    Bar chart(s): RMSD (with CI whiskers if present) and optional TM-score,
    one bar per condition vs baseline.
    """
    has_tm = "tm_score" in df.columns and df["tm_score"].notna().any()
    n_panels = 2 if has_tm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    names = df["condition"].tolist()
    labels = style.condition_labels(names)
    x = range(len(names))
    failed = set(df.loc[df["likely_failed"], "condition"]) if "likely_failed" in df.columns else set()

    # RMSD panel
    ax = axes[0]
    rmsd = df["rmsd"].tolist()
    yerr = None
    if {"rmsd_lo", "rmsd_hi"}.issubset(df.columns):
        lo = (df["rmsd"] - df["rmsd_lo"]).clip(lower=0).tolist()
        hi = (df["rmsd_hi"] - df["rmsd"]).clip(lower=0).tolist()
        yerr = [lo, hi]
    bar_colors = ["#bdbdbd" if n in failed else style.PALETTE[0] for n in names]
    bars = ax.bar(x, rmsd, color=bar_colors, edgecolor="black", alpha=0.9,
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.1})
    for n, b in zip(names, bars):
        if n in failed:
            b.set_hatch("xx")
    for i, v in enumerate(rmsd):
        if math.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Protein Cα RMSD vs baseline (Å)")
    title = f"Structural distance vs {baseline_name}"
    if failed:
        title += "  (hatched = likely failed)"
    ax.set_title(title)

    if has_tm:
        ax2 = axes[1]
        tm = df["tm_score"].tolist()
        ax2.bar(x, tm, color=style.PALETTE[2], edgecolor="black", alpha=0.9)
        for i, v in enumerate(tm):
            if math.isfinite(v):
                ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=40, ha="right")
        ax2.set_ylabel("TM-score")
        ax2.set_ylim(0, 1.05)
        ax2.set_title("TM-score vs baseline")

    fig.suptitle("Structural distances vs baseline", fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "structural_distances")
