"""
Structural-distance bar chart (af3bench2 overhaul).

Single RMSD panel with:
  * TM-score encoded as bar fill colour via a diverging colormap anchored at
    TM=0.8 (plan 1.6b) — the separate TM panel is dropped.
  * an optional broken y-axis separating low-confidence/high-RMSD conditions
    from the well-behaved range (plan 1.6a).
  * two-tier confidence hatching (plan 0.4).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm

from . import style


def _tm_colormap():
    # green (high TM) -> white (0.8) -> red (low TM)
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "tm_div", ["#B2182B", "#F7F7F7", "#1B7837"]
    )


def plot_distances(
    df: pd.DataFrame,
    baseline_name: str,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    tiers: Optional[Dict[str, str]] = None,
    has_dna: Optional[Dict[str, bool]] = None,  # Issue 6 fix
) -> List[Path]:
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)
    x = np.arange(len(names))
    rmsd = df["rmsd"].tolist()
    tiers = tiers or {}

    has_tm = "tm_score" in df.columns and df["tm_score"].notna().any()
    cmap = _tm_colormap()
    norm = TwoSlopeNorm(vmin=0.4, vcenter=0.8, vmax=1.0)
    tm_vals = df["tm_score"].tolist() if has_tm else [float("nan")] * len(names)
    bar_colors = []
    for i in range(len(names)):
        if has_tm and math.isfinite(tm_vals[i]):
            bar_colors.append(cmap(norm(tm_vals[i])))
        else:
            bar_colors.append(style.PALETTE[0])

    yerr = None
    if {"rmsd_lo", "rmsd_hi"}.issubset(df.columns):
        lo = (df["rmsd"] - df["rmsd_lo"]).clip(lower=0).tolist()
        hi = (df["rmsd_hi"] - df["rmsd"]).clip(lower=0).tolist()
        yerr = [lo, hi]

    # Decide whether a broken axis is warranted: a clear gap between a high
    # cluster (low-confidence/artifact) and the rest.
    finite_rmsd = np.array([r for r in rmsd if math.isfinite(r)])
    artifact_vals = [rmsd[i] for i in range(len(names))
                     if tiers.get(names[i]) == "likely_artifact" and math.isfinite(rmsd[i])]
    normal_vals = [rmsd[i] for i in range(len(names))
                   if tiers.get(names[i]) != "likely_artifact" and math.isfinite(rmsd[i])]
    do_break = bool(artifact_vals and normal_vals
                    and min(artifact_vals) - max(normal_vals) > 1.0)

    if do_break:
        return _plot_broken(df, names, labels, x, rmsd, yerr, bar_colors, tiers,
                            baseline_name, plots_dir, cmap, norm, has_tm,
                            float(max(normal_vals)), float(min(artifact_vals)),
                            has_dna)  # Issue 6 fix

    fig, ax = plt.subplots(figsize=(max(6.5, len(names) * 1.0), 5.2))
    _draw_bars(ax, x, rmsd, yerr, bar_colors, names, tiers)
    _finish(ax, x, labels, rmsd, baseline_name, tiers, names, has_dna)  # Issue 6 fix
    if has_tm:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="TM-score (green≥0.8 conserved, red<0.6 divergent)",
                     shrink=0.85)
    fig.suptitle("Structural distance vs baseline (bar fill = TM-score)",
                 fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "structural_distances")


def _draw_bars(ax, x, rmsd, yerr, bar_colors, names, tiers):
    bars = ax.bar(x, rmsd, color=bar_colors, edgecolor="black", alpha=0.95,
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.1})
    for i, n in enumerate(names):
        t = tiers.get(n, "ok")
        if t != "ok":
            bars[i].set_hatch(style.TIER_HATCH.get(t, ""))
    return bars


def _finish(ax, x, labels, rmsd, baseline_name, tiers, names=None, has_dna=None):
    for i, v in enumerate(rmsd):
        if math.isfinite(v):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            # Issue 6 fix: Add DNA marker
            if has_dna and names and i < len(names) and has_dna.get(names[i], False):
                ax.text(i, v + 0.35, "⊕DNA", ha="center", va="bottom",
                        fontsize=6.5, color="#009E73", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Protein Cα RMSD vs baseline (Å)")


def _plot_broken(df, names, labels, x, rmsd, yerr, bar_colors, tiers,
                 baseline_name, plots_dir, cmap, norm, has_tm, low_top, high_bot,
                 has_dna=None):  # Issue 6 fix
    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, sharex=True, figsize=(max(6.5, len(names) * 1.0), 6.2),
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
    )
    for ax in (ax_hi, ax_lo):
        _draw_bars(ax, x, rmsd, yerr, bar_colors, names, tiers)

    ax_lo.set_ylim(0, low_top * 1.25)
    ax_hi.set_ylim(high_bot * 0.95, max(r for r in rmsd if math.isfinite(r)) * 1.12)
    ax_hi.set_title("low-confidence conditions", fontsize=9, loc="left", color=style.C_FAIL)

    # broken-axis diagonal marks
    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(bottom=False)
    d = 0.008
    kw = dict(transform=ax_hi.transAxes, color="k", clip_on=False, linewidth=1)
    ax_hi.plot((-d, +d), (-d, +d), **kw)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw2 = dict(transform=ax_lo.transAxes, color="k", clip_on=False, linewidth=1)
    ax_lo.plot((-d, +d), (1 - d * 2.2, 1 + d * 2.2), **kw2)
    ax_lo.plot((1 - d, 1 + d), (1 - d * 2.2, 1 + d * 2.2), **kw2)

    for i, v in enumerate(rmsd):
        if not math.isfinite(v):
            continue
        ax = ax_hi if v >= high_bot * 0.95 else ax_lo
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        # Issue 6 fix: Add DNA marker
        if has_dna and i < len(names) and has_dna.get(names[i], False):
            ax.text(i, v + 0.35, "⊕DNA", ha="center", va="bottom",
                    fontsize=6.5, color="#009E73", style="italic")

    ax_lo.set_xticks(x)
    ax_lo.set_xticklabels(labels, rotation=40, ha="right")
    ax_lo.set_ylabel("Protein Cα RMSD (Å)")
    ax_lo.yaxis.set_label_coords(-0.07, 0.75)

    if has_tm:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=(ax_hi, ax_lo),
                     label="TM-score (green≥0.8, red<0.6)", shrink=0.7)
    fig.suptitle("Structural distance vs baseline (broken axis; bar fill = TM-score)",
                 fontweight="bold")
    return style.save(fig, plots_dir, "structural_distances")
