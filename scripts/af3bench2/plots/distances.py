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
    has_dna: Optional[Dict[str, bool]] = None,
) -> List[Path]:
    """
    Structural distances with shared sort order and clear failed condition separation (§ 3).
    """
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)
    rmsd = df["rmsd"].tolist()
    tiers = tiers or {}
    
    # Sort by ascending valid RMSD, then failed conditions separated (§ 3)
    valid_names, failed_names = style.split_conditions(names)
    valid_rmsd = [(names.index(n), rmsd[names.index(n)], n) for n in valid_names if n in names]
    failed_rmsd = [(names.index(n), rmsd[names.index(n)], n) for n in failed_names if n in names]
    
    # Sort valid by RMSD
    valid_rmsd.sort(key=lambda x: x[1] if math.isfinite(x[1]) else float('inf'))
    
    # Create new sorted order: valid first, then gap, then failed
    sorted_indices = [x[0] for x in valid_rmsd] + [x[0] for x in failed_rmsd]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_rmsd = [rmsd[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_names))

    has_tm = "tm_score" in df.columns and df["tm_score"].notna().any()
    cmap = _tm_colormap()
    norm = TwoSlopeNorm(vmin=0.4, vcenter=0.8, vmax=1.0)
    tm_vals = [df.iloc[sorted_indices[i]]["tm_score"] if has_tm else float("nan") 
               for i in range(len(sorted_indices))]
    
    bar_colors = []
    for i in range(len(sorted_names)):
        name = sorted_names[i]
        if style.is_failed_condition(name):
            bar_colors.append(style.PTM_COLORS["failed"])
        elif has_tm and math.isfinite(tm_vals[i]):
            bar_colors.append(cmap(norm(tm_vals[i])))
        else:
            ptm_group = style.get_ptm_group(name)
            bar_colors.append(style.PTM_COLORS.get(ptm_group, "#7F7F7F"))

    yerr = None
    if {"rmsd_lo", "rmsd_hi"}.issubset(df.columns):
        lo = [(df.iloc[sorted_indices[i]]["rmsd"] - df.iloc[sorted_indices[i]]["rmsd_lo"]).clip(lower=0) 
              for i in range(len(sorted_indices))]
        hi = [(df.iloc[sorted_indices[i]]["rmsd_hi"] - df.iloc[sorted_indices[i]]["rmsd"]).clip(lower=0)
              for i in range(len(sorted_indices))]
        yerr = [lo, hi]

    # Check if broken axis is warranted
    valid_max = max(x[1] for x in valid_rmsd if math.isfinite(x[1])) if valid_rmsd else 0
    failed_min = min(x[1] for x in failed_rmsd if math.isfinite(x[1])) if failed_rmsd else float('inf')
    do_break = (failed_rmsd and valid_rmsd and 
                math.isfinite(valid_max) and math.isfinite(failed_min) and
                failed_min - valid_max > 1.0)

    if do_break:
        return _plot_broken(df, sorted_names, sorted_labels, x, sorted_rmsd, yerr, bar_colors, tiers,
                            baseline_name, plots_dir, cmap, norm, has_tm,
                            float(valid_max), float(failed_min), has_dna, len(valid_rmsd))

    fig, ax = plt.subplots(figsize=(max(6.5, len(sorted_names) * 1.0), 5.2))
    _draw_bars(ax, x, sorted_rmsd, yerr, bar_colors, sorted_names, tiers)
    _finish(ax, x, sorted_labels, sorted_rmsd, baseline_name, tiers, sorted_names, has_dna, len(valid_rmsd))
    
    if has_tm:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="TM-score (green≥0.8 conserved, red<0.6 divergent)",
                     shrink=0.85)
    
    fig.suptitle("Structural distance vs baseline (sorted by valid RMSD, bar fill = TM-score)",
                 fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "structural_distances")


def _draw_bars(ax, x, rmsd, yerr, bar_colors, names, tiers):
    bars = ax.bar(x, rmsd, color=bar_colors, edgecolor="black", alpha=0.95,
                  yerr=yerr, capsize=4, error_kw={"linewidth": 1.1})
    
    # Add failed condition markers (§ 0)
    for i, n in enumerate(names):
        if style.is_failed_condition(n):
            # Draw red cross over the bar
            y_pos = rmsd[i] if math.isfinite(rmsd[i]) else 0
            ax.plot([i-0.3, i+0.3], [y_pos-0.3, y_pos+0.3], 'r-', linewidth=3, zorder=10)
            ax.plot([i-0.3, i+0.3], [y_pos+0.3, y_pos-0.3], 'r-', linewidth=3, zorder=10)
        
        # Legacy tier hatching for other low confidence
        t = tiers.get(n, "ok")
        if t != "ok" and not style.is_failed_condition(n):
            bars[i].set_hatch(style.TIER_HATCH.get(t, ""))
    return bars


def _finish(ax, x, labels, rmsd, baseline_name, tiers, names=None, has_dna=None, n_valid=None):
    # Add value labels on bars
    for i, v in enumerate(rmsd):
        if math.isfinite(v):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            # DNA marker
            if has_dna and names and i < len(names) and has_dna.get(names[i], False):
                ax.text(i, v + 0.35, "⊕DNA", ha="center", va="bottom",
                        fontsize=6.5, color="#009E73", style="italic")
    
    # Add visual separator between valid and failed conditions (§ 3)
    if n_valid is not None and n_valid > 0 and n_valid < len(x):
        ax.axvline(n_valid - 0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)
        ax.text(n_valid - 0.5, ax.get_ylim()[1] * 0.9, " model collapse", 
                rotation=90, ha="right", va="top", fontsize=8, color="red", fontweight="bold")
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Protein Cα RMSD vs baseline (Å)")
    
    # Color-code x-axis labels by condition type
    if names:
        for i, name in enumerate(names):
            ptm_group = style.get_ptm_group(name)
            color = style.PTM_COLORS["failed"] if style.is_failed_condition(name) else style.PTM_COLORS.get(ptm_group, "#7F7F7F")
            ax.get_xticklabels()[i].set_color(color)
    
    # Add noise floor reference (§ 3)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7, 
               label="noise floor (~0.5 Å)")
    ax.legend(fontsize=8)


def _plot_broken(df, names, labels, x, rmsd, yerr, bar_colors, tiers,
                 baseline_name, plots_dir, cmap, norm, has_tm, low_top, high_bot,
                 has_dna=None, n_valid=None):
    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, sharex=True, figsize=(max(6.5, len(names) * 1.0), 6.2),
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
    )
    for ax in (ax_hi, ax_lo):
        _draw_bars(ax, x, rmsd, yerr, bar_colors, names, tiers)

    ax_lo.set_ylim(0, low_top * 1.25)
    ax_hi.set_ylim(high_bot * 0.95, max(r for r in rmsd if math.isfinite(r)) * 1.12)
    ax_hi.set_title("Failed predictions (model collapse)", fontsize=9, loc="left", 
                   color=style.PTM_COLORS["failed"])

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

    # Add value labels and DNA markers
    for i, v in enumerate(rmsd):
        if not math.isfinite(v):
            continue
        ax = ax_hi if v >= high_bot * 0.95 else ax_lo
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        if has_dna and i < len(names) and has_dna.get(names[i], False):
            ax.text(i, v + 0.35, "⊕DNA", ha="center", va="bottom",
                    fontsize=6.5, color="#009E73", style="italic")

    # Add visual separator (§ 3)
    if n_valid is not None and n_valid > 0 and n_valid < len(x):
        ax_lo.axvline(n_valid - 0.5, color="red", linestyle="-", linewidth=2, alpha=0.7)
        ax_lo.text(n_valid - 0.5, ax_lo.get_ylim()[1] * 0.9, " model collapse", 
                   rotation=90, ha="right", va="top", fontsize=8, color="red", fontweight="bold")

    ax_lo.set_xticks(x)
    ax_lo.set_xticklabels(labels, rotation=40, ha="right")
    ax_lo.set_ylabel("Protein Cα RMSD (Å)")
    ax_lo.yaxis.set_label_coords(-0.07, 0.75)
    
    # Color-code x-axis labels
    for i, name in enumerate(names):
        ptm_group = style.get_ptm_group(name)
        color = style.PTM_COLORS["failed"] if style.is_failed_condition(name) else style.PTM_COLORS.get(ptm_group, "#7F7F7F")
        ax_lo.get_xticklabels()[i].set_color(color)
    
    # Add noise floor reference
    ax_lo.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                  label="noise floor (~0.5 Å)")
    ax_lo.legend(fontsize=8)

    if has_tm:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=(ax_hi, ax_lo),
                     label="TM-score (green≥0.8, red<0.6)", shrink=0.7)
    
    fig.suptitle("Structural distance vs baseline (broken axis; bar fill = TM-score)",
                 fontweight="bold")
    
    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Grey × = model collapse (pTM < 0.25, mean PAE > 25 Å); "
                         "excluded from biological interpretation.", 
             fontsize=8, style="italic", color="#666666")
    
    return style.save(fig, plots_dir, "structural_distances")
