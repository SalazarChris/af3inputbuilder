"""Confidence summary and PAE figures (af3bench2 overhaul)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


def classify_tiers(df: pd.DataFrame) -> Dict[str, str]:
    """
    Two-tier confidence classification (plan 0.4).

    Returns {condition: tier} where tier ∈ {ok, low_confidence, likely_artifact}.
    """
    tiers: Dict[str, str] = {}
    for _, row in df.iterrows():
        tiers[row["condition"]] = style.classify_tier(
            row.get("ptm", float("nan")), row.get("iptm", float("nan"))
        )
    return tiers


def detect_failed(df: pd.DataFrame) -> set:
    """
    Legacy single-tier failure flag, retained for backward compatibility.

    af3bench2 prefers :func:`classify_tiers`; this is a thin wrapper that
    returns all non-ok conditions so existing callers still get a conservative
    "exclude" set without contradicting the authoritative tier classification.
    """
    return {c for c, t in classify_tiers(df).items() if t != "ok"}


def likely_artifacts(df: pd.DataFrame) -> set:
    """Conditions in the ``likely_artifact`` tier (excluded from grids)."""
    return {c for c, t in classify_tiers(df).items() if t == "likely_artifact"}


def plot_confidence_summary(
    df: pd.DataFrame,
    plots_dir: Path,
    seed_sd: Optional[Dict[str, Dict[str, float]]] = None,
    label_map: Optional[Dict[str, str]] = None,
    ptm_group: Optional[Dict[str, str]] = None,
    plot_seed_strip: bool = False,
) -> List[Path]:
    """
    Confidence summary: three-panel bar chart (§ 1) with PTM group colours,
    failed condition quarantine, and proper axis scaling.
    
    Returns to bar chart format per improvement guide while implementing
    proper visual treatment for failed conditions and PTM group organization.
    """
    if not {"ptm", "iptm", "plddt_mean"}.issubset(df.columns):
        return []

    tiers = classify_tiers(df)
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)
    valid_names, failed_names = style.split_conditions(names)

    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    
    # Sort conditions within PTM groups by salt tier for visibility (§ 1)
    sorted_names = []
    for ptm in ["unmodified", "SEP102", "TPO101", "DNA"]:
        group_conditions = [n for n in names if style.get_ptm_group(n) == ptm]
        # Sort by salt tier
        group_conditions.sort(key=lambda x: (
            0 if "nax1_" in x else 1 if "nax10_" in x else 2 if "nax100_" in x else 3
        ))
        sorted_names.extend(group_conditions)
    
    # Reorder data according to sorted names
    name_to_idx = {name: i for i, name in enumerate(names)}
    sorted_indices = [name_to_idx[name] for name in sorted_names]
    sorted_labels = [labels[i] for i in sorted_indices]
    x = np.arange(len(sorted_names))

    # Panel 1: pTM
    ptm_vals = [df.iloc[name_to_idx[name]]["ptm"] for name in sorted_names]
    bars1 = []
    for i, name in enumerate(sorted_names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if style.is_failed_condition(name):
            color = style.PTM_COLORS["failed"]
        bar = ax1.bar([i], [ptm_vals[i]], color=color, edgecolor="black", alpha=0.9)
        bars1.append(bar)
        
        # Add failed condition marker (§ 0)
        if style.is_failed_condition(name):
            # Draw diagonal cross
            ax1.plot([i-0.3, i+0.3], [ptm_vals[i]-0.05, ptm_vals[i]+0.05], 'r-', linewidth=2)
            ax1.plot([i-0.3, i+0.3], [ptm_vals[i]+0.05, ptm_vals[i]-0.05], 'r-', linewidth=2)
    
    ax1.set_ylabel("pTM")
    ax1.set_title("Predicted TM-score")
    ax1.set_ylim(0, 1)
    
    # Panel 2: ipTM  
    iptm_vals = [df.iloc[name_to_idx[name]]["iptm"] for name in sorted_names]
    bars2 = []
    for i, name in enumerate(sorted_names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if style.is_failed_condition(name):
            color = style.PTM_COLORS["failed"]
        bar = ax2.bar([i], [iptm_vals[i]], color=color, edgecolor="black", alpha=0.9)
        bars2.append(bar)
        
        # Add failed condition marker (§ 0)
        if style.is_failed_condition(name):
            ax2.plot([i-0.3, i+0.3], [iptm_vals[i]-0.05, iptm_vals[i]+0.05], 'r-', linewidth=2)
            ax2.plot([i-0.3, i+0.3], [iptm_vals[i]+0.05, iptm_vals[i]-0.05], 'r-', linewidth=2)
    
    ax2.set_ylabel("ipTM")
    ax2.set_title("Interface predicted TM-score")
    ax2.set_ylim(0, 1)
    
    # Panel 3: mean pLDDT - rescaled axis (§ 1)
    plddt_vals = [df.iloc[name_to_idx[name]]["plddt_mean"] for name in sorted_names]
    finite_plddt = [v for v in plddt_vals if math.isfinite(v)]
    if finite_plddt:
        plddt_min = max(60, min(finite_plddt) - 5)
        plddt_max = 100
    else:
        plddt_min, plddt_max = 60, 100
        
    bars3 = []
    for i, name in enumerate(sorted_names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if style.is_failed_condition(name):
            color = style.PTM_COLORS["failed"]
        bar = ax3.bar([i], [plddt_vals[i]], color=color, edgecolor="black", alpha=0.9)
        bars3.append(bar)
        
        # Add failed condition marker (§ 0)
        if style.is_failed_condition(name):
            ax3.plot([i-0.3, i+0.3], [plddt_vals[i]-2, plddt_vals[i]+2], 'r-', linewidth=2)
            ax3.plot([i-0.3, i+0.3], [plddt_vals[i]+2, plddt_vals[i]-2], 'r-', linewidth=2)
    
    ax3.set_ylabel("Mean pLDDT")
    ax3.set_title("Per-residue confidence")
    ax3.set_ylim(plddt_min, plddt_max)

    # Add secondary x-axis showing salt tier (§ 1)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_labels, rotation=45, ha="right")
        
        # Color-coded tick labels by PTM group
        for i, name in enumerate(sorted_names):
            ptm_group_name = style.get_ptm_group(name)
            color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
            if style.is_failed_condition(name):
                color = style.PTM_COLORS["failed"]
            ax.get_xticklabels()[i].set_color(color)
    
    # Add legend for PTM groups
    from matplotlib.lines import Line2D
    handles = []
    for ptm_group in ["unmodified", "SEP102", "TPO101", "DNA"]:
        color = style.PTM_COLORS.get(ptm_group, "#7F7F7F")
        handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                             markerfacecolor=color, markeredgecolor="black",
                             label=ptm_group))
    # Add failed condition legend
    handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                         markerfacecolor=style.PTM_COLORS["failed"], 
                         markeredgecolor="black", label="failed prediction"))
    
    fig.legend(handles=handles, title="PTM group", loc="upper right", 
               bbox_to_anchor=(0.98, 0.95), fontsize=8)
    
    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Grey × = model collapse (pTM < 0.25, mean PAE > 25 Å); "
                         "excluded from biological interpretation.", 
             fontsize=8, style="italic", color="#666666")

    fig.suptitle("Confidence summary by PTM group and salt tier", 
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    return style.save(fig, plots_dir, "confidence_summary")

    # Note: seed strip functionality preserved but moved to separate function
    # if plot_seed_strip is requested


def plot_seed_decomposition(
    df: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    per_seed: Optional[Dict[str, List[float]]] = None,
) -> List[Path]:
    """
    Per-condition seed strip plot (plan 1.4b / 2.6).

    When ``per_seed`` (condition -> list of per-seed mean pTM) is available it is
    drawn directly; otherwise the per-seed spread is approximated from the
    reported seed SD as a ±1 SD band around the condition mean, so the figure
    is always produced.
    """
    names = df["condition"].tolist()
    if not names:
        return []
    labels = style.short_labels(names, label_map)
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.9), 5))
    for i, name in enumerate(names):
        mean = df.iloc[i].get("ptm", float("nan"))
        if per_seed and name in per_seed and len(per_seed[name]):
            pts = np.asarray(per_seed[name], dtype=float)
            jit = (np.random.default_rng(i).random(pts.size) - 0.5) * 0.25
            ax.scatter(np.full(pts.size, i) + jit, pts, s=18, alpha=0.7,
                       color=style.PALETTE[0], zorder=3)
            cmean = float(np.nanmean(pts))
            # flag seed outliers (>2× within-seed SD from the condition mean)
            sd = float(np.nanstd(pts, ddof=1)) if pts.size >= 2 else 0.0
            if sd > 0:
                for p in pts:
                    if abs(p - cmean) > 2 * sd:
                        ax.scatter([i], [p], s=70, marker="*", color=style.C_FAIL,
                                   zorder=4)
        else:
            sd = df.iloc[i].get("ptm_seed_sd", 0.0) or 0.0
            if math.isfinite(mean):
                ax.errorbar([i], [mean], yerr=[[sd], [sd]], fmt="o",
                            color=style.PALETTE[0], capsize=4, zorder=3)
        if math.isfinite(mean):
            ax.hlines(mean, i - 0.3, i + 0.3, color="black", linewidth=1.4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("pTM (per seed)" if per_seed else "pTM (mean ± seed SD)")
    ax.set_title("Seed-level pTM dispersion per condition\n"
                 "(★ = seed >2×SD from condition mean)", fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "seed_variance_by_condition")


def plot_pae(
    df: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    cross_chain: Optional[Dict[str, Dict[str, float]]] = None,
    baseline_name: Optional[str] = None,
) -> List[Path]:
    """
    PAE comparison with broken y-axis (§ 2) and failed condition treatment.
    
    Shows within-protein PAE with proper handling of failed conditions
    that have PAE ≈ 30 Å, using broken axis to reveal detail in valid range.
    """
    if "mean_pae" not in df.columns or not df["mean_pae"].notna().any():
        return []
    
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)
    x = np.arange(len(names))
    vals = df["mean_pae"].tolist()
    
    valid_names, failed_names = style.split_conditions(names)
    valid_vals = [vals[names.index(n)] for n in valid_names if n in names]
    failed_vals = [vals[names.index(n)] for n in failed_names if n in names]
    
    # Determine if broken axis is needed (§ 2)
    valid_max = max(v for v in valid_vals if math.isfinite(v)) if valid_vals else 0
    failed_min = min(v for v in failed_vals if math.isfinite(v)) if failed_vals else float('inf')
    use_broken_axis = (failed_vals and valid_vals and 
                      math.isfinite(valid_max) and math.isfinite(failed_min) and
                      failed_min - valid_max > 10)  # 10 Å gap threshold
    
    if use_broken_axis:
        # Create broken axis layout
        fig, (ax_hi, ax_lo) = plt.subplots(
            2, 1, sharex=True, figsize=(max(7, len(names) * 1.1), 7),
            gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.08}
        )
        
        # Draw bars on both axes
        bars_lo = []
        bars_hi = []
        for i, name in enumerate(names):
            ptm_group_name = style.get_ptm_group(name)
            color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
            if style.is_failed_condition(name):
                color = style.PTM_COLORS["failed"]
            
            bar_lo = ax_lo.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bar_hi = ax_hi.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bars_lo.append(bar_lo)
            bars_hi.append(bar_hi)
            
            # Add failed condition markers (§ 0)
            if style.is_failed_condition(name):
                for ax in [ax_lo, ax_hi]:
                    ax.plot([i-0.3, i+0.3], [vals[i]-1, vals[i]+1], 'r-', linewidth=2, zorder=10)
                    ax.plot([i-0.3, i+0.3], [vals[i]+1, vals[i]-1], 'r-', linewidth=2, zorder=10)
        
        # Set axis limits
        ax_lo.set_ylim(0, valid_max + 2)
        ax_hi.set_ylim(failed_min - 2, max(v for v in vals if math.isfinite(v)) + 2)
        ax_hi.set_title("Failed predictions (model collapse)", fontsize=9, loc="left", 
                       color=style.PTM_COLORS["failed"])
        
        # Draw broken axis indicators
        ax_hi.spines["bottom"].set_visible(False)
        ax_lo.spines["top"].set_visible(False)
        ax_hi.tick_params(bottom=False)
        
        d = 0.008
        kwargs = dict(transform=ax_hi.transAxes, color="k", clip_on=False, linewidth=1)
        ax_hi.plot((-d, +d), (-d, +d), **kwargs)
        ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs2 = dict(transform=ax_lo.transAxes, color="k", clip_on=False, linewidth=1)
        ax_lo.plot((-d, +d), (1 - d * 2.5, 1 + d * 2.5), **kwargs2)
        ax_lo.plot((1 - d, 1 + d), (1 - d * 2.5, 1 + d * 2.5), **kwargs2)
        
        ax_lo.set_ylabel("Mean PAE (Å)")
        ax_lo.yaxis.set_label_coords(-0.07, 0.75)
        
        # Add baseline reference line (§ 2)
        if baseline_name and baseline_name in names:
            bi = names.index(baseline_name)
            b_pae = vals[bi]
            if math.isfinite(b_pae):
                target_ax = ax_lo if b_pae <= valid_max + 2 else ax_hi
                target_ax.axhline(b_pae, color="gray", linestyle="--", linewidth=1.0,
                                 label=f"baseline PAE ({b_pae:.1f} Å)")
                target_ax.legend(fontsize=8)
        
        fig.suptitle("PAE comparison (broken axis for failed predictions)", 
                     fontweight="bold")
        
    else:
        # Standard single axis
        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 5.5))
        bars = []
        for i, name in enumerate(names):
            ptm_group_name = style.get_ptm_group(name)
            color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
            if style.is_failed_condition(name):
                color = style.PTM_COLORS["failed"]
            
            bar = ax.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bars.append(bar)
            
            # Add failed condition markers (§ 0)
            if style.is_failed_condition(name):
                ax.plot([i-0.3, i+0.3], [vals[i]-1, vals[i]+1], 'r-', linewidth=2, zorder=10)
                ax.plot([i-0.3, i+0.3], [vals[i]+1, vals[i]-1], 'r-', linewidth=2, zorder=10)
        
        # Add baseline reference line (§ 2)
        if baseline_name and baseline_name in names:
            bi = names.index(baseline_name)
            b_pae = vals[bi]
            if math.isfinite(b_pae):
                ax.axhline(b_pae, color="gray", linestyle="--", linewidth=1.0,
                          label=f"baseline PAE ({b_pae:.1f} Å)")
                ax.legend(fontsize=8)
        
        ax.set_ylabel("Mean PAE (Å)")
        ax.set_title("PAE per condition", fontweight="bold")

    # Set x-axis for all cases
    target_ax = ax_lo if use_broken_axis else ax
    target_ax.set_xticks(x)
    target_ax.set_xticklabels(labels, rotation=40, ha="right")
    
    # Color-code x-axis labels by PTM group
    for i, name in enumerate(names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if style.is_failed_condition(name):
            color = style.PTM_COLORS["failed"]
        target_ax.get_xticklabels()[i].set_color(color)
    
    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Grey × = model collapse (pTM < 0.25, mean PAE > 25 Å); "
                         "excluded from biological interpretation.", 
             fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    return style.save(fig, plots_dir, "pae_comparison")
