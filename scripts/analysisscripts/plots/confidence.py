"""Confidence summary and PAE figures (af3bench2 overhaul)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


def plot_baseline_violins(baseline_name: str, baseline_data: dict, plots_dir: Path) -> List[Path]:
    """
    Violin plots for baseline ensemble metrics (Cycle 11: baseline ensemble characterization).
    
    Generates distribution plots for pTM, ipTM, pLDDT, and PAE across baseline seeds.
    Used to quantify natural variability and identify multi-modal distributions.
    
    Returns list of generated file paths.
    """
    if not baseline_data or not baseline_data.get("n_samples", 0):
        return []
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Baseline Ensemble Distribution: {baseline_name}", 
                 fontweight="bold", fontsize=12)
    
    metrics = [
        ("pTM", "ptm", 0, 0, (0, 1)),
        ("ipTM", "iptm", 0, 1, (0, 1)),
        ("pLDDT", "plddt", 1, 0, (0, 100)),
        ("PAE", "pae", 1, 1, (0, 50)),
    ]
    
    for (title, key, row, col, lim) in metrics:
        ax = axes[row, col]
        vals = baseline_data.get(key, [])
        
        if key == "pae" and isinstance(vals, float):
            # Single PAE value - create a simple box
            if not np.isnan(vals):
                ax.boxplot([vals], positions=[0], widths=0.5)
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(lim)
        else:
            # Distribution plot
            finite_vals = [v for v in vals if math.isfinite(v)]
            if finite_vals:
                # Violin plot
                parts = ax.violinplot(finite_vals, positions=[0], widths=0.8,
                                     showmeans=True, showmedians=True)
                for pc in parts['bodies']:
                    pc.set_facecolor(style.C_DISP)
                    pc.set_alpha(0.7)
                for partname in ('cmeans','cmedians','cmins','cmaxes'):
                    if partname in parts:
                        parts[partname].set_color('black')
                        parts[partname].set_linewidth(1.5)
                ax.set_xlim(-0.5, 0.5)
                ax.set_ylim(lim)
        
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f"{title} (n={len(vals) if isinstance(vals, list) else 1})", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "baseline_violins")


def classify_tiers(df: pd.DataFrame) -> Dict[str, str]:
    """
    Two-tier confidence classification (plan 0.4).

    Returns {condition: tier} where tier ∈ {ok, low_confidence, likely_artifact}.
    """
    tiers: Dict[str, str] = {}
    for _, row in df.iterrows():
        tiers[row["condition"]] = style.classify_tier(
            row.get("plddt_mean", float("nan")), row.get("mean_pae", float("nan"))
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


def _baseline_seed_sd(df: pd.DataFrame, baseline_name: Optional[str],
                      col: str) -> Optional[float]:
    """Return the baseline condition's seed-SD value for a column, if present."""
    if baseline_name is None or col not in df.columns:
        return None
    sub = df.loc[df["condition"] == baseline_name, col]
    if sub.empty:
        return None
    val = sub.iloc[0]
    return float(val) if (val is not None and math.isfinite(val)) else None


def plot_baseline_diagnostics(
    df: pd.DataFrame,
    plots_dir: Path,
    baseline_name: str,
    per_seed_ptm: Optional[List[float]] = None,
    seed_ids: Optional[List[str]] = None,
) -> List[Path]:
    """
    Baseline per-seed pTM distribution diagnostic (Fix 7).

    Renders a strip plot of per-seed pTM for the baseline condition with the
    ensemble mean and ±2 SD guides, flagging any seed beyond 2 SD as a potential
    outlier.  Produced as a standalone ``baseline_diagnostics`` figure so the
    confidence_summary layout is untouched.
    """
    if per_seed_ptm is None or len(per_seed_ptm) == 0:
        return []
    pts = np.asarray([p for p in per_seed_ptm if math.isfinite(p)], dtype=float)
    if pts.size == 0:
        return []

    mean = float(np.nanmean(pts))
    sd = float(np.nanstd(pts, ddof=1)) if pts.size >= 2 else 0.0
    lo2, hi2 = mean - 2 * sd, mean + 2 * sd

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    rng = np.random.default_rng(0)
    jit = (rng.random(pts.size) - 0.5) * 0.3
    outlier = np.abs(pts - mean) > 2 * sd if sd > 0 else np.zeros(pts.size, dtype=bool)

    ax.scatter(jit[~outlier], pts[~outlier], s=45, color=style.PALETTE[0],
               alpha=0.8, zorder=3, label="per-seed pTM")
    if outlier.any():
        ax.scatter(jit[outlier], pts[outlier], s=90, marker="*",
                   color=style.C_FAIL, zorder=4, label="seed > 2 SD (outlier)")
        if seed_ids is not None and len(seed_ids) == pts.size:
            for k in np.where(outlier)[0]:
                ax.annotate(str(seed_ids[k]), (jit[k], pts[k]),
                            xytext=(6, 0), textcoords="offset points",
                            fontsize=style.FS_ANNOTATION, color=style.C_FAIL)

    ax.axhline(mean, color="black", linewidth=1.4, zorder=2,
               label=f"ensemble mean ({mean:.3f})")
    ax.axhline(hi2, color="gray", linestyle="--", linewidth=1.0, zorder=2,
               label=f"±2 SD ({sd:.3f})")
    ax.axhline(lo2, color="gray", linestyle="--", linewidth=1.0, zorder=2)

    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_ylabel("pTM (per seed)", fontsize=style.FS_AXIS_LABEL)
    ax.tick_params(labelsize=style.FS_TICK_LABEL)
    ax.set_title(f"Baseline per-seed pTM distribution\n{baseline_name}",
                 fontweight="bold", fontsize=style.FS_SUBPLOT_TITLE)
    ax.legend(fontsize=style.FS_ANNOTATION, loc="best")
    fig.tight_layout()
    return style.save(fig, plots_dir, "baseline_diagnostics")


def plot_ptm_scatter(
    df: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    ion_tier: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """
    pTM vs ipTM scatter plot with PTM group colours (Cycle 2 improvement).
    
    Each point represents a condition. Color indicates PTM group, and failed
    conditions are marked with a cross. The plot helps visualize the correlation
    between global and interface confidence scores.
    
    Returns list of generated file paths.
    """
    if not {"ptm", "iptm"}.issubset(df.columns):
        return []
    
    names = df["condition"].tolist()
    labels = style.short_condition_labels(names, ion_tier)
    
    # Extract values
    ptm_vals = df["ptm"].tolist()
    iptm_vals = df["iptm"].tolist()
    
    # Classify tiers for artifact marking
    tiers = classify_tiers(df)
    def _is_art(n: str) -> bool:
        return tiers.get(n, "ok") == "likely_artifact"
    
    # Color by PTM group
    ptm_colors = []
    for name in names:
        ptm_grp = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_grp, "#7F7F7F")
        if _is_art(name):
            color = style.PTM_COLORS["failed"]
        ptm_colors.append(color)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, name in enumerate(names):
        ax.scatter([ptm_vals[i]], [iptm_vals[i]], 
                   s=80, color=ptm_colors[i], edgecolor="black",
                   alpha=0.8, zorder=3, label=labels[i])
        
        # Mark failed conditions with a cross
        if _is_art(name):
            ax.plot([ptm_vals[i]-0.03, ptm_vals[i]+0.03], 
                   [iptm_vals[i]-0.03, iptm_vals[i]+0.03], 
                   'r-', linewidth=2, zorder=4)
            ax.plot([ptm_vals[i]-0.03, ptm_vals[i]+0.03], 
                   [iptm_vals[i]+0.03, iptm_vals[i]-0.03], 
                   'r-', linewidth=2, zorder=4)
    
    ax.set_xlabel("pTM (global confidence)", fontsize=style.FS_AXIS_LABEL)
    ax.set_ylabel("ipTM (interface confidence)", fontsize=style.FS_AXIS_LABEL)
    ax.set_title("pTM vs ipTM by PTM group", fontweight="bold", fontsize=style.FS_SUBPLOT_TITLE)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=style.FS_TICK_LABEL)
    
    # Add diagonal reference line (pTM == ipTM)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, zorder=1,
            label="y=x (equal confidence)")
    
    # Legend for PTM groups
    from matplotlib.lines import Line2D
    handles = []
    for ptm_grp in ["unmodified", "SEP102", "TPO101", "DNA"]:
        color = style.PTM_COLORS.get(ptm_grp, "#7F7F7F")
        handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=8,
                             markerfacecolor=color, markeredgecolor="black",
                             label=ptm_grp))
    handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                         markerfacecolor=style.PTM_COLORS["failed"], 
                         markeredgecolor="black", label="failed prediction"))
    ax.legend(handles=handles, loc="lower right", fontsize=style.FS_ANNOTATION)
    
    # Add correlation coefficient annotation
    finite_mask = np.isfinite(ptm_vals) & np.isfinite(iptm_vals)
    if np.sum(finite_mask) >= 2:
        corr = np.corrcoef([ptm_vals[i] for i in range(len(ptm_vals)) if finite_mask[i]],
                          [iptm_vals[i] for i in range(len(iptm_vals)) if finite_mask[i]])[0, 1]
        if np.isfinite(corr):
            ax.text(0.02, 0.98, f"r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=style.FS_ANNOTATION, verticalalignment="top",
                    fontfamily="monospace")
    
    ax.grid(True, alpha=0.3, zorder=0)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "ptm_scatter")


def plot_confidence_summary(
    df: pd.DataFrame,
    plots_dir: Path,
    seed_sd: Optional[Dict[str, Dict[str, float]]] = None,
    label_map: Optional[Dict[str, str]] = None,
    ptm_group: Optional[Dict[str, str]] = None,
    plot_seed_strip: bool = False,
    ion_tier: Optional[Dict[str, str]] = None,
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
    # Artifact marking is data-driven from the macromolecule-scoped tier
    # (protein pLDDT + macromolecule PAE), never the legacy hardcoded name set.
    def _is_art(n: str) -> bool:
        return tiers.get(n, "ok") == "likely_artifact"
    names = df["condition"].tolist()
    # Fix 6: consistent short factor-string labels.
    labels = style.short_condition_labels(names, ion_tier)
    valid_names, failed_names = style.split_conditions(names)

    # Fix 1: only annotate bar values for a small set of reference bars when the
    # condition count is high (avoid a solid block of overlapping text).
    annotate_values = len(names) > 15

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
    # Helper: per-panel min/max/reference annotation gating (Fix 1).  When the
    # condition count exceeds 15, annotate only the min, max, and reference bars
    # to avoid a solid block of overlapping text.
    def _annotate_panel(ax, vals, fmt, offset):
        finite_idx = [i for i, v in enumerate(vals) if math.isfinite(v)]
        if not finite_idx:
            return
        if annotate_values:
            mn = min(finite_idx, key=lambda i: vals[i])
            mx = max(finite_idx, key=lambda i: vals[i])
            ref_idx = [i for i, n in enumerate(sorted_names)
                       if n == baseline_for_ref]
            to_label = {mn, mx} | set(ref_idx)
        else:
            to_label = set(finite_idx)
        for i in to_label:
            ax.text(i, vals[i] + offset, fmt.format(vals[i]), ha="center",
                    va="bottom", fontsize=style.FS_ANNOTATION, rotation=90)

    # Resolve the reference condition (is_reference flag) for annotation/error bars.
    baseline_for_ref = None
    if "is_reference" in df.columns:
        ref_rows = df.loc[df["is_reference"].astype(bool), "condition"].tolist()
        if ref_rows:
            baseline_for_ref = ref_rows[0]

    def _seed_err(metric, name):
        if seed_sd and name in seed_sd:
            sd = seed_sd[name].get(metric)
            if sd is not None and math.isfinite(sd):
                return float(sd)
        return None

    bars1 = []
    for i, name in enumerate(sorted_names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if _is_art(name):
            color = style.PTM_COLORS["failed"]
        err = _seed_err("ptm", name)
        bar = ax1.bar([i], [ptm_vals[i]], color=color, edgecolor="black", alpha=0.9,
                      yerr=[err] if err is not None else None, capsize=3,
                      error_kw={"linewidth": 0.9})
        bars1.append(bar)
        
        # Add failed condition marker (§ 0)
        if _is_art(name):
            # Draw diagonal cross
            ax1.plot([i-0.3, i+0.3], [ptm_vals[i]-0.05, ptm_vals[i]+0.05], 'r-', linewidth=2)
            ax1.plot([i-0.3, i+0.3], [ptm_vals[i]+0.05, ptm_vals[i]-0.05], 'r-', linewidth=2)
    
    ax1.set_ylabel("pTM", fontsize=style.FS_AXIS_LABEL)
    ax1.set_title("Predicted TM-score", fontsize=style.FS_SUBPLOT_TITLE)
    ax1.set_ylim(0, 1)
    _annotate_panel(ax1, ptm_vals, "{:.2f}", 0.02)
    
    # Panel 2: ipTM  
    iptm_vals = [df.iloc[name_to_idx[name]]["iptm"] for name in sorted_names]
    bars2 = []
    for i, name in enumerate(sorted_names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if _is_art(name):
            color = style.PTM_COLORS["failed"]
        err = _seed_err("iptm", name)
        bar = ax2.bar([i], [iptm_vals[i]], color=color, edgecolor="black", alpha=0.9,
                      yerr=[err] if err is not None else None, capsize=3,
                      error_kw={"linewidth": 0.9})
        bars2.append(bar)
        
        # Add failed condition marker (§ 0)
        if _is_art(name):
            ax2.plot([i-0.3, i+0.3], [iptm_vals[i]-0.05, iptm_vals[i]+0.05], 'r-', linewidth=2)
            ax2.plot([i-0.3, i+0.3], [iptm_vals[i]+0.05, iptm_vals[i]-0.05], 'r-', linewidth=2)
    
    ax2.set_ylabel("ipTM", fontsize=style.FS_AXIS_LABEL)
    ax2.set_title("Interface predicted TM-score", fontsize=style.FS_SUBPLOT_TITLE)
    ax2.set_ylim(0, 1)
    _annotate_panel(ax2, iptm_vals, "{:.2f}", 0.02)
    
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
        if _is_art(name):
            color = style.PTM_COLORS["failed"]
        err = _seed_err("plddt_mean", name)
        bar = ax3.bar([i], [plddt_vals[i]], color=color, edgecolor="black", alpha=0.9,
                      yerr=[err] if err is not None else None, capsize=3,
                      error_kw={"linewidth": 0.9})
        bars3.append(bar)
        
        # Add failed condition marker (§ 0)
        if _is_art(name):
            ax3.plot([i-0.3, i+0.3], [plddt_vals[i]-2, plddt_vals[i]+2], 'r-', linewidth=2)
            ax3.plot([i-0.3, i+0.3], [plddt_vals[i]+2, plddt_vals[i]-2], 'r-', linewidth=2)
    
    ax3.set_ylabel("Mean pLDDT", fontsize=style.FS_AXIS_LABEL)
    ax3.set_title("Per-residue confidence", fontsize=style.FS_SUBPLOT_TITLE)
    ax3.set_ylim(plddt_min, plddt_max)
    _annotate_panel(ax3, plddt_vals, "{:.0f}", (plddt_max - plddt_min) * 0.02)

    # Add secondary x-axis showing salt tier (§ 1)
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_labels, rotation=45, ha="right",
                           fontsize=style.FS_TICK_LABEL)
        ax.tick_params(labelsize=style.FS_TICK_LABEL)
        
        # Color-coded tick labels by PTM group
        for i, name in enumerate(sorted_names):
            ptm_group_name = style.get_ptm_group(name)
            color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
            if _is_art(name):
                color = style.PTM_COLORS["failed"]
            ax.get_xticklabels()[i].set_color(color)
    
    # Add legend for PTM groups
    from matplotlib.lines import Line2D
    handles = []
    for ptm_grp in ["unmodified", "SEP102", "TPO101", "DNA"]:
        color = style.PTM_COLORS.get(ptm_grp, "#7F7F7F")
        handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                             markerfacecolor=color, markeredgecolor="black",
                             label=ptm_grp))
    # Add failed condition legend
    handles.append(Line2D([0], [0], marker="s", linestyle="", markersize=8,
                         markerfacecolor=style.PTM_COLORS["failed"], 
                         markeredgecolor="black", label="failed prediction"))
    
    fig.legend(handles=handles, title="PTM group", loc="upper right", 
               bbox_to_anchor=(0.98, 0.95), fontsize=style.FS_ANNOTATION)
    
    # Footnote (§ 0) plus baseline pTM heterogeneity caveat (Fix 7).
    foot = ("Error bars: seed-level SD.  Grey × = collapsed prediction "
            "(protein pLDDT < 50 AND macromolecule PAE > 25 Å); excluded from interpretation. "
            "Note: pTM/ipTM are full-system scores (include free Na\u207a/Cl\u207b/water tokens) and "
            "are deflated by solvent count; the reliability tier uses protein pLDDT + "
            "macromolecule-scoped PAE, so low pTM/ipTM at high ionic strength does not by "
            "itself indicate a failed protein prediction.")
    base_sd = _baseline_seed_sd(df, baseline_for_ref, "ptm_seed_sd")
    if base_sd is not None and base_sd > 0.03:
        foot += (f"\nBaseline pTM seed SD = {base_sd:.3f} (> 0.03): baseline "
                 f"confidence is heterogeneous across seeds; interpret ΔpTM with caution.")
    fig.text(0.02, 0.02, foot, fontsize=style.FS_ANNOTATION, style="italic",
             color="#666666")

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
    # Data-driven artifact marking from the macromolecule-scoped tier.
    _tiers = classify_tiers(df)
    def _is_art(n: str) -> bool:
        return _tiers.get(n, "ok") == "likely_artifact"
    
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
            if _is_art(name):
                color = style.PTM_COLORS["failed"]
            
            bar_lo = ax_lo.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bar_hi = ax_hi.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bars_lo.append(bar_lo)
            bars_hi.append(bar_hi)
            
            # Add failed condition markers (§ 0)
            if _is_art(name):
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
        
        ax_lo.set_ylabel("Macromolecule mean PAE (Å)")
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
            if _is_art(name):
                color = style.PTM_COLORS["failed"]
            
            bar = ax.bar([i], [vals[i]], color=color, edgecolor="black", alpha=0.9)
            bars.append(bar)
            
            # Add failed condition markers (§ 0)
            if _is_art(name):
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
        
        ax.set_ylabel("Macromolecule mean PAE (Å)")
        ax.set_title("PAE per condition", fontweight="bold")

    # Set x-axis for all cases
    target_ax = ax_lo if use_broken_axis else ax
    target_ax.set_xticks(x)
    target_ax.set_xticklabels(labels, rotation=40, ha="right")
    
    # Color-code x-axis labels by PTM group
    for i, name in enumerate(names):
        ptm_group_name = style.get_ptm_group(name)
        color = style.PTM_COLORS.get(ptm_group_name, "#7F7F7F")
        if _is_art(name):
            color = style.PTM_COLORS["failed"]
        target_ax.get_xticklabels()[i].set_color(color)
    
    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Mean PAE is macromolecule-scoped (protein+nucleic tokens only; "
                         "excludes free ion/water). Grey × = collapsed prediction "
                         "(protein pLDDT < 50 AND macromolecule PAE > 25 Å).",
             fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    return style.save(fig, plots_dir, "pae_comparison")
