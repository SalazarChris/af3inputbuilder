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
    Confidence scatter (plan 1.4a): x = pTM, y = ipTM, marker size = mean pLDDT,
    colour = PTM group.  ``likely_artifact`` conditions are drawn as X markers.

    Replaces the original three-panel bar chart, which forced the reader to join
    pTM/ipTM/pLDDT across panels.

    plot_seed_strip: if True and actual per-seed data is available in seed_sd,
    also render the seed decomposition strip plot.  Off by default because the
    approximation from a single SD value is not informative enough to show
    unconditionally.
    """
    if "ptm" not in df.columns or not df["ptm"].notna().any():
        return []

    tiers = classify_tiers(df)
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    for i, name in enumerate(names):
        row = df.iloc[i]
        ptm = row.get("ptm", float("nan"))
        iptm = row.get("iptm", float("nan"))
        pl = row.get("plddt_mean", float("nan"))
        if not (math.isfinite(ptm) and math.isfinite(iptm)):
            continue
        grp = (ptm_group or {}).get(name, "none")
        color = style.ptm_color(grp)
        size = max(40.0, (pl / 100.0) * 320.0) if math.isfinite(pl) else 80.0
        tier = tiers.get(name, "ok")
        is_ref = bool(row.get("is_reference", False))

        if tier == "likely_artifact":
            ax.scatter(ptm, iptm, s=size, marker="X", color=color,
                       edgecolor="black", linewidth=1.2, zorder=4)
        else:
            ax.scatter(ptm, iptm, s=size, marker="o", color=color,
                       edgecolor=("black" if is_ref else "white"),
                       linewidth=(2.2 if is_ref else 0.8),
                       alpha=0.9, zorder=3)
        ax.annotate(labels[i], (ptm, iptm), fontsize=6.5,
                    xytext=(4, 4), textcoords="offset points")

    # Confidence guide lines
    ax.axvline(style.LOW_CONF_IPTM, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(style.LOW_CONF_IPTM, color="gray", linestyle=":", linewidth=0.8)
    ax.text(style.LOW_CONF_IPTM, ax.get_ylim()[0], " ipTM=0.40", fontsize=6,
            color="gray", va="bottom")

    ax.set_xlabel("pTM")
    ax.set_ylabel("ipTM")
    ax.set_title("Confidence landscape (marker size = mean pLDDT)\n"
                 "X = likely artifact · bold ring = reference", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    # PTM-group legend
    if ptm_group:
        from matplotlib.lines import Line2D
        groups = sorted(set(ptm_group.get(n, "none") for n in names))
        handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=style.ptm_color(g), markeredgecolor="white",
                   label=(g if g != "none" else "unmodified"))
            for g in groups
        ]
        ax.legend(handles=handles, title="PTM group", loc="lower right", fontsize=7)

    fig.tight_layout()
    out = style.save(fig, plots_dir, "confidence_summary")

    # Supplementary: per-seed strip (plan 1.4b / 2.6) — only when actual
    # per-seed data is passed; the SD-approximation fallback is not shown
    # by default because it looks authoritative but is derived from one number.
    if plot_seed_strip:
        per_seed_data = None
        if seed_sd:
            # seed_sd is {condition: {ptm: sd, ...}} — not per-seed points.
            # Only render if caller passes actual per-seed lists via a different
            # mechanism; here we just honour the flag.
            pass
        out += plot_seed_decomposition(df, plots_dir, label_map,
                                       per_seed=per_seed_data)
    return out


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
    PAE comparison: two-panel layout when cross-chain data available.
    
    Panel 1: within-protein PAE (all conditions)
    Panel 2: cross-chain PAE (DNA conditions only)
    
    This separates protein fold confidence from protein-DNA interface confidence.
    
    cross_chain: {condition: {"within": x, "cross": y}}  (cross may be NaN).
    """
    if "mean_pae" not in df.columns or not df["mean_pae"].notna().any():
        return []
    names = df["condition"].tolist()
    labels = style.short_labels(names, label_map)
    x = np.arange(len(names))

    # Check if we have meaningful cross-chain data
    has_cross = (cross_chain and 
                 any(math.isfinite(cross_chain.get(n, {}).get("cross", float("nan"))) 
                     for n in names))
    
    if has_cross:
        # Two-panel layout: within-protein (all) + cross-chain (DNA only)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(names) * 1.4), 5.5),
                                       gridspec_kw={"width_ratios": [2.5, 1.5]})
        
        # Panel 1: Within-protein PAE (all conditions)
        within = [cross_chain.get(n, {}).get("within", df.iloc[i].get("mean_pae"))
                  for i, n in enumerate(names)]
        ax1.bar(x, within, color=style.PALETTE[0], edgecolor="black", alpha=0.9)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=40, ha="right")
        ax1.set_ylabel("Mean PAE (Å)")
        ax1.set_title("Within-protein PAE (all conditions)", fontweight="bold", fontsize=10)
        
        # Baseline reference
        if baseline_name and baseline_name in names:
            bi = names.index(baseline_name)
            b_pae_within = within[bi]
            if math.isfinite(b_pae_within):
                ax1.axhline(b_pae_within, color="gray", linestyle="--", linewidth=1.0,
                           label="baseline")
                ax1.legend(fontsize=8)
        
        # Panel 2: Cross-chain PAE (DNA conditions only)
        dna_indices = [i for i, n in enumerate(names) 
                       if math.isfinite(cross_chain.get(n, {}).get("cross", float("nan")))]
        if dna_indices:
            dna_names = [names[i] for i in dna_indices]
            dna_labels = [labels[i] for i in dna_indices]
            dna_cross = [cross_chain[names[i]]["cross"] for i in dna_indices]
            x_dna = np.arange(len(dna_indices))
            
            ax2.bar(x_dna, dna_cross, color=style.PALETTE[4], edgecolor="black", alpha=0.9)
            ax2.set_xticks(x_dna)
            ax2.set_xticklabels(dna_labels, rotation=40, ha="right")
            ax2.set_ylabel("Mean PAE (Å)")
            ax2.set_title("Protein-DNA interface PAE (DNA conditions)", 
                         fontweight="bold", fontsize=10)
        else:
            ax2.text(0.5, 0.5, "No DNA conditions", ha="center", va="center",
                    transform=ax2.transAxes, fontsize=10, color="gray")
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        fig.suptitle("PAE decomposition: protein fold vs protein-DNA interface confidence",
                     fontweight="bold", fontsize=11, y=0.98)
    else:
        # Single panel: global mean PAE
        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.1), 5.5))
        vals = df["mean_pae"].tolist()
        bars = ax.bar(x, vals, color=style.PALETTE[3], edgecolor="black", alpha=0.9)
        
        # Baseline reference line
        if baseline_name and baseline_name in names:
            bi = names.index(baseline_name)
            b_pae = df.iloc[bi].get("mean_pae", float("nan"))
            if math.isfinite(b_pae):
                ax.axhline(b_pae, color="gray", linestyle="--", linewidth=1.0,
                           label="baseline mean PAE")
                ax.legend(fontsize=8)
        
        if "is_reference" in df.columns:
            for bi in df.index[df["is_reference"]].tolist():
                bars[bi].set_hatch("//")
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel("Mean PAE (Å)")
        ax.set_title("PAE per condition", fontweight="bold")

    fig.tight_layout()
    return style.save(fig, plots_dir, "pae_comparison")
