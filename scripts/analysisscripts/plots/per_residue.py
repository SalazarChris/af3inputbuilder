"""
Per-residue displacement figures (af3bench2 overhaul).

Two stacked panels sharing the residue-number x-axis:

  1. Displacement (Å): condition-vs-baseline mean with a 95% CI ribbon, the
     baseline ensemble RMSF noise envelope, an optional between-replicate IQR
     band for heterogeneous conditions, FDR significance dots, a significance
     rug on the x-axis, and a low-pLDDT amber overlay.
  2. pLDDT: reference vs condition per-residue confidence, with a ΔpLDDT label
     at PTM sites that lose confidence.

Plan items implemented: 1.1a (rug replaces strip), 1.1b (ΔpLDDT label),
1.1c (adaptive y-scale), 1.1d (low-pLDDT overlay), 1.1e (dashed small-N CI),
1.1f (between-replicate IQR band), 0.4 (two-tier confidence styling),
0.1 (baseline composition annotation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import style


def _ptm_positions(ptm_labels: List[str]) -> List[int]:
    out = []
    for lab in ptm_labels or []:
        digits = "".join(ch for ch in lab if ch.isdigit())
        if digits:
            out.append(int(digits))
    return out


def _adaptive_ceiling(disp_hi: np.ndarray, floor: float = 5.0) -> float:
    """Per-condition y-max (plan 1.1c): tight to the data, never below floor."""
    vals = disp_hi[np.isfinite(disp_hi)]
    if vals.size == 0:
        return floor
    return float(max(np.ceil(np.nanmax(vals) * 1.15), floor))


def plot_profile(
    profile: dict,
    plots_dir: Path,
    ref_name: str,
    baseline_warning: Optional[str] = None,
    has_dna: bool = False,
    y_ceiling: Optional[float] = None,  # NEW: standardized y-axis (§ 7a)
) -> List[Path]:
    """
    Render one per-residue figure with standardized y-axis and failed condition treatment (§ 7).
    
    NEW in implementation guide:
    - Standardized y-axis ceiling across all plots (§ 7a)
    - Failed condition banner for collapsed predictions (§ 7b)
    - Three-tier significance strip (future enhancement)
    """
    name = profile["name"]
    label = profile.get("label_short", name)
    x = np.asarray(profile["res_numbers"], dtype=float)
    disp = np.asarray(profile["disp_mean"], dtype=float)
    lo = np.asarray(profile.get("disp_lo", disp), dtype=float)
    hi = np.asarray(profile.get("disp_hi", disp), dtype=float)
    rmsf = np.asarray(profile.get("baseline_rmsf", np.full_like(disp, np.nan)), dtype=float)
    sig = np.asarray(profile.get("significant", np.zeros_like(disp, dtype=bool)), dtype=bool)
    ref_pl = np.asarray(profile["ref_plddt"], dtype=float)
    cond_pl = np.asarray(profile["cond_plddt"], dtype=float)
    chain_ids = profile.get("chain_ids", [])
    n_c = profile.get("n_samples", 0)
    n_b = profile.get("n_samples_base", 0)
    pr_ref = profile.get("per_residue_ref", ref_name)
    ref_lbl = style.short_labels([pr_ref])[0] if pr_ref else "baseline"
    tier = profile.get("tier", "ok")
    small_n = n_c < 20
    
    # Data-driven collapse flag from the profile (set in analysis.run from the
    # macromolecule-scoped collapsed set), not the legacy hardcoded name list.
    is_failed = bool(profile.get("is_collapsed", False)) or profile.get("tier") == "likely_artifact"

    iqr_lo = profile.get("disp_iqr_lo")
    iqr_hi = profile.get("disp_iqr_hi")
    has_iqr_band = (
        iqr_lo is not None and iqr_hi is not None
        and (profile.get("n_clusters", 1) > 4
             or profile.get("dominant_fraction", 1.0) < 0.70)
    )

    # Standardized figure size for consistent significance strips (§ 9)
    fig = plt.figure(figsize=(14, 7.5))  # Increased height for better significance strip visibility
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.4], hspace=0.10)
    ax_disp = fig.add_subplot(gs[0])
    ax_pl = fig.add_subplot(gs[1], sharex=ax_disp)

    # Cycle 9: Get cluster assignment for title
    cluster_info = ""
    n_clusters = profile.get("n_clusters", 1)
    dominant_frac = profile.get("dominant_fraction", 1.0)
    if n_clusters > 1:
        cluster_info = f" (n_clusters={n_clusters}, dom={dominant_frac:.0%})"
    
    # Cycle 11: Note if baseline is multi-cluster
    baseline_cluster_info = profile.get("baseline_cluster_info", {})
    if baseline_cluster_info:
        base_clusters = baseline_cluster_info.get("n_clusters", 1)
        base_dominant = baseline_cluster_info.get("dominant_fraction", 1.0)
        if base_clusters > 1:
            cluster_info += f" (baseline n_cl={base_clusters}, dom={base_dominant:.0%})"
    
    title = f"Per-residue displacement vs {ref_lbl}: {style.short_labels([name])[0]}{cluster_info}"
    if tier != "ok":
        title += f"   [{style.TIER_LABEL.get(tier, tier)}]"
    if has_dna:
        title += "  [protein+DNA co-fold]"
    
    title_color = "black"
    if is_failed:
        title_color = style.PTM_COLORS["failed"]

    fig.suptitle(title, fontweight="bold", fontsize=11, color=title_color)

    # Failed condition banner (§ 7b)
    if is_failed:
        banner_text = ("⚠ COLLAPSED PREDICTION — protein pLDDT < 50 AND macromolecule PAE > 25 Å.\n"
                      "Displacement values are not biologically interpretable.")
        fig.text(0.5, 0.95, banner_text, ha="center", va="top", fontsize=10,
                color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                         edgecolor="red", linewidth=2))

    # --- low-pLDDT amber overlay (plan 1.1d) ---
    _shade_low_plddt(ax_disp, x, cond_pl)

    # --- between-replicate IQR band (plan 1.1f) ---
    if has_iqr_band and not is_failed:  # Don't show IQR for failed conditions
        il = np.asarray(iqr_lo, dtype=float)
        ih = np.asarray(iqr_hi, dtype=float)
        ax_disp.fill_between(x, il, ih, color="#7B68A6", alpha=0.20, zorder=1,
                             label="between-replicate IQR")

    # --- baseline noise envelope ---
    if np.any(np.isfinite(rmsf)) and not is_failed:
        ax_disp.fill_between(x, 0, rmsf, color=style.C_NOISE, alpha=0.35,
                             label=f"{ref_lbl} noise (RMSF)", zorder=2)
        # Fix 5: dashed reference at the mean baseline RMSF for this condition.
        mean_rmsf = float(np.nanmean(rmsf))
        if np.isfinite(mean_rmsf):
            ax_disp.axhline(mean_rmsf, color="#444444", linestyle="--",
                            linewidth=0.8, zorder=2,
                            label=f"mean {ref_lbl} RMSF")
        
        # Cycle 4: Add RMSF threshold line for significance visual reference
        # Threshold is typically ~1.0-1.5 Å for meaningful displacement
        rmsf_threshold = 1.0  # Significance threshold in Å
        ax_disp.axhline(rmsf_threshold, color="#999999", linestyle=":", 
                       linewidth=0.7, zorder=1,
                       label=f"RMSF threshold ({rmsf_threshold} Å)")
        
        # Cycle 8: Add significance threshold line at mean baseline RMSF
        ax_disp.axhline(mean_rmsf, color="#666666", linestyle="-.", 
                       linewidth=0.8, zorder=1,
                       label=f"significance threshold ({mean_rmsf:.2f} Å)")

    # --- 95% CI ribbon: dashed outline for small N or grey for failed (§ 7b) ---
    if np.any(hi > lo):
        ci_label = f"95% CI (n={n_c}{'  ⚠ small N' if small_n else ''})"
        ci_color = style.PTM_COLORS["failed"] if is_failed else style.C_DISP
        ci_alpha = 0.15 if is_failed else 0.25
        
        if small_n and not is_failed:
            ax_disp.plot(x, lo, color=ci_color, linewidth=0.8, linestyle="--", zorder=3)
            ax_disp.plot(x, hi, color=ci_color, linewidth=0.8, linestyle="--", zorder=3,
                         label=ci_label)
        else:
            ax_disp.fill_between(x, lo, hi, color=ci_color, alpha=ci_alpha, zorder=3,
                                 label=ci_label)

    # Main displacement line: dashed for failed conditions (§ 7b)
    line_style = "--" if is_failed else "-"
    line_color = style.PTM_COLORS["failed"] if is_failed else style.C_DISP
    ax_disp.plot(x, disp, color=line_color, linewidth=1.0, linestyle=line_style, zorder=4,
                 label="mean displacement")

    # Significance markers: confidence-gated (FDR<0.05 AND pLDDT>=70 both states)
    if sig.any() and not is_failed:
        ax_disp.scatter(x[sig], disp[sig], s=12, color=style.C_SIG, zorder=5,
                        label="significant motion (FDR<0.05, pLDDT≥70)")

    ax_disp.set_ylabel("Cα displacement (Å)")
    
    # Standardized y-axis ceiling (§ 7a)
    if y_ceiling is not None and np.isfinite(y_ceiling):
        ceiling = y_ceiling
    else:
        # Use profile-specific ceiling if provided, otherwise adaptive
        ceiling = profile.get("y_ceiling")
        if not (ceiling and np.isfinite(ceiling)):
            ceiling = _adaptive_ceiling(hi)
    
    ax_disp.set_ylim(0, ceiling)
    ax_disp.legend(loc="upper right", ncol=2, fontsize=7)
    plt.setp(ax_disp.get_xticklabels(), visible=False)

    # --- significance rug on the x-axis: remove for failed conditions (§ 7b) ---
    if sig.any() and not is_failed:
        ax_disp.scatter(x[sig], np.full(int(sig.sum()), 0.0),
                        marker="|", s=80, color=style.C_SIG, zorder=6,
                        clip_on=False)

    # baseline composition annotation (plan 0.1)
    if baseline_warning:
        ax_disp.text(0.01, 0.97, baseline_warning, transform=ax_disp.transAxes,
                     fontsize=6.5, va="top", ha="left", style="italic",
                     color="#555555",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#FFF6E5",
                               ec="#E69F00", alpha=0.9))

    # --- pLDDT panel: grey for failed conditions (§ 7b) ---
    ref_color = style.PTM_COLORS["failed"] if is_failed else style.C_REF
    cond_color = style.PTM_COLORS["failed"] if is_failed else style.C_COND
    
    ax_pl.plot(x, ref_pl, color=ref_color, linewidth=0.9, label=f"{ref_lbl} pLDDT")
    ax_pl.plot(x, cond_pl, color=cond_color, linewidth=0.9, label="condition pLDDT")
    ax_pl.axhline(70, color="gray", linestyle=":", linewidth=0.7)
    ax_pl.set_ylim(0, 100)
    ax_pl.set_ylabel("pLDDT")
    ax_pl.set_xlabel("Residue number")
    ax_pl.legend(loc="lower right", ncol=2, fontsize=7)

    # --- chain boundaries ---
    if len(chain_ids) == len(x):
        for i in range(1, len(chain_ids)):
            if chain_ids[i] != chain_ids[i - 1]:
                xb = (x[i] + x[i - 1]) / 2
                for ax in (ax_disp, ax_pl):
                    ax.axvline(xb, color="gray", linestyle=":", linewidth=0.7)

    # --- PTM site markers: always show for reference (§ 7a) ---
    res_set = {int(v): k for k, v in enumerate(x) if np.isfinite(v)}
    ptm_positions = _ptm_positions(profile.get("ptm_labels", []))
    
    # Always add PTM site marker as grey dashed line for spatial reference (§ 7a)
    for pos in ptm_positions:  # only the condition's actual PTM sites (no default)
        if pos in res_set:
            marker_color = style.PTM_COLORS["failed"] if is_failed else style.C_PTM
            for ax in (ax_disp, ax_pl):
                ax.axvline(pos, color=marker_color, linestyle="--", linewidth=1.3, zorder=5)
            ax_disp.text(pos, ceiling * 0.97, f"PTM {pos}", color=marker_color,
                         fontsize=8, ha="center", va="top", rotation=90)
            
            # ΔpLDDT label: skip for failed conditions
            if not is_failed:
                idx = res_set[pos]
                d_pl = cond_pl[idx] - ref_pl[idx]
                if np.isfinite(d_pl) and abs(d_pl) > 10:
                    ax_pl.annotate(f"ΔpLDDT = {d_pl:+.0f} at PTM",
                                   (pos, max(cond_pl[idx], ref_pl[idx])),
                                   xytext=(0, 12), textcoords="offset points",
                                   fontsize=7, color=style.C_FAIL, ha="center",
                                   fontweight="bold")

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]
    return style.save(fig, plots_dir, f"per_residue_{safe}")


def _shade_low_plddt(ax, x: np.ndarray, cond_pl: np.ndarray, cutoff: float = 70.0) -> None:
    """Amber-shade contiguous regions where condition pLDDT < cutoff (plan 1.1d)."""
    low = np.isfinite(cond_pl) & (cond_pl < cutoff)
    if not low.any():
        return
    labelled = False
    start = None
    for i in range(len(x)):
        if low[i] and start is None:
            start = i
        is_end = (not low[i]) or (i == len(x) - 1)
        if start is not None and is_end:
            end = i if not low[i] else i
            x0 = x[start] - 0.5
            x1 = x[end] + (0.5 if low[i] else -0.5)
            ax.axvspan(x0, x1, color="#E69F00", alpha=0.13, zorder=0,
                       label=None if labelled else "pLDDT < 70")
            labelled = True
            start = None
