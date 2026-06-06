"""
Factorial-design figures: PTM group × salt tier (af3bench2 overhaul).

  panel_per_residue       grid of displacement profiles per (PTM group, tier).
  concentration_response  mean displacement vs salt tier; PTM panel + DNA panel
                          split (plan 1.3c), ligand-multiplier superscripts
                          (1.3b), rank-swap callout (1.3a).
  ptm_effect_grid         heatmap distinguishing not-measured / n.s. / measured
                          (1.2a), IQR + heterogeneity warning per cell (1.2b),
                          noise-floor boundary marker (1.2c), DNA separator
                          (1.2d).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import style


def _tier_key(t: str) -> float:
    if t == "0x":
        return 0.0
    try:
        return float(t.rstrip("x"))
    except ValueError:
        return 9_999.0


def plot_panel_per_residue(
    cells: Dict[tuple, dict],
    ptm_order: List[str],
    tier_order: List[str],
    baseline_name: str,
    y_ceiling: float,
    plots_dir: Path,
) -> List[Path]:
    n_rows = len(ptm_order)
    n_cols = len(tier_order)
    if n_rows == 0 or n_cols == 0:
        return []

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.suptitle(
        "Per-residue displacement vs baseline\n"
        "rows = PTM group   |   columns = salt-ion tier (Na+Cl; water excluded)",
        fontweight="bold", fontsize=11,
    )

    for ri, ptm in enumerate(ptm_order):
        for ci, tier in enumerate(tier_order):
            ax = axes[ri][ci]
            if ri == 0:
                ax.set_title(tier, fontsize=9)
            if ci == 0:
                ax.set_ylabel(ptm if ptm != "none" else "unmodified", fontsize=8)

            cell = cells.get((ptm, tier))
            if cell is None:
                ax.set_facecolor("#f4f4f4")
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.grid(False)
                continue

            x = np.asarray(cell["res_numbers"], dtype=float)
            disp = np.asarray(cell["disp_mean"], dtype=float)
            rmsf = np.asarray(cell.get("baseline_rmsf", np.full_like(disp, np.nan)), dtype=float)

            if np.any(np.isfinite(rmsf)):
                ax.fill_between(x, 0, rmsf, color=style.C_NOISE, alpha=0.3, zorder=1)
            ax.plot(x, disp, color=style.C_DISP, linewidth=0.7, zorder=2)
            ax.set_ylim(0, y_ceiling)

            mean_d = cell.get("mean_disp", float(np.nanmean(disp)))
            txt = f"μ={mean_d:.1f}Å"
            if cell.get("tier") == "likely_artifact":
                txt += "  ✗"
            ax.text(0.97, 0.95, txt, ha="right", va="top", transform=ax.transAxes,
                    fontsize=7, color=style.C_DISP)

            for lab in cell.get("ptm_labels", []):
                digits = "".join(c for c in lab if c.isdigit())
                if digits and int(digits) in set(int(v) for v in x if np.isfinite(v)):
                    ax.axvline(int(digits), color=style.C_PTM, linestyle="--", linewidth=1.0)

    for ax in axes[-1]:
        ax.set_xlabel("Residue number", fontsize=8)

    fig.tight_layout()
    return style.save(fig, plots_dir, "panel_per_residue")


def plot_concentration_response(
    data: Dict[str, Dict[str, dict]],
    ptm_order: List[str],
    baseline_name: str,
    plots_dir: Path,
    dna_groups: Optional[set] = None,
    ligand_mult: Optional[Dict[str, Dict[str, int]]] = None,
    confound_note: Optional[str] = None,
) -> List[Path]:
    """
    Concentration-response with failed point marking and clear CI labels (§ 6).
    
    data: {ptm_group: {tier: {"mean","lo","hi","n"}}}
    dna_groups: PTM-group names that are DNA conditions (drawn in a right panel).
    ligand_mult: {ptm_group: {tier: multiplier}} for per-point superscripts.
    """
    if not data:
        return []
    dna_groups = dna_groups or set()
    ptm_groups = [g for g in ptm_order if g in data and g not in dna_groups]
    dna_present = [g for g in data if g in dna_groups]

    n_panels = 2 if dna_present else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.5 if n_panels == 1 else 11, 4.8),
        gridspec_kw={"width_ratios": [3, 1.4]} if n_panels == 2 else None,
    )
    if n_panels == 1:
        axes = [axes]

    def _draw(ax, groups, title):
        for ptm in groups:
            tiers = sorted(data[ptm].keys(), key=_tier_key)
            xs = list(range(len(tiers)))
            means = [data[ptm][t]["mean"] for t in tiers]
            lo = [max(0.0, data[ptm][t]["mean"] - data[ptm][t]["lo"]) for t in tiers]
            hi = [max(0.0, data[ptm][t]["hi"] - data[ptm][t]["mean"]) for t in tiers]
            
            # Use updated color scheme (§ 9)
            ptm_display = ptm if ptm != "none" else "unmodified"
            color = style.PTM_COLORS.get(ptm_display, "#7F7F7F")
            
            # Identify failed points (§ 6)
            failed_mask = [style.is_failed_condition(f"{ptm}_nax{t.rstrip('x')}_") for t in tiers]
            
            # Plot valid points as solid markers
            valid_xs = [x for x, failed in zip(xs, failed_mask) if not failed]
            valid_means = [m for m, failed in zip(means, failed_mask) if not failed]
            valid_lo = [l for l, failed in zip(lo, failed_mask) if not failed]
            valid_hi = [h for h, failed in zip(hi, failed_mask) if not failed]
            
            if valid_xs:
                ax.errorbar(valid_xs, valid_means, yerr=[valid_lo, valid_hi], 
                           marker="o", linewidth=1.8, capsize=4, color=color, 
                           label=ptm_display, markersize=6, linestyle="-")
            
            # Plot failed points as hollow markers with dashed connection (§ 6)
            failed_xs = [x for x, failed in zip(xs, failed_mask) if failed]
            failed_means = [m for m, failed in zip(means, failed_mask) if failed]
            
            if failed_xs:
                ax.scatter(failed_xs, failed_means, marker="o", s=40, 
                          facecolors="none", edgecolors=style.PTM_COLORS["failed"],
                          linewidths=2, zorder=5)
                # Dashed line to failed points if there are preceding valid points
                if valid_xs and failed_xs:
                    last_valid_x = max(valid_xs)
                    last_valid_mean = means[last_valid_x]
                    first_failed_x = min(failed_xs)
                    first_failed_mean = failed_means[0]
                    ax.plot([last_valid_x, first_failed_x], [last_valid_mean, first_failed_mean],
                           color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            
            # ligand-multiplier superscripts (plan 1.3b)
            if ligand_mult and ptm in ligand_mult:
                for xi, t in zip(xs, tiers):
                    mult = ligand_mult[ptm].get(t)
                    if mult:
                        ax.annotate(f"{mult}×", (xi, means[xs.index(xi)]),
                                    xytext=(0, 8), textcoords="offset points",
                                    fontsize=6, color=color, ha="center")
            ax.set_xticks(xs)
            ax.set_xticklabels(tiers)
        ax.set_xlabel("Salt-ion tier (Na+Cl; ligand co-scales at 5:1)")
        ax.set_title(title, fontsize=10)
        
        # Add confound note as annotation box (§ 6)
        if confound_note:
            ax.annotate(
                f"⚠ {confound_note}",
                xy=(0.01, 0.98), xycoords="axes fraction",
                va="top", ha="left", fontsize=7, color="#B05800",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E69F00", alpha=0.9),
            )
        
        # Add noise floor band (§ 6)
        ax.axhspan(0, 2, color="lightgray", alpha=0.3, label="noise floor")

    _draw(axes[0], ptm_groups, "PTM conditions")
    axes[0].set_ylabel("Mean Cα displacement vs baseline (Å)")
    
    # Enhanced legend with failed point explanation (§ 6)
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker="o", linestyle="", markersize=6,
                         markerfacecolor="none", markeredgecolor=style.PTM_COLORS["failed"],
                         markeredgewidth=2, label="collapsed prediction"))
    axes[0].legend(handles=handles, title="PTM group", fontsize=7)

    # rank-swap detection between adjacent tiers (plan 1.3a)
    _annotate_rank_swaps(axes[0], data, ptm_groups)

    if dna_present:
        _draw(axes[1], dna_present, "DNA conditions")
        axes[1].legend(fontsize=7)

    # Clear error bar labeling (§ 6)
    sub = "Error bars: 95% CI across ensemble replicates per condition"
    fig.suptitle(f"Concentration–response  |  baseline: {baseline_name}\n{sub}",
                 fontweight="bold", fontsize=10)
    
    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Grey open circles = model collapse; excluded from biological interpretation", 
             fontsize=8, style="italic", color="#666666")
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "concentration_response")


def _annotate_rank_swaps(ax, data, groups) -> None:
    """Mark adjacent tiers where two PTM groups exchange displacement rank."""
    if len(groups) < 2:
        return
    all_tiers = sorted({t for g in groups for t in data[g]}, key=_tier_key)
    swap_count = 0
    for k in range(len(all_tiers) - 1):
        t0, t1 = all_tiers[k], all_tiers[k + 1]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                ga, gb = groups[a], groups[b]
                if not all(t in data[ga] and t in data[gb] for t in (t0, t1)):
                    continue
                d0 = data[ga][t0]["mean"] - data[gb][t0]["mean"]
                d1 = data[ga][t1]["mean"] - data[gb][t1]["mean"]
                if np.isfinite(d0) and np.isfinite(d1) and d0 * d1 < 0:
                    xc = k + 0.5
                    yc = np.mean([data[ga][t0]["mean"], data[gb][t0]["mean"],
                                  data[ga][t1]["mean"], data[gb][t1]["mean"]])
                    ax.scatter([xc], [yc], marker="P", s=90, color="black", zorder=6)
                    # Offset successive annotations vertically to avoid overlap
                    y_offset = 10 + swap_count * 14
                    ax.annotate(f"rank swap at {t1}", (xc, yc),
                                xytext=(6, y_offset), textcoords="offset points",
                                fontsize=7, fontweight="bold")
                    swap_count += 1


def plot_ptm_effect_grid(
    grid: np.ndarray,
    ns_mask: np.ndarray,
    ptm_order: List[str],
    tier_order: List[str],
    baseline_name: str,
    plots_dir: Path,
    measured_mask: Optional[np.ndarray] = None,
    iqr_grid: Optional[np.ndarray] = None,
    nclusters_grid: Optional[np.ndarray] = None,
    noise_grid: Optional[np.ndarray] = None,
    dna_rows: Optional[List[int]] = None,
    artifact_mask: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    PTM effect grid with rescaled color for valid cells only (§ 5) and failed condition treatment.
    """
    if grid.size == 0:
        return []
    n_rows, n_cols = grid.shape
    
    # Create artifact mask from failed conditions if not provided (§ 5)
    if artifact_mask is None:
        artifact_mask = np.zeros((n_rows, n_cols), dtype=bool)
        for ri, ptm in enumerate(ptm_order):
            for ci, tier in enumerate(tier_order):
                # Construct typical condition name pattern to check
                condition_patterns = [
                    f"oct4_seg_chain_b_{ptm.lower()}_{tier.lower().replace('x', 'x')}_",
                    f"oct4_seg_chain_b_nax{tier.rstrip('x')}_" if ptm == "unmodified" else f"oct4_seg_chain_b_{ptm.lower()}_nax{tier.rstrip('x')}_"
                ]
                for pattern in condition_patterns:
                    if any(style.is_failed_condition(fc) and pattern in fc for fc in style.FAILED_CONDITIONS):
                        artifact_mask[ri, ci] = True
                        break
    
    # Compute color scale using only valid cells (§ 5)
    valid_mask = np.isfinite(grid) & (~artifact_mask)
    if measured_mask is not None:
        valid_mask = valid_mask & measured_mask
    
    if np.any(valid_mask):
        valid_values = grid[valid_mask]
        vmin = 0  # Always start from 0
        vmax = float(np.nanmax(valid_values))
    else:
        vmin, vmax = 0, 1
    
    fig, ax = plt.subplots(figsize=(max(5, n_cols * 1.7 + 1), max(3.5, n_rows * 1.35 + 1)))
    
    # Create masked array for display
    display_grid = np.copy(grid)
    display_grid[artifact_mask] = np.nan  # Hide artifact values from colormap
    masked = np.ma.masked_invalid(display_grid)
    
    im = ax.imshow(masked, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean Cα displacement (Å)", shrink=0.85)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(tier_order)
    ax.set_yticklabels([p if p != "unmodified" else "unmodified" for p in ptm_order])
    ax.set_xlabel("Salt-ion tier (Na+Cl; ligand co-scales at 5:1)")
    ax.set_ylabel("PTM group")
    ax.set_title(
        f"PTM × salt+ligand tier effect grid\n"
        f"baseline: {baseline_name}\n"
        f"Color scale: valid predictions only (0–{vmax:.1f} Å)",
        fontweight="bold", fontsize=9,
    )
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

    # Cell annotations with improved failed condition treatment (§ 5)
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = grid[ri, ci]
            measured = (measured_mask is None) or bool(measured_mask[ri, ci])
            is_artifact = bool(artifact_mask[ri, ci])
            
            if not measured:
                # not-measured: leave white, no text (plan 1.2a)
                ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                       facecolor="white", edgecolor="#dddddd",
                                       zorder=2))
                continue
            
            if is_artifact:
                # Failed condition: grey with red cross, remove numerical value (§ 5)
                ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                       facecolor=style.PTM_COLORS["failed"], 
                                       edgecolor="black", zorder=2))
                # Draw red cross
                ax.plot([ci-0.3, ci+0.3], [ri-0.3, ri+0.3], 'r-', linewidth=3, zorder=4)
                ax.plot([ci-0.3, ci+0.3], [ri+0.3, ri-0.3], 'r-', linewidth=3, zorder=4)
                ax.text(ci, ri, "collapsed\nprediction", ha="center", va="center", 
                        fontsize=6, color="white", fontweight="bold", zorder=5)
                continue
            
            if not np.isfinite(v):
                # measured but no value → n.s. light-gray hatched
                ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                       facecolor="#ECECEC", hatch="//",
                                       edgecolor="#BBBBBB", zorder=2))
                ax.text(ci, ri, "n.s.", ha="center", va="center", fontsize=7,
                        color="#666666")
                continue

            # Valid measured value
            txt = f"{v:.1f}"
            ns = bool(ns_mask[ri, ci])
            if ns:
                txt += "\nn.s."
            if iqr_grid is not None and np.isfinite(iqr_grid[ri, ci]):
                txt += f"\n(IQR {iqr_grid[ri, ci]:.1f})"
            if nclusters_grid is not None and nclusters_grid[ri, ci] > 4:
                txt += " ⚠"
            
            # Bold border for key biological contrasts (§ 5)
            is_key_contrast = (
                (ri == ptm_order.index("TPO101") if "TPO101" in ptm_order else False and 
                 ci == tier_order.index("1x") if "1x" in tier_order else False) or
                (ri == ptm_order.index("SEP102") if "SEP102" in ptm_order else False and 
                 ci == tier_order.index("1x") if "1x" in tier_order else False)
            )
            
            if is_key_contrast:
                ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                       fill=False, edgecolor="black",
                                       linewidth=3, zorder=3))
            
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=7,
                    fontweight="bold",
                    color="white" if v > vmax * 0.6 else "black")

            # noise-floor boundary marker (plan 1.2c)
            if noise_grid is not None and np.isfinite(noise_grid[ri, ci]):
                if abs(v - noise_grid[ri, ci]) < 0.5:
                    ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           fill=False, edgecolor="black",
                                           linestyle="--", linewidth=1.6, zorder=3))

    # Add 0x column as n/a for PTM conditions (§ 5)
    if "0x" not in tier_order:
        # This would need to be handled at data preparation level
        pass

    # DNA separator (plan 1.2d)
    if dna_rows:
        top = min(dna_rows)
        ax.axhline(top - 0.5, color="black", linewidth=2.2, zorder=4)
        ax.annotate(
            "DNA conditions\n(qualitatively distinct perturbation)",
            xy=(-0.02, top - 0.5),
            xycoords=("axes fraction", "data"),
            ha="right", va="center", fontsize=7.5, fontweight="bold",
            annotation_clip=False,
            color=style.PTM_COLORS["DNA"],
        )

    # Add footnote for failed conditions (§ 0)
    fig.text(0.02, 0.02, "Grey × = model collapse; values not interpretable", 
             fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    return style.save(fig, plots_dir, "ptm_effect_grid")
