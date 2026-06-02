"""
Factorial-design figures: PTM group x ion tier.

  panel_per_residue       grid of displacement profiles, one cell per
                          (PTM group, ion tier), with the baseline noise band.
  concentration_response  mean displacement vs ion tier, one line per PTM group,
                          with 95% CI whiskers.
  ptm_effect_grid         heatmap of mean displacement; cells within noise are
                          marked "n.s.".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

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
    """
    cells: {(ptm_group, ion_tier): profile-like dict with res_numbers,
            disp_mean, baseline_rmsf, ptm_labels, mean_disp}
    """
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
        f"Per-residue displacement vs {baseline_name}\n"
        f"rows = PTM group   |   columns = salt-ion tier (Na+Cl; water excluded)",
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
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
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
            ax.text(0.97, 0.95, f"μ={mean_d:.1f}Å", ha="right", va="top",
                    transform=ax.transAxes, fontsize=7, color=style.C_DISP)

            # PTM marker
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
) -> List[Path]:
    """
    data: {ptm_group: {tier: {"mean": m, "lo": lo, "hi": hi}}}
    """
    if not data:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for ptm in ptm_order:
        if ptm not in data:
            continue
        tiers = sorted(data[ptm].keys(), key=_tier_key)
        xs = list(range(len(tiers)))
        means = [data[ptm][t]["mean"] for t in tiers]
        lo = [data[ptm][t]["mean"] - data[ptm][t]["lo"] for t in tiers]
        hi = [data[ptm][t]["hi"] - data[ptm][t]["mean"] for t in tiers]
        color = style.ptm_color(ptm)
        label = ptm if ptm != "none" else "unmodified"
        ax.errorbar(xs, means, yerr=[lo, hi], marker="o", linewidth=1.8,
                    capsize=4, color=color, label=label, markersize=6)
        ax.set_xticks(xs)
        ax.set_xticklabels(tiers)

    ax.set_xlabel("Salt-ion tier (Na+Cl; water held separate)")
    ax.set_ylabel("Mean Cα displacement vs baseline (Å)")
    ax.set_title(f"Concentration–response  |  baseline: {baseline_name}", fontweight="bold")
    ax.legend(title="PTM group")
    fig.tight_layout()
    return style.save(fig, plots_dir, "concentration_response")


def plot_ptm_effect_grid(
    grid: np.ndarray,            # (rows, cols) mean displacement
    ns_mask: np.ndarray,         # (rows, cols) True where within noise
    ptm_order: List[str],
    tier_order: List[str],
    baseline_name: str,
    plots_dir: Path,
) -> List[Path]:
    if grid.size == 0:
        return []
    fig, ax = plt.subplots(
        figsize=(max(4, len(tier_order) * 1.5 + 1), max(3, len(ptm_order) * 1.2 + 1))
    )
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean Cα displacement (Å)", shrink=0.85)

    ax.set_xticks(range(len(tier_order)))
    ax.set_yticks(range(len(ptm_order)))
    ax.set_xticklabels(tier_order)
    ax.set_yticklabels([p if p != "none" else "unmodified" for p in ptm_order])
    ax.set_xlabel("Salt-ion tier (Na+Cl)")
    ax.set_ylabel("PTM group")
    ax.set_title(f"PTM × concentration effect grid\nbaseline: {baseline_name}",
                 fontweight="bold")
    ax.grid(False)

    vmax = float(np.nanmax(grid)) if np.any(np.isfinite(grid)) else 1.0
    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            v = grid[ri, ci]
            if not np.isfinite(v):
                continue
            txt = f"{v:.1f}"
            if ns_mask[ri, ci]:
                txt += "\nn.s."
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=8,
                    fontweight="bold",
                    color="white" if v > vmax * 0.6 else "black")
    fig.tight_layout()
    return style.save(fig, plots_dir, "ptm_effect_grid")
