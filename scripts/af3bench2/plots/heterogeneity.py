"""
Heterogeneity visualisations (af3bench2 additions).

  plot_heterogeneity_overview   one row per condition, dominant vs minority
                                cluster fraction, n_clusters label, an inset
                                RMSD sparkline, tier dividers (plan 3.1).
  plot_condition_cluster_portrait  two-panel focus for high-heterogeneity
                                conditions: confidence scatter by cluster +
                                per-cluster displacement profiles (plan 3.2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


_TIER_ORDER = {"high": 0, "moderate": 1, "low": 2}


def plot_heterogeneity_overview(
    variance_df: pd.DataFrame,
    plots_dir: Path,
    ptm_group: Optional[Dict[str, str]] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    if variance_df is None or variance_df.empty:
        return []
    df = variance_df.copy()
    df["_tier_rank"] = df["heterogeneity_tier"].map(_TIER_ORDER).fillna(1)
    df = df.sort_values(["_tier_rank", "cluster_entropy_bits"],
                        ascending=[True, False]).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.6 * n + 1.5)))
    ptm_group = ptm_group or {}

    y_positions = []
    for i, row in df.iterrows():
        y = n - i
        y_positions.append(y)
        cond = row["condition"]
        grp = ptm_group.get(cond, "none")
        color = style.ptm_color(grp)
        dom = float(row["dominant_cluster_fraction"])
        # dominant (solid) + minority (hatched) bar, total width 1.0
        ax.barh(y, dom, color=color, edgecolor="black", height=0.6, zorder=3)
        if dom < 1.0:
            ax.barh(y, 1.0 - dom, left=dom, color=color, edgecolor="black",
                    height=0.6, hatch="////", alpha=0.55, zorder=3)
        # n_clusters label
        ax.text(1.02, y, f"{int(row['n_structural_clusters'])} cl",
                va="center", ha="left", fontsize=7)
        # RMSD sparkline inset to the right
        _sparkline(ax, row, y)

    # tier dividers
    labels = style.short_labels(df["condition"].tolist(), label_map)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Cluster fraction (solid = dominant, hatched = minority)")
    ax.set_title("Within-condition structural heterogeneity\n"
                 "(ordered by cluster entropy; most heterogeneous at top)",
                 fontweight="bold")

    prev_tier = None
    for i, row in df.iterrows():
        tier = row["heterogeneity_tier"]
        if tier != prev_tier:
            y = n - i + 0.5
            ax.axhline(y, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            ax.text(-0.01, y - 0.3, tier, ha="right", va="top", fontsize=7,
                    color=style.HETERO_COLOR.get(tier, "black"),
                    transform=ax.get_yaxis_transform(), fontweight="bold")
            prev_tier = tier

    fig.tight_layout()
    return style.save(fig, plots_dir, "heterogeneity_overview")


def _sparkline(ax, row, y) -> None:
    """5-bin RMSD histogram drawn in axis-fraction space to the right of a bar."""
    rmsd_max = float(row.get("rmsd_max_angstrom", 0) or 0)
    if rmsd_max <= 0:
        return
    # crude distribution proxy from median/iqr/max if raw values absent
    med = float(row.get("rmsd_median_angstrom", 0) or 0)
    # We just draw a marker for median position on a 0..max mini-axis.
    x0, width = 1.16, 0.16
    ax.plot([x0, x0 + width], [y, y], color="#bbbbbb", linewidth=0.8,
            transform=ax.get_yaxis_transform(), clip_on=False)
    frac = med / rmsd_max if rmsd_max else 0
    ax.scatter([x0 + width * frac], [y], s=12, color=style.C_DISP,
               transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
    ax.text(x0 + width + 0.005, y, f"{rmsd_max:.0f}Å", va="center", ha="left",
            fontsize=6, color="#999999", transform=ax.get_yaxis_transform())


def plot_condition_cluster_portrait(
    condition: str,
    label_short: str,
    summary,
    ensemble,
    per_cluster_disp: Optional[Dict[int, np.ndarray]],
    res_numbers: Optional[np.ndarray],
    baseline_rmsf: Optional[np.ndarray],
    plots_dir: Path,
) -> List[Path]:
    """
    Two-panel portrait for a high-heterogeneity condition (plan 3.2).

    Left: per-replicate confidence scatter (mean pLDDT vs pTM) coloured by
          structural cluster.  Right: per-cluster mean displacement profile.
    """
    labels = np.asarray(summary.cluster_assignments, dtype=int)
    if labels.size == 0:
        return []

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"Cluster portrait: {label_short}", fontweight="bold")

    ptm = np.asarray(getattr(ensemble, "ptm", np.empty(0)), dtype=float)
    plddt = np.asarray(getattr(ensemble, "plddt_mean", np.empty(0)), dtype=float)
    uniq, counts = np.unique(labels, return_counts=True)
    dominant = uniq[np.argmax(counts)]
    palette = style.PALETTE

    # --- left: confidence scatter by cluster ---
    if ptm.size == labels.size and plddt.size == labels.size:
        for k, cid in enumerate(uniq):
            mask = labels == cid
            face = palette[k % len(palette)]
            filled = (cid == dominant)
            ax_l.scatter(plddt[mask], ptm[mask], s=55,
                         facecolor=(face if filled else "none"),
                         edgecolor=face, linewidth=1.5,
                         label=f"Cluster {cid} (n={int(mask.sum())})", zorder=3)
        ax_l.axvline(float(np.nanmean(plddt)), color="gray", linestyle=":", linewidth=0.8)
        ax_l.axhline(float(np.nanmean(ptm)), color="gray", linestyle=":", linewidth=0.8)
        ax_l.set_xlabel("Mean pLDDT (per replicate)")
        ax_l.set_ylabel("pTM (per replicate)")
        ax_l.set_title("Confidence by structural cluster", fontsize=10)
        ax_l.legend(fontsize=7)
    else:
        ax_l.text(0.5, 0.5, "per-replicate scores unavailable",
                  ha="center", va="center", transform=ax_l.transAxes)

    # --- right: per-cluster displacement profile ---
    if per_cluster_disp and res_numbers is not None:
        x = np.asarray(res_numbers, dtype=float)
        if baseline_rmsf is not None and np.any(np.isfinite(baseline_rmsf)):
            ax_r.fill_between(x, 0, baseline_rmsf, color=style.C_NOISE, alpha=0.3,
                              zorder=1, label="baseline noise")
        for k, cid in enumerate(uniq):
            prof = per_cluster_disp.get(int(cid))
            if prof is None:
                continue
            ls = "-" if cid == dominant else "--"
            ax_r.plot(x, prof, linestyle=ls, linewidth=1.2,
                      color=palette[k % len(palette)],
                      label=f"Cluster {cid}" + (" (dominant)" if cid == dominant else ""))
        ax_r.set_xlabel("Residue number")
        ax_r.set_ylabel("Cα displacement (Å)")
        ax_r.set_title("Per-cluster displacement profile", fontsize=10)
        ax_r.legend(fontsize=7)
    else:
        ax_r.text(0.5, 0.5, "per-cluster profile unavailable",
                  ha="center", va="center", transform=ax_r.transAxes)

    fig.tight_layout()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in condition)[:120]
    return style.save(fig, plots_dir, f"cluster_portrait_{safe}")
