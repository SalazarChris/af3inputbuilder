"""Cluster figures: dendrogram + clustered RMSD heatmap with cluster strips."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from . import style

try:
    from scipy.cluster.hierarchy import dendrogram
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


# A categorical, colourblind-friendly cluster palette
_CLUSTER_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#999999",
    "#117733", "#882255",
]


def cluster_color(cid: int) -> str:
    return _CLUSTER_COLORS[(cid - 1) % len(_CLUSTER_COLORS)]


def plot_cluster_heatmap(
    names: List[str],
    matrix: np.ndarray,
    cl: dict,
    plots_dir: Path,
    cut_height: float = None,
) -> List[Path]:
    """
    Clustered RMSD heatmap with a top dendrogram and a cluster-colour strip.

    names    original (sorted) order matching ``matrix`` rows/cols
    matrix   N×N RMSD
    cl       result of cluster.cluster_conditions
    """
    n = len(names)
    if n < 2:
        return []

    order = cl["order"]
    ordered_names = cl["ordered_names"]
    labels_map = cl["labels"]
    Z = cl["linkage"]
    reordered = matrix[np.ix_(order, order)]
    disp_labels = style.condition_labels(ordered_names, max_len=20)

    fig = plt.figure(figsize=(max(7, n * 0.75 + 2), max(7, n * 0.75 + 2)))
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.1, 0.18, max(6, n * 0.6)],
        width_ratios=[max(6, n * 0.6), 0.35],
        hspace=0.03, wspace=0.04,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[2, 0])
    ax_cb = fig.add_subplot(gs[2, 1])

    # --- dendrogram ---
    if HAS_SCIPY and Z is not None:
        dendrogram(
            Z, ax=ax_dend, no_labels=True, color_threshold=0,
            link_color_func=lambda _: "#555555",
        )
        if cut_height is not None and np.isfinite(cut_height):
            ax_dend.axhline(cut_height, color=style.C_DISP, linestyle="--",
                            linewidth=1.0)
            ax_dend.text(0.99, cut_height, f" cut {cut_height:.1f} Å",
                         color=style.C_DISP, fontsize=7, va="bottom", ha="right",
                         transform=ax_dend.get_yaxis_transform())
        ax_dend.set_xticks([])
    ax_dend.set_yticks([])
    for s in ax_dend.spines.values():
        s.set_visible(False)
    ax_dend.set_title("Structural clustering of conditions (Cα RMSD)",
                      fontweight="bold")
    ax_dend.grid(False)

    # --- cluster colour strip ---
    for i, nm in enumerate(ordered_names):
        ax_strip.add_patch(plt.Rectangle((i, 0), 1, 1,
                                         color=cluster_color(labels_map[nm]),
                                         ec="white", lw=0.5))
    ax_strip.set_xlim(0, n)
    ax_strip.set_ylim(0, 1)
    ax_strip.set_xticks([])
    ax_strip.set_yticks([0.5])
    ax_strip.set_yticklabels(["cluster"], fontsize=7)
    ax_strip.grid(False)
    for s in ax_strip.spines.values():
        s.set_visible(False)

    # --- heatmap ---
    masked = np.ma.masked_invalid(reordered)
    im = ax_main.imshow(masked, cmap="YlOrRd", vmin=0, aspect="auto")
    fig.colorbar(im, cax=ax_cb, label="Protein Cα RMSD (Å)")
    ax_main.set_xticks(range(n))
    ax_main.set_yticks(range(n))
    ax_main.set_xticklabels(disp_labels, rotation=45, ha="right", fontsize=8)
    ax_main.set_yticklabels(disp_labels, fontsize=8)
    ax_main.grid(False)

    vmax = float(np.nanmax(reordered)) if np.any(np.isfinite(reordered)) else 1.0
    for i in range(n):
        for j in range(n):
            v = reordered[i, j]
            if math.isfinite(v):
                ax_main.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                             color="white" if v > vmax * 0.6 else "black")

    # cluster legend
    cids = sorted(set(labels_map.values()))
    handles = [Patch(facecolor=cluster_color(c), label=f"cluster {c}") for c in cids]
    ax_main.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                   fontsize=7, title="clusters", title_fontsize=8)

    return style.save(fig, plots_dir, "structural_clustering")
