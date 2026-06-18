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
    label_map: dict = None,
    ion_tier: dict = None,
    artifact_names: "List[str] | None" = None,
    separation: "dict | None" = None,
) -> List[Path]:
    """
    Clustered RMSD heatmap with a top dendrogram and a cluster-colour strip.

    names    original (sorted) order matching ``matrix`` rows/cols
    matrix   N×N RMSD
    cl       result of cluster.cluster_conditions
    label_map / ion_tier  used to render short axis labels (Fix 6)
    artifact_names        conditions excluded as artifacts, noted in caption (Fix 2)
    """
    n = len(names)
    if n < 2:
        return []

    order = cl["order"]
    ordered_names = cl["ordered_names"]
    labels_map = cl["labels"]
    Z = cl["linkage"]
    reordered = matrix[np.ix_(order, order)]

    # Fix 6: short-form, factor-string labels for the axes and dendrogram leaves.
    disp_labels = style.short_condition_labels(ordered_names, ion_tier)

    # Fix 1: scale tick density / font / rotation by condition count.
    many = n > 20
    tick_fs = 6 if many else 8
    x_rotation = 90 if many else 45

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

    # --- dendrogram (Fix 6: short leaf labels) ---
    if HAS_SCIPY and Z is not None:
        dendrogram(
            Z, ax=ax_dend, no_labels=True, color_threshold=0,
            link_color_func=lambda _: "#555555",
        )
        if cut_height is not None and np.isfinite(cut_height):
            ax_dend.axhline(cut_height, color=style.C_DISP, linestyle="--",
                            linewidth=1.0)
            ax_dend.text(0.99, cut_height, f" cut {cut_height:.1f} Å",
                         color=style.C_DISP, fontsize=style.FS_TICK_LABEL,
                         va="bottom", ha="right",
                         transform=ax_dend.get_yaxis_transform())
        # Within-condition sampling-noise floor: clusters split below this line
        # are not separable from AF3 ensemble noise (see cycle-5 diagnostic).
        if separation and separation.get("within_condition_noise_floor_A") is not None:
            nf = float(separation["within_condition_noise_floor_A"])
            if np.isfinite(nf):
                ax_dend.axhline(nf, color="#B2182B", linestyle=":", linewidth=1.4)
                ax_dend.text(0.99, nf, f" within-condition noise floor {nf:.1f} Å",
                             color="#B2182B", fontsize=style.FS_TICK_LABEL,
                             va="bottom", ha="right",
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
    ax_strip.set_yticklabels(["cluster"], fontsize=style.FS_TICK_LABEL)
    ax_strip.grid(False)
    for s in ax_strip.spines.values():
        s.set_visible(False)

    # --- heatmap ---
    masked = np.ma.masked_invalid(reordered)
    im = ax_main.imshow(masked, cmap="YlOrRd", vmin=0, aspect="auto")
    cbar = fig.colorbar(im, cax=ax_cb, label="Protein Cα RMSD (Å)")
    cbar.set_label("Protein Cα RMSD (Å)", fontsize=style.FS_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=style.FS_TICK_LABEL)
    ax_main.set_xticks(range(n))
    ax_main.set_yticks(range(n))
    ax_main.set_xticklabels(disp_labels, rotation=x_rotation, ha="right",
                            fontsize=tick_fs)
    ax_main.set_yticklabels(disp_labels, fontsize=tick_fs)
    ax_main.grid(False)

    # Annotate cell values only when the grid is small enough to stay legible.
    vmax = float(np.nanmax(reordered)) if np.any(np.isfinite(reordered)) else 1.0
    if not many:
        for i in range(n):
            for j in range(n):
                v = reordered[i, j]
                if math.isfinite(v):
                    ax_main.text(j, i, f"{v:.1f}", ha="center", va="center",
                                 fontsize=6,
                                 color="white" if v > vmax * 0.6 else "black")

    # cluster legend
    cids = sorted(set(labels_map.values()))
    handles = [Patch(facecolor=cluster_color(c), label=f"cluster {c}") for c in cids]
    ax_main.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.05, 1.0),
                   fontsize=style.FS_TICK_LABEL, title="clusters", title_fontsize=8)

    # Cluster separability vs sampling noise (cycle-5 diagnostic): if the
    # between-condition distances are within the within-condition noise floor,
    # the clusters are exploratory and must not be read as distinct states.
    if separation and separation.get("separation_adequate") is False:
        bm = separation.get("between_condition_median_rmsd_A")
        nf = separation.get("within_condition_noise_floor_A")
        fig.text(0.02, 0.045,
                 f"EXPLORATORY: between-condition RMSD (median {bm} Å) is within the "
                 f"within-condition sampling noise floor ({nf} Å). Clusters are NOT robustly "
                 "separable from AF3 ensemble noise — do not interpret as distinct states.",
                 fontsize=style.FS_ANNOTATION, fontweight="bold", color="#B2182B")

    # Fix 2: note that artifact conditions are excluded from this heatmap so
    # they cannot distort dendrogram topology without annotation.
    if artifact_names:
        short_art = ", ".join(style.short_condition_labels(sorted(artifact_names), ion_tier))
        fig.text(0.02, 0.01,
                 f"Artifact conditions excluded from clustering: {short_art}.",
                 fontsize=style.FS_ANNOTATION, style="italic", color="#666666")

    # Fix 6: retain the full condition names in the saved image metadata so the
    # mapping from short label to full name is recoverable from the file.
    metadata = {"Description": "; ".join(
        f"{s}={f}" for s, f in zip(disp_labels, ordered_names))}

    return style.save(fig, plots_dir, "structural_clustering", metadata=metadata)
