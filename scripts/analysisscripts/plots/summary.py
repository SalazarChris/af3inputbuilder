"""
Summary visualizations for large-scale structural analysis (new module).

This module provides redesigned visualizations for the complex (>80 conditions)
structural analysis data, focusing on:
1. PTM × Concentration effect grids with improved readability
2. Hierarchically clustered structural distance heatmaps
3. Concentration response curves with error bars
4. Quality score dashboards
5. Clustering overviews
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# sklearn is optional for dimensionality reduction plots
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from . import style


def plot_ptm_concentration_effect_grid(
    df: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    ion_tier: Optional[Dict[str, str]] = None,
    baseline_name: Optional[str] = None,
) -> List[Path]:
    """
    Redesigned PTM × Concentration effect grid with improved readability.
    
    Improvements:
    - Perceptually uniform diverging colormap (coolwarm) centered at baseline
    - Numeric values instead of "n.s." text
    - Asterisk significance indicators (p<0.05, p<0.01, p<0.001)
    - Separator lines between PTM groups
    - Larger font sizes, bold for significant values
    - Clear title and labeling
    
    Parameters:
        df: DataFrame with columns: condition, ptm_group, ion_tier, 
            mean_disp, p_value, significant
        plots_dir: Output directory for figures
        label_map: Optional mapping from condition names to short labels
        ion_tier: Optional mapping for ion/salt tier labels
        baseline_name: Baseline condition name for centering colormap
        
    Returns:
        List of generated file paths
    """
    if not {"ptm_group", "ion_tier", "mean_disp", "p_value", "significant"}.issubset(df.columns):
        return []
    
    # Create pivot table for heatmap
    pivot = df.pivot_table(
        index="ptm_group", 
        columns="ion_tier", 
        values="mean_disp",
        aggfunc="mean"
    )
    
    # Create significance mask
    sig_mask = df.pivot_table(
        index="ptm_group", 
        columns="ion_tier", 
        values="significant",
        aggfunc="any"
    ).fillna(False)
    
    # Create p-value matrix for asterisks
    pval_matrix = df.pivot_table(
        index="ptm_group", 
        columns="ion_tier", 
        values="p_value",
        aggfunc="min"
    )
    
    # Determine PTM order
    ptm_order = ["unmodified", "SEP102", "TPO101", "TPO101+SEP102", "DNA"]
    ptm_order = [g for g in ptm_order if g in pivot.index]
    
    # Determine ion tier order
    ion_tiers = sorted(pivot.columns.tolist())
    
    # Reorder pivot table
    pivot = pivot.reindex(ptm_order, columns=ion_tiers)
    sig_mask = sig_mask.reindex(ptm_order, columns=ion_tiers)
    pval_matrix = pval_matrix.reindex(ptm_order, columns=ion_tiers)
    
    # Create figure with improved layout
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use coolwarm diverging colormap centered at 0
    # First, compute the value range for proper centering
    max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
    vmax = max_abs if max_abs > 0 else 5.0
    vmin = -vmax
    
    # Plot heatmap with coolwarm colormap
    im = ax.imshow(
        pivot.values,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower"
    )
    
    # Add colorbar with proper label
    cbar = fig.colorbar(im, ax=ax, label="Mean Cα displacement (Å)", shrink=0.8)
    cbar.set_label("Mean Cα displacement (Å)", fontsize=style.FS_AXIS_LABEL)
    
    # Set up axes
    ax.set_xticks(range(len(ion_tiers)))
    ax.set_yticks(range(len(ptm_order)))
    ax.set_xticklabels(ion_tiers, fontsize=10)
    ax.set_yticklabels(ptm_order, fontsize=10)
    
    ax.set_xlabel("Salt concentration (NaCl ×)", fontsize=12, fontweight="bold")
    ax.set_ylabel("PTM state", fontsize=12, fontweight="bold")
    
    # Set title with proper formatting
    ax.set_title(
        "PTM-Dependent Structural Response to Salt Concentration",
        fontweight="bold",
        fontsize=14,
        pad=20
    )
    
    # Add separator lines between PTM groups
    # Find where PTM state changes (unmodified, SEP102, TPO101, DNA)
    ptm_changes = []
    prev_grp = None
    for i, grp in enumerate(ptm_order):
        if prev_grp is not None and not _same_ptm_category(prev_grp, grp):
            ptm_changes.append(i)
        prev_grp = grp
    
    for y_pos in ptm_changes:
        ax.axhline(y_pos - 0.5, color="black", linewidth=1.5, linestyle="-")
    
    # Add significance markers as asterisks
    for i, ptm in enumerate(ptm_order):
        for j, tier in enumerate(ion_tiers):
            if sig_mask.loc[ptm, tier]:
                pval = pval_matrix.loc[ptm, tier]
                if pval < 0.001:
                    marker = "***"
                    fontsize = 11
                    fontweight = "bold"
                elif pval < 0.01:
                    marker = "**"
                    fontsize = 10
                    fontweight = "bold"
                elif pval < 0.05:
                    marker = "*"
                    fontsize = 9
                    fontweight = "normal"
                else:
                    continue
                
                ax.text(
                    j, i, marker,
                    ha="center", va="center",
                    fontsize=fontsize, fontweight=fontweight,
                    color="black"
                )
    
    # Add numeric values in cells (smaller font)
    for i, ptm in enumerate(ptm_order):
        for j, tier in enumerate(ion_tiers):
            val = pivot.loc[ptm, tier]
            if not np.isnan(val):
                # Use smaller font for non-significant values
                color = "white" if abs(val) > vmax * 0.6 else "black"
                fontsize = 8
                
                # Add asterisk to significant values in text
                if sig_mask.loc[ptm, tier]:
                    pval = pval_matrix.loc[ptm, tier]
                    if pval < 0.05:
                        val_str = f"{val:.1f}*" if math.isfinite(val) else "n.s."
                        fontsize = 9
                    else:
                        val_str = f"{val:.1f}" if math.isfinite(val) else "n.s."
                else:
                    val_str = f"{val:.1f}" if math.isfinite(val) else "n.s."
                
                ax.text(j, i, val_str, ha="center", va="center",
                       fontsize=fontsize, color=color, fontweight="normal")
    
    # Add baseline reference line (horizontal at y=0 in data space)
    if baseline_name:
        # Find baseline's PTM group and add horizontal line
        baseline_grp = df[df["condition"] == baseline_name]["ptm_group"].iloc[0] if baseline_name in df["condition"].values else "unmodified"
        if baseline_grp in ptm_order:
            baseline_idx = ptm_order.index(baseline_grp)
            ax.axhline(baseline_idx - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "ptm_concentration_grid")


def _same_ptm_category(g1: str, g2: str) -> bool:
    """Check if two PTM groups belong to the same category (unmod, SEP, TPO, DNA)."""
    if g1 == g2:
        return True
    if g1 in ("unmodified", "none", "") and g2 in ("unmodified", "none", ""):
        return True
    if g1.startswith("SEP") and g2.startswith("SEP"):
        return True
    if g1.startswith("TPO") and g2.startswith("TPO"):
        return True
    return False


def plot_structural_distance_heatmap(
    rmsd_matrix: np.ndarray,
    condition_names: List[str],
    ptm_groups: Dict[str, str],
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    perturbation_annotations: Optional[Dict[str, List[str]]] = None,
) -> List[Path]:
    """
    Hierarchically clustered structural distance heatmap.
    
    Features:
    - Dendrograms on both axes showing clustering relationships
    - Color-coded by PTM state and perturbation annotations
    - Clear annotation of cluster membership
    - Scalable to 80+ conditions
    - Generic: works with any perturbation factors (DNA, protein, small molecule, etc.)
    
    Parameters:
        rmsd_matrix: N×N RMSD matrix
        condition_names: List of condition names
        ptm_groups: Mapping from condition to PTM group
        plots_dir: Output directory
        label_map: Optional label mapping
        perturbation_annotations: Optional mapping from condition to list of
            perturbation factors for annotation (e.g., ["DNA"], ["SEP102", "TPO101"])
        
    Returns:
        List of generated file paths
    """
    n = len(condition_names)
    if n < 2:
        return []
    
    # Default: no additional annotations beyond PTM
    if perturbation_annotations is None:
        perturbation_annotations = {n: [] for n in condition_names}
    
    # Perform hierarchical clustering
    condensed = squareform(rmsd_matrix, checks=False)
    Z = linkage(condensed, method="average")
    
    # Get cluster assignments at data-driven threshold
    # Use median RMSD as cut height
    finite_rmsd = rmsd_matrix[np.triu_indices(n, k=1)]
    finite_rmsd = finite_rmsd[np.isfinite(finite_rmsd)]
    cut_height = float(np.median(finite_rmsd)) if finite_rmsd.size > 0 else 3.0
    
    labels = fcluster(Z, t=cut_height, criterion="distance")
    
    # Get leaf order
    def get_leaf_order(tree):
        """Get the order of leaves from a linkage tree."""
        n_leaves = tree.shape[0] + 1
        order = []
        
        def traverse(node):
            if node < n_leaves:
                order.append(int(node))
            else:
                left = int(tree[node - n_leaves, 0])
                right = int(tree[node - n_leaves, 1])
                traverse(left)
                traverse(right)
        
        traverse(2 * n_leaves - 2)  # Root node for n_leaves
        return order
    
    leaf_order = get_leaf_order(Z)
    ordered_names = [condition_names[i] for i in leaf_order]
    
    # Reorder matrix
    reordered = rmsd_matrix[np.ix_(leaf_order, leaf_order)]
    
    # Create figure with dendrograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 12),
                            gridspec_kw={"height_ratios": [1, 4], 
                                        "width_ratios": [1, 4],
                                        "wspace": 0.05, "hspace": 0.05})
    
    # Top dendrogram
    ax_top = axes[0, 1]
    dendrogram(Z, ax=ax_top, color_threshold=0, no_labels=True,
              link_color_func=lambda _: "#555555")
    ax_top.axhline(cut_height, color=style.C_DISP, linestyle="--", linewidth=1.0)
    ax_top.set_xticks([])
    ax_top.set_ylabel("RMSD (Å)", fontsize=10)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    
    # Left dendrogram
    ax_left = axes[1, 0]
    # Transpose for left-side dendrogram
    Z_left = linkage(squareform(reordered, checks=False), method="average")
    dendrogram(Z_left, ax=ax_left, orientation="left", color_threshold=0,
              link_color_func=lambda _: "#555555")
    ax_left.axvline(cut_height, color=style.C_DISP, linestyle="--", linewidth=1.0)
    ax_left.set_yticks([])
    ax_left.set_xlabel("RMSD (Å)", fontsize=10)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)
    
    # Heatmap
    ax_heat = axes[1, 1]
    im = ax_heat.imshow(reordered, cmap="viridis", aspect="auto")
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8, label="Cα RMSD (Å)")
    cbar.set_label("Cα RMSD (Å)", fontsize=10)
    
    # Condition annotations (PTM group + perturbation factors)
    n = len(ordered_names)
    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    
    # X-axis labels (bottom) - PTM group with perturbation annotations
    ax_heat.set_xticklabels(ordered_names, rotation=90, fontsize=8)
    for i, name in enumerate(ordered_names):
        ptm = ptm_groups.get(name, "none")
        color = style.PTM_COLORS.get(ptm, "#777777")
        ax_heat.get_xticklabels()[i].set_color(color)
        
        # Add perturbation annotations as superscript or suffix
        annos = perturbation_annotations.get(name, [])
        if annos:
            # Boldface if any perturbation present
            ax_heat.get_xticklabels()[i].set_fontweight("bold")
    
    # Y-axis labels (left) - same as X
    ax_heat.set_yticklabels(ordered_names, fontsize=8)
    for i, name in enumerate(ordered_names):
        ptm = ptm_groups.get(name, "none")
        color = style.PTM_COLORS.get(ptm, "#777777")
        ax_heat.get_yticklabels()[i].set_color(color)
        
        annos = perturbation_annotations.get(name, [])
        if annos:
            ax_heat.get_yticklabels()[i].set_fontweight("bold")
    
    # Cluster boundary lines
    unique_clusters = np.unique(labels[leaf_order])
    prev_label = None
    for i, lbl in enumerate(labels[leaf_order]):
        if prev_label is not None and lbl != prev_label:
            ax_heat.axhline(i - 0.5, color="white", linewidth=1.5)
            ax_heat.axvline(i - 0.5, color="white", linewidth=1.5)
        prev_label = lbl
    
    ax_heat.set_title("Structural Clustering (RMSD)", fontweight="bold", fontsize=12)
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "structural_distance_heatmap")


def plot_concentration_response_faceted(
    df: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
    perturbation_groups: Optional[Dict[str, List[str]]] = None,
) -> List[Path]:
    """
    Faceted concentration response curves by perturbation state.
    
    Features:
    - Separate panel for each unique perturbation state (PTM + ligands)
    - X-axis: salt concentration
    - Y-axis: RMSD relative to baseline with confidence intervals
    - Baseline reference line at y=0
    - Generic handling of any ligand/modification combination
    
    Parameters:
        df: DataFrame with columns: condition, ptm_group, ion_tier,
            mean_disp, lo, hi, significant
        plots_dir: Output directory
        label_map: Optional label mapping
        perturbation_groups: Optional mapping from condition to list of
            perturbation factors (e.g., ["SEP102", "DNA"], ["TPO101"])
        
    Returns:
        List of generated file paths
    """
    if not {"ptm_group", "ion_tier", "mean_disp", "lo", "hi", "significant"}.issubset(df.columns):
        return []
    
    # Extract perturbation state for each condition if not provided
    if perturbation_groups is None:
        # Default: use PTM group as the sole perturbation factor
        perturbation_groups = {
            name: [row["ptm_group"]]
            for name, row in df.iterrows()
        }
    
    # Get unique perturbation states
    all_perturbations = set()
    for perts in perturbation_groups.values():
        all_perturbations.update(perts)
    
    # Remove "unmodified" if there are actual modifications (simplifies legend)
    if len(all_perturbations - {"unmodified", "none", ""}) > 0:
        perturbation_states = sorted(
            p for p in all_perturbations 
            if p not in ("unmodified", "none", "")
        )
        # Add unmodified back at the end if present
        if "unmodified" in all_perturbations:
            perturbation_states.append("unmodified")
    else:
        perturbation_states = sorted(all_perturbations)
    
    # Get conditions grouped by perturbation state
    conditions_by_perturbation: Dict[str, List[str]] = {}
    for name, perts in perturbation_groups.items():
        for p in perts:
            if p not in conditions_by_perturbation:
                conditions_by_perturbation[p] = []
            conditions_by_perturbation[p].append(name)
    
    # Get ion tiers (salt concentrations)
    ion_tiers = sorted(df["ion_tier"].unique())
    
    # Determine number of panels (one per perturbation state)
    n_panels = len(perturbation_states)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows),
                            squeeze=False)
    axes = axes.flatten()
    
    for idx, perturbation in enumerate(perturbation_states):
        ax = axes[idx]
        
        # Get conditions with this perturbation
        relevant_conditions = conditions_by_perturbation.get(perturbation, [])
        ax_data = df[df["condition"].isin(relevant_conditions)].copy()
        
        if ax_data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                   transform=ax.transAxes, fontsize=10)
            continue
        
        # Sort by ion tier
        ion_tiers_in_data = sorted(ax_data["ion_tier"].unique())
        
        # Plot each condition in this perturbation group
        # Use the condition's full PTM_group to determine color
        unique_ptms = ax_data["ptm_group"].unique()
        
        for ptm in sorted(unique_ptms):
            ptm_data = ax_data[ax_data["ptm_group"] == ptm]
            
            if ptm_data.empty:
                continue
            
            # Get data sorted by ion tier
            sorted_data = ptm_data.set_index("ion_tier").loc[ion_tiers_in_data].reset_index()
            
            if sorted_data.empty or len(sorted_data) < 1:
                continue
            
            x_vals = [ion_tiers_in_data.index(tier) for tier in sorted_data["ion_tier"]]
            y_vals = sorted_data["mean_disp"].values
            y_lo = sorted_data["lo"].values
            y_hi = sorted_data["hi"].values
            
            # Use PTM-specific color
            color = style.PTM_COLORS.get(ptm, "#777777")
            
            # Determine marker style based on number of conditions
            # If multiple conditions for same PTM, differentiate them
            marker = "o"
            if len(unique_ptms) > 1:
                # Use different marker for each PTM within perturbation group
                pass
            
            ax.errorbar(x_vals, y_vals, yerr=[y_vals - y_lo, y_hi - y_vals],
                       fmt="o-", color=color, linewidth=2, markersize=6, capsize=3,
                       label=ptm, alpha=0.8)
        
        # Add baseline reference line (y=0)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        
        # Configure axis
        ax.set_xticks(range(len(ion_tiers_in_data)))
        ax.set_xticklabels(ion_tiers_in_data, fontsize=9)
        ax.set_xlabel("Salt concentration (×)", fontsize=11)
        ax.set_ylabel("RMSD vs baseline (Å)", fontsize=11)
        ax.set_title(f"Perturbation: {perturbation}", fontweight="bold", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        
        # Add significance markers
        for i, tier in enumerate(ion_tiers_in_data):
            tier_data = ax_data[ax_data["ion_tier"] == tier]
            if not tier_data.empty and tier_data["significant"].any():
                ax.plot(i, tier_data["mean_disp"].iloc[0], "b*", markersize=10)
    
    # Hide empty subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Concentration-Dependent Structural Response", 
                fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "concentration_response_faceted")


def plot_quality_dashboard(
    df_conf: pd.DataFrame,
    df_dist: pd.DataFrame,
    plots_dir: Path,
) -> List[Path]:
    """
    Multi-panel quality dashboard showing model confidence metrics.
    
    Features:
    - pTM distribution as histogram
    - ipTM vs pTM scatter
    - pLDDT vs PAE scatter
    - Quality tier distribution as bar chart
    
    Parameters:
        df_conf: Confidence summary DataFrame
        df_dist: Distance DataFrame
        plots_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    if not {"ptm", "iptm", "plddt_mean", "mean_pae"}.issubset(df_conf.columns):
        return []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: pTM distribution
    ax1 = axes[0, 0]
    ptm_vals = df_conf["ptm"].dropna()
    ax1.hist(ptm_vals, bins=20, color=style.PALETTE[0], alpha=0.7, edgecolor="black")
    ax1.axvline(ptm_vals.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {ptm_vals.mean():.3f}")
    ax1.set_xlabel("pTM", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("pTM Distribution", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: ipTM vs pTM scatter
    ax2 = axes[0, 1]
    iptm_vals = df_conf["iptm"].dropna()
    ax2.scatter(ptm_vals, iptm_vals, s=50, color=style.PALETTE[1], alpha=0.7,
               edgecolor="black")
    ax2.set_xlabel("pTM", fontsize=11)
    ax2.set_ylabel("ipTM", fontsize=11)
    ax2.set_title("ipTM vs pTM", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    if len(ptm_vals) >= 2:
        corr = np.corrcoef(ptm_vals, iptm_vals)[0, 1]
        if np.isfinite(corr):
            ax2.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax2.transAxes,
                    fontsize=10, verticalalignment="top",
                    fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="white"))
    
    # Panel 3: pLDDT vs PAE scatter
    ax3 = axes[1, 0]
    plddt_vals = df_conf["plddt_mean"].dropna()
    pae_vals = df_conf["mean_pae"].dropna()
    
    # Filter valid values
    valid = np.isfinite(plddt_vals) & np.isfinite(pae_vals)
    ax3.scatter(plddt_vals[valid], pae_vals[valid], s=50, color=style.PALETTE[2],
               alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Mean pLDDT", fontsize=11)
    ax3.set_ylabel("Mean PAE (Å)", fontsize=11)
    ax3.set_title("pLDDT vs PAE", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Quality tier distribution
    ax4 = axes[1, 1]
    if "confidence_tier" in df_conf.columns:
        tier_counts = df_conf["confidence_tier"].value_counts()
        colors = [style.TIER_BADGE.get(t, "#777777") for t in tier_counts.index]
        ax4.bar(range(len(tier_counts)), tier_counts.values, color=colors,
               edgecolor="black")
        ax4.set_xticks(range(len(tier_counts)))
        ax4.set_xticklabels(tier_counts.index, fontsize=9, rotation=45, ha="right")
        ax4.set_ylabel("Count", fontsize=11)
        ax4.set_title("Quality Tier Distribution", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y")
    else:
        ax4.text(0.5, 0.5, "No quality tier data", ha="center", va="center",
                transform=ax4.transAxes, fontsize=10)
    
    fig.suptitle("Model Quality Dashboard", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "quality_dashboard")


def plot_clustering_overview(
    cluster_labels: Dict[str, int],
    ptm_groups: Dict[str, str],
    has_dna: Dict[str, bool],
    rmsd_matrix: np.ndarray,
    condition_names: List[str],
    plots_dir: Path,
) -> List[Path]:
    """
    Clustering overview: bar plot of cluster assignments + centroid heatmap.
    
    Parameters:
        cluster_labels: Mapping from condition to cluster ID
        ptm_groups: Mapping from condition to PTM group
        has_dna: Mapping from condition to DNA presence
        rmsd_matrix: RMSD matrix for centroid computation
        condition_names: List of condition names
        plots_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Bar plot of cluster assignments
    ax1 = axes[0]
    
    # Count conditions per cluster
    cluster_counts = {}
    for name, cluster in cluster_labels.items():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    
    # Sort by cluster ID
    sorted_clusters = sorted(cluster_counts.keys())
    counts = [cluster_counts[c] for c in sorted_clusters]
    
    # Color by cluster
    colors = [style.cluster_color(c) for c in sorted_clusters]
    
    ax1.bar(range(len(sorted_clusters)), counts, color=colors, edgecolor="black")
    ax1.set_xticks(range(len(sorted_clusters)))
    ax1.set_xticklabels([f"C{c}" for c in sorted_clusters], fontsize=10)
    ax1.set_ylabel("Number of conditions", fontsize=11)
    ax1.set_title("Cluster Assignment Distribution", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Annotate with counts
    for i, (c, count) in enumerate(zip(sorted_clusters, counts)):
        ax1.text(i, count + 0.1, str(count), ha="center", va="bottom", fontsize=9)
    
    # Heatmap of cluster centroids
    ax2 = axes[1]
    
    # Compute cluster centroids (mean of members)
    unique_clusters = sorted(set(cluster_labels.values()))
    n_clusters = len(unique_clusters)
    
    # Get condition order
    ordered_names = sorted(condition_names, key=lambda x: cluster_labels.get(x, 0))
    
    # Reorder matrix
    idx_order = [condition_names.index(n) for n in ordered_names]
    reordered = rmsd_matrix[np.ix_(idx_order, idx_order)]
    
    im = ax2.imshow(reordered, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax2, shrink=0.8, label="Cα RMSD (Å)")
    
    # Set labels
    ax2.set_xticks(range(len(ordered_names)))
    ax2.set_yticks(range(len(ordered_names)))
    ax2.set_xticklabels(ordered_names, rotation=90, fontsize=7)
    ax2.set_yticklabels(ordered_names, fontsize=7)
    
    # Color by PTM group
    for i, name in enumerate(ordered_names):
        ptm = ptm_groups.get(name, "none")
        color = style.PTM_COLORS.get(ptm, "#777777")
        ax2.get_xticklabels()[i].set_color(color)
        ax2.get_yticklabels()[i].set_color(color)
        if has_dna.get(name, False):
            ax2.get_xticklabels()[i].set_fontweight("bold")
            ax2.get_yticklabels()[i].set_fontweight("bold")
    
    ax2.set_title("Clustered RMSD Matrix", fontweight="bold")
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "clustering_overview")


def plot_residue_fluctuation_profile(
    rmsf_data: pd.DataFrame,
    plots_dir: Path,
    domain_annotations: Optional[Dict[str, List[tuple]]] = None,
    ptm_sites: Optional[List[int]] = None,
) -> List[Path]:
    """
    Per-residue fluctuation profile with domain annotation.
    
    Parameters:
        rmsf_data: DataFrame with columns: chain_id, residue_number, rmsf_A
        domain_annotations: Optional dict mapping chain to list of (start, end) tuples
        ptm_sites: Optional list of residue numbers with PTM sites
        plots_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    if not {"chain_id", "residue_number", "rmsf_A"}.issubset(rmsf_data.columns):
        return []
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # Plot RMSF by chain
    for chain in rmsf_data["chain_id"].unique():
        chain_data = rmsf_data[rmsf_data["chain_id"] == chain].sort_values("residue_number")
        
        x = chain_data["residue_number"].values
        y = chain_data["rmsf_A"].values
        
        ax.plot(x, y, label=f"Chain {chain}", linewidth=1.5, alpha=0.8)
    
    # Add domain annotations if provided
    if domain_annotations:
        for chain, domains in domain_annotations.items():
            for start, end in domains:
                ax.axvspan(start, end, alpha=0.1, color=style.PALETTE[2],
                          label=f"Domain ({start}-{end})" if chain == rmsf_data["chain_id"].iloc[0] else "")
    
    # Add PTM site markers
    if ptm_sites:
        for site in ptm_sites:
            ax.axvline(site, color=style.PTM_COLORS.get("TPO", "#CC79A7"),
                      linestyle="--", linewidth=1.0, alpha=0.7,
                      label="PTM site" if site == ptm_sites[0] else "")
    
    # Add threshold lines (mean ± std)
    all_rmsf = rmsf_data["rmsf_A"].dropna()
    mean_rmsf = all_rmsf.mean()
    std_rmsf = all_rmsf.std()
    
    ax.axhline(mean_rmsf, color="black", linestyle="-", linewidth=1.0,
              label=f"Mean: {mean_rmsf:.2f} Å")
    ax.axhline(mean_rmsf + std_rmsf, color="gray", linestyle="--", linewidth=0.8,
              label=f"Mean + 1σ")
    ax.axhline(mean_rmsf - std_rmsf, color="gray", linestyle="--", linewidth=0.8)
    
    ax.set_xlabel("Residue number", fontsize=12)
    ax.set_ylabel("RMSF (Å)", fontsize=12)
    ax.set_title("Per-Residue Fluctuation Profile", fontweight="bold", fontsize=14)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return style.save(fig, plots_dir, "residue_fluctuation_profile")


def plot_dimensionality_reduction(
    embeddings: np.ndarray,
    condition_names: List[str],
    ptm_groups: Dict[str, str],
    plots_dir: Path,
    method: str = "pca",
    annotations: Optional[Dict[str, List[str]]] = None,
) -> List[Path]:
    """
    2D dimensionality reduction visualization (PCA, t-SNE, UMAP).
    
    Features:
    - Generic: works with any perturbation factors
    - Color-coded by perturbation state
    - Scalable visualization for large datasets
    
    Parameters:
        embeddings: N×2 array of reduced coordinates
        condition_names: List of condition names
        ptm_groups: Mapping from condition to PTM group
        plots_dir: Output directory
        method: Reduction method name (pca, tsne, umap)
        annotations: Optional mapping from condition to list of
            perturbation factors for annotation
        
    Returns:
        List of generated file paths
    """
    if not HAS_SKLEARN:
        log.warning("sklearn not installed; dimensionality reduction skipped.")
        return []
    
    if embeddings.shape[1] != 2:
        return []
    
    # Default: no annotations
    if annotations is None:
        annotations = {n: [] for n in condition_names}
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by PTM group
    for name in condition_names:
        ptm = ptm_groups.get(name, "none")
        color = style.PTM_COLORS.get(ptm, "#777777")
        
        idx = condition_names.index(name)
        x, y = embeddings[idx]
        annos = annotations.get(name, [])
        
        # Marker style based on perturbation count
        marker = "o"
        if len(annos) > 1:
            marker = "D"  # diamond for multiple perturbations
        elif len(annos) == 1 and "DNA" in annos:
            marker = "s"  # square for DNA (but could be any ligand)
        
        ax.scatter(x, y, s=80, color=color, marker=marker, alpha=0.8,
                  edgecolor="black", linewidth=0.5,
                  label=f"{ptm}{'+' if annos else ''}")
        
        # Add condition label (only for a subset to avoid clutter)
        if len(condition_names) <= 30 or idx % max(1, len(condition_names) // 10) == 0:
            ax.text(x, y, name.split("_")[0], fontsize=7, ha="center", va="center")
    
    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
    ax.set_title(f"{method.upper()} Projection of Structural Conditions", 
                fontweight="bold", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="best")
    
    fig.tight_layout()
    return style.save(fig, plots_dir, f"{method}_embedding")
