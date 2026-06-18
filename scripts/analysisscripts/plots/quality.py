"""
Quality assessment visualizations for analysisscripts.

Provides multi-panel quality dashboards showing:
- Model confidence distributions
- Quality metric correlations
- Cluster confidence breakdowns
- Per-condition quality assessment
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


def plot_confidence_distributions(
    df_conf: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """
    Multi-panel confidence distribution dashboard.
    
    Panels:
    1. pTM distribution with baseline comparison
    2. ipTM distribution with baseline comparison
    3. pLDDT distribution with baseline comparison
    4. PAE distribution with baseline comparison
    
    Parameters:
        df_conf: Confidence summary DataFrame
        plots_dir: Output directory
        label_map: Optional condition label mapping
        
    Returns:
        List of generated file paths
    """
    if not {"ptm", "iptm", "plddt_mean", "mean_pae"}.issubset(df_conf.columns):
        return []
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = [
        ("ptm", "pTM", 0, (0, 1)),
        ("iptm", "ipTM", 1, (0, 1)),
        ("plddt_mean", "pLDDT", 2, (60, 100)),
        ("mean_pae", "Mean PAE (Å)", 3, (0, 30)),
    ]
    
    for ax, (col, title, idx, lim) in zip(axes, metrics):
        vals = df_conf[col].dropna()
        
        # Histogram
        ax.hist(vals, bins=20, color=style.PALETTE[0], alpha=0.7, 
               edgecolor="black", density=False)
        
        # Add mean line
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=2,
                  label=f"Mean: {vals.mean():.2f}")
        
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"{title} Distribution", fontweight="bold")
        ax.set_xlim(lim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Model Confidence Distributions", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "confidence_distributions")


def plot_quality_correlations(
    df_conf: pd.DataFrame,
    plots_dir: Path,
) -> List[Path]:
    """
    Scatter plot matrix showing quality metric correlations.
    
    Panels:
    - pTM vs ipTM
    - pTM vs pLDDT
    - ipTM vs pLDDT
    - pLDDT vs PAE (inverse correlation expected)
    
    Parameters:
        df_conf: Confidence summary DataFrame
        plots_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    if not {"ptm", "iptm", "plddt_mean", "mean_pae"}.issubset(df_conf.columns):
        return []
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # pTM vs ipTM
    ax1 = axes[0, 0]
    valid = np.isfinite(df_conf["ptm"]) & np.isfinite(df_conf["iptm"])
    ax1.scatter(df_conf.loc[valid, "ptm"], df_conf.loc[valid, "iptm"],
               s=60, color=style.PALETTE[0], alpha=0.7, edgecolor="black")
    ax1.set_xlabel("pTM", fontsize=11)
    ax1.set_ylabel("ipTM", fontsize=11)
    ax1.set_title("pTM vs ipTM", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    if valid.sum() >= 2:
        corr = np.corrcoef(df_conf.loc[valid, "ptm"], df_conf.loc[valid, "iptm"])[0, 1]
        if np.isfinite(corr):
            ax1.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax1.transAxes,
                    fontsize=10, verticalalignment="top",
                    fontfamily="monospace")
    
    # pTM vs pLDDT
    ax2 = axes[0, 1]
    valid = np.isfinite(df_conf["ptm"]) & np.isfinite(df_conf["plddt_mean"])
    ax2.scatter(df_conf.loc[valid, "ptm"], df_conf.loc[valid, "plddt_mean"],
               s=60, color=style.PALETTE[1], alpha=0.7, edgecolor="black")
    ax2.set_xlabel("pTM", fontsize=11)
    ax2.set_ylabel("pLDDT", fontsize=11)
    ax2.set_title("pTM vs pLDDT", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # ipTM vs pLDDT
    ax3 = axes[1, 0]
    valid = np.isfinite(df_conf["iptm"]) & np.isfinite(df_conf["plddt_mean"])
    ax3.scatter(df_conf.loc[valid, "iptm"], df_conf.loc[valid, "plddt_mean"],
               s=60, color=style.PALETTE[2], alpha=0.7, edgecolor="black")
    ax3.set_xlabel("ipTM", fontsize=11)
    ax3.set_ylabel("pLDDT", fontsize=11)
    ax3.set_title("ipTM vs pLDDT", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # pLDDT vs PAE (inverse correlation expected)
    ax4 = axes[1, 1]
    valid = np.isfinite(df_conf["plddt_mean"]) & np.isfinite(df_conf["mean_pae"])
    ax4.scatter(df_conf.loc[valid, "plddt_mean"], df_conf.loc[valid, "mean_pae"],
               s=60, color=style.PALETTE[3], alpha=0.7, edgecolor="black")
    ax4.set_xlabel("pLDDT", fontsize=11)
    ax4.set_ylabel("Mean PAE (Å)", fontsize=11)
    ax4.set_title("pLDDT vs PAE (inverse corr expected)", fontweight="bold")
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle("Quality Metric Correlations", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "quality_correlations")


def plot_cluster_confidence_breakdown(
    df_cluster_conf: pd.DataFrame,
    plots_dir: Path,
) -> List[Path]:
    """
    Per-cluster confidence breakdown.
    
    Panels:
    - Mean pTM per cluster (with SD)
    - Mean pLDDT per cluster
    - Cluster size distribution
    
    Parameters:
        df_cluster_conf: Cluster confidence breakdown DataFrame
        plots_dir: Output directory
        
    Returns:
        List of generated file paths
    """
    if not {"cluster", "mean_ptm_per_cluster", "sd_ptm_per_cluster", 
            "mean_plddt_per_cluster", "n_replicates_per_cluster"}.issubset(df_cluster_conf.columns):
        return []
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: Mean pTM per cluster
    ax1 = axes[0]
    cluster_ptm = df_cluster_conf.groupby("cluster")["mean_ptm_per_cluster"].mean()
    cluster_ptm_std = df_cluster_conf.groupby("cluster")["sd_ptm_per_cluster"].mean()
    
    ax1.bar(cluster_ptm.index, cluster_ptm.values, 
           yerr=cluster_ptm_std.values, capsize=3,
           color=style.PALETTE[0], edgecolor="black")
    ax1.set_xlabel("Cluster", fontsize=11)
    ax1.set_ylabel("Mean pTM", fontsize=11)
    ax1.set_title("Cluster Confidence (pTM)", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    
    # Panel 2: Mean pLDDT per cluster
    ax2 = axes[1]
    cluster_plddt = df_cluster_conf.groupby("cluster")["mean_plddt_per_cluster"].mean()
    
    ax2.bar(cluster_plddt.index, cluster_plddt.values,
           color=style.PALETTE[1], edgecolor="black")
    ax2.set_xlabel("Cluster", fontsize=11)
    ax2.set_ylabel("Mean pLDDT", fontsize=11)
    ax2.set_title("Cluster Confidence (pLDDT)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    
    # Panel 3: Cluster size distribution
    ax3 = axes[2]
    cluster_sizes = df_cluster_conf.groupby("cluster")["n_replicates_per_cluster"].sum()
    
    ax3.bar(cluster_sizes.index, cluster_sizes.values,
           color=style.PALETTE[2], edgecolor="black")
    ax3.set_xlabel("Cluster", fontsize=11)
    ax3.set_ylabel("Number of replicates", fontsize=11)
    ax3.set_title("Cluster Size Distribution", fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    
    # Annotate counts
    for i, (c, count) in enumerate(cluster_sizes.items()):
        ax3.text(c, count + 0.1, str(count), ha="center", va="bottom", fontsize=9)
    
    fig.suptitle("Cluster Confidence Breakdown", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "cluster_confidence_breakdown")


def plot_quality_by_condition(
    df_conf: pd.DataFrame,
    df_dist: pd.DataFrame,
    plots_dir: Path,
    label_map: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """
    Per-condition quality assessment.
    
    Panels:
    - pTM and RMSD scatter (showing confidence vs deviation tradeoff)
    - pLDDT vs significant residues (showing confidence vs motion)
    - Quality tier distribution as stacked bar
    
    Parameters:
        df_conf: Confidence summary DataFrame
        df_dist: Distance DataFrame
        plots_dir: Output directory
        label_map: Optional label mapping
        
    Returns:
        List of generated file paths
    """
    if not {"condition", "ptm", "plddt_mean"}.issubset(df_conf.columns):
        return []
    
    # Merge confidence and distance data
    merged = df_conf.merge(df_dist[["condition", "rmsd", "n_significant"]], 
                          on="condition", how="left")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Panel 1: pTM vs RMSD scatter
    ax1 = axes[0]
    valid = np.isfinite(merged["ptm"]) & np.isfinite(merged["rmsd"])
    ax1.scatter(merged.loc[valid, "ptm"], merged.loc[valid, "rmsd"],
               s=80, color=style.PALETTE[0], alpha=0.7, edgecolor="black")
    ax1.set_xlabel("pTM", fontsize=11)
    ax1.set_ylabel("RMSD vs baseline (Å)", fontsize=11)
    ax1.set_title("Confidence vs Structural Deviation", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: pLDDT vs significant residues
    ax2 = axes[1]
    valid = np.isfinite(merged["plddt_mean"]) & np.isfinite(merged["n_significant"])
    ax2.scatter(merged.loc[valid, "plddt_mean"], merged.loc[valid, "n_significant"],
               s=80, color=style.PALETTE[1], alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Mean pLDDT", fontsize=11)
    ax2.set_ylabel("Significant residues (FDR<0.05)", fontsize=11)
    ax2.set_title("Confidence vs Local Motion", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Quality tier distribution
    ax3 = axes[2]
    if "confidence_tier" in merged.columns:
        tier_counts = merged["confidence_tier"].value_counts()
        colors = [style.TIER_BADGE.get(t, "#777777") for t in tier_counts.index]
        ax3.bar(range(len(tier_counts)), tier_counts.values, color=colors,
               edgecolor="black")
        ax3.set_xticks(range(len(tier_counts)))
        ax3.set_xticklabels(tier_counts.index, fontsize=9, rotation=45, ha="right")
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title("Quality Tier Distribution", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(0.5, 0.5, "No quality tier data", ha="center", va="center",
                transform=ax3.transAxes, fontsize=10)
    
    fig.suptitle("Per-Condition Quality Assessment", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    
    return style.save(fig, plots_dir, "quality_by_condition")
