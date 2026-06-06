"""
Collapsed prediction detection for AF3 ensemble analysis.

Implements the detection criteria from the simplified specification:
1. pTM < 0.25
2. mean PAE > 25.0 Å  
3. within-condition n_structural_clusters > 10

Also provides helper functions for handling collapsed conditions throughout the pipeline.
"""

from __future__ import annotations

from typing import Dict, Set, Optional
import pandas as pd


def detect_collapsed_conditions(
    confidence_df: pd.DataFrame,
    heterogeneity_data: Optional[Dict[str, Dict]] = None,
    pae_threshold: float = 25.0,
    ptm_threshold: float = 0.25,
    cluster_threshold: int = 10
) -> Set[str]:
    """
    Detect collapsed conditions using the criteria from the simplified spec.
    
    Parameters
    ----------
    confidence_df : pd.DataFrame
        DataFrame from confidence_summary.csv with columns: 
        'condition', 'ptm', 'mean_pae'
    heterogeneity_data : dict, optional
        Dict mapping condition names to heterogeneity data with key 'n_clusters'
    pae_threshold : float
        Mean PAE threshold (default: 25.0 Å)
    ptm_threshold : float  
        pTM threshold (default: 0.25)
    cluster_threshold : int
        n_structural_clusters threshold (default: 10)
    
    Returns
    -------
    set
        Set of condition names that are collapsed
    """
    collapsed = set()
    
    for _, row in confidence_df.iterrows():
        condition = row['condition']
        ptm = row.get('ptm')
        mean_pae = row.get('mean_pae')
        
        # Check pTM and PAE criteria
        if pd.isna(ptm) or pd.isna(mean_pae):
            continue
            
        meets_ptm_pae = (ptm < ptm_threshold) and (mean_pae > pae_threshold)
        
        # Check cluster criterion if heterogeneity data is available
        meets_cluster = False
        if heterogeneity_data and condition in heterogeneity_data:
            n_clusters = heterogeneity_data[condition].get('n_clusters', 0)
            meets_cluster = n_clusters > cluster_threshold
        else:
            # If no heterogeneity data, just use pTM+PAE
            meets_cluster = True
        
        # Condition is collapsed if it meets all available criteria
        if meets_ptm_pae and meets_cluster:
            collapsed.add(condition)
    
    return collapsed


def add_collapsed_flags_to_dataframes(
    confidence_df: pd.DataFrame,
    structural_distances_df: pd.DataFrame,
    collapsed_conditions: Set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add is_collapsed column to dataframes.
    
    Parameters
    ----------
    confidence_df : pd.DataFrame
        DataFrame from confidence_summary.csv
    structural_distances_df : pd.DataFrame
        DataFrame from structural_distances.csv  
    collapsed_conditions : set
        Set of collapsed condition names
    
    Returns
    -------
    tuple
        Updated confidence_df, structural_distances_df with is_collapsed column
    """
    # Add to confidence summary
    confidence_df = confidence_df.copy()
    confidence_df['is_collapsed'] = confidence_df['condition'].isin(collapsed_conditions)
    
    # Add to structural distances
    structural_distances_df = structural_distances_df.copy()
    structural_distances_df['is_collapsed'] = structural_distances_df['condition'].isin(collapsed_conditions)
    
    # Also unify likely_failed and likely_artifact into is_collapsed
    if 'likely_failed' in structural_distances_df.columns:
        structural_distances_df['is_collapsed'] = (
            structural_distances_df['is_collapsed'] | 
            structural_distances_df['likely_failed']
        )
    
    return confidence_df, structural_distances_df


def get_display_name_map() -> Dict[str, str]:
    """
    Get the display name mapping with footnote symbols.
    
    Returns
    -------
    dict
        Mapping from raw condition names to display names with footnotes
    """
    return {
        # Valid conditions
        "oct4_seg_chain_b_nax1_clx1_smilesx10_no_msa": "Unmod 1x (baseline)",
        "oct4_seg_chain_b_nax10_clx10_smilesx100_no_msa": "Unmod 10x",
        "oct4_seg_chain_b_sep102_nax1_clx1_smilesx10_no_msa": "SEP102 1x",
        "oct4_seg_chain_b_sep102_nax10_clx10_smilesx100_no_msa": "SEP102 10x",
        "oct4_seg_chain_b_tpo101_nax1_clx1_smilesx10_no_msa": "TPO101 1x ‡",
        "oct4_seg_chain_b_tpo101_nax10_clx10_smilesx100_no_msa": "TPO101 10x",
        "oct4_seg_chainb_dna_no_msa": "DNA (no ions)",
        "oct4_seg_chainb_dna_ions_no_msa": "DNA + ions",
        
        # Collapsed conditions
        "oct4_seg_chain_b_nax100_clx100_smilesx1000_no_msa": "Unmod 100x †",
        "oct4_seg_chain_b_sep102_nax100_clx100_smilesx1000_no_msa": "SEP102 100x †",
        "oct4_seg_chain_b_tpo101_nax100_clx100_smilesx1000_no_msa": "TPO101 100x †",
    }


def get_heterogeneity_tier(
    n_clusters: int, 
    dominant_fraction: float, 
    is_collapsed: bool = False
) -> str:
    """
    Get heterogeneity tier using simplified criteria.
    
    Parameters
    ----------
    n_clusters : int
        Number of structural clusters
    dominant_fraction : float
        Fraction of replicates in largest cluster
    is_collapsed : bool
        Whether condition is collapsed
    
    Returns
    -------
    str
        Heterogeneity tier: 'collapsed', 'high', 'moderate', or 'low'
    """
    if is_collapsed:
        return "collapsed"
    
    if n_clusters > 5 or dominant_fraction < 0.50:
        return "high"
    
    if n_clusters in [2, 3, 4, 5]:  # n_clusters in [2, 5]
        return "moderate"
    
    # n_clusters == 1
    return "low"


def is_tpo101_1x_high_heterogeneity(condition_name: str) -> bool:
    """
    Check if condition is TPO101 1x which requires special high-heterogeneity handling.
    
    Parameters
    ----------
    condition_name : str
        Condition name
    
    Returns
    -------
    bool
        True if condition is TPO101 1x
    """
    return condition_name == "oct4_seg_chain_b_tpo101_nax1_clx1_smilesx10_no_msa"


def get_footnote_symbols() -> Dict[str, str]:
    """
    Get footnote symbol explanations.
    
    Returns
    -------
    dict
        Mapping from footnote symbols to explanations
    """
    return {
        "†": "Collapsed prediction — model collapse from ion saturation (pTM < 0.25, mean PAE > 25 Å). Not biologically interpretable.",
        "‡": "High within-condition heterogeneity (6 clusters, dominant_fraction=0.62). Mean displacement reflects ensemble spread, not a single conformational shift.",
    }