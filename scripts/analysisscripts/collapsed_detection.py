"""
Collapsed prediction detection for AF3 ensemble analysis.

A "collapsed" condition is one whose *macromolecular* structure prediction is
not interpretable — not merely one whose AF3 summary scores are low.

Scope matters.  In AF3 every explicit ion (Na+, Cl-) and water is represented as
its own token, and the positions of free solvent are not determined by the
macromolecular fold.  Consequently the *full-system* pTM/ipTM and the mean of
the *full* PAE matrix are diluted/inflated by physically meaningless
solvent–solvent and solvent–macromolecule entries, and they degrade purely as a
function of how many ions/waters were placed in the input — independent of
whether the protein/complex itself was predicted well.  Judging "model
collapse" from those system-wide scores therefore mislabels well-folded
high-ionic-strength conditions as failures.

The criteria below are scope-consistent and macromolecule-centric:

1. protein mean pLDDT < ``plddt_threshold`` (default 50) — AlphaFold's
   "very low confidence" band, below which a model "should not be interpreted"
   (Jumper et al. 2021, Nature; AlphaFold/AlphaFold-DB confidence guidance).
   pLDDT is a per-residue local-confidence measure evaluated only on protein
   Cα here, so it is unaffected by solvent token count.
2. macromolecule-scoped mean PAE > ``pae_threshold`` (default 25 Å) — computed
   over protein + nucleic tokens only (see ``model.ConditionModel.mean_pae``),
   indicating no reliable relative positioning of the macromolecular tokens.
3. within-condition n_structural_clusters > ``cluster_threshold`` (default 10),
   when heterogeneity data is available.

Both confidence criteria must be met (conjunctive) so that only genuinely
uninterpretable predictions are quarantined; either signal alone can have a
benign explanation (flexible termini depressing mean pLDDT; multi-domain
flexibility raising PAE).

Also provides helper functions for handling collapsed conditions throughout the
pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Set, Optional
import pandas as pd

log = logging.getLogger("af3bench2.collapsed")


def detect_collapsed_conditions(
    confidence_df: pd.DataFrame,
    heterogeneity_data: Optional[Dict[str, Dict]] = None,
    pae_threshold: float = 25.0,
    plddt_threshold: float = 50.0,
    cluster_threshold: int = 10
) -> Set[str]:
    """
    Detect collapsed conditions from macromolecule-scoped confidence.

    Parameters
    ----------
    confidence_df : pd.DataFrame
        DataFrame from confidence_summary.csv.  Uses ``plddt_mean`` (protein
        mean pLDDT) and ``mean_pae`` (macromolecule-scoped mean PAE).  If
        ``plddt_mean`` is absent it falls back to the legacy full-system
        ``ptm`` < 0.25 criterion and logs a warning, since that path is
        contaminated by solvent tokens and is not recommended.
    heterogeneity_data : dict, optional
        Dict mapping condition names to heterogeneity data with key 'n_clusters'
    pae_threshold : float
        Macromolecule mean PAE threshold (default: 25.0 Å)
    plddt_threshold : float
        Protein mean pLDDT threshold (default: 50.0, AlphaFold "very low" band)
    cluster_threshold : int
        n_structural_clusters threshold (default: 10)

    Returns
    -------
    set
        Set of condition names whose macromolecular prediction is collapsed.
    """
    collapsed = set()
    has_plddt = "plddt_mean" in confidence_df.columns
    if not has_plddt:
        log.warning(
            "confidence_df has no 'plddt_mean' column; falling back to the "
            "legacy full-system pTM<0.25 collapse criterion, which is inflated "
            "by free ion/water tokens. Prefer protein pLDDT."
        )

    for _, row in confidence_df.iterrows():
        condition = row['condition']
        mean_pae = row.get('mean_pae')  # macromolecule-scoped (see model.py)

        if has_plddt:
            plddt = row.get('plddt_mean')
            if pd.isna(plddt) or pd.isna(mean_pae):
                continue
            meets_conf = (plddt < plddt_threshold) and (mean_pae > pae_threshold)
        else:
            ptm = row.get('ptm')
            if pd.isna(ptm) or pd.isna(mean_pae):
                continue
            meets_conf = (ptm < 0.25) and (mean_pae > pae_threshold)

        # Check cluster criterion if heterogeneity data is available
        if heterogeneity_data and condition in heterogeneity_data:
            n_clusters = heterogeneity_data[condition].get('n_clusters', 0)
            meets_cluster = n_clusters > cluster_threshold
        else:
            # If no heterogeneity data, rely on the confidence criteria alone.
            meets_cluster = True

        if meets_conf and meets_cluster:
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


def get_heterogeneity_tier(
    n_clusters: int, 
    dominant_fraction: float, 
    is_collapsed: bool = False
) -> str:
    """Within-condition heterogeneity tier.

    Delegates to :func:`heterogeneity._assign_tier` (the canonical, dominance-
    aware definition) so the two call sites cannot diverge.
    """
    from .heterogeneity import _assign_tier
    return _assign_tier(n_clusters, dominant_fraction, is_collapsed)