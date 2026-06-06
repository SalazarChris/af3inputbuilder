"""
Within-condition structural heterogeneity (af3bench2 addition).

The original pipeline only clustered *between* conditions.  Several of the
overhaul plan's features (0.3, 1.1f, 1.2b, 2.1, 2.2, 3.1, 3.2) require knowing
how reproducible the AF3 ensemble is *within* a single condition: how many
distinct structural clusters its replicates fall into, how dominant the largest
cluster is, and the spread of replicate-to-mean RMSD.

This module computes that from an :class:`EnsembleModel` and packages it as a
``HeterogeneitySummary`` per condition plus a per-cluster confidence breakdown.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as geom

log = logging.getLogger("af3bench2.heterogeneity")

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


@dataclass
class HeterogeneitySummary:
    condition: str
    n_replicates: int = 0
    rmsd_median: float = float("nan")
    rmsd_iqr: float = float("nan")
    rmsd_max: float = float("nan")
    n_clusters: int = 1
    dominant_fraction: float = 1.0
    cluster_entropy: float = 0.0
    ptm_cv: float = float("nan")
    plddt_cv: float = float("nan")
    tier: str = "low"
    # Per-replicate cluster assignment (1-based), parallel to ensemble axis 0.
    cluster_assignments: List[int] = field(default_factory=list)
    # Per-replicate RMSD to the ensemble mean structure.
    rmsd_to_mean: List[float] = field(default_factory=list)


def _cv(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    m = float(np.mean(v))
    if abs(m) < 1e-9:
        return float("nan")
    return float(np.std(v, ddof=1) / abs(m))


def _pairwise_rmsd(coords: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Symmetric S×S Cα RMSD matrix between ensemble replicates."""
    S = coords.shape[0]
    mat = np.zeros((S, S), dtype=np.float64)
    for i in range(S):
        for j in range(i + 1, S):
            _, _, rmsd = geom.kabsch(coords[i][fit_mask], coords[j][fit_mask])
            mat[i, j] = mat[j, i] = rmsd
    return mat


def _entropy_bits(fractions: np.ndarray) -> float:
    f = fractions[fractions > 0]
    if f.size <= 1:
        return 0.0
    return float(-np.sum(f * np.log2(f)))


def _assign_tier(n_clusters: int, dominant_fraction: float, is_collapsed: bool = False) -> str:
    """heterogeneity_tier per simplified spec criteria."""
    if is_collapsed:
        return "collapsed"
    
    if n_clusters > 5 or dominant_fraction < 0.50:
        return "high"
    
    if n_clusters in [2, 3, 4, 5]:  # n_clusters in [2, 5]
        return "moderate"
    
    # n_clusters == 1
    return "low"


def summarize_condition(
    condition: str,
    ensemble,
    plddt_cutoff: float = 50.0,
    cluster_threshold: float = 3.0,
    is_collapsed: bool = False,
) -> HeterogeneitySummary:
    """
    Compute within-condition heterogeneity from an EnsembleModel.

    cluster_threshold is the Cα RMSD (Å) cut height used to group replicates
    into structural clusters.
    """
    summ = HeterogeneitySummary(condition=condition)
    if ensemble is None or not getattr(ensemble, "has_structural_ensemble", False):
        # Degenerate: single representative (or none) → perfectly reproducible
        summ.n_replicates = int(getattr(ensemble, "n_samples", 0) or 0)
        summ.ptm_cv = _cv(getattr(ensemble, "ptm", np.empty(0)))
        summ.plddt_cv = _cv(getattr(ensemble, "plddt_mean", np.empty(0)))
        return summ

    coords = np.asarray(ensemble.ca_coords, dtype=np.float64)
    plddts = np.asarray(ensemble.ca_plddts, dtype=np.float64)
    S, N, _ = coords.shape
    summ.n_replicates = S

    # Too few replicates for reliable clustering — treat as single cluster.
    # With n<10 a single outlier replicate creates a spurious "cluster" that
    # represents 20% of the ensemble, making the tier assignment meaningless.
    if S < 10:
        summ.n_clusters = 1
        summ.dominant_fraction = 1.0
        summ.tier = "collapsed" if is_collapsed else "low"
        summ.cluster_assignments = [1] * S
        summ.ptm_cv = _cv(ensemble.ptm)
        summ.plddt_cv = _cv(ensemble.plddt_mean)
        log.debug("%s: n=%d < 10, skipping within-condition clustering", condition, S)
        # Still compute RMSD spread for the variance summary table
        fit_mask = np.all(plddts > plddt_cutoff, axis=0)
        if fit_mask.sum() < 3:
            fit_mask = np.ones(N, dtype=bool)
        aligned = geom.superpose_stack_to_mean(coords, plddts, plddt_cutoff)
        mean = aligned.mean(axis=0)
        rmsd_to_mean = np.sqrt(
            np.mean(np.sum((aligned[:, fit_mask] - mean[fit_mask]) ** 2, axis=2), axis=1)
        )
        summ.rmsd_to_mean = [float(x) for x in rmsd_to_mean]
        summ.rmsd_median = float(np.median(rmsd_to_mean))
        summ.rmsd_iqr = float(np.percentile(rmsd_to_mean, 75) - np.percentile(rmsd_to_mean, 25))
        summ.rmsd_max = float(np.max(rmsd_to_mean))
        return summ

    # Fitting mask: residues confident in all frames (fall back to all)
    fit_mask = np.all(plddts > plddt_cutoff, axis=0)
    if fit_mask.sum() < 3:
        fit_mask = np.ones(N, dtype=bool)

    # Superpose to mean, RMSD of each replicate to the ensemble mean
    aligned = geom.superpose_stack_to_mean(coords, plddts, plddt_cutoff)
    mean = aligned.mean(axis=0)
    rmsd_to_mean = np.sqrt(
        np.mean(np.sum((aligned[:, fit_mask] - mean[fit_mask]) ** 2, axis=2), axis=1)
    )
    summ.rmsd_to_mean = [float(x) for x in rmsd_to_mean]
    summ.rmsd_median = float(np.median(rmsd_to_mean))
    summ.rmsd_iqr = float(np.percentile(rmsd_to_mean, 75) - np.percentile(rmsd_to_mean, 25))
    summ.rmsd_max = float(np.max(rmsd_to_mean))

    # Confidence dispersion
    summ.ptm_cv = _cv(ensemble.ptm)
    summ.plddt_cv = _cv(ensemble.plddt_mean)

    # Cluster replicates by pairwise RMSD
    labels = np.ones(S, dtype=int)
    if HAS_SCIPY and S >= 2:
        pw = _pairwise_rmsd(aligned, fit_mask)
        condensed = squareform(pw, checks=False)
        if condensed.size and np.any(condensed > 0):
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=cluster_threshold, criterion="distance")
    summ.cluster_assignments = [int(c) for c in labels]

    uniq, counts = np.unique(labels, return_counts=True)
    fractions = counts / counts.sum()
    summ.n_clusters = int(uniq.size)
    summ.dominant_fraction = float(fractions.max())
    summ.cluster_entropy = _entropy_bits(fractions)
    summ.tier = _assign_tier(summ.n_clusters, summ.dominant_fraction, is_collapsed)
    return summ


def per_cluster_confidence(
    condition: str,
    ensemble,
    summary: HeterogeneitySummary,
) -> List[dict]:
    """
    Per-cluster confidence breakdown (plan 2.2): mean/sd pTM, mean pLDDT, n, and
    a dominant-cluster flag.  Returns a list of row dicts.
    """
    rows: List[dict] = []
    if not summary.cluster_assignments:
        return rows
    labels = np.asarray(summary.cluster_assignments, dtype=int)
    ptm = np.asarray(getattr(ensemble, "ptm", np.empty(0)), dtype=np.float64)
    plddt = np.asarray(getattr(ensemble, "plddt_mean", np.empty(0)), dtype=np.float64)

    uniq, counts = np.unique(labels, return_counts=True)
    dominant = uniq[np.argmax(counts)] if uniq.size else 1
    for cid in uniq:
        mask = labels == cid
        c_ptm = ptm[mask] if ptm.size == labels.size else np.empty(0)
        c_pl = plddt[mask] if plddt.size == labels.size else np.empty(0)

        def _m(a):
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if a.size else float("nan")

        def _s(a):
            a = a[np.isfinite(a)]
            return float(np.std(a, ddof=1)) if a.size >= 2 else 0.0

        rows.append({
            "condition": condition,
            "cluster": int(cid),
            "n_replicates_per_cluster": int(mask.sum()),
            "mean_ptm_per_cluster": round(_m(c_ptm), 4),
            "sd_ptm_per_cluster": round(_s(c_ptm), 4),
            "mean_plddt_per_cluster": round(_m(c_pl), 4),
            "cluster_is_dominant": bool(cid == dominant),
        })
    return rows


# ---------------------------------------------------------------------------
# Per-residue dispersion (plan 2.3)
# ---------------------------------------------------------------------------

def bimodality_coefficient(x: np.ndarray) -> float:
    """
    Ellison's sample bimodality coefficient:
        BC = (skew² + 1) / (kurt + 3·(n−1)² / ((n−2)(n−3)))
    BC > 0.555 (uniform reference) suggests bimodality.  NaN for n < 4.
    """
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 4:
        return float("nan")
    m = v.mean()
    s = v.std(ddof=0)
    if s < 1e-12:
        return 0.0
    z = (v - m) / s
    skew = np.mean(z ** 3)
    kurt = np.mean(z ** 4) - 3.0  # excess kurtosis
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


def per_residue_dispersion(disp_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Per-residue SD, IQR and bimodality flag across replicates (plan 2.3).

    disp_matrix: (S, M) per-sample per-residue displacement.
    Returns dict of (M,) arrays: 'sd', 'iqr', 'bimodal'.
    """
    D = np.asarray(disp_matrix, dtype=np.float64)
    if D.ndim == 1:
        D = D[None, :]
    S, M = D.shape
    sd = np.nanstd(D, axis=0, ddof=1) if S >= 2 else np.zeros(M)
    q75 = np.nanpercentile(D, 75, axis=0)
    q25 = np.nanpercentile(D, 25, axis=0)
    iqr = q75 - q25
    bimodal = np.zeros(M, dtype=bool)
    if S >= 4:
        for j in range(M):
            bc = bimodality_coefficient(D[:, j])
            bimodal[j] = bool(np.isfinite(bc) and bc > 0.555)
    return {"sd": sd, "iqr": iqr, "bimodal": bimodal}
