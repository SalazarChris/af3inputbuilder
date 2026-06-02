"""
Structural clustering of conditions.

Builds an all-vs-all protein Calpha RMSD matrix between condition
representatives (identity-aware Kabsch), then groups conditions by
hierarchical clustering.  The result drives the clustered heatmap, the
dendrogram, and the cluster-coloured PyMOL scenes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geometry as geom

log = logging.getLogger("af3bench.cluster")

try:
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


def pairwise_rmsd(
    conditions: Dict[str, "object"],
    plddt_cutoff: float = 50.0,
) -> Tuple[List[str], np.ndarray]:
    """
    Symmetric N×N protein Calpha RMSD matrix between condition representatives.

    Returns (names sorted, matrix) with zeros on the diagonal and NaN where an
    alignment could not be formed.
    """
    names = sorted(conditions)
    n = len(names)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            al = geom.align(conditions[names[i]], conditions[names[j]], plddt_cutoff)
            mat[i, j] = mat[j, i] = al["rmsd"]
    return names, mat


def _cut_to_k(Z, n: int, k: int) -> np.ndarray:
    """
    Cut a linkage matrix to exactly ``k`` clusters by stopping the agglomeration
    after ``n - k`` merges.  This is robust to tied merge heights where scipy's
    ``criterion='maxclust'`` can collapse everything into one cluster.
    """
    k = max(1, min(k, n))
    # Union-find over original observations [0..n-1]; cluster ids in Z are
    # >= n for merged nodes.
    parent = list(range(2 * n - 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    n_merges = n - k
    for i in range(n_merges):
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        new_id = n + i
        parent[find(left)] = new_id
        parent[find(right)] = new_id

    roots = {}
    labels = np.empty(n, dtype=int)
    next_id = 1
    for obs in range(n):
        r = find(obs)
        if r not in roots:
            roots[r] = next_id
            next_id += 1
        labels[obs] = roots[r]
    return labels


def cluster_conditions(
    names: List[str],
    matrix: np.ndarray,
    method: str = "average",
    threshold: float = 3.0,
    n_clusters: Optional[int] = None,
) -> dict:
    """
    Hierarchical clustering of the RMSD matrix.

    threshold    distance (Å) cut for criterion='distance' (used when
                 n_clusters is None).
    n_clusters   if given, cut the tree to exactly this many clusters.

    Returns dict:
        labels    {name: cluster_id (1-based)}
        order     list[int] leaf order for heatmap display
        linkage   the linkage matrix Z (or None)
        ordered_names  names in leaf order
        n_clusters     number of clusters formed
    """
    n = len(names)
    if n == 0:
        return {"labels": {}, "order": [], "linkage": None,
                "ordered_names": [], "n_clusters": 0}
    if n == 1:
        return {"labels": {names[0]: 1}, "order": [0], "linkage": None,
                "ordered_names": list(names), "n_clusters": 1}

    if not HAS_SCIPY:
        return {"labels": {nm: 1 for nm in names}, "order": list(range(n)),
                "linkage": None, "ordered_names": list(names), "n_clusters": 1}

    # Fill NaNs with a large finite distance so clustering is well-defined
    filled = matrix.copy()
    finite_max = np.nanmax(filled[np.isfinite(filled)]) if np.any(np.isfinite(filled)) else 1.0
    filled[~np.isfinite(filled)] = finite_max * 1.5
    np.fill_diagonal(filled, 0.0)
    # enforce symmetry
    filled = (filled + filled.T) / 2.0

    condensed = squareform(filled, checks=False)
    Z = linkage(condensed, method=method)
    order = list(leaves_list(Z))

    if n_clusters is not None and n_clusters >= 1:
        flat = _cut_to_k(Z, n, n_clusters)
    else:
        flat = fcluster(Z, t=threshold, criterion="distance")

    labels = {names[i]: int(flat[i]) for i in range(n)}

    # Relabel clusters so the leftmost leaf gets cluster 1 (stable display)
    remap: Dict[int, int] = {}
    next_id = 1
    for leaf in order:
        cid = int(flat[leaf])
        if cid not in remap:
            remap[cid] = next_id
            next_id += 1
    labels = {nm: remap[c] for nm, c in labels.items()}

    return {
        "labels": labels,
        "order": order,
        "linkage": Z,
        "ordered_names": [names[i] for i in order],
        "n_clusters": int(len(set(labels.values()))),
    }


def cluster_summary(
    labels: Dict[str, int],
    conditions: Dict[str, "object"],
) -> List[dict]:
    """Human-readable per-cluster membership with shared factor description."""
    clusters: Dict[int, List[str]] = {}
    for name, cid in labels.items():
        clusters.setdefault(cid, []).append(name)

    out = []
    for cid in sorted(clusters):
        members = sorted(clusters[cid])
        ptms = sorted({
            (c if (c := "+".join(conditions[m].ptm_labels)) else "none")
            for m in members
        })
        has_dna = any(conditions[m].n_nucleic_residues > 0 for m in members)
        out.append({
            "cluster": cid,
            "n_members": len(members),
            "members": members,
            "ptm_groups": ptms,
            "any_dna": has_dna,
        })
    return out
