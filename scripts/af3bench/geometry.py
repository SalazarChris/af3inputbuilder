"""
Geometry — residue-identity-aware superposition and distance metrics.

The original pipeline aligned structures by array position (``[:min(len)]``),
implicitly assuming residue *i* in one model equals residue *i* in another.
This module instead matches residues by their ``(chain_id, residue_number)``
key, so alignment is correct even when chain order differs or DNA shifts the
indexing.

All superposition is performed on protein Calpha atoms only.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("af3bench.geometry")

try:
    import tmtools
    HAS_TMTOOLS = True
except ImportError:  # pragma: no cover
    HAS_TMTOOLS = False


ResidueKey = Tuple[str, int]


# ---------------------------------------------------------------------------
# Core Kabsch
# ---------------------------------------------------------------------------

def kabsch(ref: np.ndarray, mob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Kabsch superposition of ``mob`` onto ``ref``.  Both (N, 3), already paired
    row-for-row (same N).

    Returns (R_mat (3,3), t_vec (3,), rmsd).  Apply as ``mob @ R_mat.T + t_vec``.
    """
    if len(ref) != len(mob):
        raise ValueError(f"kabsch requires equal-length inputs, got {len(ref)} vs {len(mob)}")
    if len(ref) == 0:
        return np.eye(3), np.zeros(3), float("nan")

    r_mean, m_mean = ref.mean(0), mob.mean(0)
    rc, mc = ref - r_mean, mob - m_mean
    H = mc.T @ rc
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt = Vt.copy()
        Vt[-1] *= -1
        R_mat = Vt.T @ U.T
    t_vec = r_mean - m_mean @ R_mat.T
    rmsd = float(np.sqrt(np.mean(np.sum((rc - mc @ R_mat.T) ** 2, axis=1))))
    return R_mat, t_vec, rmsd


# ---------------------------------------------------------------------------
# Residue-identity matching
# ---------------------------------------------------------------------------

def match_indices(
    keys_ref: Sequence[ResidueKey],
    keys_mob: Sequence[ResidueKey],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return parallel index arrays into ref and mob for residues present in both,
    matched on (chain_id, residue_number) and ordered by the reference.
    """
    mob_lookup: Dict[ResidueKey, int] = {}
    for j, k in enumerate(keys_mob):
        mob_lookup.setdefault(k, j)  # first occurrence wins

    ref_idx: List[int] = []
    mob_idx: List[int] = []
    for i, k in enumerate(keys_ref):
        j = mob_lookup.get(k)
        if j is not None:
            ref_idx.append(i)
            mob_idx.append(j)
    return np.asarray(ref_idx, dtype=int), np.asarray(mob_idx, dtype=int)


# ---------------------------------------------------------------------------
# Alignment + RMSD between two ConditionModels
# ---------------------------------------------------------------------------

def align(
    ref,
    mob,
    plddt_cutoff: float = 50.0,
) -> dict:
    """
    Align ``mob`` onto ``ref`` using shared, high-confidence protein Calpha atoms.

    Returns a dict:
        R, t            transform mapping mob -> ref frame
        rmsd            RMSD over the fitting set
        n_fit           number of residues used to fit
        n_shared        number of residues shared (chain,resnum)
        ref_idx,mob_idx parallel index arrays over the *shared* set (ordered by ref)
    """
    ref_idx, mob_idx = match_indices(ref.ca_keys, mob.ca_keys)
    n_shared = len(ref_idx)
    if n_shared < 3:
        return {
            "R": np.eye(3), "t": np.zeros(3), "rmsd": float("nan"),
            "n_fit": 0, "n_shared": n_shared,
            "ref_idx": ref_idx, "mob_idx": mob_idx,
        }

    r_coords = ref.ca_coords[ref_idx]
    m_coords = mob.ca_coords[mob_idx]
    r_pl = ref.ca_plddts[ref_idx]
    m_pl = mob.ca_plddts[mob_idx]

    mask = (r_pl > plddt_cutoff) & (m_pl > plddt_cutoff)
    if mask.sum() < 3:
        mask = np.ones(n_shared, dtype=bool)

    R_mat, t_vec, rmsd = kabsch(r_coords[mask], m_coords[mask])
    return {
        "R": R_mat, "t": t_vec, "rmsd": rmsd,
        "n_fit": int(mask.sum()), "n_shared": n_shared,
        "ref_idx": ref_idx, "mob_idx": mob_idx,
    }


def per_residue_displacement(
    ref,
    mob,
    R_mat: np.ndarray,
    t_vec: np.ndarray,
    ref_idx: np.ndarray,
    mob_idx: np.ndarray,
) -> Tuple[np.ndarray, List[ResidueKey]]:
    """
    Per-residue protein Calpha displacement (Angstrom) over the shared residue
    set, after applying the protein superposition to ``mob``.

    Returns (displacement (M,), residue keys length M ordered by ref).
    """
    if len(ref_idx) == 0:
        return np.empty(0), []
    r = ref.ca_coords[ref_idx]
    m = mob.ca_coords[mob_idx] @ R_mat.T + t_vec
    disp = np.sqrt(np.sum((r - m) ** 2, axis=1))
    keys = [ref.ca_keys[i] for i in ref_idx]
    return disp, keys


def nucleic_displacement(
    ref,
    mob,
    R_mat: np.ndarray,
    t_vec: np.ndarray,
) -> Tuple[np.ndarray, List[ResidueKey]]:
    """Nucleic C4' displacement over shared (chain,resnum) keys, protein frame."""
    if ref.n_nucleic_residues == 0 or mob.n_nucleic_residues == 0:
        return np.empty(0), []
    ref_idx, mob_idx = match_indices(ref.na_keys, mob.na_keys)
    if len(ref_idx) == 0:
        return np.empty(0), []
    r = ref.na_coords[ref_idx]
    m = mob.na_coords[mob_idx] @ R_mat.T + t_vec
    disp = np.sqrt(np.sum((r - m) ** 2, axis=1))
    keys = [ref.na_keys[i] for i in ref_idx]
    return disp, keys


# ---------------------------------------------------------------------------
# Ensemble RMSF (intrinsic structural noise)
# ---------------------------------------------------------------------------

def superpose_stack_to_mean(
    coords: np.ndarray,
    plddts: Optional[np.ndarray] = None,
    plddt_cutoff: float = 50.0,
    n_iter: int = 3,
) -> np.ndarray:
    """
    Iteratively superpose every frame of ``coords`` (S, N, 3) onto the running
    mean structure.  Returns the superposed stack (S, N, 3).

    A pLDDT mask (S, N) restricts the fitting atoms to confident residues if
    provided; displacement/RMSF is still reported over all N residues.
    """
    if coords.ndim != 3 or coords.shape[0] < 1:
        return coords
    S, N, _ = coords.shape
    aligned = coords.astype(np.float64).copy()

    # Fitting mask: residues confident in *all* frames
    if plddts is not None and plddts.shape == (S, N):
        fit_mask = np.all(plddts > plddt_cutoff, axis=0)
        if fit_mask.sum() < 3:
            fit_mask = np.ones(N, dtype=bool)
    else:
        fit_mask = np.ones(N, dtype=bool)

    ref = aligned[0]
    for _ in range(max(1, n_iter)):
        for s in range(S):
            R_mat, t_vec, _ = kabsch(ref[fit_mask], aligned[s][fit_mask])
            aligned[s] = aligned[s] @ R_mat.T + t_vec
        ref = aligned.mean(axis=0)
    return aligned


def ensemble_rmsf(
    coords: np.ndarray,
    plddts: Optional[np.ndarray] = None,
    plddt_cutoff: float = 50.0,
) -> np.ndarray:
    """
    Per-residue RMSF (Angstrom) of an ensemble (S, N, 3): the RMS deviation of
    each residue's Calpha about the ensemble mean after internal superposition.

    Returns (N,) array, or empty if fewer than 2 frames.
    """
    if coords.ndim != 3 or coords.shape[0] < 2 or coords.shape[1] == 0:
        return np.empty(0)
    aligned = superpose_stack_to_mean(coords, plddts, plddt_cutoff)
    mean = aligned.mean(axis=0)                       # (N, 3)
    sq = np.sum((aligned - mean) ** 2, axis=2)        # (S, N) squared distance
    return np.sqrt(np.mean(sq, axis=0))               # (N,) RMS over samples


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------

def tm_score(ref, mob) -> Tuple[float, float]:
    """
    TM-score (normalised to ref, normalised to mob) on shared protein Calpha.

    Uses real one-letter sequence when available on the models
    (``seq1``/``seq_letters``), otherwise falls back to poly-alanine.
    """
    if not HAS_TMTOOLS:
        return float("nan"), float("nan")
    ref_idx, mob_idx = match_indices(ref.ca_keys, mob.ca_keys)
    k = len(ref_idx)
    if k < 5:
        return float("nan"), float("nan")
    try:
        c1 = np.asarray(ref.ca_coords[ref_idx], dtype=np.float64)
        c2 = np.asarray(mob.ca_coords[mob_idx], dtype=np.float64)
        seq1 = _seq_for(ref, ref_idx, k)
        seq2 = _seq_for(mob, mob_idx, k)
        result = tmtools.tm_align(c1, c2, seq1, seq2)
        return float(result.tm_norm_chain1), float(result.tm_norm_chain2)
    except Exception as exc:  # noqa: BLE001
        log.warning("TM-score failed for %s vs %s: %s", mob.name, ref.name, exc)
        return float("nan"), float("nan")


def _seq_for(model, idx: np.ndarray, k: int) -> str:
    letters = getattr(model, "ca_seq_letters", None)
    if letters and len(letters) >= max(idx, default=-1) + 1:
        try:
            return "".join(letters[i] for i in idx)
        except Exception:  # noqa: BLE001
            pass
    return "A" * k
