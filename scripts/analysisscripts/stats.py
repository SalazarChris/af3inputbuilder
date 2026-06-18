"""
Statistics — ensemble confidence intervals and per-residue significance.

The scientific upgrade: every structural measurement is accompanied by an
estimate of AF3's own sampling spread, so a displacement can be judged against
noise rather than reported as a bare point estimate.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("af3bench.stats")

try:
    from scipy import stats as _sp_stats
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _hierarchical_indices(seed_labels, n_boot, rng):
    """Two-stage (cluster) bootstrap index sets respecting seed grouping.

    AF3 replicates are seeds × samples-per-seed; samples sharing a seed are
    correlated, so resampling all rows independently underestimates uncertainty.
    Each iteration resamples seeds with replacement, then samples within each
    chosen seed (standard hierarchical bootstrap).  Returns a list of index
    arrays, or None when seed grouping is unusable (<2 seeds) so the caller
    falls back to the ordinary bootstrap.
    """
    labels = np.asarray(seed_labels)
    seeds = np.unique(labels)
    if seeds.size < 2:
        return None
    members = {int(s): np.where(labels == s)[0] for s in seeds}
    seed_arr = np.array(list(members.keys()))
    out = []
    for _ in range(n_boot):
        chosen = rng.choice(seed_arr, size=seed_arr.size, replace=True)
        idx = np.concatenate([
            rng.choice(members[int(s)], size=members[int(s)].size, replace=True)
            for s in chosen
        ])
        out.append(idx)
    return out


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    statistic: str = "mean",
    rng: Optional[np.random.Generator] = None,
    seed_labels: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap (point, lo, hi) for a 1-D sample.

    statistic: "mean" or "median".
    seed_labels: optional per-value seed id; when given (and >= 2 seeds), a
        hierarchical (cluster) bootstrap respecting the seed grouping is used
        instead of the ordinary i.i.d. bootstrap.
    Returns (point_estimate, ci_lo, ci_hi); NaNs if fewer than 2 finite values.
    """
    v = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(v)
    vf = v[finite_mask]
    func = np.mean if statistic == "mean" else np.median
    if vf.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(func(vf))
    if vf.size < 2:
        return point, point, point

    rng = rng or np.random.default_rng(12345)
    alpha = (1.0 - ci) / 2.0

    # Hierarchical bootstrap when seed grouping is available and aligned.
    hidx = None
    if seed_labels is not None and len(seed_labels) == v.size:
        sl = np.asarray(seed_labels)[finite_mask]
        hidx = _hierarchical_indices(sl, n_boot, rng)
    if hidx is not None:
        boot = np.array([func(vf[ix]) for ix in hidx], dtype=np.float64)
    else:
        idx = rng.integers(0, vf.size, size=(n_boot, vf.size))
        boot = func(vf[idx], axis=1)
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    return point, lo, hi


def column_bootstrap_ci(
    matrix: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    seed_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap mean + CI for every column of an (S, N) matrix by resampling
    rows (samples).  Returns (mean (N,), lo (N,), hi (N,)).

    seed_labels: optional per-row seed id; when given (>= 2 seeds) a
    hierarchical (cluster) bootstrap respecting the seed grouping is used.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] == 0:
        n = M.shape[1] if M.ndim == 2 else 0
        nan = np.full(n, np.nan)
        return nan, nan.copy(), nan.copy()
    S, N = M.shape
    mean = np.nanmean(M, axis=0)
    if S < 2:
        return mean, mean.copy(), mean.copy()
    rng = rng or np.random.default_rng(12345)
    alpha = (1.0 - ci) / 2.0

    hidx = None
    if seed_labels is not None and len(seed_labels) == S:
        hidx = _hierarchical_indices(np.asarray(seed_labels), n_boot, rng)
    if hidx is not None:
        boot = np.stack([np.nanmean(M[ix], axis=0) for ix in hidx], axis=0)
    else:
        idx = rng.integers(0, S, size=(n_boot, S))
        boot = np.nanmean(M[idx], axis=1)            # (n_boot, N)
    lo = np.percentile(boot, 100 * alpha, axis=0)
    hi = np.percentile(boot, 100 * (1 - alpha), axis=0)
    return mean, lo, hi


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction.

    Returns (reject (bool array), qvalues).  NaN p-values are passed through as
    non-significant with NaN q.
    """
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    q = np.full(n, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return reject, q

    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    q_sorted = ranked * m / ranks
    # enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0, 1)
    reject[order] = q[order] <= alpha
    return reject, q


# ---------------------------------------------------------------------------
# Per-residue displacement statistics for a condition vs a baseline ensemble
# ---------------------------------------------------------------------------

def displacement_significance(
    cond_disp: np.ndarray,       # (S_cond, M) per-sample displacement vs baseline mean
    baseline_rmsf: np.ndarray,   # (M,) baseline intrinsic per-residue RMSF
    alpha: float = 0.05,
    ref_plddt: Optional[np.ndarray] = None,   # (M,) baseline per-residue pLDDT
    cond_plddt: Optional[np.ndarray] = None,  # (M,) condition per-residue pLDDT
    plddt_floor: float = 70.0,
    seed_labels: Optional[np.ndarray] = None,  # (S,) seed id per condition sample
) -> Dict[str, np.ndarray]:
    """
    For each residue, test whether the condition's displacement distribution
    (across its samples) exceeds the baseline's intrinsic structural noise.

    A one-sample test compares the condition's per-sample displacement against
    the baseline RMSF for that residue (the noise floor).  P-values are FDR
    corrected across residues.

    Interpretability gate (per-residue confidence).  A displacement is only
    interpretable as *motion* if the residue is confidently placed; otherwise a
    large "displacement" merely reflects that AF3 does not know where to put a
    disordered/low-confidence residue (e.g. a flexible terminus), which is
    placement uncertainty, not conformational change.  When ``ref_plddt`` and
    ``cond_plddt`` are supplied, ``significant`` additionally requires the
    residue's pLDDT to clear ``plddt_floor`` (default 70 — AlphaFold's
    "confident" band floor; Jumper et al. 2021) in *both* the baseline and the
    condition.  The ungated statistical result is preserved separately as
    ``significant_stat`` so the raw observable is never lost.

    Returns dict with keys:
        mean, lo, hi      per-residue displacement mean and 95% CI
        pval, qval        raw and FDR-adjusted p-values
        baseline_rmsf     noise floor used
        significant_stat  q <= alpha AND CI-lo > baseline RMSF (statistics only)
        confident_residue residue confidently placed in both states (or all-True
                          when pLDDT not supplied)
        significant       significant_stat AND confident_residue (interpretable)
    """
    D = np.asarray(cond_disp, dtype=np.float64)
    if D.ndim == 1:
        D = D[None, :]
    S, M = D.shape
    mean, lo, hi = column_bootstrap_ci(D, seed_labels=seed_labels)

    rmsf = np.asarray(baseline_rmsf, dtype=np.float64)
    if rmsf.size < M:
        rmsf = np.pad(rmsf, (0, M - rmsf.size), constant_values=np.nan)
    rmsf = rmsf[:M]

    pval = np.full(M, np.nan)
    if HAS_SCIPY and S >= 3:
        for j in range(M):
            col = D[:, j]
            col = col[np.isfinite(col)]
            floor = rmsf[j] if np.isfinite(rmsf[j]) else 0.0
            if col.size >= 3 and np.ptp(col) > 0:
                # one-sided: is displacement greater than the noise floor?
                t = _sp_stats.ttest_1samp(col, popmean=floor)
                # convert to one-sided (greater)
                if np.isfinite(t.statistic):
                    p_two = t.pvalue
                    pval[j] = p_two / 2 if t.statistic > 0 else 1 - p_two / 2
            elif col.size >= 1:
                pval[j] = 0.0 if np.mean(col) > floor else 1.0

    reject, qval = benjamini_hochberg(pval, alpha=alpha)
    beats_noise = np.greater(lo, rmsf, where=np.isfinite(lo) & np.isfinite(rmsf),
                             out=np.zeros(M, dtype=bool))
    significant_stat = reject & beats_noise

    # Per-residue confidence (interpretability) gate.
    if ref_plddt is not None and cond_plddt is not None:
        rp = np.asarray(ref_plddt, dtype=np.float64)
        cp = np.asarray(cond_plddt, dtype=np.float64)
        confident = (
            np.greater_equal(rp, plddt_floor, where=np.isfinite(rp),
                             out=np.zeros(M, dtype=bool))
            & np.greater_equal(cp, plddt_floor, where=np.isfinite(cp),
                               out=np.zeros(M, dtype=bool))
        )
    else:
        confident = np.ones(M, dtype=bool)

    significant = significant_stat & confident
    return {
        "mean": mean, "lo": lo, "hi": hi,
        "pval": pval, "qval": qval,
        "baseline_rmsf": rmsf,
        "significant_stat": significant_stat,
        "confident_residue": confident,
        "significant": significant,
    }


def summarize_scores(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, sd, n) for a list of seed-level scores; SD uses ddof=1."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), 0.0, 0
    sd = float(np.std(v, ddof=1)) if v.size >= 2 else 0.0
    return float(np.mean(v)), sd, int(v.size)
