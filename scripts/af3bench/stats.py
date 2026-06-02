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

def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    statistic: str = "mean",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap (point, lo, hi) for a 1-D sample.

    statistic: "mean" or "median".
    Returns (point_estimate, ci_lo, ci_hi); NaNs if fewer than 2 finite values.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    func = np.mean if statistic == "mean" else np.median
    point = float(func(v))
    if v.size < 2:
        return point, point, point

    rng = rng or np.random.default_rng(12345)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = func(v[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    return point, lo, hi


def column_bootstrap_ci(
    matrix: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap mean + CI for every column of an (S, N) matrix by resampling
    rows (samples).  Returns (mean (N,), lo (N,), hi (N,)).
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
    idx = rng.integers(0, S, size=(n_boot, S))
    boot = np.nanmean(M[idx], axis=1)            # (n_boot, N)
    alpha = (1.0 - ci) / 2.0
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
) -> Dict[str, np.ndarray]:
    """
    For each residue, test whether the condition's displacement distribution
    (across its samples) exceeds the baseline's intrinsic structural noise.

    A one-sample test compares the condition's per-sample displacement against
    the baseline RMSF for that residue (the noise floor).  P-values are FDR
    corrected across residues.

    Returns dict with keys:
        mean, lo, hi   per-residue displacement mean and 95% CI
        pval, qval     raw and FDR-adjusted p-values
        significant    bool array (q <= alpha AND CI-lo > baseline RMSF)
    """
    D = np.asarray(cond_disp, dtype=np.float64)
    if D.ndim == 1:
        D = D[None, :]
    S, M = D.shape
    mean, lo, hi = column_bootstrap_ci(D)

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
    significant = reject & beats_noise
    return {
        "mean": mean, "lo": lo, "hi": hi,
        "pval": pval, "qval": qval,
        "baseline_rmsf": rmsf,
        "significant": significant,
    }


def summarize_scores(values: List[float]) -> Tuple[float, float, int]:
    """Return (mean, sd, n) for a list of seed-level scores; SD uses ddof=1."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), 0.0, 0
    sd = float(np.std(v, ddof=1)) if v.size >= 2 else 0.0
    return float(np.mean(v)), sd, int(v.size)
