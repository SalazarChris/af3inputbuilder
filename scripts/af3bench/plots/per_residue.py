"""
Per-residue displacement figures — the core "where does it move, and is it
real?" visual.

Each figure has three stacked panels sharing the residue-number x-axis:

  1. Displacement (Angstrom): condition-vs-baseline mean with a 95% CI ribbon,
     plus the baseline ensemble RMSF as a noise envelope.  A residue only
     "counts" when its displacement CI clears the noise.
  2. Significance track: a thin strip colouring residues whose displacement
     significantly exceeds noise (FDR-controlled).
  3. pLDDT: reference vs condition per-residue confidence.

PTM sites are marked with a labelled vertical line on every panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from . import style


def _ptm_positions(ptm_labels: List[str]) -> List[int]:
    out = []
    for lab in ptm_labels or []:
        digits = "".join(ch for ch in lab if ch.isdigit())
        if digits:
            out.append(int(digits))
    return out


def plot_profile(
    profile: dict,
    plots_dir: Path,
    ref_name: str,
) -> List[Path]:
    """
    Render one per-residue figure from a profile dict with keys:

        name            condition name
        res_numbers     (M,) residue numbers (x-axis)
        chain_ids       (M,) chain id per residue
        disp_mean       (M,) mean displacement (A)
        disp_lo,disp_hi (M,) 95% CI
        baseline_rmsf   (M,) baseline noise envelope (A)
        significant     (M,) bool
        ref_plddt       (M,) reference per-residue pLDDT
        cond_plddt      (M,) condition per-residue pLDDT
        ptm_labels      list[str]
        y_ceiling       float (shared across panels) or None
        n_samples       int (condition ensemble size)
        n_samples_base  int (baseline ensemble size)
    """
    name = profile["name"]
    x = np.asarray(profile["res_numbers"], dtype=float)
    disp = np.asarray(profile["disp_mean"], dtype=float)
    lo = np.asarray(profile.get("disp_lo", disp), dtype=float)
    hi = np.asarray(profile.get("disp_hi", disp), dtype=float)
    rmsf = np.asarray(profile.get("baseline_rmsf", np.full_like(disp, np.nan)), dtype=float)
    sig = np.asarray(profile.get("significant", np.zeros_like(disp, dtype=bool)), dtype=bool)
    ref_pl = np.asarray(profile["ref_plddt"], dtype=float)
    cond_pl = np.asarray(profile["cond_plddt"], dtype=float)
    chain_ids = profile.get("chain_ids", [])

    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[3.0, 0.25, 1.6],
        hspace=0.08,
    )
    ax_disp = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[1], sharex=ax_disp)
    ax_pl = fig.add_subplot(gs[2], sharex=ax_disp)

    n_c = profile.get("n_samples", 0)
    n_b = profile.get("n_samples_base", 0)
    fig.suptitle(
        f"Per-residue displacement vs {ref_name}: {name}",
        fontweight="bold", fontsize=11,
    )

    # --- Panel 1: displacement + CI + noise envelope ---
    if np.any(np.isfinite(rmsf)):
        ax_disp.fill_between(
            x, 0, rmsf, color=style.C_NOISE, alpha=0.35,
            label=f"baseline noise (RMSF, n={n_b})", zorder=1,
        )
    if np.any(hi > lo):
        ax_disp.fill_between(
            x, lo, hi, color=style.C_DISP, alpha=0.25,
            label=f"95% CI (n={n_c})", zorder=2,
        )
    ax_disp.plot(x, disp, color=style.C_DISP, linewidth=1.0, zorder=3,
                 label="mean displacement")

    # mark significant residues along the curve
    if sig.any():
        ax_disp.scatter(
            x[sig], disp[sig], s=10, color=style.C_SIG, zorder=4,
            label="exceeds noise (FDR<0.05)",
        )

    ax_disp.set_ylabel("Cα displacement (Å)")
    y_ceiling = profile.get("y_ceiling")
    if y_ceiling and np.isfinite(y_ceiling):
        ax_disp.set_ylim(0, y_ceiling)
    else:
        ax_disp.set_ylim(bottom=0)
    ax_disp.legend(loc="upper right", ncol=2)
    plt.setp(ax_disp.get_xticklabels(), visible=False)

    # --- Panel 2: significance track ---
    ax_sig.set_ylim(0, 1)
    ax_sig.set_yticks([0.5])
    ax_sig.set_yticklabels(["sig."], fontsize=7)
    ax_sig.grid(False)
    if len(x) > 1:
        width = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    else:
        width = 1.0
    for xi, s in zip(x, sig):
        if s:
            ax_sig.axvspan(xi - width / 2, xi + width / 2, color=style.C_SIG, alpha=0.8)
    plt.setp(ax_sig.get_xticklabels(), visible=False)
    for spine in ("left",):
        ax_sig.spines[spine].set_visible(False)

    # --- Panel 3: pLDDT ---
    ax_pl.plot(x, ref_pl, color=style.C_REF, linewidth=0.9, label="baseline pLDDT")
    ax_pl.plot(x, cond_pl, color=style.C_COND, linewidth=0.9, label="condition pLDDT")
    ax_pl.set_ylim(0, 100)
    ax_pl.set_ylabel("pLDDT")
    ax_pl.set_xlabel("Residue number")
    ax_pl.legend(loc="lower right", ncol=2)

    # --- chain boundaries ---
    if len(chain_ids) == len(x):
        for i in range(1, len(chain_ids)):
            if chain_ids[i] != chain_ids[i - 1]:
                xb = (x[i] + x[i - 1]) / 2
                for ax in (ax_disp, ax_sig, ax_pl):
                    ax.axvline(xb, color="gray", linestyle=":", linewidth=0.7)

    # --- PTM site markers ---
    res_set = set(int(v) for v in x if np.isfinite(v))
    for pos in _ptm_positions(profile.get("ptm_labels", [])):
        if pos in res_set:
            for ax in (ax_disp, ax_pl):
                ax.axvline(pos, color=style.C_PTM, linestyle="--", linewidth=1.3, zorder=5)
            ax_disp.text(
                pos, ax_disp.get_ylim()[1] * 0.97, f"PTM {pos}",
                color=style.C_PTM, fontsize=8, ha="center", va="top",
                rotation=90,
            )

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]
    return style.save(fig, plots_dir, f"per_residue_{safe}")
