#!/usr/bin/env python3
"""
af3_analysis.py  —  AF3 Analysis Pipeline (overhaul build)
===========================================================

Entry point for the ``analysisscripts`` package implementing the enhanced
analysis pipeline with baseline ensemble characterization, multi-cycle
analysis framework, and confidence-aware statistical analysis.

What analysisscripts adds
-------------------------
Correctness (Part 0):
  * baseline-composition warning surfaced in findings + on every baseline plot
  * ligand:salt co-variation confound check + per-point ligand annotation
  * two-tier confidence (low_confidence vs likely_artifact) replacing the single
    "likely failed" flag, propagated to every figure
  * within-condition heterogeneity (n_clusters, dominant fraction, RMSD IQR)
    fed into the PTM grid and per-residue plots

Metrics (Part 2):
  * condition_variance_summary.csv, cluster_confidence_breakdown.csv
  * per-residue SD / IQR / bimodality columns
  * fold-divergence verdict from the TM-score (condition_pairs.csv)
  * short condition labels (label_short)

Figures (Parts 1 & 3):
  * per-residue: adaptive y-scale, low-pLDDT amber overlay, significance rug,
    ΔpLDDT PTM annotation, dashed small-N CI, between-replicate IQR band
  * PTM grid: not-measured vs n.s. encoding, IQR + heterogeneity markers,
    noise-floor boundary, DNA separator
  * concentration-response: PTM/DNA panel split, ligand-multiplier labels,
    rank-swap callout
  * NEW: baseline ensemble violin plots, pTM/ipTM scatter, cluster portraits

Cycle 11 (Baseline Ensemble Characterization):
  * Violin plots for pTM, ipTM, pLDDT, PAE across baseline seeds
  * Baseline structural clustering to detect substates
  * Pairwise RMSD distribution within baseline ensemble
  * Confidence variability metrics (CV for pTM, ipTM, pLDDT)

Usage
-----
  python af3_analysis.py --models <dir> [--baseline <name>] [--output <dir>]
                         [--chains A,B] [--pymol] [--tm]
                         [--plddt-cutoff 50] [--n-bootstrap 2000] [--fdr 0.05]
                         [--max-samples N] [--dpi 300] [--formats png,pdf]
                         [--cluster-threshold 3.0] [--no-plots]
"""

import sys
from pathlib import Path

# Make the package importable when this file is run directly as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from analysisscripts.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
