#!/usr/bin/env python3
"""
af3_bench.py  —  AF3 Condition Comparison Pipeline (thin entry point)
=====================================================================

This file is now a backward-compatible shim. The implementation lives in the
``af3bench`` package next to this file:

    af3bench/io.py         discovery, CIF/JSON loading, ensemble loading, writers
    af3bench/model.py      ConditionModel, EnsembleModel, ExperimentStructure
    af3bench/geometry.py   residue-identity-aware Kabsch, RMSD, RMSF, displacement
    af3bench/factors.py    experimental-factor parsing (single source of truth)
    af3bench/stats.py      bootstrap CIs, per-residue significance + FDR
    af3bench/plots/        publication-quality figures (shared style)
    af3bench/pymol.py      PyMOL scene scripts
    af3bench/report.py     findings.json / findings.md
    af3bench/analysis.py   baseline-vs-conditions orchestration
    af3bench/cli.py        command-line interface

The original 2,480-line monolith is preserved as af3_bench_legacy.py.bak.

Scientific upgrade over the original
-------------------------------------
Every structural measurement now carries an ensemble-derived confidence
interval. Per-residue displacement is tested against the baseline's own
intrinsic structural noise (RMSF across AF3 samples), so a shift is only
interpreted as real when its 95% CI clears the noise band (FDR-controlled).

Usage
-----
  python af3_bench.py --models <dir> [--baseline <name>] [--output <dir>]
                      [--chains A,B] [--pymol] [--tm]
                      [--plddt-cutoff 50] [--n-bootstrap 2000] [--fdr 0.05]
                      [--max-samples N] [--dpi 300] [--formats png,pdf]
                      [--no-plots]
"""

import sys
from pathlib import Path

# Make the package importable when this file is run directly as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from af3bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
