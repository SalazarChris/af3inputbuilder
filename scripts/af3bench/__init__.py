"""
af3bench — AF3 condition-comparison analysis package
====================================================

A modular re-implementation of the original monolithic ``af3_bench.py``.

The package separates concerns into:

  io        discovery, JSON/CIF loading, ensemble loading, table writers
  model     data containers (ConditionModel, EnsembleModel, ExperimentStructure)
  geometry  residue-identity-aware Kabsch alignment, RMSD, RMSF, displacement
  factors   single source of truth for experimental factor parsing
  stats     ensemble statistics, bootstrap CIs, per-residue significance + FDR
  plots     publication-quality figures (shared style)
  pymol     PyMOL scene scripts
  report    machine- and human-readable findings
  cli       command-line entry point

The scientific upgrade over the original is the use of the full AF3 sample
ensemble (seed-*_sample-*) to put confidence intervals on every structural
measurement, so a per-residue displacement can be judged against AF3's own
sampling noise.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
