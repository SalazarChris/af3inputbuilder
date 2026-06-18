"""
analysisscripts — AF3 condition-comparison analysis pipeline
=============================================================

Enhanced analysis pipeline for AlphaFold 3 predictions with:
- Baseline ensemble characterization (violin plots, clustering)
- Within-condition ensemble heterogeneity metrics
- Confidence-aware statistical analysis with bootstrap CIs
- Multi-cycle analysis framework (11+ cycles implemented)

A modular re-implementation of the original analysis pipeline with
scientific improvements for structural comparison across conditions.

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
"""

__version__ = "2.1.0"

__all__ = ["__version__"]
