"""Visualization module for analysisscripts analysis results.

Provides publication-quality figures with improved visual design principles:
- Perceptually uniform colormaps
- Colorblind-friendly palettes
- Clear hierarchy and typography
- Proper handling of large datasets (>80 conditions)

Modules:
    confidence  - pTM/ipTM/pLDDT/PAE figures
    distances   - RMSD/TM-score visualizations
    factorial   - PTM × concentration effect grids
    per_residue - Residue-level displacement profiles
    clusters    - Structural clustering visualizations
    heterogeneity - Within-condition heterogeneity plots
    style       - Shared styling configuration
    summary     - Summary visualizations (new: cluster heatmap, PCA)
    quality     - Quality dashboard (new: confidence metrics)
"""

from . import confidence, distances, factorial, per_residue, clusters, heterogeneity, style, summary, quality

__all__ = ['confidence', 'distances', 'factorial', 'per_residue', 'clusters', 'heterogeneity', 'style', 'summary', 'quality']
