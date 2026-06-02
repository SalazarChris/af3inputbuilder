"""
Command-line interface for af3bench.

Backward compatible with the original af3_bench.py invocation
(``--models``, ``--baseline``, ``--output``, ``--chains``, ``--pymol``,
``--tm``) and adds rigor/visual controls.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from . import __version__
from . import analysis
from .geometry import HAS_TMTOOLS
from .plots import style

log = logging.getLogger("af3bench")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="af3bench2",
        description="AF3 condition-comparison pipeline (ensemble-aware, overhaul build).",
    )
    p.add_argument("--models", "-m", required=True, type=Path,
                   help="Root folder; each subdirectory is one AF3 condition.")
    p.add_argument("--baseline", "-b", default=None,
                   help="Baseline condition name (auto-detected if omitted).")
    p.add_argument("--output", "-o", default=Path("af3_results"), type=Path,
                   help="Output directory (default: af3_results).")
    p.add_argument("--chains", default=None,
                   help="Restrict alignment to these protein chain IDs, e.g. A,B.")
    p.add_argument("--pymol", action="store_true", help="Generate PyMOL .pml scripts.")
    p.add_argument("--tm", action="store_true",
                   help="Compute TM-score (requires tmtools).")

    p.add_argument("--plddt-cutoff", type=float, default=50.0,
                   help="pLDDT cutoff for fitting atoms (default: 50).")
    p.add_argument("--n-bootstrap", type=int, default=2000,
                   help="Bootstrap resamples for CIs (default: 2000).")
    p.add_argument("--fdr", type=float, default=0.05,
                   help="FDR alpha for per-residue significance (default: 0.05).")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap the number of ensemble samples per condition.")
    p.add_argument("--n-clusters", type=int, default=None,
                   help="Cut the structural dendrogram into exactly this many clusters "
                        "(default: distance-based cut via --cluster-threshold).")
    p.add_argument("--cluster-threshold", type=float, default=3.0,
                   help="RMSD (Å) cut height for structural clustering (default: 3.0).")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    p.add_argument("--formats", default="png",
                   help="Comma-separated figure formats, e.g. png,pdf (default: png).")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument("--version", action="version", version=f"af3bench2 {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S", level=logging.INFO,
    )
    args = build_parser().parse_args(argv)

    chain_filter = [c.strip() for c in args.chains.split(",")] if args.chains else None
    compute_tm = args.tm and HAS_TMTOOLS
    if args.tm and not HAS_TMTOOLS:
        log.warning("--tm requested but tmtools not installed; skipping TM-score.")

    style.configure(dpi=args.dpi, formats=[f.strip() for f in args.formats.split(",") if f.strip()])

    log.info("=== af3bench2 %s ===", __version__)
    log.info("Models: %s", args.models.resolve())
    log.info("Output: %s", args.output.resolve())

    analysis.run(
        models_dir=args.models.resolve(),
        output_dir=args.output.resolve(),
        baseline_arg=args.baseline,
        chain_filter=chain_filter,
        compute_tm=compute_tm,
        pymol=args.pymol,
        plddt_cutoff=args.plddt_cutoff,
        n_bootstrap=args.n_bootstrap,
        fdr_alpha=args.fdr,
        max_samples=args.max_samples,
        make_plots=not args.no_plots,
        n_clusters=args.n_clusters,
        cluster_threshold=args.cluster_threshold,
    )


if __name__ == "__main__":
    main()
