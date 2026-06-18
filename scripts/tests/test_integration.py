"""End-to-end integration test for analysisscripts.analysis.run.

Builds synthetic ConditionModel / EnsembleModel objects (bypassing CIF/JSON
loading) and runs the full pipeline with plots + PyMOL, verifying it completes
and emits the expected tables/columns after all the scope/noise/baseline fixes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

from analysisscripts import analysis as A
from analysisscripts.model import ConditionModel, EnsembleModel

N = 20          # protein residues
S = 10          # replicates (2 seeds x 5 samples)
RNG = np.random.default_rng(0)


def _cond(name, dna=False, ptm=None, bend=0.0, ptm_val=0.85, plddt=95.0):
    c = ConditionModel(name, Path(f"{name}/model.cif"))
    # helix-like backbone; a per-condition hinge bend on the C-terminal half
    # creates real internal conformational differences (survive Kabsch).
    idx = np.arange(N, dtype=float)
    base = np.stack([idx, 2.0 * np.sin(idx * 0.5), 2.0 * np.cos(idx * 0.5)], axis=1)
    half = N // 2
    base[half:, 2] += bend * (idx[half:] - half)
    c.ca_coords = base
    c.ca_plddts = np.full(N, plddt)
    c.ca_chain_ids = ["A"] * N
    c.ca_res_indices = list(range(1, N + 1))
    c.ptm = ptm_val
    c.iptm = ptm_val - 0.02
    c.ranking_score = 0.8
    c.protein_chain_ids_from_json = ["A"]
    token_chains = ["A"] * N
    if dna:
        c.na_coords = np.array([[float(i), 5.0, 0.0] for i in range(4)])
        c.na_plddts = np.full(4, 90.0)
        c.na_chain_ids = ["B"] * 4
        c.na_res_indices = list(range(1, 5))
        c.nucleic_chain_ids_from_json = ["B"]
        token_chains = token_chains + ["B"] * 4
    c.ptm_labels = [ptm] if ptm else []
    c.n_na = c.n_cl = c.n_water = 0
    T = len(token_chains)
    c.pae_matrix = np.full((T, T), 4.0)
    c.token_chain_ids = token_chains
    return c


def _ens(name, cond, spread=0.3):
    e = EnsembleModel(name=name)
    coords = np.stack([cond.ca_coords + RNG.normal(0, spread, size=(N, 3))
                       for _ in range(S)], axis=0)
    e.ca_coords = coords
    e.ca_plddts = np.full((S, N), 95.0)
    e.ca_keys = list(zip(cond.ca_chain_ids, cond.ca_res_indices))
    e.ptm = np.full(S, cond.ptm)
    e.iptm = np.full(S, cond.iptm)
    e.plddt_mean = np.full(S, 95.0)
    e.sample_paths = [Path(f"{name}/seed-{s // 5}_sample-{s % 5}/model.cif") for s in range(S)]
    return e


def test_run_end_to_end(tmp_path, monkeypatch):
    # apo baseline, DNA (shifted = apo->DNA transition), and PTM variants
    specs = {
        "pou_baseline": dict(dna=False, ptm=None, bend=0.0),
        "pou_sep102": dict(dna=False, ptm="SEP102", bend=0.15),
        "pou_dna": dict(dna=True, ptm=None, bend=1.2),       # large apo->DNA hinge
        "pou_sep102_dna": dict(dna=True, ptm="SEP102", bend=1.28),  # DNA + small PTM perturbation
    }
    conditions = {n: _cond(n, **kw) for n, kw in specs.items()}
    ensembles = {n: _ens(n, c) for n, c in conditions.items()}

    monkeypatch.setattr(A, "discover_conditions", lambda md, cf: (conditions, []))
    monkeypatch.setattr(A, "load_ensemble",
                        lambda d, eff, reference_keys=None, max_samples=None: ensembles[Path(d).name])

    out = tmp_path / "out"
    A.run(
        models_dir=tmp_path / "models", output_dir=out, baseline_arg="pou_baseline",
        chain_filter=None, compute_tm=False, pymol=True, plddt_cutoff=50.0,
        n_bootstrap=200, fdr_alpha=0.05, max_samples=None, make_plots=True,
    )

    # tables produced
    dist = pd.read_csv(out / "tables" / "structural_distances.csv")
    assert {"rmsd", "context_ref", "rmsd_vs_context_ref"}.issubset(dist.columns)

    # context-matched RMSD for the DNA+PTM condition is much smaller than its
    # RMSD vs the apo baseline (the apo->DNA transition is removed).
    row = dist.set_index("condition").loc["pou_sep102_dna"]
    assert row["context_ref"] == "pou_dna"
    assert row["rmsd_vs_context_ref"] < row["rmsd"]

    # per-residue significance is now measured vs the context ref, so the DNA+PTM
    # condition's significant-residue count (the PTM perturbation on the DNA-bound
    # state) is small — it does NOT re-flag the whole DNA-binding interface.
    nsig = dist.set_index("condition")["n_significant"]
    assert nsig["pou_sep102_dna"] <= 5

    # findings + manifest + at least one plot + pymol scene exist
    assert (out / "findings.md").exists()
    assert (out / "run_manifest.json").exists()
    assert (out / "plots" / "structural_distances.png").exists()
    assert (out / "pymol" / "06_clusters.pml").exists()
