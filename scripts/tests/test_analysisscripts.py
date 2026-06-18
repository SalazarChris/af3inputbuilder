"""
Unit tests for the analysisscripts overhaul package.

Covers the new logic added by the implementation plan: two-tier confidence
classification, within-condition heterogeneity, per-residue dispersion /
bimodality, the SMILES (ligand) multiplier in factor parsing, and the
fold-divergence verdict.

Run from the scripts directory:
    pytest tests/test_analysisscripts.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analysisscripts import heterogeneity as het
from analysisscripts.model import EnsembleModel
from analysisscripts.plots import style as style2
from analysisscripts.plots import confidence as conf2


# ---------------------------------------------------------------------------
# Two-tier confidence (plan 0.4)
# ---------------------------------------------------------------------------

def test_classify_tier_artifact():
    # genuinely collapsed fold: very-low protein pLDDT AND high macromolecule PAE
    assert style2.classify_tier(38.0, 28.0) == "likely_artifact"


def test_classify_tier_low_confidence():
    # below AlphaFold "confident" pLDDT floor (70)
    assert style2.classify_tier(62.0, 8.0) == "low_confidence"


def test_classify_tier_ok():
    # confident fold; full-system ipTM is irrelevant to the macromolecule scope
    assert style2.classify_tier(92.0, 4.0) == "ok"
    # confidently folded but heavily solvated (low full-system ipTM) is still ok
    assert style2.classify_tier(95.0, 7.5) == "ok"


def test_classify_tiers_dataframe():
    df = pd.DataFrame({
        "condition": ["good", "collapsed", "weak"],
        "plddt_mean": [92.0, 38.0, 62.0],
        "mean_pae": [4.0, 28.0, 8.0],
    })
    tiers = conf2.classify_tiers(df)
    assert tiers["good"] == "ok"
    assert tiers["collapsed"] == "likely_artifact"
    assert tiers["weak"] == "low_confidence"
    assert conf2.likely_artifacts(df) == {"collapsed"}


# ---------------------------------------------------------------------------
# Heterogeneity tiers (plan 2.1)
# ---------------------------------------------------------------------------

def test_heterogeneity_tier_assignment():
    assert het._assign_tier(1, 1.0) == "low"
    assert het._assign_tier(2, 0.85) == "low"
    assert het._assign_tier(3, 0.72) == "moderate"
    assert het._assign_tier(8, 0.30) == "high"


def test_entropy_bits_monotone():
    # a perfectly even 4-way split has more entropy than a dominated split
    even = het._entropy_bits(np.array([0.25, 0.25, 0.25, 0.25]))
    skew = het._entropy_bits(np.array([0.9, 0.05, 0.03, 0.02]))
    assert even == pytest.approx(2.0)
    assert skew < even


def _make_ensemble(coords, ptm=None, plddt=None):
    e = EnsembleModel(name="t")
    e.ca_coords = coords
    e.ca_plddts = np.full(coords.shape[:2], 90.0)
    e.ca_keys = [("B", i + 1) for i in range(coords.shape[1])]
    if ptm is not None:
        e.ptm = np.asarray(ptm, float)
    if plddt is not None:
        e.plddt_mean = np.asarray(plddt, float)
    return e


def test_summarize_condition_reproducible_is_low_tier():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1, 30, 3))
    # 10 near-identical replicates → one cluster, low heterogeneity
    coords = np.repeat(base, 10, axis=0) + rng.normal(scale=0.05, size=(10, 30, 3))
    e = _make_ensemble(coords, ptm=[0.8] * 10, plddt=[95.0] * 10)
    summ = het.summarize_condition("t", e, plddt_cutoff=50.0, cluster_threshold=3.0)
    assert summ.n_replicates == 10
    assert summ.n_clusters == 1
    assert summ.tier == "low"


def test_summarize_condition_two_states_is_heterogeneous():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(30, 3)) * 5.0
    # second state: a genuine conformational change (a loop region displaced),
    # not a rigid translation (which superposition would remove).
    b = a.copy()
    b[20:] += np.array([10.0, 0.0, 0.0])
    coords = np.concatenate([
        np.repeat(a[None], 5, axis=0) + rng.normal(scale=0.05, size=(5, 30, 3)),
        np.repeat(b[None], 5, axis=0) + rng.normal(scale=0.05, size=(5, 30, 3)),
    ], axis=0)
    e = _make_ensemble(coords, ptm=[0.8] * 10, plddt=[95.0] * 10)
    summ = het.summarize_condition("t", e, plddt_cutoff=50.0, cluster_threshold=3.0)
    assert summ.n_clusters >= 2
    assert summ.dominant_fraction <= 0.6


# ---------------------------------------------------------------------------
# Per-residue dispersion + bimodality (plan 2.3)
# ---------------------------------------------------------------------------

def test_bimodality_coefficient_detects_two_peaks():
    rng = np.random.default_rng(2)
    unimodal = rng.normal(0, 1, size=200)
    bimodal = np.concatenate([rng.normal(-3, 0.3, 100), rng.normal(3, 0.3, 100)])
    bc_uni = het.bimodality_coefficient(unimodal)
    bc_bi = het.bimodality_coefficient(bimodal)
    assert bc_bi > 0.555
    assert bc_bi > bc_uni


def test_per_residue_dispersion_shapes():
    rng = np.random.default_rng(3)
    D = rng.normal(5, 1, size=(20, 8))
    out = het.per_residue_dispersion(D)
    assert out["sd"].shape == (8,)
    assert out["iqr"].shape == (8,)
    assert out["bimodal"].shape == (8,)
    assert np.all(out["iqr"] >= 0)


# ---------------------------------------------------------------------------
# Factor parsing: SMILES multiplier (plan 0.2 / 2.4)
# ---------------------------------------------------------------------------

def test_factors_counts_smiles_multiplier(tmp_path):
    import json
    from analysisscripts import factors
    data = {
        "sequences": [
            {"protein": {"id": "B", "sequence": "ACDEFGHIK"}},
            {"ligand": {"id": ["A", "C"], "ccdCodes": ["NA"]}},
            {"ligand": {"id": ["L", "M"], "ccdCodes": ["CL"]}},
            {"ligand": {"id": [f"W{i}" for i in range(20)], "smiles": "O"}},
        ]
    }
    p = tmp_path / "x_data.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    f = factors.parse_condition_factors(p)
    assert f["n_na"] == 2
    assert f["n_cl"] == 2
    assert f["n_smiles"] == 20  # 20 water/SMILES entities
    assert f["has_real_ligand"] is False


# ---------------------------------------------------------------------------
# Fold-divergence verdict (plan 2.5)
# ---------------------------------------------------------------------------

def test_fold_divergence_verdicts():
    from analysisscripts.analysis import _fold_divergence
    df = pd.DataFrame({
        "condition": ["a", "b", "c"],
        # thresholds: conserved >=0.80, similar (same fold) 0.50-0.80,
        # divergent (different fold) <0.50  (Zhang & Skolnick 2004)
        "tm_score": [0.92, 0.70, 0.42],
        "rmsd": [1.0, 4.0, 9.0],
    })
    rows, warnings = _fold_divergence(df, "baseline", compute_tm=True)
    verdicts = {r["condition_b"]: r["fold_consistency_verdict"] for r in rows}
    assert verdicts["a"] == "conserved"
    assert verdicts["b"] == "similar"
    assert verdicts["c"] == "divergent"
    assert any("c vs baseline" in w for w in warnings)
    # TM=0.52 is the same fold (>0.5), must NOT be flagged divergent
    df2 = pd.DataFrame({"condition": ["d"], "tm_score": [0.52], "rmsd": [6.0]})
    rows2, warn2 = _fold_divergence(df2, "baseline", compute_tm=True)
    assert rows2[0]["fold_consistency_verdict"] == "similar"
    assert warn2 == []


# ---------------------------------------------------------------------------
# Context-matched references (DNA-context stratification)
# ---------------------------------------------------------------------------

def test_select_context_references():
    from types import SimpleNamespace as NS
    from analysisscripts import analysis as A
    conds = ["pou_baseline", "pou_dna", "pou_sep102", "pou_tpo101_dna"]
    struct = NS(
        ion_tier={c: "0x" for c in conds},
        has_dna={"pou_baseline": False, "pou_dna": True,
                 "pou_sep102": False, "pou_tpo101_dna": True},
        ptm_group={"pou_baseline": "none", "pou_dna": "DNA",
                   "pou_sep102": "SEP102", "pou_tpo101_dna": "DNA+TPO101"},
    )
    refs, byctx = A.select_context_references(conds, "pou_baseline", struct)
    assert byctx[False] == "pou_baseline" and byctx[True] == "pou_dna"
    assert refs["pou_tpo101_dna"] == "pou_dna"
    assert refs["pou_sep102"] == "pou_baseline"


def test_context_matched_verdict_ignores_apo_to_dna_shift():
    from types import SimpleNamespace as NS
    import pandas as pd
    from analysisscripts import analysis as A
    from analysisscripts.heterogeneity import HeterogeneitySummary

    def H(rm, dom=1.0):
        s = HeterogeneitySummary(condition="x")
        s.rmsd_median = rm; s.tier = "low"; s.n_clusters = 1
        s.dominant_fraction = dom; s.rmsd_iqr = 1.0
        return s

    # large RMSD vs apo baseline (9 A) but tiny vs DNA context ref (1.4 A)
    df_dist = pd.DataFrame([dict(
        condition="pou_tpo101_dna", rmsd=9.0, rmsd_lo=8.6, tm_score=0.64,
        n_significant=40, context_ref="pou_dna",
        rmsd_vs_context_ref=1.4, rmsd_vs_context_ref_lo=1.2)])
    df_conf = pd.DataFrame([dict(condition="pou_tpo101_dna", delta_ptm=0.3, delta_iptm=0.3)])
    hetero = {"pou_tpo101_dna": H(2.0, dom=0.9), "pou_dna": H(1.5)}
    conditions = {"pou_tpo101_dna": NS(n_nucleic_residues=16)}
    struct = NS(ptm_group={"pou_tpo101_dna": "TPO101"},
                ion_tier={"pou_tpo101_dna": "0x"}, ligand_mult={})
    rows = A._scientific_summary_rows(
        ["pou_tpo101_dna"], conditions, df_dist, df_conf, hetero, struct,
        {"pou_tpo101_dna": "TPO101_DNA"}, set(), {},
        baseline_noise_rmsd=7.5, baseline_name="pou_baseline")
    r = rows[0]
    assert r["structural_shift_basis"] == "context_matched"
    # judged on the 1.4 A perturbation (within DNA-context noise), NOT the 9 A apo->DNA shift
    assert r["structural_shift"] == "no_shift_detected"
