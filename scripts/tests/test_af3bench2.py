"""
Unit tests for the af3bench2 overhaul package.

Covers the new logic added by the implementation plan: two-tier confidence
classification, within-condition heterogeneity, per-residue dispersion /
bimodality, the SMILES (ligand) multiplier in factor parsing, and the
fold-divergence verdict.

Run from the scripts directory:
    pytest tests/test_af3bench2.py -v
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

from af3bench2 import heterogeneity as het
from af3bench2.model import EnsembleModel
from af3bench2.plots import style as style2
from af3bench2.plots import confidence as conf2


# ---------------------------------------------------------------------------
# Two-tier confidence (plan 0.4)
# ---------------------------------------------------------------------------

def test_classify_tier_artifact():
    # collapsed model from the real dataset
    assert style2.classify_tier(0.20, 0.14) == "likely_artifact"


def test_classify_tier_low_confidence():
    assert style2.classify_tier(0.55, 0.35) == "low_confidence"


def test_classify_tier_ok():
    assert style2.classify_tier(0.73, 0.71) == "ok"


def test_classify_tiers_dataframe():
    df = pd.DataFrame({
        "condition": ["good", "collapsed", "weak"],
        "ptm": [0.73, 0.20, 0.55],
        "iptm": [0.71, 0.14, 0.30],
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
    from af3bench2 import factors
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
    from af3bench2.analysis import _fold_divergence
    df = pd.DataFrame({
        "condition": ["a", "b", "c"],
        "tm_score": [0.92, 0.70, 0.52],
        "rmsd": [1.0, 4.0, 7.0],
    })
    rows, warnings = _fold_divergence(df, "baseline", compute_tm=True)
    verdicts = {r["condition_b"]: r["fold_consistency_verdict"] for r in rows}
    assert verdicts["a"] == "conserved"
    assert verdicts["b"] == "similar"
    assert verdicts["c"] == "divergent"
    assert any("c vs baseline" in w for w in warnings)
