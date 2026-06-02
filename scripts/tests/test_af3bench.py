"""
Unit + smoke tests for the af3bench package.

Run from the scripts directory:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the package importable
_SCRIPTS = Path(__file__).resolve().parent.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from af3bench import geometry as geom
from af3bench import stats as st
from af3bench import factors
from af3bench.model import ConditionModel, EnsembleModel


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_kabsch_identity():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(20, 3))
    R, t, rmsd = geom.kabsch(coords, coords)
    assert rmsd == pytest.approx(0.0, abs=1e-9)


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(30, 3))
    # rotate + translate
    theta = 0.7
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    mob = ref @ Rz.T + np.array([5.0, -2.0, 1.0])
    R, t, rmsd = geom.kabsch(ref, mob)
    recovered = mob @ R.T + t
    assert np.allclose(recovered, ref, atol=1e-6)
    assert rmsd == pytest.approx(0.0, abs=1e-6)


def test_match_indices_by_identity():
    keys_ref = [("A", 1), ("A", 2), ("A", 3), ("B", 1)]
    keys_mob = [("B", 1), ("A", 3), ("A", 1), ("A", 2)]
    ri, mi = geom.match_indices(keys_ref, keys_mob)
    # ordered by ref
    assert [keys_ref[i] for i in ri] == keys_ref
    assert [keys_mob[j] for j in mi] == keys_ref


def test_match_indices_partial_overlap():
    keys_ref = [("A", 1), ("A", 2), ("A", 99)]
    keys_mob = [("A", 1), ("A", 2)]
    ri, mi = geom.match_indices(keys_ref, keys_mob)
    assert len(ri) == 2
    assert [keys_ref[i] for i in ri] == [("A", 1), ("A", 2)]


def test_ensemble_rmsf_zero_for_identical_frames():
    coords = np.tile(np.random.default_rng(2).normal(size=(10, 3)), (5, 1, 1))
    rmsf = geom.ensemble_rmsf(coords)
    assert rmsf.shape == (10,)
    assert np.allclose(rmsf, 0.0, atol=1e-6)


def test_ensemble_rmsf_positive_for_jittered():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(10, 3))
    frames = np.stack([base + rng.normal(scale=0.3, size=(10, 3)) for _ in range(8)])
    rmsf = geom.ensemble_rmsf(frames)
    assert rmsf.shape == (10,)
    assert np.all(rmsf > 0)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(4)
    vals = rng.normal(loc=5.0, scale=1.0, size=200)
    m, lo, hi = st.bootstrap_ci(vals, n_boot=500, rng=rng)
    assert lo < m < hi
    assert m == pytest.approx(5.0, abs=0.3)


def test_bootstrap_ci_single_value():
    m, lo, hi = st.bootstrap_ci(np.array([3.0]))
    assert m == lo == hi == 3.0


def test_benjamini_hochberg_basic():
    p = np.array([0.001, 0.01, 0.2, 0.8])
    reject, q = st.benjamini_hochberg(p, alpha=0.05)
    assert reject[0]
    assert not reject[3]
    assert np.all((q[np.isfinite(q)] >= 0) & (q[np.isfinite(q)] <= 1))


def test_benjamini_hochberg_handles_nan():
    p = np.array([0.001, np.nan, 0.5])
    reject, q = st.benjamini_hochberg(p)
    assert reject[0]
    assert not reject[1]


def test_column_bootstrap_ci_shapes():
    M = np.random.default_rng(5).normal(size=(6, 12))
    mean, lo, hi = st.column_bootstrap_ci(M, n_boot=200)
    assert mean.shape == lo.shape == hi.shape == (12,)
    assert np.all(lo <= mean + 1e-9)
    assert np.all(hi >= mean - 1e-9)


def test_displacement_significance_flags_real_motion():
    # one residue moves a lot, others stay within noise
    rng = np.random.default_rng(6)
    S, M = 10, 5
    disp = np.abs(rng.normal(scale=0.2, size=(S, M)))
    disp[:, 2] += 5.0  # residue index 2 moves clearly
    rmsf = np.full(M, 0.3)
    res = st.displacement_significance(disp, rmsf, alpha=0.05)
    assert res["significant"][2]
    assert not res["significant"][0]


# ---------------------------------------------------------------------------
# Factors (the ccdCodes fix + ion/water separation)
# ---------------------------------------------------------------------------

def _write_data_json(tmp_path, sequences):
    import json
    p = tmp_path / "x_data.json"
    p.write_text(json.dumps({"sequences": sequences}), encoding="utf-8")
    return p


def test_factors_reads_ccdcodes_plural(tmp_path):
    seqs = [
        {"protein": {"id": "B", "sequence": "ACDEFG", "modifications": []}},
        {"ligand": {"id": ["A"], "ccdCodes": ["NA"]}},
        {"ligand": {"id": ["C"], "ccdCodes": ["CL"]}},
        {"ligand": {"id": ["D", "E"], "smiles": "O"}},
    ]
    f = factors.parse_condition_factors(_write_data_json(tmp_path, seqs))
    assert f["n_na"] == 1
    assert f["n_cl"] == 1
    assert f["n_water"] == 2
    assert not f["has_real_ligand"]


def test_factors_separates_water_from_ions(tmp_path):
    seqs = [
        {"protein": {"id": "B", "sequence": "ACDEFG", "modifications": []}},
        {"ligand": {"id": ["A", "C", "D"], "ccdCodes": ["NA"]}},
        {"ligand": {"id": ["x" + str(i) for i in range(30)], "smiles": "O"}},
    ]
    f = factors.parse_condition_factors(_write_data_json(tmp_path, seqs))
    assert f["n_na"] == 3
    assert f["n_water"] == 30
    # tier must be based on ions only, not water


def test_factors_reads_ptm(tmp_path):
    seqs = [
        {"protein": {"id": "B", "sequence": "ACDEFG",
                     "modifications": [{"ptmType": "TPO", "ptmPosition": 101}]}},
    ]
    f = factors.parse_condition_factors(_write_data_json(tmp_path, seqs))
    assert f["ptm_labels"] == ["TPO101"]


def test_factors_detects_real_ligand(tmp_path):
    seqs = [
        {"protein": {"id": "B", "sequence": "ACDEFG", "modifications": []}},
        {"ligand": {"id": ["L"], "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}},
    ]
    f = factors.parse_condition_factors(_write_data_json(tmp_path, seqs))
    assert f["has_real_ligand"]


def test_classify_nonpolymer():
    assert factors.classify_nonpolymer({"ccdCodes": ["NA"]}) == "ion"
    assert factors.classify_nonpolymer({"smiles": "O"}) == "water"
    assert factors.classify_nonpolymer({"ccdCodes": ["HOH"]}) == "water"
    assert factors.classify_nonpolymer({"smiles": "CCO"}) == "ligand"


def test_build_experiment_structure_ion_tier_excludes_water():
    c1 = ConditionModel("c_1x", Path("a"))
    c1.n_na, c1.n_cl, c1.n_water = 1, 1, 10
    c10 = ConditionModel("c_10x", Path("b"))
    c10.n_na, c10.n_cl, c10.n_water = 10, 10, 100
    conditions = {"c_1x": c1, "c_10x": c10}
    struct = factors.build_experiment_structure(conditions)
    assert struct.ion_tier["c_1x"] == "1x"
    assert struct.ion_tier["c_10x"] == "10x"


# ---------------------------------------------------------------------------
# Confidence QC
# ---------------------------------------------------------------------------

def test_detect_failed_flags_outlier():
    import pandas as pd
    from af3bench.plots import confidence as cf
    df = pd.DataFrame({
        "condition": [f"c{i}" for i in range(6)],
        "iptm": [0.8, 0.82, 0.79, 0.81, 0.80, 0.14],   # last one collapsed
        "mean_pae": [6.0, 6.5, 5.8, 6.2, 6.1, 30.0],
    })
    failed = cf.detect_failed(df)
    assert "c5" in failed
    assert "c0" not in failed


def test_detect_failed_robust_to_many_failures():
    # When 3 of 8 collapse, Tukey fences alone miss them; absolute thresholds catch them.
    import pandas as pd
    from af3bench.plots import confidence as cf
    df = pd.DataFrame({
        "condition": [f"c{i}" for i in range(8)],
        "iptm":     [0.71, 0.68, 0.70, 0.67, 0.14, 0.14, 0.14, 0.69],
        "mean_pae": [6.0, 7.0, 6.5, 8.0, 30.0, 30.5, 31.0, 6.2],
    })
    failed = cf.detect_failed(df)
    assert {"c4", "c5", "c6"}.issubset(failed)
    assert "c0" not in failed


def test_baseline_resolution_prefers_clean():
    from af3bench.analysis import resolve_baseline
    clean = ConditionModel("clean_nax1", Path("a"))
    clean.n_na = clean.n_cl = 1
    modded = ConditionModel("tpo101_nax1", Path("b"))
    modded.n_na = modded.n_cl = 1
    modded.ptm_labels = ["TPO101"]
    conds = {"clean_nax1": clean, "tpo101_nax1": modded}
    name, reason = resolve_baseline(conds, None)
    assert name == "clean_nax1"


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def test_cluster_conditions_groups_similar():
    from af3bench import cluster as cl
    names = ["a1", "a2", "b1", "b2"]
    # two tight groups: a* near each other, b* near each other, far apart
    M = np.array([
        [0.0, 0.5, 9.0, 9.2],
        [0.5, 0.0, 9.1, 9.0],
        [9.0, 9.1, 0.0, 0.4],
        [9.2, 9.0, 0.4, 0.0],
    ])
    res = cl.cluster_conditions(names, M, threshold=3.0)
    labels = res["labels"]
    assert labels["a1"] == labels["a2"]
    assert labels["b1"] == labels["b2"]
    assert labels["a1"] != labels["b1"]
    assert res["n_clusters"] == 2


def test_cluster_conditions_n_clusters_override():
    from af3bench import cluster as cl
    names = ["a", "b", "c"]
    M = np.array([[0, 1, 2], [1, 0, 1.5], [2, 1.5, 0]], dtype=float)
    res = cl.cluster_conditions(names, M, n_clusters=3)
    assert res["n_clusters"] == 3
    assert len(set(res["labels"].values())) == 3


def test_cluster_conditions_handles_nan():
    from af3bench import cluster as cl
    names = ["a", "b", "c"]
    M = np.array([[0, 1, np.nan], [1, 0, 2], [np.nan, 2, 0]], dtype=float)
    res = cl.cluster_conditions(names, M, threshold=1.5)
    assert set(res["labels"]) == set(names)
    assert res["linkage"] is not None
