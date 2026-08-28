"""
Tests for the builder integration — manifest → JobBuilder.

Verifies that:
1. A condition resolves into the correct entities
2. Modifications are correctly attached to proteins
3. DNA/RNA/ligands/ions are correctly added
4. The resulting JobBuilder produces valid AF3 JSON
5. Both POU and Protein X datasets work through the same code path
"""

import pytest
from pathlib import Path

from af3_builder.condition_manifest import (
    MasterManifest,
    load_master_manifest,
    load_protein_registry,
    load_construct_registry,
    load_modification_registry,
    load_nucleic_acid_registry,
    load_ligand_registry,
    load_ion_registry,
    load_af3_compatibility_registry,
    load_covalent_bond_registry,
)
from af3_builder.condition_manifest.builder import (
    ConditionSpec,
    ResolvedEntity,
    resolve_condition,
    build_job,
    build_all_jobs,
    _int_to_chain,
    _chain_to_int,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POU_DIR = Path(__file__).resolve().parents[4] / "testdata" / "pou2" / "registries"
PROTEIN_X_DIR = Path(__file__).resolve().parents[4] / "testdata" / "protein_x" / "registries"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pou_registries():
    return {
        "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
        "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
        "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
        "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
        "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
        "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        "covalent_bond_registry": load_covalent_bond_registry(POU_DIR / "covalent_bond_registry.csv"),
    }


def _load_px_registries():
    return {
        "protein_registry": load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv"),
        "construct_registry": load_construct_registry(PROTEIN_X_DIR / "construct_registry.csv"),
        "modification_registry": load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv"),
        "nucleic_acid_registry": load_nucleic_acid_registry(PROTEIN_X_DIR / "nucleic_acid_registry.csv"),
        "ligand_registry": load_ligand_registry(PROTEIN_X_DIR / "ligand_registry.csv"),
        "ion_registry": load_ion_registry(PROTEIN_X_DIR / "ion_registry.csv"),
        "af3_compatibility_registry": load_af3_compatibility_registry(PROTEIN_X_DIR / "af3_compatibility_registry.csv"),
    }


# ---------------------------------------------------------------------------
# Chain letter helpers
# ---------------------------------------------------------------------------

class TestChainHelpers:
    def test_int_to_chain_basic(self):
        assert _int_to_chain(0) == "A"
        assert _int_to_chain(1) == "B"
        assert _int_to_chain(25) == "Z"
        assert _int_to_chain(26) == "AA"

    def test_chain_to_int_basic(self):
        assert _chain_to_int("A") == 0
        assert _chain_to_int("B") == 1
        assert _chain_to_int("Z") == 25
        assert _chain_to_int("AA") == 26

    def test_roundtrip(self):
        for i in range(52):
            assert _chain_to_int(_int_to_chain(i)) == i


# ---------------------------------------------------------------------------
# Condition resolution
# ---------------------------------------------------------------------------

class TestResolveCondition:
    """Test that resolve_condition produces correct ConditionSpec."""

    @pytest.fixture
    def pou_manifest(self):
        return load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )

    def test_baseline_resolves(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_baseline", **regs)
        assert spec.condition_id == "pou_baseline"
        assert spec.condition_name == "Baseline (POU)"
        # Baseline has no modifications
        for p in spec.proteins:
            assert len(p.modifications) == 0

    def test_tpo101_has_modification(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_tpo101", **regs)
        # Should have protein with one modification
        assert len(spec.proteins) >= 1
        has_phospho = any(
            m["modification_id"] == "phospho_T235"
            for p in spec.proteins
            for m in p.modifications
        )
        assert has_phospho

    def test_tpo101_sep102_has_two_modifications(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_tpo101_sep102", **regs)
        all_mods = [m for p in spec.proteins for m in p.modifications]
        mod_ids = {m["modification_id"] for m in all_mods}
        assert "phospho_T235" in mod_ids
        assert "phospho_S236" in mod_ids

    def test_dna_condition_has_dna_entity(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_dna", **regs)
        assert len(spec.dna) >= 1
        assert spec.dna[0].entity_id == "ref_dna_duplex"
        assert spec.dna[0].sequence != ""

    def test_no_dna_in_baseline(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_baseline", **regs)
        assert len(spec.dna) == 0

    def test_ions_present(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_baseline", **regs)
        assert len(spec.ions) >= 1
        assert spec.ions[0].ccd_code == "MG"

    def test_unknown_condition_raises(self, pou_manifest):
        regs = _load_pou_registries()
        with pytest.raises(ValueError, match="not found"):
            resolve_condition(pou_manifest, "nonexistent", **regs)

    def test_protein_has_sequence(self, pou_manifest):
        regs = _load_pou_registries()
        spec = resolve_condition(pou_manifest, "pou_baseline", **regs)
        assert len(spec.proteins) >= 1
        assert len(spec.proteins[0].sequence) > 0


# ---------------------------------------------------------------------------
# Job building
# ---------------------------------------------------------------------------

class TestBuildJob:
    """Test that build_job produces a valid JobBuilder."""

    @pytest.fixture
    def pou_manifest(self):
        return load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )

    def test_build_baseline_job(self, pou_manifest):
        regs = _load_pou_registries()
        jb = build_job(pou_manifest, "pou_baseline", seeds=[42], **regs)
        job_dict = jb.to_dict()
        assert job_dict["name"] == "Baseline (POU)"
        assert job_dict["modelSeeds"] == [42]
        assert len(job_dict["sequences"]) >= 1
        # Baseline should have protein but no modifications
        protein_seq = job_dict["sequences"][0]
        assert "protein" in protein_seq
        assert "modifications" not in protein_seq["protein"]

    def test_build_tpo101_job_has_modification(self, pou_manifest):
        regs = _load_pou_registries()
        jb = build_job(pou_manifest, "pou_tpo101", seeds=[1], **regs)
        job_dict = jb.to_dict()
        # Find the protein sequence
        protein_seqs = [s for s in job_dict["sequences"] if "protein" in s]
        assert len(protein_seqs) >= 1
        # Should have modifications
        prot = protein_seqs[0]["protein"]
        assert "modifications" in prot
        assert len(prot["modifications"]) >= 1

    def test_build_dna_job_has_dna_entity(self, pou_manifest):
        regs = _load_pou_registries()
        jb = build_job(pou_manifest, "pou_dna", seeds=[1], **regs)
        job_dict = jb.to_dict()
        # Should have both protein and DNA sequences
        types = [list(s.keys())[0] for s in job_dict["sequences"]]
        assert "protein" in types
        assert "dna" in types

    def test_build_with_no_seeds(self, pou_manifest):
        regs = _load_pou_registries()
        jb = build_job(pou_manifest, "pou_baseline", **regs)
        job_dict = jb.to_dict()
        # Should have generated a random seed
        assert len(job_dict["modelSeeds"]) == 1

    def test_build_all_jobs(self, pou_manifest):
        regs = _load_pou_registries()
        jobs = build_all_jobs(pou_manifest, seeds=[1], **regs)
        # 8 original + 14 Priority 1 + 4 methylation + 2 SUMO = 28 verified_native
        # O-GlcNAc (2 conditions) is uncertain and excluded by default
        assert len(jobs) == 28
        for cid, jb in jobs.items():
            job_dict = jb.to_dict()
            assert "name" in job_dict
            assert "sequences" in job_dict
            assert len(job_dict["sequences"]) >= 1

    def test_build_all_jobs_with_uncertain(self, pou_manifest):
        """build_all_jobs with allow_uncertain=True builds all conditions."""
        regs = _load_pou_registries()
        jobs = build_all_jobs(pou_manifest, seeds=[1], allow_uncertain=True, **regs)
        # All 30 conditions (8 original + 14 Priority 1 + 8 Priority 2)
        assert len(jobs) == 30


# ---------------------------------------------------------------------------
# Protein X — different protein, same code
# ---------------------------------------------------------------------------

class TestProteinXIntegration:
    """Test that a completely different protein works through the same code."""

    @pytest.fixture
    def px_manifest(self):
        return load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
            entities_path=PROTEIN_X_DIR / "condition_entities.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
        )

    def test_kinase_baseline_resolves(self, px_manifest):
        regs = _load_px_registries()
        spec = resolve_condition(px_manifest, "kinase_baseline", **regs)
        assert spec.condition_id == "kinase_baseline"
        assert len(spec.proteins) >= 1
        assert spec.proteins[0].sequence != ""

    def test_kinase_ligand_has_ligand(self, px_manifest):
        regs = _load_px_registries()
        spec = resolve_condition(px_manifest, "kinase_ligand_A", **regs)
        assert len(spec.ligands) >= 1
        assert spec.ligands[0].ccd_code == "ATP"

    def test_kinase_partner_job(self, px_manifest):
        """Partner conditions should resolve (partner is biological, not an AF3 entity type)."""
        regs = _load_px_registries()
        spec = resolve_condition(px_manifest, "kinase_partner", **regs)
        # The partner is recorded in the manifest but isn't a separate AF3 entity
        # (it's the same protein in a different conformation)
        # The key test is that resolution doesn't crash
        assert spec.condition_id == "kinase_partner"

    def test_kinase_builds_all_jobs(self, px_manifest):
        regs = _load_px_registries()
        jobs = build_all_jobs(px_manifest, seeds=[1], **regs)
        assert len(jobs) == 8
        # Ligand conditions should have ligand sequences
        ligand_job = jobs["kinase_ligand_A"].to_dict()
        types = [list(s.keys())[0] for s in ligand_job["sequences"]]
        assert "ligand" in types

    def test_same_code_different_protein(self, px_manifest):
        """Both POU and Protein X use exactly the same build_job function."""
        pou_manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        pou_regs = _load_pou_registries()
        px_regs = _load_px_registries()

        pou_job = build_job(pou_manifest, "pou_baseline", seeds=[1], **pou_regs)
        px_job = build_job(px_manifest, "kinase_baseline", seeds=[1], **px_regs)

        # Both produce valid job dicts
        pou_dict = pou_job.to_dict()
        px_dict = px_job.to_dict()

        assert pou_dict["dialect"] == "alphafold3"
        assert px_dict["dialect"] == "alphafold3"
        assert len(pou_dict["sequences"]) >= 1
        assert len(px_dict["sequences"]) >= 1
