"""
Tests for AF3 chemical/representation compatibility.

Verifies:
1. Native PTM representation (phosphoserine → SEP)
2. Custom representation (status = verified_custom)
3. Unsupported representation
4. Missing CCD code detection
5. Missing custom CCD detection
6. Valid covalent modification
7. Invalid covalent modification
8. Multiple PTMs on one condition
9. Two unrelated proteins using the same PTM representation
10. CCD code validation against reference.py
11. AF3RepresentationStatus enum behavior
12. Ligand AF3 status
13. Ion AF3 status
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
    validate_manifest,
    AF3RepresentationStatus,
    is_known_ccd_code,
)
from af3_builder.condition_manifest.builder import (
    resolve_condition,
    build_job,
    _find_af3_representation,
)
from af3_builder.condition_manifest.registries import (
    AF3CompatibilityRecord,
    CovalentBondRecord,
    ModificationRecord,
    LigandRecord,
    IonRecord,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POU_DIR = Path(__file__).resolve().parents[4] / "testdata" / "pou2" / "registries"
PROTEIN_X_DIR = Path(__file__).resolve().parents[4] / "testdata" / "protein_x" / "registries"


# ---------------------------------------------------------------------------
# 1. AF3RepresentationStatus enum
# ---------------------------------------------------------------------------

class TestAF3RepresentationStatus:
    """Test the status enum."""

    def test_all_statuses_defined(self):
        statuses = list(AF3RepresentationStatus)
        assert len(statuses) == 5
        assert AF3RepresentationStatus.VERIFIED_NATIVE.value == "verified_native"
        assert AF3RepresentationStatus.VERIFIED_CUSTOM.value == "verified_custom"
        assert AF3RepresentationStatus.REPRESENTATION_POSSIBLE.value == "representation_possible"
        assert AF3RepresentationStatus.REPRESENTATION_UNCERTAIN.value == "representation_uncertain"
        assert AF3RepresentationStatus.UNSUPPORTED.value == "unsupported"

    def test_case_insensitive_lookup(self):
        assert AF3RepresentationStatus("VERIFIED_NATIVE") == AF3RepresentationStatus.VERIFIED_NATIVE
        assert AF3RepresentationStatus("Verified_Native") == AF3RepresentationStatus.VERIFIED_NATIVE

    def test_invalid_value_returns_none(self):
        assert AF3RepresentationStatus._missing_("bogus") is None


# ---------------------------------------------------------------------------
# 2. CCD code validation
# ---------------------------------------------------------------------------

class TestCCDCodeValidation:
    """Test CCD code checking against reference.py."""

    def test_known_ccd_codes_exist(self):
        # These should be in core/reference.py
        assert is_known_ccd_code("SEP")
        assert is_known_ccd_code("TPO")
        assert is_known_ccd_code("ALY")
        assert is_known_ccd_code("MG")
        assert is_known_ccd_code("ATP")

    def test_unknown_ccd_code(self):
        assert not is_known_ccd_code("ZZZ")
        assert not is_known_ccd_code("NONEXISTENT")


# ---------------------------------------------------------------------------
# 3. AF3CompatibilityRecord properties
# ---------------------------------------------------------------------------

class TestAF3CompatibilityRecord:
    """Test the extended AF3CompatibilityRecord."""

    def test_status_property_verified_native(self):
        rec = AF3CompatibilityRecord(
            representation_id="test",
            af3_status="verified_native",
        )
        assert rec.status == AF3RepresentationStatus.VERIFIED_NATIVE

    def test_status_property_unsupported(self):
        rec = AF3CompatibilityRecord(
            representation_id="test",
            af3_status="unsupported",
        )
        assert rec.status == AF3RepresentationStatus.UNSUPPORTED

    def test_status_property_empty_returns_uncertain(self):
        rec = AF3CompatibilityRecord(representation_id="test")
        assert rec.status == AF3RepresentationStatus.REPRESENTATION_UNCERTAIN

    def test_needs_custom_ccd(self):
        rec = AF3CompatibilityRecord(
            representation_id="test",
            custom_ccd_required="true",
        )
        assert rec.needs_custom_ccd is True

    def test_needs_covalent_bond(self):
        rec = AF3CompatibilityRecord(
            representation_id="test",
            covalent_bond_required="true",
            bond_entity_1="A",
            bond_residue_1="100",
            bond_atom_1="SG",
            bond_entity_2="E",
            bond_residue_2="1",
            bond_atom_2="C",
        )
        assert rec.needs_covalent_bond is True


# ---------------------------------------------------------------------------
# 4. Native PTM representation
# ---------------------------------------------------------------------------

class TestNativePTMRepresentation:
    """Test that native PTMs resolve to correct CCD codes."""

    @pytest.fixture
    def regs(self):
        return {
            "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        }

    def test_phosphothreonine_resolves_to_tpo(self, regs):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        spec = resolve_condition(manifest, "pou_tpo101", **regs)
        # Find the modification
        all_mods = [m for p in spec.proteins for m in p.modifications]
        tpo_mods = [m for m in all_mods if m["modification_id"] == "phospho_T235"]
        assert len(tpo_mods) == 1
        assert tpo_mods[0]["ccd_code"] == "TPO"
        assert tpo_mods[0]["af3_status"] == "verified_native"

    def test_phosphoserine_resolves_to_sep(self, regs):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        spec = resolve_condition(manifest, "pou_sep102", **regs)
        all_mods = [m for p in spec.proteins for m in p.modifications]
        sep_mods = [m for m in all_mods if m["modification_id"] == "phospho_S236"]
        assert len(sep_mods) == 1
        assert sep_mods[0]["ccd_code"] == "SEP"
        assert sep_mods[0]["af3_status"] == "verified_native"

    def test_no_warnings_for_native_ptms(self, regs):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        spec = resolve_condition(manifest, "pou_tpo101_sep102", **regs)
        # Should have no UNSUPPORTED or UNCERTAIN warnings
        assert not spec.has_unsupported_representations


# ---------------------------------------------------------------------------
# 5. Multiple PTMs
# ---------------------------------------------------------------------------

class TestMultiplePTMs:
    """Test conditions with multiple modifications."""

    @pytest.fixture
    def regs(self):
        return {
            "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        }

    def test_double_phosphorylation(self, regs):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        spec = resolve_condition(manifest, "pou_tpo101_sep102", **regs)
        all_mods = [m for p in spec.proteins for m in p.modifications]
        assert len(all_mods) == 2
        mod_ids = {m["modification_id"] for m in all_mods}
        assert mod_ids == {"phospho_T235", "phospho_S236"}
        # Both should be verified native
        for m in all_mods:
            assert m["af3_status"] == "verified_native"

    def test_builds_job_with_multiple_mods(self, regs):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        jb = build_job(manifest, "pou_tpo101_sep102", seeds=[1], **regs)
        job_dict = jb.to_dict()
        protein_seqs = [s for s in job_dict["sequences"] if "protein" in s]
        prot = protein_seqs[0]["protein"]
        assert "modifications" in prot
        assert len(prot["modifications"]) == 2


# ---------------------------------------------------------------------------
# 6. Two unrelated proteins using the same PTM
# ---------------------------------------------------------------------------

class TestProteinAgnosticPTM:
    """Test that the same PTM representation works for different proteins."""

    def test_phosphorylation_works_for_both_proteins(self):
        pou_regs = {
            "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        }
        px_regs = {
            "protein_registry": load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(PROTEIN_X_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(PROTEIN_X_DIR / "nucleic_acid_registry.csv"),
            "ligand_registry": load_ligand_registry(PROTEIN_X_DIR / "ligand_registry.csv"),
            "ion_registry": load_ion_registry(PROTEIN_X_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(PROTEIN_X_DIR / "af3_compatibility_registry.csv"),
        }

        pou_manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        px_manifest = load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
            entities_path=PROTEIN_X_DIR / "condition_entities.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
        )

        # POU phosphorylation
        pou_spec = resolve_condition(pou_manifest, "pou_tpo101", **pou_regs)
        pou_mods = [m for p in pou_spec.proteins for m in p.modifications]
        assert any(m["ccd_code"] == "TPO" for m in pou_mods)

        # Protein X phosphorylation (different protein, same PTM representation)
        px_spec = resolve_condition(px_manifest, "kinase_phospho", **px_regs)
        px_mods = [m for p in px_spec.proteins for m in p.modifications]
        assert any(m["ccd_code"] == "TPO" for m in px_mods)

        # Both use the same AF3 representation
        assert pou_mods[0]["af3_status"] == px_mods[0]["af3_status"]


# ---------------------------------------------------------------------------
# 7. Ligand AF3 status
# ---------------------------------------------------------------------------

class TestLigandAF3Status:
    """Test that ligand records carry AF3 status."""

    def test_ligand_with_ccd_has_possible_status(self):
        lig = LigandRecord(
            ligand_id="test",
            ccd_code="ATP",
        )
        assert lig.status == AF3RepresentationStatus.REPRESENTATION_POSSIBLE

    def test_ligand_with_explicit_status(self):
        lig = LigandRecord(
            ligand_id="test",
            ccd_code="ATP",
            af3_status="verified_native",
        )
        assert lig.status == AF3RepresentationStatus.VERIFIED_NATIVE

    def test_ligand_without_ccd_or_smiles(self):
        lig = LigandRecord(ligand_id="test")
        assert lig.status == AF3RepresentationStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# 8. Ion AF3 status
# ---------------------------------------------------------------------------

class TestIonAF3Status:
    """Test that ion records carry AF3 status."""

    def test_ion_with_ccd_has_possible_status(self):
        ion = IonRecord(ion_id="test", ccd_code="MG")
        assert ion.status == AF3RepresentationStatus.REPRESENTATION_POSSIBLE

    def test_ion_without_ccd(self):
        ion = IonRecord(ion_id="test")
        assert ion.status == AF3RepresentationStatus.UNSUPPORTED


# ---------------------------------------------------------------------------
# 9. Validation catches AF3 issues
# ---------------------------------------------------------------------------

class TestValidationAF3Checks:
    """Test that validation detects AF3 representation problems."""

    def test_validation_passes_for_pou(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
            exclusivity_path=POU_DIR / "exclusivity_rules.csv",
        )
        result = validate_manifest(
            manifest,
            protein_registry=load_protein_registry(POU_DIR / "protein_registry.csv"),
            construct_registry=load_construct_registry(POU_DIR / "construct_registry.csv"),
            modification_registry=load_modification_registry(POU_DIR / "modification_registry.csv"),
            nucleic_acid_registry=load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            ion_registry=load_ion_registry(POU_DIR / "ion_registry.csv"),
            af3_registry=load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        )
        assert result.is_valid, result.summary()

    def test_validation_passes_for_protein_x(self):
        manifest = load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
            entities_path=PROTEIN_X_DIR / "condition_entities.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
        )
        result = validate_manifest(
            manifest,
            protein_registry=load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv"),
            construct_registry=load_construct_registry(PROTEIN_X_DIR / "construct_registry.csv"),
            modification_registry=load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv"),
            nucleic_acid_registry=load_nucleic_acid_registry(PROTEIN_X_DIR / "nucleic_acid_registry.csv"),
            ligand_registry=load_ligand_registry(PROTEIN_X_DIR / "ligand_registry.csv"),
            ion_registry=load_ion_registry(PROTEIN_X_DIR / "ion_registry.csv"),
            af3_registry=load_af3_compatibility_registry(PROTEIN_X_DIR / "af3_compatibility_registry.csv"),
        )
        assert result.is_valid, result.summary()

    def test_validation_flags_unsupported_modification(self):
        """A modification with unsupported AF3 status should be flagged."""
        from af3_builder.condition_manifest.manifest import (
            ConditionRecord, ConditionModificationRecord,
        )
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Test"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="bad_mod",
            sequence_position="10", construct_id="X",
        )

        mod_reg = {
            "bad_mod": ModificationRecord(
                modification_id="bad_mod",
                modification_name="Bad Mod",
                modification_class="unknown",
            )
        }
        af3_reg = {
            "rep_bad": AF3CompatibilityRecord(
                representation_id="rep_bad",
                modification_id="bad_mod",
                af3_status="unsupported",
            )
        }

        result = validate_manifest(
            manifest,
            modification_registry=mod_reg,
            af3_registry=af3_reg,
        )
        assert not result.is_valid
        assert any("UNSUPPORTED" in e for e in result.errors)

    def test_validation_flags_missing_custom_ccd(self):
        """A custom CCD without an ID should be flagged."""
        from af3_builder.condition_manifest.manifest import (
            ConditionRecord, ConditionModificationRecord,
        )
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Test"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="custom_mod",
            sequence_position="10", construct_id="X",
        )

        mod_reg = {
            "custom_mod": ModificationRecord(
                modification_id="custom_mod",
                modification_name="Custom Mod",
            )
        }
        af3_reg = {
            "rep_custom": AF3CompatibilityRecord(
                representation_id="rep_custom",
                modification_id="custom_mod",
                af3_status="verified_custom",
                custom_ccd_required="true",
                # No custom_ccd_id!
            )
        }

        result = validate_manifest(
            manifest,
            modification_registry=mod_reg,
            af3_registry=af3_reg,
        )
        assert not result.is_valid
        assert any("custom_ccd_id" in e for e in result.errors)

    def test_covalent_bond_validation(self):
        """Test covalent bond validation."""
        from af3_builder.condition_manifest.manifest import ConditionRecord
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Test"
        )

        # Valid bond
        valid_bond = CovalentBondRecord(
            bond_id="bond_1",
            entity_1_type="protein", entity_1_id="A",
            residue_1="100", atom_1="SG",
            entity_2_type="ligand", entity_2_id="E",
            residue_2="1", atom_2="C",
            bond_type="covalent",
            af3_status="verified_native",
        )
        result = validate_manifest(
            manifest,
            covalent_bond_registry={"bond_1": valid_bond},
        )
        assert result.is_valid

        # Unsupported bond
        unsupported_bond = CovalentBondRecord(
            bond_id="bond_2",
            entity_1_type="protein", entity_1_id="A",
            residue_1="100", atom_1="SG",
            entity_2_type="ligand", entity_2_id="E",
            residue_2="1", atom_2="C",
            af3_status="unsupported",
        )
        result = validate_manifest(
            manifest,
            covalent_bond_registry={"bond_2": unsupported_bond},
        )
        assert not result.is_valid
        assert any("UNSUPPORTED" in e for e in result.errors)

        # Bond with invalid residue position
        bad_pos_bond = CovalentBondRecord(
            bond_id="bond_3",
            entity_1_type="protein", entity_1_id="A",
            residue_1="abc", atom_1="SG",
            entity_2_type="ligand", entity_2_id="E",
            residue_2="1", atom_2="C",
        )
        result = validate_manifest(
            manifest,
            covalent_bond_registry={"bond_3": bad_pos_bond},
        )
        assert not result.is_valid
        assert any("bond_3" in e and "not a valid integer" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 10. build_all_jobs rejects unsupported
# ---------------------------------------------------------------------------

class TestBuildAllJobsRejectsUnsupported:
    """Test that build_all_jobs raises for unsupported representations."""

    def test_build_all_jobs_verified_native_for_pou(self):
        """build_all_jobs without allow_uncertain builds only verified_native conditions."""
        from af3_builder.condition_manifest.builder import build_all_jobs
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        regs = {
            "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
            "covalent_bond_registry": load_covalent_bond_registry(POU_DIR / "covalent_bond_registry.csv"),
        }
        jobs = build_all_jobs(manifest, seeds=[1], **regs)
        # 8 original + 14 Priority 1 + 4 methylation + 4 SUMO/Ub (with bonds) = 30
        # O-GlcNAc is uncertain and excluded by default
        assert len(jobs) == 30

    def test_build_all_jobs_with_uncertain_for_pou(self):
        """build_all_jobs with allow_uncertain=True builds all non-unsupported conditions."""
        from af3_builder.condition_manifest.builder import build_all_jobs
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        regs = {
            "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
            "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
            "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
            "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
            "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
            "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
            "covalent_bond_registry": load_covalent_bond_registry(POU_DIR / "covalent_bond_registry.csv"),
        }
        jobs = build_all_jobs(manifest, seeds=[1], allow_uncertain=True, **regs)
        # All 32 conditions (8 original + 14 Priority 1 + 10 Priority 2) should build
        assert len(jobs) == 32
