"""
End-to-end tests for the condition_manifest → JobBuilder → AF3 JSON pipeline.

Covers all 12 mandatory test cases specified in the task:
    1.  Existing unmodified condition
    2.  Existing modified protein condition
    3.  Multiple modifications
    4.  DNA
    5.  Ligand
    6.  Protein partner (explicit entity)
    7.  Complex condition
    8.  Invalid residue mapping
    9.  Unsupported representation
    10. Missing custom CCD
    11. Invalid covalent bond
    12. Genericity (two unrelated proteins)

Plus JSON fixture verification for representative conditions.
"""

import json
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
)
from af3_builder.condition_manifest.builder import (
    ConditionSpec,
    ResolvedEntity,
    resolve_condition,
    build_job,
    build_all_jobs,
    _mods_to_af3_format,
    _validate_spec_for_build,
    _int_to_chain,
)
from af3_builder.condition_manifest.manifest import (
    ConditionRecord,
    ConditionModificationRecord,
    ConditionEntityRecord,
)
from af3_builder.condition_manifest.registries import (
    AF3CompatibilityRecord,
    CovalentBondRecord,
    ModificationRecord,
    ProteinRecord,
    ConstructRecord,
    LigandRecord,
    IonRecord,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POU_DIR = Path(__file__).resolve().parents[4] / "testdata" / "pou2" / "registries"
PROTEIN_X_DIR = Path(__file__).resolve().parents[4] / "testdata" / "protein_x" / "registries"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pou_registries():
    return {
        "protein_registry": load_protein_registry(POU_DIR / "protein_registry.csv"),
        "construct_registry": load_construct_registry(POU_DIR / "construct_registry.csv"),
        "modification_registry": load_modification_registry(POU_DIR / "modification_registry.csv"),
        "nucleic_acid_registry": load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv"),
        "ligand_registry": load_ligand_registry(POU_DIR / "ligand_registry.csv"),
        "ion_registry": load_ion_registry(POU_DIR / "ion_registry.csv"),
        "af3_compatibility_registry": load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv"),
        "covalent_bond_registry": load_covalent_bond_registry(POU_DIR / "covalent_bond_registry.csv"),
    }


@pytest.fixture
def pou_manifest():
    return load_master_manifest(
        POU_DIR / "master_condition_manifest.csv",
        modifications_path=POU_DIR / "condition_modifications.csv",
        entities_path=POU_DIR / "condition_entities.csv",
        factors_path=POU_DIR / "condition_factors.csv",
    )


@pytest.fixture
def px_registries():
    return {
        "protein_registry": load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv"),
        "construct_registry": load_construct_registry(PROTEIN_X_DIR / "construct_registry.csv"),
        "modification_registry": load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv"),
        "nucleic_acid_registry": load_nucleic_acid_registry(PROTEIN_X_DIR / "nucleic_acid_registry.csv"),
        "ligand_registry": load_ligand_registry(PROTEIN_X_DIR / "ligand_registry.csv"),
        "ion_registry": load_ion_registry(PROTEIN_X_DIR / "ion_registry.csv"),
        "af3_compatibility_registry": load_af3_compatibility_registry(PROTEIN_X_DIR / "af3_compatibility_registry.csv"),
        "covalent_bond_registry": load_covalent_bond_registry(PROTEIN_X_DIR / "covalent_bond_registry.csv"),
    }


@pytest.fixture
def px_manifest():
    return load_master_manifest(
        PROTEIN_X_DIR / "master_condition_manifest.csv",
        modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
        entities_path=PROTEIN_X_DIR / "condition_entities.csv",
        factors_path=PROTEIN_X_DIR / "condition_factors.csv",
    )


# ===================================================================
# TEST 1 — Existing unmodified condition
# ===================================================================

class Test1_UnmodifiedCondition:
    """Verify an existing unmodified condition builds successfully."""

    def test_baseline_resolves_with_exactly_one_protein(
        self, pou_manifest, pou_registries
    ):
        spec = resolve_condition(pou_manifest, "pou_baseline", **pou_registries)
        assert spec.condition_id == "pou_baseline"
        assert len(spec.proteins) == 1, (
            f"Expected exactly 1 protein for baseline, got {len(spec.proteins)}: "
            f"{[p.entity_id for p in spec.proteins]}"
        )
        assert spec.proteins[0].entity_id == "POU_DOMAIN"
        assert len(spec.proteins[0].modifications) == 0

    def test_baseline_builds_valid_job(self, pou_manifest, pou_registries):
        jb = build_job(pou_manifest, "pou_baseline", seeds=[42], **pou_registries)
        d = jb.to_dict()
        assert d["name"] == "Baseline (POU)"
        assert d["modelSeeds"] == [42]
        assert d["dialect"] == "alphafold3"

        prots = [s for s in d["sequences"] if "protein" in s]
        assert len(prots) == 1
        prot = prots[0]["protein"]
        assert "modifications" not in prot
        assert len(prot["sequence"]) > 0

    def test_baseline_json_is_clean(self, pou_manifest, pou_registries):
        """No metadata leakage in AF3 JSON."""
        jb = build_job(pou_manifest, "pou_baseline", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        # Should NOT contain internal metadata keys
        assert "modification_id" not in prot
        assert "ccd_code" not in prot
        assert "af3_status" not in prot

    def test_baseline_has_ion(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_baseline", **pou_registries)
        assert len(spec.ions) >= 1
        assert spec.ions[0].ccd_code == "MG"

    def test_baseline_no_dna(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_baseline", **pou_registries)
        assert len(spec.dna) == 0


# ===================================================================
# TEST 2 — Existing modified protein condition
# ===================================================================

class Test2_ModifiedProteinCondition:
    """Verify an existing PTM condition resolves correctly."""

    def test_tpo101_has_single_modification(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_tpo101", **pou_registries)
        assert len(spec.proteins) == 1
        mods = spec.proteins[0].modifications
        assert len(mods) == 1
        assert mods[0]["modification_id"] == "phospho_T235"
        assert mods[0]["ccd_code"] == "TPO"
        assert mods[0]["af3_status"] == "verified_native"

    def test_tpo101_builds_valid_job(self, pou_manifest, pou_registries):
        jb = build_job(pou_manifest, "pou_tpo101", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert "modifications" in prot
        assert len(prot["modifications"]) == 1
        # AF3 format: ptmType, ptmPosition
        mod = prot["modifications"][0]
        assert mod["ptmType"] == "TPO"
        assert mod["ptmPosition"] == 235
        assert len(mod) == 2, f"Expected only ptmType+ptmPosition, got: {list(mod.keys())}"

    def test_sep102_resolves_correctly(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_sep102", **pou_registries)
        mods = spec.proteins[0].modifications
        assert len(mods) == 1
        assert mods[0]["ccd_code"] == "SEP"
        assert mods[0]["af3_status"] == "verified_native"


# ===================================================================
# TEST 3 — Multiple modifications
# ===================================================================

class Test3_MultipleModifications:
    """Verify a condition containing PTM_A + PTM_B survives into JSON."""

    def test_double_phosphorylation_resolves(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_tpo101_sep102", **pou_registries)
        mods = spec.proteins[0].modifications
        assert len(mods) == 2
        mod_ids = {m["modification_id"] for m in mods}
        assert mod_ids == {"phospho_T235", "phospho_S236"}
        for m in mods:
            assert m["af3_status"] == "verified_native"

    def test_double_phosphorylation_builds_json(self, pou_manifest, pou_registries):
        jb = build_job(pou_manifest, "pou_tpo101_sep102", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert "modifications" in prot
        assert len(prot["modifications"]) == 2
        ccd_codes = {m["ptmType"] for m in prot["modifications"]}
        assert ccd_codes == {"TPO", "SEP"}

    def test_triple_modification_condition(self, pou_manifest, pou_registries):
        """Test pou_tpo101_sep102_dna has both mods + DNA."""
        spec = resolve_condition(
            pou_manifest, "pou_tpo101_sep102_dna", **pou_registries
        )
        mods = spec.proteins[0].modifications
        assert len(mods) == 2
        assert len(spec.dna) == 1

    def test_no_warnings_for_native_ptms(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_tpo101_sep102", **pou_registries)
        assert not spec.has_unsupported_representations
        assert not spec.has_uncertain_representations


# ===================================================================
# TEST 4 — DNA
# ===================================================================

class Test4_DNA:
    """Verify Protein + DNA produces correct entities."""

    def test_dna_condition_has_protein_and_dna(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_dna", **pou_registries)
        assert len(spec.proteins) == 1
        assert spec.proteins[0].entity_id == "POU_DOMAIN"
        assert len(spec.dna) == 1
        assert spec.dna[0].entity_id == "ref_dna_duplex"

    def test_dna_sequence_resolved(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_dna", **pou_registries)
        assert len(spec.dna[0].sequence) > 0
        # DNA should be a valid DNA sequence
        valid_chars = set("ACGTNXacgtnx")
        assert all(c in valid_chars for c in spec.dna[0].sequence)

    def test_dna_builds_valid_json(self, pou_manifest, pou_registries):
        jb = build_job(pou_manifest, "pou_dna", seeds=[1], **pou_registries)
        d = jb.to_dict()
        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "dna" in types

    def test_dna_with_modification(self, pou_manifest, pou_registries):
        spec = resolve_condition(
            pou_manifest, "pou_tpo101_dna", **pou_registries
        )
        assert len(spec.proteins) == 1
        assert len(spec.proteins[0].modifications) == 1
        assert len(spec.dna) == 1


# ===================================================================
# TEST 5 — Ligand
# ===================================================================

class Test5_Ligand:
    """Verify Protein + ligand produces correct ligand entity."""

    def test_ligand_condition_resolves(self, px_manifest, px_registries):
        spec = resolve_condition(px_manifest, "kinase_ligand_A", **px_registries)
        assert len(spec.proteins) == 1
        assert len(spec.ligands) >= 1
        lig = [l for l in spec.ligands if l.entity_id == "example_ligand_A"]
        assert len(lig) == 1
        assert lig[0].ccd_code == "ATP"

    def test_ligand_builds_valid_json(self, px_manifest, px_registries):
        jb = build_job(px_manifest, "kinase_ligand_A", seeds=[1], **px_registries)
        d = jb.to_dict()
        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "ligand" in types

    def test_ligand_json_format(self, px_manifest, px_registries):
        jb = build_job(px_manifest, "kinase_ligand_A", seeds=[1], **px_registries)
        d = jb.to_dict()
        lig_seq = [s for s in d["sequences"] if "ligand" in s][0]["ligand"]
        assert "ccdCodes" in lig_seq
        assert lig_seq["ccdCodes"] == ["ATP"]

    def test_multiple_ligand_conditions(self, px_manifest, px_registries):
        """Both ATP and NAD conditions resolve correctly."""
        spec_atp = resolve_condition(px_manifest, "kinase_ligand_A", **px_registries)
        spec_nad = resolve_condition(px_manifest, "kinase_ligand_B", **px_registries)
        lig_atp = [l for l in spec_atp.ligands if l.entity_id == "example_ligand_A"]
        lig_nad = [l for l in spec_nad.ligands if l.entity_id == "example_ligand_B"]
        assert lig_atp[0].ccd_code == "ATP"
        assert lig_nad[0].ccd_code == "NAD"


# ===================================================================
# TEST 6 — Protein partner (explicit entity)
# ===================================================================

class Test6_ProteinPartner:
    """Verify Protein A + Protein B using the generic registry path."""

    def test_partner_condition_resolves(self, px_manifest, px_registries):
        """Partner is biological metadata; the protein entity provides the AF3 chain."""
        spec = resolve_condition(px_manifest, "kinase_partner", **px_registries)
        assert spec.condition_id == "kinase_partner"
        assert len(spec.proteins) == 1
        assert spec.proteins[0].entity_id == "PRKX_KINASE"

    def test_partner_builds_job(self, px_manifest, px_registries):
        jb = build_job(px_manifest, "kinase_partner", seeds=[1], **px_registries)
        d = jb.to_dict()
        prots = [s for s in d["sequences"] if "protein" in s]
        assert len(prots) == 1


# ===================================================================
# TEST 7 — Complex condition
# ===================================================================

class Test7_ComplexCondition:
    """Verify a condition with protein + ligand + multiple PTMs."""

    def test_kinase_multi_resolves(self, px_manifest, px_registries):
        spec = resolve_condition(px_manifest, "kinase_multi", **px_registries)
        assert len(spec.proteins) == 1
        mods = spec.proteins[0].modifications
        assert len(mods) == 2
        mod_ids = {m["modification_id"] for m in mods}
        assert "phospho_K42" in mod_ids
        assert "acetyl_K56" in mod_ids
        # Has ligand
        assert len(spec.ligands) >= 1

    def test_kinase_multi_builds_json(self, px_manifest, px_registries):
        jb = build_job(px_manifest, "kinase_multi", seeds=[1], **px_registries)
        d = jb.to_dict()
        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "ligand" in types
        # Check modifications
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 2
        for mod in prot["modifications"]:
            assert "ptmType" in mod
            assert "ptmPosition" in mod

    def test_pou_dna_with_modifications(self, pou_manifest, pou_registries):
        """Protein + modifications + DNA + ion."""
        spec = resolve_condition(
            pou_manifest, "pou_tpo101_sep102_dna", **pou_registries
        )
        assert len(spec.proteins) == 1
        assert len(spec.proteins[0].modifications) == 2
        assert len(spec.dna) == 1
        assert len(spec.ions) >= 1

    def test_pou_dna_with_modifications_builds_json(
        self, pou_manifest, pou_registries
    ):
        jb = build_job(
            pou_manifest, "pou_tpo101_sep102_dna", seeds=[1], **pou_registries
        )
        d = jb.to_dict()
        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "dna" in types
        # ligand = ion
        assert "ligand" in types

    def test_kinase_phospho_ligand(self, px_manifest, px_registries):
        """Protein + modification + ligand."""
        spec = resolve_condition(
            px_manifest, "kinase_phospho_ligand", **px_registries
        )
        assert len(spec.proteins) == 1
        assert len(spec.proteins[0].modifications) == 1
        assert len(spec.ligands) >= 1


# ===================================================================
# TEST 8 — Invalid residue mapping
# ===================================================================

class Test8_InvalidResidueMapping:
    """Must fail validation when modification position is out of range."""

    def test_position_outside_sequence_length(self):
        """Modification at position 999 on a 125-residue protein."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Position"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="999", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        from af3_builder.condition_manifest.registries import (
            ConstructRecord, ModificationRecord, NucleicAcidRecord,
        )
        from af3_builder.condition_manifest.builder import (
            AF3CompatibilityRecord,
        )

        # Load actual POU data for construct registry
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        mod_reg = load_modification_registry(POU_DIR / "modification_registry.csv")
        af3_reg = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")

        with pytest.raises(ValueError, match="outside construct"):
            resolve_condition(
                manifest, "c1",
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_position_zero_rejected(self):
        """Modification at position 0 should be caught."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Position"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="0", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        mod_reg = load_modification_registry(POU_DIR / "modification_registry.csv")
        af3_reg = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")

        with pytest.raises(ValueError, match="outside construct"):
            resolve_condition(
                manifest, "c1",
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_valid_position_passes(self):
        """Modification at position 235 on a 360-residue protein should pass."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Good Position"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="235", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        mod_reg = load_modification_registry(POU_DIR / "modification_registry.csv")
        af3_reg = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")

        spec = resolve_condition(
            manifest, "c1",
            construct_registry=construct_reg,
            modification_registry=mod_reg,
            af3_compatibility_registry=af3_reg,
        )
        assert len(spec.proteins) == 1
        assert len(spec.proteins[0].modifications) == 1


# ===================================================================
# TEST 9 — Unsupported representation
# ===================================================================

class Test9_UnsupportedRepresentation:
    """Must fail rather than silently generating an invalid job."""

    def test_unsupported_mod_blocks_build_job(self):
        """build_job should reject UNSUPPORTED representations."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Mod"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="bad_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "bad_mod": ModificationRecord(
                modification_id="bad_mod",
                modified_residue="UNK",
            )
        }
        af3_reg = {
            "rep_bad": AF3CompatibilityRecord(
                representation_id="rep_bad",
                modification_id="bad_mod",
                af3_status="unsupported",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="UNSUPPORTED"):
            build_job(
                manifest, "c1", seeds=[1],
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_unsupported_blocks_build_all_jobs(self):
        """build_all_jobs should also reject UNSUPPORTED."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Mod"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="bad_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "bad_mod": ModificationRecord(
                modification_id="bad_mod",
                modified_residue="UNK",
            )
        }
        af3_reg = {
            "rep_bad": AF3CompatibilityRecord(
                representation_id="rep_bad",
                modification_id="bad_mod",
                af3_status="unsupported",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="UNSUPPORTED"):
            build_all_jobs(
                manifest, seeds=[1],
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_uncertain_blocks_by_default(self):
        """UNCERTAIN representations should be rejected by default."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Uncertain Mod"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="uncertain_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "uncertain_mod": ModificationRecord(
                modification_id="uncertain_mod",
                modified_residue="UNK",
            )
        }
        af3_reg = {
            "rep_unc": AF3CompatibilityRecord(
                representation_id="rep_unc",
                modification_id="uncertain_mod",
                af3_status="representation_uncertain",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="uncertain"):
            build_job(
                manifest, "c1", seeds=[1],
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_uncertain_allowed_with_flag(self):
        """UNCERTAIN representations can be allowed with allow_uncertain=True."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Uncertain Mod"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="uncertain_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "uncertain_mod": ModificationRecord(
                modification_id="uncertain_mod",
                modified_residue="UNK",
            )
        }
        af3_reg = {
            "rep_unc": AF3CompatibilityRecord(
                representation_id="rep_unc",
                modification_id="uncertain_mod",
                ccd_code="UNK",
                af3_status="representation_uncertain",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        # Should NOT raise with allow_uncertain=True
        jb = build_job(
            manifest, "c1", seeds=[1],
            allow_uncertain=True,
            construct_registry=construct_reg,
            modification_registry=mod_reg,
            af3_compatibility_registry=af3_reg,
        )
        d = jb.to_dict()
        assert d["name"] == "Uncertain Mod"


# ===================================================================
# TEST 10 — Missing custom CCD
# ===================================================================

class Test10_MissingCustomCCD:
    """Must fail if the condition requires a custom CCD that does not exist."""

    def test_custom_ccd_without_id_rejected(self):
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Custom CCD"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="custom_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "custom_mod": ModificationRecord(
                modification_id="custom_mod",
            )
        }
        af3_reg = {
            "rep_custom": AF3CompatibilityRecord(
                representation_id="rep_custom",
                modification_id="custom_mod",
                ccd_code="CCD",  # Has a CCD code, but needs custom
                af3_status="verified_custom",
                custom_ccd_required="true",
                # No custom_ccd_id!
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="custom CCD"):
            build_job(
                manifest, "c1", seeds=[1],
                construct_registry=construct_reg,
                modification_registry=mod_reg,
                af3_compatibility_registry=af3_reg,
            )

    def test_custom_ccd_with_id_resolves(self):
        """When custom_ccd_id IS provided, resolution should succeed."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Custom CCD"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="custom_mod",
            sequence_position="10", construct_id="POU_DOMAIN",
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        mod_reg = {
            "custom_mod": ModificationRecord(
                modification_id="custom_mod",
            )
        }
        af3_reg = {
            "rep_custom": AF3CompatibilityRecord(
                representation_id="rep_custom",
                modification_id="custom_mod",
                af3_status="verified_custom",
                custom_ccd_required="true",
                custom_ccd_id="MY_CUSTOM_CCD",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        # Should NOT raise — custom_ccd_id is present
        spec = resolve_condition(
            manifest, "c1",
            construct_registry=construct_reg,
            modification_registry=mod_reg,
            af3_compatibility_registry=af3_reg,
        )
        assert len(spec.proteins) == 1
        mod = spec.proteins[0].modifications[0]
        assert mod["needs_custom_ccd"] is True
        assert mod["custom_ccd_id"] == "MY_CUSTOM_CCD"


# ===================================================================
# TEST 11 — Invalid covalent bond
# ===================================================================

class Test11_InvalidCovalentBond:
    """Must fail validation when covalent bond has invalid parameters."""

    def test_unsupported_bond_rejected(self):
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Bond"
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )
        manifest.entities["e2"] = ConditionEntityRecord(
            condition_id="c1", entity_type="ligand",
            entity_id="example_ligand_A", stoichiometry="1",
        )

        bond_reg = {
            "bad_bond": CovalentBondRecord(
                bond_id="bad_bond",
                entity_1_type="protein", entity_1_id="POU_DOMAIN",
                residue_1="100", atom_1="SG",
                entity_2_type="ligand", entity_2_id="example_ligand_A",
                residue_2="1", atom_2="C",
                af3_status="unsupported",
            )
        }
        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        ligand_reg = load_ligand_registry(POU_DIR / "ligand_registry.csv")

        result = validate_manifest(
            manifest,
            construct_registry=construct_reg,
            ligand_registry=ligand_reg,
            covalent_bond_registry=bond_reg,
        )
        assert not result.is_valid
        assert any("UNSUPPORTED" in e for e in result.errors)

    def test_bond_with_invalid_residue_position(self):
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Bond Pos"
        )

        bond_reg = {
            "bad_pos": CovalentBondRecord(
                bond_id="bad_pos",
                entity_1_type="protein", entity_1_id="A",
                residue_1="abc", atom_1="SG",
                entity_2_type="ligand", entity_2_id="B",
                residue_2="1", atom_2="C",
            )
        }

        result = validate_manifest(
            manifest,
            covalent_bond_registry=bond_reg,
        )
        assert not result.is_valid
        assert any("bad_pos" in e and "not a valid integer" in e
                    for e in result.errors)

    def test_valid_bond_passes(self):
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Good Bond"
        )

        bond_reg = {
            "good_bond": CovalentBondRecord(
                bond_id="good_bond",
                entity_1_type="protein", entity_1_id="A",
                residue_1="100", atom_1="SG",
                entity_2_type="ligand", entity_2_id="B",
                residue_2="1", atom_2="C",
                af3_status="verified_native",
            )
        }

        result = validate_manifest(
            manifest,
            covalent_bond_registry=bond_reg,
        )
        assert result.is_valid


# ===================================================================
# TEST 12 — Genericity
# ===================================================================

class Test12_Genericity:
    """Use two unrelated proteins. Both must pass through the same code path."""

    def test_pou_resolves(self, pou_manifest, pou_registries):
        spec = resolve_condition(pou_manifest, "pou_baseline", **pou_registries)
        assert spec.proteins[0].entity_id == "POU_DOMAIN"
        assert len(spec.proteins[0].sequence) > 0

    def test_protein_x_resolves(self, px_manifest, px_registries):
        spec = resolve_condition(px_manifest, "kinase_baseline", **px_registries)
        assert spec.proteins[0].entity_id == "PRKX_KINASE"
        assert len(spec.proteins[0].sequence) > 0

    def test_same_code_different_proteins(self, pou_manifest, pou_registries,
                                          px_manifest, px_registries):
        """Both POU and Protein X use exactly the same build_job function."""
        pou_job = build_job(pou_manifest, "pou_baseline", seeds=[1], **pou_registries)
        px_job = build_job(px_manifest, "kinase_baseline", seeds=[1], **px_registries)

        pou_dict = pou_job.to_dict()
        px_dict = px_job.to_dict()

        assert pou_dict["dialect"] == "alphafold3"
        assert px_dict["dialect"] == "alphafold3"

        # Different proteins, different sequences
        pou_prot = [s for s in pou_dict["sequences"] if "protein" in s][0]["protein"]
        px_prot = [s for s in px_dict["sequences"] if "protein" in s][0]["protein"]
        assert pou_prot["sequence"] != px_prot["sequence"]

    def test_pou_modified_condition(self, pou_manifest, pou_registries):
        """POU with PTM."""
        job = build_job(pou_manifest, "pou_tpo101", seeds=[1], **pou_registries)
        d = job.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert "modifications" in prot
        assert prot["modifications"][0]["ptmType"] == "TPO"

    def test_px_modified_condition(self, px_manifest, px_registries):
        """Protein X with PTM."""
        job = build_job(px_manifest, "kinase_phospho", seeds=[1], **px_registries)
        d = job.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert "modifications" in prot
        assert prot["modifications"][0]["ptmType"] == "TPO"

    def test_no_protein_specific_code(self):
        """Verify the builder code contains no protein-specific conditionals."""
        import af3_builder.condition_manifest.builder as b
        source = open(b.__file__).read()
        lines = source.split("\n")
        code_lines = [
            l for l in lines
            if not l.strip().startswith("#")
            and not l.strip().startswith('"""')
        ]
        code_text = "\n".join(code_lines)
        assert "if protein ==" not in code_text.lower()
        assert '=="OCT4"' not in code_text
        assert '=="POU5F1"' not in code_text
        assert '=="PRKX"' not in code_text


# ===================================================================
# JSON FIXTURE VERIFICATION
# ===================================================================

class TestJSONFixtures:
    """Inspect representative JSON structures for correctness."""

    def test_baseline_json_structure(self, pou_manifest, pou_registries):
        """Unmodified protein: 1 protein, no mods, correct AF3 schema."""
        jb = build_job(pou_manifest, "pou_baseline", seeds=[42], **pou_registries)
        d = jb.to_dict()

        # Top-level AF3 keys
        assert "name" in d
        assert "modelSeeds" in d
        assert "sequences" in d
        assert "dialect" in d
        assert d["dialect"] == "alphafold3"
        assert "version" in d

        # Exactly 1 protein + 1 ion-as-ligand
        prots = [s for s in d["sequences"] if "protein" in s]
        ligs = [s for s in d["sequences"] if "ligand" in s]
        assert len(prots) == 1
        assert len(ligs) == 1

        # Protein structure
        prot = prots[0]["protein"]
        assert "id" in prot
        assert "sequence" in prot
        assert isinstance(prot["sequence"], str)
        assert len(prot["sequence"]) > 0
        assert "modifications" not in prot  # No modifications

    def test_modified_protein_json_structure(self, pou_manifest, pou_registries):
        """Modified protein: correct AF3 modification format."""
        jb = build_job(pou_manifest, "pou_tpo101", seeds=[42], **pou_registries)
        d = jb.to_dict()

        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert "modifications" in prot
        assert len(prot["modifications"]) == 1

        mod = prot["modifications"][0]
        # Only AF3-compatible keys
        assert set(mod.keys()) == {"ptmType", "ptmPosition"}
        assert mod["ptmType"] == "TPO"
        assert mod["ptmPosition"] == 235

    def test_protein_plus_dna_json_structure(self, pou_manifest, pou_registries):
        """Protein + DNA: both entity types present."""
        jb = build_job(pou_manifest, "pou_dna", seeds=[42], **pou_registries)
        d = jb.to_dict()

        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "dna" in types

        dna = [s for s in d["sequences"] if "dna" in s][0]["dna"]
        assert "id" in dna
        assert "sequence" in dna
        assert len(dna["sequence"]) > 0

    def test_protein_plus_ligand_json_structure(self, px_manifest, px_registries):
        """Protein + ligand: correct CCD reference."""
        jb = build_job(px_manifest, "kinase_ligand_A", seeds=[42], **px_registries)
        d = jb.to_dict()

        lig = [s for s in d["sequences"] if "ligand" in s
               if "ccdCodes" in s.get("ligand", {})][0]["ligand"]
        assert lig["ccdCodes"] == ["ATP"]

    def test_multiple_ptms_json_structure(self, pou_manifest, pou_registries):
        """Multiple PTMs: both modifications present in AF3 format."""
        jb = build_job(
            pou_manifest, "pou_tpo101_sep102", seeds=[42], **pou_registries
        )
        d = jb.to_dict()

        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 2
        ccd_codes = {m["ptmType"] for m in prot["modifications"]}
        assert ccd_codes == {"TPO", "SEP"}
        for m in prot["modifications"]:
            assert set(m.keys()) == {"ptmType", "ptmPosition"}

    def test_complex_condition_json_structure(
        self, pou_manifest, pou_registries
    ):
        """Protein + modifications + DNA + ion in one condition."""
        jb = build_job(
            pou_manifest, "pou_tpo101_sep102_dna", seeds=[42], **pou_registries
        )
        d = jb.to_dict()

        types = [list(s.keys())[0] for s in d["sequences"]]
        assert "protein" in types
        assert "dna" in types
        assert "ligand" in types  # ion

        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 2

    def test_build_all_jobs_all_valid(self, pou_manifest, pou_registries):
        """All verified-native POU conditions build valid JSON."""
        jobs = build_all_jobs(pou_manifest, seeds=[42], **pou_registries)
        # 8 original + 14 Priority 1 + 4 methylation + 4 SUMO/Ub (with bonds) = 30
        # O-GlcNAc is uncertain and excluded by default
        assert len(jobs) == 30
        for cid, jb in jobs.items():
            d = jb.to_dict()
            assert d["dialect"] == "alphafold3"
            assert len(d["sequences"]) >= 1
            assert "name" in d
            assert "modelSeeds" in d

    def test_build_all_jobs_with_uncertain(self, pou_manifest, pou_registries):
        """build_all_jobs with allow_uncertain=True builds all conditions."""
        jobs = build_all_jobs(pou_manifest, seeds=[42], allow_uncertain=True, **pou_registries)
        # All 32 conditions
        assert len(jobs) == 32
        for cid, jb in jobs.items():
            d = jb.to_dict()
            assert d["dialect"] == "alphafold3"

    def test_protein_x_all_valid(self, px_manifest, px_registries):
        """All Protein X conditions build valid JSON."""
        jobs = build_all_jobs(px_manifest, seeds=[42], **px_registries)
        assert len(jobs) == 8
        for cid, jb in jobs.items():
            d = jb.to_dict()
            assert d["dialect"] == "alphafold3"
            assert len(d["sequences"]) >= 1

    def test_seeds_in_json(self, pou_manifest, pou_registries):
        """Seeds are preserved in JSON output."""
        jb = build_job(pou_manifest, "pou_baseline", seeds=[1, 2, 3], **pou_registries)
        d = jb.to_dict()
        assert d["modelSeeds"] == [1, 2, 3]

    def test_chain_ids_unique(self, pou_manifest, pou_registries):
        """All entity chain IDs in a job are unique."""
        jb = build_job(
            pou_manifest, "pou_tpo101_sep102_dna", seeds=[1], **pou_registries
        )
        d = jb.to_dict()
        ids = []
        for s in d["sequences"]:
            for key, val in s.items():
                eid = val.get("id")
                if isinstance(eid, list):
                    ids.extend(eid)
                elif eid is not None:
                    ids.append(eid)
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"


# ===================================================================
# MODIFICATION FORMAT CONVERSION
# ===================================================================

class TestModificationFormat:
    """Verify that internal metadata doesn't leak into AF3 JSON."""

    def test_mods_to_af3_format(self):
        internal_mods = [
            {
                "modification_id": "phospho_T235",
                "ccd_code": "TPO",
                "position": 235,
                "af3_status": "verified_native",
                "needs_custom_ccd": False,
                "custom_ccd_id": "",
            }
        ]
        af3_mods = _mods_to_af3_format(internal_mods)
        assert len(af3_mods) == 1
        assert af3_mods[0] == {"ptmType": "TPO", "ptmPosition": 235}

    def test_mods_to_af3_format_skips_empty(self):
        mods = [
            {"modification_id": "x", "ccd_code": "", "position": 5},
            {"modification_id": "y", "ccd_code": "TPO", "position": None},
            {"modification_id": "z", "ccd_code": "SEP", "position": 10},
        ]
        af3_mods = _mods_to_af3_format(mods)
        assert len(af3_mods) == 1
        assert af3_mods[0]["ptmType"] == "SEP"

    def test_mods_to_af3_format_empty_list(self):
        assert _mods_to_af3_format([]) == []


# ===================================================================
# VALIDATION HARNESS
# ===================================================================

class TestValidationHarness:
    """Test the _validate_spec_for_build function."""

    def test_rejects_unsupported(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
            )],
            representation_warnings=["UNSUPPORTED: something bad"],
        )
        with pytest.raises(ValueError, match="UNSUPPORTED"):
            _validate_spec_for_build(spec)

    def test_rejects_uncertain_by_default(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
            )],
            representation_warnings=["UNCERTAIN: something unsure"],
        )
        with pytest.raises(ValueError, match="uncertain"):
            _validate_spec_for_build(spec)

    def test_allows_uncertain_with_flag(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
            )],
            representation_warnings=["UNCERTAIN: something unsure"],
        )
        # Should not raise
        _validate_spec_for_build(spec, allow_uncertain=True)

    def test_rejects_empty_ccd(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
                modifications=[{"modification_id": "mod1", "ccd_code": ""}],
            )],
        )
        with pytest.raises(ValueError, match="no CCD code"):
            _validate_spec_for_build(spec)

    def test_rejects_missing_custom_ccd_id(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
                modifications=[{
                    "modification_id": "mod1",
                    "ccd_code": "XXX",
                    "needs_custom_ccd": True,
                    "custom_ccd_id": "",
                }],
            )],
        )
        with pytest.raises(ValueError, match="custom CCD"):
            _validate_spec_for_build(spec)

    def test_rejects_empty_protein_sequence(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="",
            )],
        )
        with pytest.raises(ValueError, match="no sequence"):
            _validate_spec_for_build(spec)

    def test_rejects_ligand_without_ccd_or_smiles(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            ligands=[ResolvedEntity(
                entity_type="ligand", entity_id="X",
                chain_id="B", ccd_code="", smiles="",
            )],
        )
        with pytest.raises(ValueError, match="neither CCD code nor SMILES"):
            _validate_spec_for_build(spec)

    def test_rejects_ion_without_ccd(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            ions=[ResolvedEntity(
                entity_type="ion", entity_id="X",
                chain_id="B", ccd_code="",
            )],
        )
        with pytest.raises(ValueError, match="no CCD code"):
            _validate_spec_for_build(spec)

    def test_clean_spec_passes(self):
        spec = ConditionSpec(
            condition_id="test", condition_name="Test",
            proteins=[ResolvedEntity(
                entity_type="protein", entity_id="X",
                chain_id="A", sequence="ACDEF",
            )],
        )
        _validate_spec_for_build(spec)  # Should not raise


# ===================================================================
# PROTEIN ENTITY TYPE
# ===================================================================

class TestProteinEntityType:
    """Verify that explicit protein entity records are handled correctly."""

    def test_explicit_protein_entity_used(self):
        """A protein entity record should determine the construct."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Explicit Protein"
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        protein_reg = load_protein_registry(POU_DIR / "protein_registry.csv")

        spec = resolve_condition(
            manifest, "c1",
            construct_registry=construct_reg,
            protein_registry=protein_reg,
        )
        assert len(spec.proteins) == 1
        assert spec.proteins[0].entity_id == "POU_DOMAIN"

    def test_protein_entity_with_modifications(self):
        """Both entity record and modification reference the same construct."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Both Sources"
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="POU_DOMAIN", stoichiometry="1",
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="235", construct_id="POU_DOMAIN",
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")
        mod_reg = load_modification_registry(POU_DIR / "modification_registry.csv")
        af3_reg = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")

        spec = resolve_condition(
            manifest, "c1",
            construct_registry=construct_reg,
            modification_registry=mod_reg,
            af3_compatibility_registry=af3_reg,
        )
        # Should have exactly 1 protein (not 2 from duplication)
        assert len(spec.proteins) == 1
        assert len(spec.proteins[0].modifications) == 1

    def test_no_protein_reference_raises(self):
        """Condition with no protein entity and no modifications should fail."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="No Protein"
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="no protein references"):
            resolve_condition(
                manifest, "c1",
                construct_registry=construct_reg,
            )

    def test_unknown_construct_raises(self):
        """Referencing a construct that doesn't exist should fail."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Bad Construct"
        )
        manifest.entities["e1"] = ConditionEntityRecord(
            condition_id="c1", entity_type="protein",
            entity_id="NONEXISTENT_CONSTRUCT", stoichiometry="1",
        )

        construct_reg = load_construct_registry(POU_DIR / "construct_registry.csv")

        with pytest.raises(ValueError, match="unknown construct"):
            resolve_condition(
                manifest, "c1",
                construct_registry=construct_reg,
            )


# ===================================================================
# CHAIN LETTER HELPERS (extended)
# ===================================================================

class TestChainHelpersExtended:
    """Additional chain helper tests."""

    def test_int_to_chain_high_values(self):
        assert _int_to_chain(25) == "Z"
        assert _int_to_chain(26) == "AA"
        assert _int_to_chain(27) == "AB"
        assert _int_to_chain(51) == "AZ"
        assert _int_to_chain(52) == "BA"

    def test_chain_id_uniqueness_in_complex_job(self, pou_manifest, pou_registries):
        """A condition with protein + DNA + ion should have unique chain IDs."""
        spec = resolve_condition(
            pou_manifest, "pou_tpo101_sep102_dna", **pou_registries
        )
        chain_ids = [e.chain_id for e in spec.all_entities]
        assert len(chain_ids) == len(set(chain_ids))


# ===================================================================
# SEPARATION OF CONCERNS
# ===================================================================

class TestSeparationOfConcerns:
    """Verify seeds are separate from condition identity."""

    def test_different_seeds_same_condition(self, pou_manifest, pou_registries):
        """Same biological condition, different seeds → different JSON, same entities."""
        jb1 = build_job(pou_manifest, "pou_tpo101", seeds=[1], **pou_registries)
        jb2 = build_job(pou_manifest, "pou_tpo101", seeds=[2], **pou_registries)

        d1 = jb1.to_dict()
        d2 = jb2.to_dict()

        # Seeds differ
        assert d1["modelSeeds"] != d2["modelSeeds"]

        # But entities are identical
        assert len(d1["sequences"]) == len(d2["sequences"])
        for s1, s2 in zip(d1["sequences"], d2["sequences"]):
            k1 = list(s1.keys())[0]
            k2 = list(s2.keys())[0]
            assert k1 == k2
            assert s1[k1] == s2[k2]

    def test_build_job_seed_compatibility(self, pou_manifest, pou_registries):
        """build_job with seeds=[1,2,3] preserves all seeds."""
        jb = build_job(pou_manifest, "pou_baseline", seeds=[1, 2, 3], **pou_registries)
        d = jb.to_dict()
        assert d["modelSeeds"] == [1, 2, 3]


class TestAF3SchemaCompatibility:
    """Regression: protein PTM modifications must use AF3-native field names.

    AF3 folding_input.py ProteinChain.from_dict() expects:
        {"ptmType": "TPO", "ptmPosition": 101}

    The pipeline must NOT generate:
        {"ccdCode": "TPO", "position": 101}
    """

    def test_ptm_fields_match_af3_schema(self, pou_manifest, pou_registries):
        """PTM objects must use ptmType/ptmPosition, not ccdCode/position."""
        jb = build_job(pou_manifest, "pou_tpo101", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 1
        mod = prot["modifications"][0]
        assert "ptmType" in mod, f"Missing ptmType; got keys: {list(mod.keys())}"
        assert "ptmPosition" in mod, f"Missing ptmPosition; got keys: {list(mod.keys())}"
        assert "ccdCode" not in mod, f"Old field ccdCode still present: {mod}"
        assert "position" not in mod, f"Old field position still present: {mod}"
        assert mod["ptmType"] == "TPO"
        assert mod["ptmPosition"] == 235

    def test_multi_ptm_fields_match_af3_schema(self, pou_manifest, pou_registries):
        """Multiple PTMs must each use ptmType/ptmPosition."""
        jb = build_job(pou_manifest, "pou_tpo101_sep102", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 2
        for mod in prot["modifications"]:
            assert "ptmType" in mod
            assert "ptmPosition" in mod
            assert "ccdCode" not in mod
            assert "position" not in mod

    def test_sep_modification_uses_af3_fields(self, pou_manifest, pou_registries):
        """SEP (phosphoserine) must use ptmType/ptmPosition."""
        jb = build_job(pou_manifest, "pou_sep102", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        mod = prot["modifications"][0]
        assert mod == {"ptmType": "SEP", "ptmPosition": 236}

    def test_no_modification_fields_leak_to_json(self, pou_manifest, pou_registries):
        """Internal metadata (modification_id, af3_status, etc.) must not appear in JSON."""
        jb = build_job(pou_manifest, "pou_tpo101", seeds=[1], **pou_registries)
        d = jb.to_dict()
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        mod = prot["modifications"][0]
        # Only ptmType and ptmPosition should be present
        assert set(mod.keys()) == {"ptmType", "ptmPosition"}

    def test_build_all_jobs_ptm_fields(self, pou_manifest, pou_registries):
        """build_all_jobs must produce correct PTM fields for every condition."""
        jobs = build_all_jobs(pou_manifest, seeds=[1], **pou_registries)
        for cid, jb in jobs.items():
            d = jb.to_dict()
            for seq in d["sequences"]:
                if "protein" in seq:
                    for mod in seq["protein"].get("modifications", []):
                        assert "ptmType" in mod, f"{cid}: missing ptmType in {mod}"
                        assert "ptmPosition" in mod, f"{cid}: missing ptmPosition in {mod}"
                        assert "ccdCode" not in mod, f"{cid}: old ccdCode in {mod}"


# ---------------------------------------------------------------------------
# Priority 2 — AF3 Representation Validation
# ---------------------------------------------------------------------------

class TestPriority2Representations:
    """Test Priority 2 biological representations through the generic pipeline."""

    def test_methylation_resolves_correctly(self, pou_manifest, pou_registries):
        """K222 monomethylation and dimethylation resolve with correct CCDs."""
        me1 = resolve_condition(pou_manifest, "oct4_monoMe_K222", **pou_registries)
        me2 = resolve_condition(pou_manifest, "oct4_diMe_K222", **pou_registries)
        # K222-me1: MLY (monomethyl-lysine)
        me1_mods = [m for p in me1.proteins for m in p.modifications]
        assert len(me1_mods) == 1
        assert me1_mods[0]["modification_id"] == "monoMe_K222"
        assert me1_mods[0]["ccd_code"] == "MLY"
        assert me1_mods[0]["af3_status"] == "verified_native"
        # K222-me2: MLZ (dimethyl-lysine)
        me2_mods = [m for p in me2.proteins for m in p.modifications]
        assert len(me2_mods) == 1
        assert me2_mods[0]["modification_id"] == "diMe_K222"
        assert me2_mods[0]["ccd_code"] == "MLZ"
        assert me2_mods[0]["af3_status"] == "verified_native"

    def test_methylation_builds_valid_json(self, pou_manifest, pou_registries):
        """K222 methylation conditions build valid AF3 JSON."""
        for cid in ["oct4_monoMe_K222", "oct4_diMe_K222"]:
            jb = build_job(pou_manifest, cid, seeds=[1], **pou_registries)
            d = jb.to_dict()
            assert d["dialect"] == "alphafold3"
            prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
            assert len(prot["modifications"]) == 1
            mod = prot["modifications"][0]
            assert mod["ptmType"] in ("MLY", "MLZ")

    def test_methylation_dna_variants(self, pou_manifest, pou_registries):
        """K222 methylation + DNA conditions include DNA entity."""
        for cid in ["oct4_monoMe_K222_dna", "oct4_diMe_K222_dna"]:
            jb = build_job(pou_manifest, cid, seeds=[1], **pou_registries)
            d = jb.to_dict()
            types = [list(s.keys())[0] for s in d["sequences"]]
            assert "protein" in types
            assert "dna" in types
            # Methylation still present
            prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
            assert len(prot["modifications"]) == 1

    def test_oglcna_resolves_uncertain(self, pou_manifest, pou_registries):
        """O-GlcNAc at S236 resolves as uncertain representation."""
        spec = resolve_condition(pou_manifest, "oct4_OGlcNAc_S236", **pou_registries)
        mods = [m for p in spec.proteins for m in p.modifications]
        assert len(mods) == 1
        assert mods[0]["modification_id"] == "OGlcNAc_S236"
        assert mods[0]["ccd_code"] == "OGT"
        assert mods[0]["af3_status"] == "representation_uncertain"
        # Should be flagged as uncertain
        assert spec.has_unsupported_representations or any(
            "UNCERTAIN" in w for w in spec.representation_warnings
        )

    def test_oglcna_builds_with_allow_uncertain(self, pou_manifest, pou_registries):
        """O-GlcNAc builds valid JSON when allow_uncertain is True."""
        jb = build_job(pou_manifest, "oct4_OGlcNAc_S236", seeds=[1],
                       allow_uncertain=True, **pou_registries)
        d = jb.to_dict()
        assert d["dialect"] == "alphafold3"
        prot = [s for s in d["sequences"] if "protein" in s][0]["protein"]
        assert len(prot["modifications"]) == 1
        assert prot["modifications"][0]["ptmType"] == "OGT"
        assert prot["modifications"][0]["ptmPosition"] == 236

    def test_sumo_builds_separate_entity_with_bond(self, pou_manifest, pou_registries):
        """K123-SUMO builds with 2 protein entities and a covalent bond."""
        jb = build_job(pou_manifest, "oct4_SUMO_K123", seeds=[1],
                       allow_uncertain=True, **pou_registries)
        d = jb.to_dict()
        assert d["dialect"] == "alphafold3"
        # Two protein entities (OCT4 + SUMO)
        prot_seqs = [s for s in d["sequences"] if "protein" in s]
        assert len(prot_seqs) == 2
        # Covalent bond present
        assert "bondedAtomPairs" in d
        assert len(d["bondedAtomPairs"]) == 1
        bond = d["bondedAtomPairs"][0]
        # Bond links residue 123 (NZ) on entity A to residue on entity B
        assert bond[0][1] == 123  # Lys 123
        assert bond[0][2] == "NZ"  # epsilon nitrogen

    def test_ubiquitin_builds_separate_entity_with_bond(self, pou_manifest, pou_registries):
        """K133-Ub builds with 2 protein entities and a covalent bond."""
        jb = build_job(pou_manifest, "oct4_UB_K133", seeds=[1],
                       allow_uncertain=True, **pou_registries)
        d = jb.to_dict()
        assert d["dialect"] == "alphafold3"
        prot_seqs = [s for s in d["sequences"] if "protein" in s]
        assert len(prot_seqs) == 2
        assert "bondedAtomPairs" in d
        assert len(d["bondedAtomPairs"]) == 1
        bond = d["bondedAtomPairs"][0]
        assert bond[0][1] == 133  # Lys 133
        assert bond[0][2] == "NZ"

    def test_sumo_dna_includes_dna(self, pou_manifest, pou_registries):
        """K123-SUMO + DNA includes all three entity types."""
        jb = build_job(pou_manifest, "oct4_SUMO_K123_dna", seeds=[1],
                       allow_uncertain=True, **pou_registries)
        d = jb.to_dict()
        types = [list(s.keys())[0] for s in d["sequences"]]
        assert types.count("protein") == 2  # OCT4 + SUMO
        assert "dna" in types
        assert "bondedAtomPairs" in d

    def test_all_priority2_build_with_allow_uncertain(self, pou_manifest, pou_registries):
        """All 10 Priority 2 conditions build with allow_uncertain=True."""
        p2_ids = [
            "oct4_monoMe_K222", "oct4_diMe_K222",
            "oct4_monoMe_K222_dna", "oct4_diMe_K222_dna",
            "oct4_OGlcNAc_S236", "oct4_OGlcNAc_S236_dna",
            "oct4_SUMO_K123", "oct4_SUMO_K123_dna",
            "oct4_UB_K133", "oct4_UB_K133_dna",
        ]
        for cid in p2_ids:
            jb = build_job(pou_manifest, cid, seeds=[1],
                           allow_uncertain=True, **pou_registries)
            d = jb.to_dict()
            assert d["dialect"] == "alphafold3", f"{cid}: bad dialect"
            assert len(d["sequences"]) >= 1, f"{cid}: no sequences"
            assert "modelSeeds" in d, f"{cid}: no seeds"

    def test_all_priority2_shared_seeds(self, pou_manifest, pou_registries):
        """All Priority 2 conditions use the same 10 seeds."""
        p2_ids = [
            "oct4_monoMe_K222", "oct4_SUMO_K123",
            "oct4_UB_K133", "oct4_OGlcNAc_S236",
        ]
        expected_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for cid in p2_ids:
            jb = build_job(pou_manifest, cid, seeds=expected_seeds,
                           allow_uncertain=True, **pou_registries)
            d = jb.to_dict()
            assert d["modelSeeds"] == expected_seeds, f"{cid}: wrong seeds"
