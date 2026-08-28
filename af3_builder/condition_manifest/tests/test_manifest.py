"""
Tests for the condition_manifest package.

Tests cover:
1. POU/OCT4 dataset loads correctly
2. All 8 POU conditions map correctly
3. Arbitrary attribute names work
4. Multiple components work
5. Protein X dataset works through identical architecture
6. Incomplete combinations are detected
7. Missing condition metadata produces explicit errors
8. Duplicate definitions produce explicit errors
9. No hard-coded protein/PTM assumptions in the code
"""

import pytest
from pathlib import Path

from af3_builder.condition_manifest.registries import (
    ProteinRecord,
    ConstructRecord,
    ModificationRecord,
    NucleicAcidRecord,
    LigandRecord,
    IonRecord,
    PartnerRecord,
    AF3CompatibilityRecord,
    ResidueMappingRecord,
    load_protein_registry,
    load_construct_registry,
    load_modification_registry,
    load_nucleic_acid_registry,
    load_ligand_registry,
    load_ion_registry,
    load_partner_registry,
    load_af3_compatibility_registry,
    load_residue_mapping_registry,
    load_csv_registry,
)
from af3_builder.condition_manifest.manifest import (
    ConditionRecord,
    ConditionModificationRecord,
    ConditionEntityRecord,
    ConditionFactorRecord,
    MasterManifest,
    load_master_manifest,
)
from af3_builder.condition_manifest.validation import (
    ValidationResult,
    validate_manifest,
)
from af3_builder.condition_manifest.inspection import (
    ManifestInspection,
    inspect_manifest,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POU_DIR = Path(__file__).resolve().parents[4] / "testdata" / "pou2" / "registries"
PROTEIN_X_DIR = Path(__file__).resolve().parents[4] / "testdata" / "protein_x" / "registries"


# ---------------------------------------------------------------------------
# 1. POU dataset loads correctly
# ---------------------------------------------------------------------------

class TestPOUDatasetLoading:
    """Test that the POU/OCT4 dataset loads correctly."""

    def test_protein_registry_loads(self):
        proteins = load_protein_registry(POU_DIR / "protein_registry.csv")
        assert "POU5F1" in proteins
        assert proteins["POU5F1"].protein_name == "OCT4 POU domain"
        assert proteins["POU5F1"].gene_name == "POU5F1"
        assert proteins["POU5F1"].uniprot_id == "Q01860"

    def test_construct_registry_loads(self):
        constructs = load_construct_registry(POU_DIR / "construct_registry.csv")
        assert "POU_DOMAIN" in constructs
        assert constructs["POU_DOMAIN"].protein_id == "POU5F1"
        assert constructs["POU_DOMAIN"].construct_sequence != ""

    def test_modification_registry_loads(self):
        mods = load_modification_registry(POU_DIR / "modification_registry.csv")
        assert "phospho_T235" in mods
        assert "phospho_S236" in mods
        assert mods["phospho_T235"].modification_class == "phosphorylation"
        assert mods["phospho_T235"].base_residue == "T"
        assert mods["phospho_T235"].modified_residue == "TPO"

    def test_nucleic_acid_registry_loads(self):
        nucs = load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv")
        assert "ref_dna_duplex" in nucs
        assert nucs["ref_dna_duplex"].entity_type == "dna"
        assert nucs["ref_dna_duplex"].sequence != ""

    def test_ligand_registry_loads(self):
        ligands = load_ligand_registry(POU_DIR / "ligand_registry.csv")
        assert "example_ligand_A" in ligands
        assert ligands["example_ligand_A"].ccd_code == "ATP"

    def test_ion_registry_loads(self):
        ions = load_ion_registry(POU_DIR / "ion_registry.csv")
        assert "mg_ion" in ions
        assert ions["mg_ion"].charge == "+2"

    def test_partner_registry_loads(self):
        partners = load_partner_registry(POU_DIR / "partner_registry.csv")
        assert "sox2_partner" in partners
        assert partners["sox2_partner"].uniprot_id == "P48431"

    def test_af3_compatibility_registry_loads(self):
        af3 = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")
        assert "rep_phospho_T235" in af3
        assert af3["rep_phospho_T235"].ccd_code == "TPO"
        # Verify T235-P and S236-P entries exist
        assert "rep_phospho_T235" in af3
        assert af3["rep_phospho_T235"].ccd_code == "TPO"
        assert "rep_phospho_S236" in af3
        assert af3["rep_phospho_S236"].ccd_code == "SEP"

    def test_residue_mapping_registry_loads(self):
        mappings = load_residue_mapping_registry(POU_DIR / "residue_mapping_registry.csv")
        assert len(mappings) >= 4
        # Check that mapping IDs are unique
        ids = list(mappings.keys())
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 2. All 8 POU conditions map correctly
# ---------------------------------------------------------------------------

class TestPOUConditions:
    """Test that all 8 POU conditions are correctly represented."""

    @pytest.fixture
    def manifest(self):
        return load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
            exclusivity_path=POU_DIR / "exclusivity_rules.csv",
        )

    def test_conditions_loaded(self, manifest):
        # 8 original + 14 Priority 1 + 10 Priority 2 = 32
        assert len(manifest.conditions) == 32

    def test_all_condition_ids_present(self, manifest):
        # Original 8 test conditions
        original = {
            "pou_baseline", "pou_tpo101", "pou_sep102",
            "pou_tpo101_sep102", "pou_dna", "pou_tpo101_dna",
            "pou_sep102_dna", "pou_tpo101_sep102_dna",
        }
        # 14 Priority 1 conditions
        priority1 = {
            "oct4_wt", "oct4_wt_dna",
            "oct4_t101p", "oct4_t101p_dna",
            "oct4_s102p", "oct4_s102p_dna",
            "oct4_t101p_s102p", "oct4_t101p_s102p_dna",
            "oct4_t235p", "oct4_t235p_dna",
            "oct4_s236p", "oct4_s236p_dna",
            "oct4_t235p_s236p", "oct4_t235p_s236p_dna",
        }
        # 10 Priority 2 conditions
        priority2 = {
            "oct4_monoMe_K222", "oct4_diMe_K222",
            "oct4_OGlcNAc_S236",
            "oct4_SUMO_K123", "oct4_UB_K133",
            "oct4_monoMe_K222_dna", "oct4_diMe_K222_dna",
            "oct4_OGlcNAc_S236_dna",
            "oct4_SUMO_K123_dna", "oct4_UB_K133_dna",
        }
        assert set(manifest.condition_ids) == original | priority1 | priority2

    def test_baseline_has_no_modifications(self, manifest):
        mods = manifest.get_modifications_for_condition("pou_baseline")
        assert len(mods) == 0

    def test_tpo101_has_one_modification(self, manifest):
        mods = manifest.get_modifications_for_condition("pou_tpo101")
        assert len(mods) == 1
        assert mods[0].modification_id == "phospho_T235"

    def test_tpo101_sep102_has_two_modifications(self, manifest):
        mods = manifest.get_modifications_for_condition("pou_tpo101_sep102")
        assert len(mods) == 2
        mod_ids = {m.modification_id for m in mods}
        assert mod_ids == {"phospho_T235", "phospho_S236"}

    def test_dna_conditions_have_dna_entity(self, manifest):
        dna_conditions = ["pou_dna", "pou_tpo101_dna", "pou_sep102_dna", "pou_tpo101_sep102_dna"]
        for cid in dna_conditions:
            entities = manifest.get_entities_for_condition(cid)
            dna_entities = [e for e in entities if e.entity_type == "dna"]
            assert len(dna_entities) == 1
            assert dna_entities[0].entity_id == "ref_dna_duplex"

    def test_no_dna_in_non_dna_conditions(self, manifest):
        non_dna = ["pou_baseline", "pou_tpo101", "pou_sep102", "pou_tpo101_sep102"]
        for cid in non_dna:
            entities = manifest.get_entities_for_condition(cid)
            dna_entities = [e for e in entities if e.entity_type == "dna"]
            assert len(dna_entities) == 0

    def test_all_conditions_have_mg_ion(self, manifest):
        for cid in manifest.condition_ids:
            entities = manifest.get_entities_for_condition(cid)
            ion_entities = [e for e in entities if e.entity_type == "ion"]
            assert len(ion_entities) >= 1
            assert ion_entities[0].entity_id == "mg_ion"


# ---------------------------------------------------------------------------
# 3. Arbitrary attribute names work
# ---------------------------------------------------------------------------

class TestArbitraryAttributes:
    """Test that the system works with any factor names, not just DNA/pTPO."""

    @pytest.fixture
    def manifest(self):
        return load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )

    def test_factor_names_are_generic(self, manifest):
        """Factor names come from data, not hard-coded."""
        factor_names = manifest.get_attribute_names()
        assert "DNA" in factor_names
        assert "pTPO101" in factor_names
        assert "pSEP102" in factor_names
        # No hard-coded 'factor_DNA' or 'factor_PTM' in the code
        assert "factor_DNA" not in factor_names

    def test_factor_values_are_data_driven(self, manifest):
        """Factor levels come from data."""
        dna_values = manifest.get_attribute_values("DNA")
        assert dna_values == {"absent", "present"}

        tpo_values = manifest.get_attribute_values("pTPO101")
        assert tpo_values == {"absent", "present"}

    def test_condition_attributes_matrix(self, manifest):
        matrix = manifest.condition_attributes_matrix()
        assert "pou_baseline" in matrix
        assert matrix["pou_baseline"]["DNA"] == "absent"
        assert matrix["pou_tpo101"]["pTPO101"] == "present"
        assert matrix["pou_tpo101_sep102_dna"]["DNA"] == "present"
        assert matrix["pou_tpo101_sep102_dna"]["pTPO101"] == "present"
        assert matrix["pou_tpo101_sep102_dna"]["pSEP102"] == "present"


# ---------------------------------------------------------------------------
# 4. Multiple components work
# ---------------------------------------------------------------------------

class TestMultipleComponents:
    """Test that conditions can have arbitrary combinations of components."""

    def test_condition_with_modification_and_dna(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        mods = manifest.get_modifications_for_condition("pou_tpo101_sep102_dna")
        entities = manifest.get_entities_for_condition("pou_tpo101_sep102_dna")
        assert len(mods) == 2  # Two PTMs
        dna_ents = [e for e in entities if e.entity_type == "dna"]
        assert len(dna_ents) == 1  # One DNA

    def test_stoichiometry_preserved(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        for cid in manifest.condition_ids:
            entities = manifest.get_entities_for_condition(cid)
            for ent in entities:
                assert ent.stoichiometry == "1"  # All are 1:1 in POU dataset


# ---------------------------------------------------------------------------
# 5. Protein X works through identical architecture
# ---------------------------------------------------------------------------

class TestProteinXDataset:
    """Test that a completely different protein works through the same architecture."""

    @pytest.fixture
    def manifest(self):
        return load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
            entities_path=PROTEIN_X_DIR / "condition_entities.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
            exclusivity_path=PROTEIN_X_DIR / "exclusivity_rules.csv",
        )

    def test_8_conditions_loaded(self, manifest):
        assert len(manifest.conditions) == 8

    def test_different_protein(self):
        proteins = load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv")
        assert "PROTEIN_X" in proteins
        assert proteins["PROTEIN_X"].gene_name == "PRKX"

    def test_different_modifications(self):
        mods = load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv")
        assert "phospho_K42" in mods
        assert "acetyl_K56" in mods
        # Different modification classes
        assert mods["acetyl_K56"].modification_class == "acetylation"

    def test_different_entities(self, manifest):
        """Protein X has ligands and partners, not DNA."""
        ligand_conds = ["kinase_ligand_A", "kinase_ligand_B", "kinase_phospho_ligand", "kinase_multi"]
        for cid in ligand_conds:
            entities = manifest.get_entities_for_condition(cid)
            ligands = [e for e in entities if e.entity_type == "ligand"]
            assert len(ligands) >= 1

        partner_conds = ["kinase_partner", "kinase_multi"]
        for cid in partner_conds:
            entities = manifest.get_entities_for_condition(cid)
            partners = [e for e in entities if e.entity_type == "partner"]
            assert len(partners) >= 1

    def test_different_factor_names(self, manifest):
        """Protein X uses completely different factor names."""
        factor_names = manifest.get_attribute_names()
        assert "phosphorylation" in factor_names
        assert "acetylation" in factor_names
        assert "ligand" in factor_names
        assert "partner" in factor_names
        # These are NOT binary — ligand has 3 levels (none, ATP, NAD)
        ligand_levels = manifest.get_attribute_values("ligand")
        assert ligand_levels == {"none", "ATP", "NAD"}

    def test_same_code_different_data(self):
        """Both datasets use the exact same loading code."""
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
        # Both loaded successfully
        assert len(pou_manifest.conditions) == 32
        assert len(px_manifest.conditions) == 8
        # POU has 11 factor types (DNA + 6 phosphorylation + methylation + SUMO + Ub + O-GlcNAc factors)
        # The exact count depends on the condition_factors.csv
        pou_attrs = pou_manifest.get_attribute_names()
        assert len(pou_attrs) >= 7  # at least the original 7 factor types
        assert len(px_manifest.get_attribute_names()) == 4


# ---------------------------------------------------------------------------
# 6. Incomplete combinations are detected
# ---------------------------------------------------------------------------

class TestIncompleteDesign:
    """Test that incomplete factorial designs are correctly identified."""

    def test_pou_dataset_structure(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        insp = inspect_manifest(manifest)
        # 32 conditions total (8 original + 14 Priority 1 + 10 Priority 2)
        assert insp.n_conditions == 32
        # 12 factor types (DNA + 6 phosphorylation + 2 methylation + SUMO + Ub + O-GlcNAc)
        assert insp.n_factors == 12

    def test_protein_x_is_incomplete(self):
        manifest = load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
        )
        insp = inspect_manifest(manifest)
        # Protein X has 4 factors with 2+ levels each — full factorial would be large
        # but only 8 conditions are observed
        assert not insp.is_complete_factorial
        assert insp.expected_n_conditions > 8
        assert len(insp.missing_combinations) > 0


# ---------------------------------------------------------------------------
# 7. Missing condition metadata produces explicit errors
# ---------------------------------------------------------------------------

class TestMissingMetadata:
    """Test that missing or invalid metadata produces clear errors."""

    def test_missing_manifest_file(self):
        with pytest.raises(FileNotFoundError):
            load_master_manifest(Path("nonexistent.csv"))

    def test_missing_required_fields(self):
        """Empty condition_id should raise."""
        with pytest.raises(ValueError, match="condition_id is required"):
            ConditionRecord(condition_id="", condition_name="test")

    def test_empty_condition_name(self):
        with pytest.raises(ValueError, match="condition_name is required"):
            ConditionRecord(condition_id="x", condition_name="")

    def test_protein_record_requires_id(self):
        with pytest.raises(ValueError, match="protein_id is required"):
            ProteinRecord(protein_id="")

    def test_modification_record_requires_id(self):
        with pytest.raises(ValueError, match="modification_id is required"):
            ModificationRecord(modification_id="")

    def test_nucleic_acid_invalid_type(self):
        with pytest.raises(ValueError, match="entity_type must be"):
            NucleicAcidRecord(entity_id="x", entity_type="protein")


# ---------------------------------------------------------------------------
# 8. Duplicate definitions produce explicit errors
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Test that duplicate IDs are detected."""

    def test_duplicate_condition_ids_in_csv(self):
        """A CSV with duplicate IDs should raise on load."""
        import tempfile
        import csv
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['condition_id', 'condition_name', 'condition_group',
                           'parent_condition_id', 'status', 'description',
                           'biological_rationale', 'experimental_tier',
                           'experimental_priority', 'notes'])
            writer.writerow(['cond_1', 'Condition 1', '', '', 'complete', '', '', '', '', ''])
            writer.writerow(['cond_1', 'Condition 1 DUPLICATE', '', '', 'complete', '', '', '', '', ''])
            tmp_path = f.name

        from af3_builder.condition_manifest.manifest import _load_rows_as_records
        with pytest.raises(ValueError, match="Duplicate ID"):
            load_csv_registry(
                Path(tmp_path),
                ConditionRecord,
                "condition_id",
            )
        Path(tmp_path).unlink()

    def test_duplicate_modification_in_junction(self):
        """Duplicate modification records should be flagged by validation."""
        manifest = MasterManifest()
        manifest.conditions["c1"] = ConditionRecord(
            condition_id="c1", condition_name="Test"
        )
        manifest.modifications["m1"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="235", construct_id="POU_DOMAIN",
        )
        manifest.modifications["m2"] = ConditionModificationRecord(
            condition_id="c1", modification_id="phospho_T235",
            sequence_position="235", construct_id="POU_DOMAIN",
        )
        result = validate_manifest(manifest)
        assert not result.is_valid
        assert any("Duplicate" in e for e in result.errors)


# ---------------------------------------------------------------------------
# 9. No hard-coded protein/PTM assumptions
# ---------------------------------------------------------------------------

class TestNoHardcodedAssumptions:
    """Verify that the code contains no protein-specific logic."""

    def test_no_oct4_in_code(self):
        """The manifest code should not reference OCT4."""
        import af3_builder.condition_manifest.manifest as m
        import af3_builder.condition_manifest.registries as r
        import af3_builder.condition_manifest.validation as v
        import af3_builder.condition_manifest.inspection as insp

        for module in [m, r, v, insp]:
            source = open(module.__file__).read()
            # Check for protein-specific logic (not in comments/docstrings)
            # Remove docstrings and comments
            lines = source.split('\n')
            code_lines = [l for l in lines if not l.strip().startswith('#') and not l.strip().startswith('"""')]
            code_text = '\n'.join(code_lines)
            # Should not contain protein-specific conditionals
            assert 'if protein ==' not in code_text.lower()
            assert 'if ptm ==' not in code_text.lower()
            assert '=="OCT4"' not in code_text
            assert '=="POU5F1"' not in code_text
            assert '=="TPO101"' not in code_text
            assert '=="SEP102"' not in code_text

    def test_generic_factor_handling(self):
        """Factors should be handled generically, not by name."""
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        # The code should work with ANY factor names
        # This is tested by loading protein_x with different factor names
        # and verifying the same functions work
        matrix = manifest.condition_attributes_matrix()
        # Should contain all factor names from data
        all_factors = set()
        for cond_factors in matrix.values():
            all_factors.update(cond_factors.keys())
        assert len(all_factors) == 12  # DNA + 6 phospho + 2 methylation + SUMO + Ub + O-GlcNAc


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test the validation system."""

    def test_valid_pou_manifest(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            modifications_path=POU_DIR / "condition_modifications.csv",
            entities_path=POU_DIR / "condition_entities.csv",
            factors_path=POU_DIR / "condition_factors.csv",
            exclusivity_path=POU_DIR / "exclusivity_rules.csv",
        )
        proteins = load_protein_registry(POU_DIR / "protein_registry.csv")
        constructs = load_construct_registry(POU_DIR / "construct_registry.csv")
        mods = load_modification_registry(POU_DIR / "modification_registry.csv")
        nucs = load_nucleic_acid_registry(POU_DIR / "nucleic_acid_registry.csv")
        ions = load_ion_registry(POU_DIR / "ion_registry.csv")
        af3 = load_af3_compatibility_registry(POU_DIR / "af3_compatibility_registry.csv")

        result = validate_manifest(
            manifest,
            protein_registry=proteins,
            construct_registry=constructs,
            modification_registry=mods,
            nucleic_acid_registry=nucs,
            ion_registry=ions,
            af3_registry=af3,
        )
        # Should be valid (no errors)
        assert result.is_valid, result.summary()

    def test_valid_protein_x_manifest(self):
        manifest = load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            modifications_path=PROTEIN_X_DIR / "condition_modifications.csv",
            entities_path=PROTEIN_X_DIR / "condition_entities.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
            exclusivity_path=PROTEIN_X_DIR / "exclusivity_rules.csv",
        )
        proteins = load_protein_registry(PROTEIN_X_DIR / "protein_registry.csv")
        constructs = load_construct_registry(PROTEIN_X_DIR / "construct_registry.csv")
        mods = load_modification_registry(PROTEIN_X_DIR / "modification_registry.csv")
        ligands = load_ligand_registry(PROTEIN_X_DIR / "ligand_registry.csv")
        ions = load_ion_registry(PROTEIN_X_DIR / "ion_registry.csv")
        partners = load_partner_registry(PROTEIN_X_DIR / "partner_registry.csv")
        af3 = load_af3_compatibility_registry(PROTEIN_X_DIR / "af3_compatibility_registry.csv")

        result = validate_manifest(
            manifest,
            protein_registry=proteins,
            construct_registry=constructs,
            modification_registry=mods,
            ligand_registry=ligands,
            ion_registry=ions,
            partner_registry=partners,
            af3_registry=af3,
        )
        assert result.is_valid, result.summary()

    def test_observed_vs_defined_conditions(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        # All observed conditions are defined
        result = validate_manifest(
            manifest,
            observed_condition_names=manifest.condition_names,
        )
        assert result.is_valid

    def test_unknown_condition_detected(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        result = validate_manifest(
            manifest,
            observed_condition_names=manifest.condition_names + ["unknown_condition"],
        )
        assert any("unknown_condition" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Inspection tests
# ---------------------------------------------------------------------------

class TestInspection:
    """Test the design inspection module."""

    def test_pou_inspection(self):
        manifest = load_master_manifest(
            POU_DIR / "master_condition_manifest.csv",
            factors_path=POU_DIR / "condition_factors.csv",
        )
        insp = inspect_manifest(manifest)
        assert insp.n_conditions == 32
        assert insp.n_factors == 12
        summary = insp.summary()
        assert "32" in summary  # 32 conditions
        assert "12" in summary  # 12 factors

    def test_protein_x_inspection(self):
        manifest = load_master_manifest(
            PROTEIN_X_DIR / "master_condition_manifest.csv",
            factors_path=PROTEIN_X_DIR / "condition_factors.csv",
        )
        insp = inspect_manifest(manifest)
        assert insp.n_conditions == 8
        assert insp.n_factors == 4
        assert not insp.is_complete_factorial
        assert insp.expected_n_conditions > 8
