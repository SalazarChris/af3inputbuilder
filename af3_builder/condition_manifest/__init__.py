"""
Condition Manifest — the authoritative registry of experimental conditions.

This subpackage lives inside af3_builder and provides:

* **Registries** — reusable catalogs of proteins, modifications, nucleic acids,
  ligands, ions, partners, and AF3 representation metadata.
* **Master condition manifest** — the authoritative registry of all experimental
  conditions and their biological components.
* **Builder integration** — resolves a condition into a fully populated
  ``JobBuilder`` with all entities, modifications, and bonds.

The manifest is the **input specification** for the builder. When you define
a condition, you're telling the builder exactly what to model.

Adding a new protein is a **data-entry operation**, not a programming operation.
"""

from .registries import (
    ProteinRecord,
    ConstructRecord,
    ModificationRecord,
    NucleicAcidRecord,
    LigandRecord,
    IonRecord,
    PartnerRecord,
    AF3CompatibilityRecord,
    CovalentBondRecord,
    ResidueMappingRecord,
    AF3RepresentationStatus,
    is_known_ccd_code,
    load_csv_registry,
    load_protein_registry,
    load_construct_registry,
    load_modification_registry,
    load_nucleic_acid_registry,
    load_ligand_registry,
    load_ion_registry,
    load_partner_registry,
    load_af3_compatibility_registry,
    load_covalent_bond_registry,
    load_residue_mapping_registry,
)

from .manifest import (
    ConditionRecord,
    ConditionModificationRecord,
    ConditionEntityRecord,
    ConditionFactorRecord,
    MasterManifest,
    load_master_manifest,
    load_condition_modifications,
    load_condition_entities,
    load_condition_factors,
)

from .validation import (
    ValidationResult,
    validate_manifest,
)

from .inspection import (
    ManifestInspection,
    inspect_manifest,
)

# Builder integration — this is the key piece
from .builder import (
    ConditionSpec,
    ResolvedEntity,
    resolve_condition,
    build_job,
    build_all_jobs,
)

__all__ = [
    # Registries
    "ProteinRecord",
    "ConstructRecord",
    "ModificationRecord",
    "NucleicAcidRecord",
    "LigandRecord",
    "IonRecord",
    "PartnerRecord",
    "AF3CompatibilityRecord",
    "ResidueMappingRecord",
    "load_csv_registry",
    "load_protein_registry",
    "load_construct_registry",
    "load_modification_registry",
    "load_nucleic_acid_registry",
    "load_ligand_registry",
    "load_ion_registry",
    "load_partner_registry",
    "load_af3_compatibility_registry",
    "load_residue_mapping_registry",
    # Manifest
    "ConditionRecord",
    "ConditionModificationRecord",
    "ConditionEntityRecord",
    "ConditionFactorRecord",
    "MasterManifest",
    "load_master_manifest",
    "load_condition_modifications",
    "load_condition_entities",
    "load_condition_factors",
    # Validation
    "ValidationResult",
    "validate_manifest",
    # Inspection
    "ManifestInspection",
    "inspect_manifest",
    # AF3 representation
    "AF3RepresentationStatus",
    "CovalentBondRecord",
    "is_known_ccd_code",
    "load_covalent_bond_registry",
    # Builder integration
    "ConditionSpec",
    "ResolvedEntity",
    "resolve_condition",
    "build_job",
    "build_all_jobs",
]
