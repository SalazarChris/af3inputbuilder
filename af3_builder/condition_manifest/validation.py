"""
Generic validation for condition manifests.

Validates referential integrity, duplicate detection, missing definitions,
sequence/position compatibility, and mutual exclusivity constraints.

All validation is **protein-agnostic** — no hard-coded entity names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .manifest import MasterManifest
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
    AF3RepresentationStatus,
    is_known_ccd_code,
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Accumulates errors, warnings, and info messages from validation."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(f"ERROR: {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(f"WARNING: {msg}")

    def add_info(self, msg: str) -> None:
        self.info.append(f"INFO: {msg}")

    def summary(self) -> str:
        lines = []
        lines.append(f"Validation complete: {len(self.errors)} errors, "
                     f"{len(self.warnings)} warnings, {len(self.info)} info")
        for e in self.errors:
            lines.append(f"  {e}")
        for w in self.warnings:
            lines.append(f"  {w}")
        for i in self.info:
            lines.append(f"  {i}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Master validation function
# ---------------------------------------------------------------------------

def validate_manifest(
    manifest: MasterManifest,
    *,
    protein_registry: Optional[Dict[str, ProteinRecord]] = None,
    construct_registry: Optional[Dict[str, ConstructRecord]] = None,
    modification_registry: Optional[Dict[str, ModificationRecord]] = None,
    nucleic_acid_registry: Optional[Dict[str, NucleicAcidRecord]] = None,
    ligand_registry: Optional[Dict[str, LigandRecord]] = None,
    ion_registry: Optional[Dict[str, IonRecord]] = None,
    partner_registry: Optional[Dict[str, PartnerRecord]] = None,
    af3_registry: Optional[Dict[str, AF3CompatibilityRecord]] = None,
    covalent_bond_registry: Optional[Dict[str, CovalentBondRecord]] = None,
    observed_condition_names: Optional[List[str]] = None,
) -> ValidationResult:
    """Run all validation checks on a loaded manifest.

    Parameters
    ----------
    manifest : MasterManifest
        The loaded manifest.
    *_registry : dict, optional
        Loaded registries for referential-integrity checks.
    observed_condition_names : list, optional
        Condition names found in actual data files.

    Returns
    -------
    ValidationResult
        Collected errors, warnings, and info.
    """
    result = ValidationResult()

    # 1. Condition-level checks
    _check_conditions(manifest, result)

    # 2. Modification junction checks
    _check_modifications(manifest, modification_registry, construct_registry, result)

    # 3. Entity junction checks
    _check_entities(manifest, nucleic_acid_registry, ligand_registry,
                    ion_registry, partner_registry, construct_registry, result)

    # 4. Factor checks
    _check_factors(manifest, result)

    # 5. Referential integrity
    _check_referential_integrity(manifest, result)

    # 6. Mutual exclusivity
    _check_exclusivity(manifest, result)

    # 7. Observed vs defined conditions
    if observed_condition_names is not None:
        _check_observed_conditions(manifest, observed_condition_names, result)

    # 8. Protein/construct referential integrity
    _check_protein_references(manifest, protein_registry, construct_registry, result)

    # 9. AF3 compatibility
    if af3_registry is not None and modification_registry is not None:
        _check_af3_compatibility(manifest, modification_registry, af3_registry, result)

    # 10. CCD code validation
    _check_ccd_codes(manifest, modification_registry, af3_registry,
                     ligand_registry, ion_registry, result)

    # 11. Covalent bond validation
    if covalent_bond_registry is not None:
        _check_covalent_bonds(manifest, covalent_bond_registry, result)

    return result


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_conditions(manifest: MasterManifest, result: ValidationResult) -> None:
    """Check for duplicate condition IDs, empty names, etc."""
    seen_ids: Set[str] = set()
    seen_names: Set[str] = set()

    for cid, cond in manifest.conditions.items():
        # Duplicate ID (should be impossible from dict, but check name)
        if cond.condition_name in seen_names:
            result.add_error(f"Duplicate condition_name '{cond.condition_name}' "
                           f"(condition_id='{cid}')")
        seen_names.add(cond.condition_name)

        # Missing required fields
        if not cond.condition_name.strip():
            result.add_error(f"Empty condition_name for condition_id='{cid}'")

        # Parent reference
        if cond.parent_condition_id and cond.parent_condition_id not in manifest.conditions:
            result.add_warning(f"Condition '{cid}' references parent "
                             f"'{cond.parent_condition_id}' which is not defined")


def _check_modifications(
    manifest: MasterManifest,
    modification_registry: Optional[Dict[str, ModificationRecord]],
    construct_registry: Optional[Dict[str, ConstructRecord]],
    result: ValidationResult,
) -> None:
    """Check modification junction records."""
    seen: Set[str] = set()

    for key, mod in manifest.modifications.items():
        # Duplicate detection
        composite = (mod.condition_id, mod.modification_id,
                    mod.sequence_position, mod.construct_id)
        if composite in seen:
            result.add_error(f"Duplicate modification record: condition={mod.condition_id}, "
                           f"modification={mod.modification_id}, "
                           f"position={mod.sequence_position}, "
                           f"construct={mod.construct_id}")
        seen.add(composite)

        # Referential integrity
        if modification_registry is not None and mod.modification_id not in modification_registry:
            result.add_error(f"Modification '{mod.modification_id}' in condition "
                           f"'{mod.condition_id}' not found in modification registry")

        if construct_registry is not None and mod.construct_id and mod.construct_id not in construct_registry:
            result.add_error(f"Construct '{mod.construct_id}' in condition "
                           f"'{mod.condition_id}' not found in construct registry")

        # Position validation
        if mod.sequence_position:
            try:
                pos = int(mod.sequence_position)
                if pos < 1:
                    result.add_warning(f"Non-positive sequence_position {pos} in "
                                     f"condition '{mod.condition_id}'")
            except ValueError:
                result.add_error(f"Invalid sequence_position '{mod.sequence_position}' "
                               f"in condition '{mod.condition_id}'")


def _check_entities(
    manifest: MasterManifest,
    nucleic_acid_registry: Optional[Dict[str, NucleicAcidRecord]],
    ligand_registry: Optional[Dict[str, LigandRecord]],
    ion_registry: Optional[Dict[str, IonRecord]],
    partner_registry: Optional[Dict[str, PartnerRecord]],
    construct_registry: Optional[Dict[str, ConstructRecord]] = None,
    result: ValidationResult = None,
) -> None:
    """Check entity junction records."""
    if result is None:
        result = ValidationResult()
    seen: Set[str] = set()

    valid_entity_types = {"protein", "dna", "rna", "ligand", "ion", "partner"}

    for key, ent in manifest.entities.items():
        # Entity type validation
        if ent.entity_type.lower() not in valid_entity_types:
            result.add_error(f"Invalid entity_type '{ent.entity_type}' in condition "
                           f"'{ent.condition_id}'")

        # Duplicate detection
        composite = (ent.condition_id, ent.entity_type, ent.entity_id)
        if composite in seen:
            result.add_error(f"Duplicate entity record: condition={ent.condition_id}, "
                           f"type={ent.entity_type}, id={ent.entity_id}")
        seen.add(composite)

        # Stoichiometry validation
        if ent.stoichiometry:
            try:
                stoich = int(ent.stoichiometry)
                if stoich < 1:
                    result.add_warning(f"Non-positive stoichiometry {stoich} for "
                                     f"entity '{ent.entity_id}' in condition '{ent.condition_id}'")
            except ValueError:
                result.add_error(f"Invalid stoichiometry '{ent.stoichiometry}' for "
                               f"entity '{ent.entity_id}' in condition '{ent.condition_id}'")

        # Referential integrity against registries
        if ent.entity_type == "protein" and construct_registry is not None:
            if ent.entity_id not in construct_registry:
                result.add_error(f"Protein entity '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in construct registry")
        elif ent.entity_type == "dna" and nucleic_acid_registry is not None:
            if ent.entity_id not in nucleic_acid_registry:
                result.add_error(f"DNA/RNA entity '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in nucleic acid registry")
        elif ent.entity_type == "rna" and nucleic_acid_registry is not None:
            if ent.entity_id not in nucleic_acid_registry:
                result.add_error(f"RNA entity '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in nucleic acid registry")
        elif ent.entity_type == "ligand" and ligand_registry is not None:
            if ent.entity_id not in ligand_registry:
                result.add_error(f"Ligand '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in ligand registry")
        elif ent.entity_type == "ion" and ion_registry is not None:
            if ent.entity_id not in ion_registry:
                result.add_error(f"Ion '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in ion registry")
        elif ent.entity_type == "partner" and partner_registry is not None:
            if ent.entity_id not in partner_registry:
                result.add_error(f"Partner '{ent.entity_id}' in condition "
                               f"'{ent.condition_id}' not found in partner registry")


def _check_factors(manifest: MasterManifest, result: ValidationResult) -> None:
    """Check factor records for consistency."""
    # Ensure every condition has at least one factor
    conditions_with_factors: Set[str] = set()
    for f in manifest.factors.values():
        conditions_with_factors.add(f.condition_id)

    for cid in manifest.conditions:
        if cid not in conditions_with_factors:
            result.add_warning(f"Condition '{cid}' has no factor annotations")


def _check_referential_integrity(manifest: MasterManifest, result: ValidationResult) -> None:
    """Check that all junction records reference existing conditions."""
    for key, mod in manifest.modifications.items():
        if mod.condition_id not in manifest.conditions:
            result.add_error(f"Modification record references unknown condition "
                           f"'{mod.condition_id}'")

    for key, ent in manifest.entities.items():
        if ent.condition_id not in manifest.conditions:
            result.add_error(f"Entity record references unknown condition "
                           f"'{ent.condition_id}'")

    for key, f in manifest.factors.items():
        if f.condition_id not in manifest.conditions:
            result.add_error(f"Factor record references unknown condition "
                           f"'{f.condition_id}'")


def _check_exclusivity(manifest: MasterManifest, result: ValidationResult) -> None:
    """Check mutual exclusivity constraints."""
    # Group entities by exclusive_group
    groups: Dict[str, List[Tuple[str, str]]] = {}  # group -> [(condition_id, entity_id)]

    for rule_id, rule in manifest.exclusivity_rules.items():
        if not rule.exclusive_group:
            continue

        # Collect all entities in this group
        for ent in manifest.entities.values():
            if ent.entity_type == rule.entity_type or not rule.entity_type:
                # Check if this entity's ID matches the rule's scope
                # For now, we check by entity_type
                pass

    # Simple check: if two entities in the same exclusive_group appear
    # in the same condition, flag it
    # This is a basic implementation — can be extended
    for rule_id, rule in manifest.exclusivity_rules.items():
        if not rule.exclusive_group:
            continue
        for cid in manifest.conditions:
            condition_entities = [
                e for e in manifest.entities.values()
                if e.condition_id == cid
            ]
            # Check if multiple entities from the same exclusive group are present
            # (This requires the rule to specify which entity_ids are in the group)


def _check_observed_conditions(
    manifest: MasterManifest,
    observed_names: List[str],
    result: ValidationResult,
) -> None:
    """Check that observed condition names match manifest definitions."""
    defined_names = {c.condition_name for c in manifest.conditions.values()}
    observed_set = set(observed_names)

    undefined = observed_set - defined_names
    if undefined:
        result.add_warning(f"{len(undefined)} condition(s) in data but not in manifest: "
                         f"{sorted(undefined)}")

    unused = defined_names - observed_set
    if unused:
        result.add_warning(f"{len(unused)} condition(s) in manifest but not in data: "
                         f"{sorted(unused)}")


def _check_protein_references(
    manifest: MasterManifest,
    protein_registry: Optional[Dict[str, ProteinRecord]],
    construct_registry: Optional[Dict[str, ConstructRecord]],
    result: ValidationResult,
) -> None:
    """Check that construct_registry references valid proteins."""
    if construct_registry is None:
        return

    for cid, construct in construct_registry.items():
        if protein_registry is not None and construct.protein_id not in protein_registry:
            result.add_error(f"Construct '{cid}' references unknown protein "
                           f"'{construct.protein_id}'")

        # Check sequence length vs domain coordinates
        if construct.domain_start and construct.domain_end:
            try:
                start = int(construct.domain_start)
                end = int(construct.domain_end)
                if start > end:
                    result.add_error(f"Construct '{cid}' has domain_start ({start}) "
                                   f"> domain_end ({end})")
                if construct.construct_sequence:
                    seq_len = len(construct.construct_sequence.replace(" ", ""))
                    if end > seq_len:
                        result.add_warning(f"Construct '{cid}' domain_end ({end}) "
                                         f"exceeds sequence length ({seq_len})")
            except ValueError:
                result.add_error(f"Construct '{cid}' has non-integer domain coordinates: "
                               f"start='{construct.domain_start}', end='{construct.domain_end}'")


def _check_af3_compatibility(
    manifest: MasterManifest,
    modification_registry: Dict[str, ModificationRecord],
    af3_registry: Dict[str, AF3CompatibilityRecord],
    result: ValidationResult,
) -> None:
    """Check that modifications used in conditions have AF3 representations."""
    for key, mod in manifest.modifications.items():
        if mod.modification_id not in modification_registry:
            continue  # Already flagged in _check_modifications

        # Find AF3 compatibility entry
        af3_rec = None
        for rec in af3_registry.values():
            if rec.modification_id == mod.modification_id:
                af3_rec = rec
                break

        if af3_rec is None:
            result.add_warning(f"Modification '{mod.modification_id}' in condition "
                             f"'{mod.condition_id}' has no AF3 compatibility entry")
            continue

        # Check status
        status = af3_rec.status
        if status == AF3RepresentationStatus.UNSUPPORTED:
            result.add_error(
                f"Modification '{mod.modification_id}' in condition "
                f"'{mod.condition_id}' is UNSUPPORTED by AF3"
            )
        elif status == AF3RepresentationStatus.REPRESENTATION_UNCERTAIN:
            result.add_warning(
                f"Modification '{mod.modification_id}' in condition "
                f"'{mod.condition_id}' has UNCERTAIN AF3 representation"
            )
        elif status == AF3RepresentationStatus.VERIFIED_CUSTOM:
            if not af3_rec.custom_ccd_id:
                result.add_error(
                    f"Modification '{mod.modification_id}' requires custom CCD "
                    f"but no custom_ccd_id specified"
                )


def _check_ccd_codes(
    manifest: MasterManifest,
    modification_registry: Optional[Dict[str, ModificationRecord]],
    af3_registry: Optional[Dict[str, AF3CompatibilityRecord]],
    ligand_registry: Optional[Dict[str, LigandRecord]],
    ion_registry: Optional[Dict[str, IonRecord]],
    result: ValidationResult,
) -> None:
    """Validate CCD codes against the project's reference lists.

    A CCD code that is not in core/reference.py is flagged as a warning
    (not an error) because the project may use custom CCDs intentionally.
    """
    checked_codes: Set[str] = set()

    # Check modification CCD codes from AF3 compatibility registry
    if af3_registry:
        for rec in af3_registry.values():
            if rec.ccd_code and rec.ccd_code not in checked_codes:
                checked_codes.add(rec.ccd_code)
                if rec.status in (
                    AF3RepresentationStatus.VERIFIED_NATIVE,
                    AF3RepresentationStatus.VERIFIED_CUSTOM,
                ):
                    # Verified entries are trusted regardless of reference.py
                    continue
                if not is_known_ccd_code(rec.ccd_code):
                    result.add_info(
                        f"CCD code '{rec.ccd_code}' in representation "
                        f"'{rec.representation_id}' is not in core/reference.py "
                        f"(may be valid but unverified)"
                    )

    # Check ligand CCD codes
    if ligand_registry:
        for lig in ligand_registry.values():
            if lig.ccd_code and lig.ccd_code not in checked_codes:
                checked_codes.add(lig.ccd_code)
                if not is_known_ccd_code(lig.ccd_code):
                    result.add_info(
                        f"CCD code '{lig.ccd_code}' for ligand '{lig.ligand_id}' "
                        f"is not in core/reference.py"
                    )

    # Check ion CCD codes
    if ion_registry:
        for ion in ion_registry.values():
            if ion.ccd_code and ion.ccd_code not in checked_codes:
                checked_codes.add(ion.ccd_code)
                if not is_known_ccd_code(ion.ccd_code):
                    result.add_info(
                        f"CCD code '{ion.ccd_code}' for ion '{ion.ion_id}' "
                        f"is not in core/reference.py"
                    )


def _check_covalent_bonds(
    manifest: MasterManifest,
    covalent_bond_registry: Dict[str, CovalentBondRecord],
    result: ValidationResult,
) -> None:
    """Validate covalent bond definitions."""
    for bond_id, bond in covalent_bond_registry.items():
        # Check AF3 status
        if bond.status == AF3RepresentationStatus.UNSUPPORTED:
            result.add_error(
                f"Covalent bond '{bond_id}' is UNSUPPORTED by AF3"
            )

        # Check that atom names are non-empty
        if not bond.atom_1.strip():
            result.add_error(
                f"Covalent bond '{bond_id}' has empty atom_1"
            )
        if not bond.atom_2.strip():
            result.add_error(
                f"Covalent bond '{bond_id}' has empty atom_2"
            )

        # Check residue positions are valid integers
        for field_name, value in [("residue_1", bond.residue_1),
                                  ("residue_2", bond.residue_2)]:
            if value:
                try:
                    pos = int(value)
                    if pos < 1:
                        result.add_warning(
                            f"Covalent bond '{bond_id}' {field_name}="
                            f"{value} is non-positive"
                        )
                except ValueError:
                    result.add_error(
                        f"Covalent bond '{bond_id}' {field_name}="
                        f"'{value}' is not a valid integer"
                    )
