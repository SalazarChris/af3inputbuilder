"""
Reusable registry records and CSV loading functions.

Each registry is a flat CSV file with an explicit primary key.
Loading produces a dict keyed by the primary key, plus a list for ordered
iteration.  Referential integrity is checked by the validation module, not
here — registries are pure data containers.
"""

from __future__ import annotations

import csv
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts, stripping whitespace from keys."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append({k.strip(): (v.strip() if v is not None else "") for k, v in row.items()})
        return rows


def _opt(val: str) -> Optional[str]:
    """Return *val* as-is if non-empty, else ``None``."""
    return val if val else None


def _opt_int(val: str) -> Optional[int]:
    """Return *val* as int if non-empty, else ``None``."""
    return int(val) if val else None


def _opt_float(val: str) -> Optional[float]:
    """Return *val* as float if non-empty, else ``None``."""
    return float(val) if val else None


def _bool_from_csv(val: str) -> bool:
    """Interpret common CSV boolean representations."""
    return val.strip().lower() in ("true", "1", "yes", "y")


def _list_from_csv(val: str, sep: str = ";") -> List[str]:
    """Split a semicolon-delimited field into a list, filtering empties."""
    if not val:
        return []
    return [v.strip() for v in val.split(sep) if v.strip()]


# ---------------------------------------------------------------------------
# AF3 representation status
# ---------------------------------------------------------------------------

class AF3RepresentationStatus(Enum):
    """Formal status of an AF3 representation.

    Distinguishes what is *technically possible* in AF3 from what is
    *verified to work*.

    VERIFIED_NATIVE
        AF3 supports this natively via a standard CCD code.
        No custom CCD required.

    VERIFIED_CUSTOM
        AF3 can represent this, but requires a custom CCD definition.
        The custom CCD must be provided at runtime.

    REPRESENTATION_POSSIBLE
        A representation likely exists but has not been verified.
        Requires manual verification before use.

    REPRESENTATION_UNCERTAIN
        It is unclear whether AF3 can represent this.
        Do NOT use in production without investigation.

    UNSUPPORTED
        AF3 cannot represent this in its current form.
        Do NOT generate jobs for conditions requiring this.
    """
    VERIFIED_NATIVE = "verified_native"
    VERIFIED_CUSTOM = "verified_custom"
    REPRESENTATION_POSSIBLE = "representation_possible"
    REPRESENTATION_UNCERTAIN = "representation_uncertain"
    UNSUPPORTED = "unsupported"

    @classmethod
    def _missing_(cls, value: str):
        """Allow case-insensitive lookup."""
        for member in cls:
            if member.value == value.lower().strip():
                return member
        return None


# Valid AF3 status strings for CSV loading
_AF3_STATUS_VALUES = {s.value for s in AF3RepresentationStatus}


# ---------------------------------------------------------------------------
# Known CCD codes from core/reference.py
# ---------------------------------------------------------------------------
# This set is populated lazily to avoid import-time side effects.
_KNOWN_CCD_CODES: Optional[set] = None

def _get_known_ccd_codes() -> set:
    """Return the set of CCD codes defined in core/reference.py.

    This avoids duplicating the reference lists — we validate against
    the single source of truth.
    """
    global _KNOWN_CCD_CODES
    if _KNOWN_CCD_CODES is None:
        try:
            from af3_builder.core.reference import (
                ALL_COMMON_LIGANDS, PROTEIN_PTMS, RNA_MODIFICATIONS, DNA_MODIFICATIONS,
            )
            codes = set()
            for code, _ in ALL_COMMON_LIGANDS:
                codes.add(code.strip().upper())
            for code, _ in PROTEIN_PTMS:
                codes.add(code.strip().upper())
            for code, _ in RNA_MODIFICATIONS:
                codes.add(code.strip().upper())
            for code, _ in DNA_MODIFICATIONS:
                codes.add(code.strip().upper())
            _KNOWN_CCD_CODES = codes
        except ImportError:
            _KNOWN_CCD_CODES = set()
    return _KNOWN_CCD_CODES


def is_known_ccd_code(code: str) -> bool:
    """Check whether a CCD code is in the project's reference lists.

    This does NOT check the PDB — it checks whether the code is
    defined in ``core/reference.py``.
    """
    return code.strip().upper() in _get_known_ccd_codes()


# ---------------------------------------------------------------------------
# Generic CSV loader
# ---------------------------------------------------------------------------

def load_csv_registry(
    path: Path,
    record_cls: type,
    id_field: str,
) -> Dict[str, Any]:
    """Load a CSV file into a dict of dataclass records keyed by *id_field*.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    record_cls : type
        A dataclass whose field names match the CSV column names.
    id_field : str
        The column to use as the dictionary key.

    Returns
    -------
    dict
        Mapping from *id_field* value to record instance.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If duplicate IDs are found.
    """
    rows = _read_csv(path)
    records: Dict[str, Any] = {}
    for row in rows:
        # Build kwargs matching dataclass fields
        kwargs: Dict[str, Any] = {}
        for f in record_cls.__dataclass_fields__:
            if f in row:
                kwargs[f] = row[f]
        rec = record_cls(**kwargs)
        key = getattr(rec, id_field)
        if key in records:
            raise ValueError(f"Duplicate ID '{key}' in {path}")
        records[key] = rec
    return records


# ---------------------------------------------------------------------------
# Protein registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProteinRecord:
    """A single protein entry."""
    protein_id: str = ""
    protein_name: str = ""
    gene_name: str = ""
    uniprot_id: str = ""
    species: str = ""
    sequence_id: str = ""
    sequence_version: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.protein_id:
            raise ValueError("protein_id is required")


def load_protein_registry(path: Path) -> Dict[str, ProteinRecord]:
    return load_csv_registry(path, ProteinRecord, "protein_id")


# ---------------------------------------------------------------------------
# Construct / domain registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConstructRecord:
    """A construct or domain within a protein."""
    construct_id: str = ""
    protein_id: str = ""
    construct_name: str = ""
    domain_id: str = ""
    domain_name: str = ""
    domain_start: str = ""   # 1-based, inclusive
    domain_end: str = ""     # 1-based, inclusive
    construct_sequence: str = ""
    sequence_length: str = ""
    source: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.construct_id:
            raise ValueError("construct_id is required")
        if not self.protein_id:
            raise ValueError("protein_id is required")


def load_construct_registry(path: Path) -> Dict[str, ConstructRecord]:
    return load_csv_registry(path, ConstructRecord, "construct_id")


# ---------------------------------------------------------------------------
# Modification registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModificationRecord:
    """A reusable modification definition (PTM, base modification, etc.).

    This record describes the **biological and chemical** identity of a
    modification.  It does NOT describe how AF3 represents it — that
    belongs in ``AF3CompatibilityRecord``.

    The ``modified_residue`` field is kept for backward compatibility
    but should be considered a hint, not the authoritative AF3 CCD code.
    """
    modification_id: str = ""
    modification_name: str = ""
    modification_class: str = ""       # phosphorylation, acetylation, etc.
    base_residue: str = ""             # standard residue (e.g. "S", "T", "K")
    modified_residue: str = ""         # AF3 CCD code hint (e.g. "SEP", "TPO")
    chemical_description: str = ""
    molecular_formula: str = ""
    formal_charge: str = ""
    evidence_level: str = ""           # A_CONFIRMED_FUNCTIONAL, etc.
    evidence_source: str = ""
    functional_evidence: str = ""
    representation_class: str = ""     # native_residue, custom_residue, separate_entity, covalent_adduct, unsupported
    notes: str = ""

    def __post_init__(self):
        if not self.modification_id:
            raise ValueError("modification_id is required")


def load_modification_registry(path: Path) -> Dict[str, ModificationRecord]:
    return load_csv_registry(path, ModificationRecord, "modification_id")


# ---------------------------------------------------------------------------
# Nucleic acid registry (DNA and RNA share the same schema)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NucleicAcidRecord:
    """A reusable nucleic acid entity (DNA or RNA).

    Supports optional modification via ``modification_id`` which references
    the modification_registry.  The AF3 representation of that modification
    is resolved through the AF3 compatibility registry.
    """
    entity_id: str = ""
    entity_type: str = ""          # "dna" or "rna"
    name: str = ""
    sequence: str = ""
    sequence_version: str = ""
    role: str = ""
    modification_id: str = ""      # references modification_registry (optional)
    source: str = ""
    evidence_level: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.entity_id:
            raise ValueError("entity_id is required")
        if self.entity_type and self.entity_type.lower() not in ("dna", "rna"):
            raise ValueError(f"entity_type must be 'dna' or 'rna', got '{self.entity_type}'")


def load_nucleic_acid_registry(path: Path) -> Dict[str, NucleicAcidRecord]:
    return load_csv_registry(path, NucleicAcidRecord, "entity_id")


# ---------------------------------------------------------------------------
# Ligand registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LigandRecord:
    """A reusable ligand definition.

    The ``ccd_code`` is the AF3 representation.  The ``af3_status``
    records whether this representation has been verified.
    """
    ligand_id: str = ""
    ligand_name: str = ""
    ccd_code: str = ""
    smiles: str = ""
    inchi: str = ""
    ligand_role: str = ""
    source: str = ""
    evidence_level: str = ""
    custom_ccd_required: str = ""   # "true"/"false"
    custom_ccd_id: str = ""
    af3_status: str = ""            # AF3RepresentationStatus value
    notes: str = ""

    def __post_init__(self):
        if not self.ligand_id:
            raise ValueError("ligand_id is required")

    @property
    def status(self) -> AF3RepresentationStatus:
        """Return af3_status as a proper enum value."""
        if not self.af3_status:
            # Infer from ccd_code presence
            if self.ccd_code:
                return AF3RepresentationStatus.REPRESENTATION_POSSIBLE
            if self.smiles:
                return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN
            return AF3RepresentationStatus.UNSUPPORTED
        try:
            return AF3RepresentationStatus(self.af3_status.lower().strip())
        except ValueError:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN

    @property
    def needs_custom_ccd(self) -> bool:
        return self.custom_ccd_required.lower().strip() in ("true", "1", "yes")


def load_ligand_registry(path: Path) -> Dict[str, LigandRecord]:
    return load_csv_registry(path, LigandRecord, "ligand_id")


# ---------------------------------------------------------------------------
# Ion registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IonRecord:
    """A reusable ion definition.

    Ions are represented as ligands in AF3 (CCD code in a ligand entity).
    """
    ion_id: str = ""
    ion_name: str = ""
    charge: str = ""
    ccd_code: str = ""
    role: str = ""
    concentration: str = ""
    af3_status: str = ""      # AF3RepresentationStatus value
    notes: str = ""

    def __post_init__(self):
        if not self.ion_id:
            raise ValueError("ion_id is required")

    @property
    def status(self) -> AF3RepresentationStatus:
        if not self.af3_status:
            if self.ccd_code:
                return AF3RepresentationStatus.REPRESENTATION_POSSIBLE
            return AF3RepresentationStatus.UNSUPPORTED
        try:
            return AF3RepresentationStatus(self.af3_status.lower().strip())
        except ValueError:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN


def load_ion_registry(path: Path) -> Dict[str, IonRecord]:
    return load_csv_registry(path, IonRecord, "ion_id")


# ---------------------------------------------------------------------------
# Partner registry (protein–protein interactions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartnerRecord:
    """A reusable interacting partner definition."""
    partner_id: str = ""
    partner_name: str = ""
    protein_id: str = ""       # references protein_registry.protein_id
    uniprot_id: str = ""
    species: str = ""
    sequence_id: str = ""
    role: str = ""             # e.g. "co-activator", "target", "dimerizer"
    interaction_evidence: str = ""
    source: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.partner_id:
            raise ValueError("partner_id is required")


def load_partner_registry(path: Path) -> Dict[str, PartnerRecord]:
    return load_csv_registry(path, PartnerRecord, "partner_id")


# ---------------------------------------------------------------------------
# AF3 compatibility registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AF3CompatibilityRecord:
    """Maps a biological modification/entity to its AF3 representation.

    This is the **technical** layer — it describes how AF3 can represent
    something, not whether it is biologically relevant.

    A biologically valid modification does NOT automatically imply AF3
    support.  This record makes that distinction explicit.

    Fields
    ------
    representation_id : str
        Primary key.
    modification_id : str
        References modification_registry.  May be empty for entities
        that are not modifications (e.g. ligands, ions).
    entity_type : str
        "protein", "dna", "rna", "ligand", "ion".
    component_type : str
        How AF3 handles this:
        - "native_polymer_mod": standard residue replacement (e.g. SEP for pS)
        - "custom_polymer_mod": requires custom CCD for the residue
        - "separate_entity": covalently attached as a separate molecule
        - "covalent_adduct": attached via covalent bond definition
        - "unsupported": cannot be represented
    ccd_code : str
        PDB CCD code for native representations.  Empty if custom or unsupported.
    custom_ccd_required : str
        "true" if a custom CCD definition must be provided.
    custom_ccd_id : str
        Identifier for the custom CCD (used to look up the definition).
    covalent_bond_required : str
        "true" if a bondedAtomPairs entry is needed.
    bond_entity_1 : str
        Chain/entity ID for bond endpoint 1 (e.g. protein chain).
    bond_residue_1 : str
        Residue position for bond endpoint 1.
    bond_atom_1 : str
        Atom name for bond endpoint 1 (e.g. "SG" for cysteine).
    bond_entity_2 : str
        Chain/entity ID for bond endpoint 2.
    bond_residue_2 : str
        Residue position for bond endpoint 2.
    bond_atom_2 : str
        Atom name for bond endpoint 2.
    af3_status : str
        From AF3RepresentationStatus enum:
        verified_native, verified_custom, representation_possible,
        representation_uncertain, unsupported.
    verification_source : str
        How this representation was verified.
    notes : str
        Free text.
    """
    representation_id: str = ""
    modification_id: str = ""
    entity_type: str = ""
    component_type: str = ""
    ccd_code: str = ""
    custom_ccd_required: str = ""   # "true"/"false"
    custom_ccd_id: str = ""
    covalent_bond_required: str = "" # "true"/"false"
    bond_entity_1: str = ""
    bond_residue_1: str = ""
    bond_atom_1: str = ""
    bond_entity_2: str = ""
    bond_residue_2: str = ""
    bond_atom_2: str = ""
    af3_status: str = ""
    verification_source: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.representation_id:
            raise ValueError("representation_id is required")

    @property
    def status(self) -> AF3RepresentationStatus:
        """Return the af3_status as a proper enum value."""
        if not self.af3_status:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN
        try:
            return AF3RepresentationStatus(self.af3_status.lower().strip())
        except ValueError:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN

    @property
    def needs_custom_ccd(self) -> bool:
        return self.custom_ccd_required.lower().strip() in ("true", "1", "yes")

    @property
    def needs_covalent_bond(self) -> bool:
        return self.covalent_bond_required.lower().strip() in ("true", "1", "yes")


def load_af3_compatibility_registry(path: Path) -> Dict[str, AF3CompatibilityRecord]:
    return load_csv_registry(path, AF3CompatibilityRecord, "representation_id")


# ---------------------------------------------------------------------------
# Covalent bond registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CovalentBondRecord:
    """A reusable covalent bond definition.

    Represents a bond between two entities (e.g. protein ↔ ubiquitin,
    protein ↔ custom CCD).  This maps directly to AF3's ``bondedAtomPairs``.

    The bond is specified by entity references and atom names, not by
    hard-coded chain letters — chain assignment happens at resolution time.
    """
    bond_id: str = ""
    entity_1_type: str = ""      # "protein", "dna", "rna", "ligand"
    entity_1_id: str = ""        # registry ID of the first entity
    residue_1: str = ""          # residue position on entity 1
    atom_1: str = ""             # atom name on entity 1 (e.g. "SG")
    entity_2_type: str = ""      # "protein", "dna", "rna", "ligand"
    entity_2_id: str = ""        # registry ID of the second entity
    residue_2: str = ""          # residue position on entity 2
    atom_2: str = ""             # atom name on entity 2
    bond_type: str = ""          # e.g. "covalent", "disulfide", "thioester"
    af3_status: str = ""         # AF3RepresentationStatus value
    notes: str = ""

    def __post_init__(self):
        if not self.bond_id:
            raise ValueError("bond_id is required")
        if not self.entity_1_id or not self.entity_2_id:
            raise ValueError(f"bond '{self.bond_id}' requires both entity_1_id and entity_2_id")
        if not self.atom_1 or not self.atom_2:
            raise ValueError(f"bond '{self.bond_id}' requires both atom_1 and atom_2")

    @property
    def status(self) -> AF3RepresentationStatus:
        if not self.af3_status:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN
        try:
            return AF3RepresentationStatus(self.af3_status.lower().strip())
        except ValueError:
            return AF3RepresentationStatus.REPRESENTATION_UNCERTAIN


def load_covalent_bond_registry(path: Path) -> Dict[str, CovalentBondRecord]:
    return load_csv_registry(path, CovalentBondRecord, "bond_id")


# ---------------------------------------------------------------------------
# Residue mapping registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResidueMappingRecord:
    """Maps residue numbering between systems for a construct."""
    mapping_id: str = ""
    construct_id: str = ""       # references construct_registry
    sequence_position: str = ""  # full-length position (1-based)
    construct_position: str = "" # construct/domain position (1-based)
    domain_position: str = ""    # domain position (1-based), if applicable
    reference_position: str = "" # literature/database position
    reference_species: str = ""
    reference_database: str = "" # e.g. "UniProt", "PDB"
    reference_accession: str = ""
    numbering_mapping_status: str = ""  # "exact", "offset", "gap", "uncertain"
    offset: str = ""             # integer offset if status is "offset"
    notes: str = ""

    def __post_init__(self):
        if not self.mapping_id:
            raise ValueError("mapping_id is required")
        if not self.construct_id:
            raise ValueError("construct_id is required")


def load_residue_mapping_registry(path: Path) -> Dict[str, ResidueMappingRecord]:
    return load_csv_registry(path, ResidueMappingRecord, "mapping_id")
