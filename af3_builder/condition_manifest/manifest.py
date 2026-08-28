"""
Master condition manifest and junction tables.

The manifest is the authoritative registry of all experimental conditions.
Each condition references biological entities (proteins, nucleic acids,
ligands, ions, modifications, partners) through junction tables, keeping
the data normalized.

Design principles
-----------------
* **No hard-coded protein names.**
* **No assumption about experimental factors.**
* **No assumption about the number of modifications, entities, or factors.**
* **Mutual exclusivity and compatibility rules are data, not code.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .registries import (
    _read_csv,
    _opt,
    _opt_int,
    _bool_from_csv,
    _list_from_csv,
)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionRecord:
    """A single experimental condition in the master manifest.

    This is the biological specification — it does NOT describe how to
    build an AF3 JSON file.
    """
    condition_id: str = ""
    condition_name: str = ""
    condition_group: str = ""
    parent_condition_id: str = ""
    status: str = "planned"
    description: str = ""
    biological_rationale: str = ""
    experimental_tier: str = ""
    experimental_priority: str = ""
    notes: str = ""

    def __post_init__(self):
        if not self.condition_id:
            raise ValueError("condition_id is required")
        if not self.condition_name:
            raise ValueError("condition_name is required")


@dataclass(frozen=True)
class ConditionModificationRecord:
    """Junction table: which modifications apply to which condition at which position."""
    condition_id: str = ""
    modification_id: str = ""
    sequence_position: str = ""   # position on the construct/sequence
    construct_id: str = ""        # which construct carries the modification
    stoichiometry: str = "1"
    evidence_level: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ConditionEntityRecord:
    """Junction table: which entities (DNA, RNA, ligand, ion, partner) are present in a condition."""
    condition_id: str = ""
    entity_type: str = ""      # "dna", "rna", "ligand", "ion", "partner"
    entity_id: str = ""        # references the appropriate registry
    stoichiometry: str = "1"
    role: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ConditionFactorRecord:
    """Experimental factor annotation for a condition.

    Factors describe what experimental variables distinguish conditions.
    They are NOT hard-coded — they come from data.
    """
    condition_id: str = ""
    factor_name: str = ""
    factor_level: str = ""     # value of the factor for this condition
    factor_role: str = ""      # e.g. "treatment", "control", "baseline"
    notes: str = ""


@dataclass(frozen=True)
class ExclusivityRule:
    """A mutual-exclusivity constraint between entities or modifications."""
    rule_id: str = ""
    exclusive_group: str = ""    # entities sharing a group cannot co-occur
    exclusive_key: str = ""      # composite key for the constraint
    entity_type: str = ""        # what kind of entity this applies to
    description: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Master manifest — container that holds all loaded data
# ---------------------------------------------------------------------------

class MasterManifest:
    """In-memory representation of all condition and registry data.

    Provides lookup methods and referential-integrity helpers.
    """

    def __init__(self):
        self.conditions: Dict[str, ConditionRecord] = {}
        self.modifications: Dict[str, ConditionModificationRecord] = {}
        self.entities: Dict[str, ConditionEntityRecord] = {}
        self.factors: Dict[str, ConditionFactorRecord] = {}
        self.exclusivity_rules: Dict[str, ExclusivityRule] = {}

    # -- helpers ------------------------------------------------------------

    @property
    def condition_ids(self) -> List[str]:
        return sorted(self.conditions.keys())

    @property
    def condition_names(self) -> List[str]:
        return [self.conditions[cid].condition_name for cid in self.condition_ids]

    def get_conditions_by_group(self, group: str) -> List[ConditionRecord]:
        return [c for c in self.conditions.values() if c.condition_group == group]

    def get_modifications_for_condition(self, condition_id: str) -> List[ConditionModificationRecord]:
        return [m for m in self.modifications.values() if m.condition_id == condition_id]

    def get_entities_for_condition(self, condition_id: str) -> List[ConditionEntityRecord]:
        return [e for e in self.entities.values() if e.condition_id == condition_id]

    def get_factors_for_condition(self, condition_id: str) -> List[ConditionFactorRecord]:
        return [f for f in self.factors.values() if f.condition_id == condition_id]

    def get_factor_values(self, factor_name: str) -> Dict[str, str]:
        """Return {condition_id: factor_level} for a given factor."""
        return {
            f.condition_id: f.factor_level
            for f in self.factors.values()
            if f.factor_name == factor_name
        }

    def get_attribute_names(self) -> List[str]:
        """Return sorted unique factor names across all conditions."""
        return sorted({f.factor_name for f in self.factors.values()})

    def get_attribute_values(self, factor_name: str) -> Set[str]:
        """Return the set of observed levels for a factor."""
        return {
            f.factor_level
            for f in self.factors.values()
            if f.factor_name == factor_name
        }

    def condition_attributes_matrix(self) -> Dict[str, Dict[str, str]]:
        """Return {condition_id: {factor_name: factor_level}}."""
        matrix: Dict[str, Dict[str, str]] = {}
        for f in self.factors.values():
            matrix.setdefault(f.condition_id, {})[f.factor_name] = f.factor_level
        return matrix


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def _load_rows_as_records(
    path: Path,
    record_cls: type,
) -> List[Any]:
    """Load a CSV and return a list of dataclass records."""
    if not path.exists():
        return []
    rows = _read_csv(path)
    records = []
    for row in rows:
        kwargs = {}
        for f in record_cls.__dataclass_fields__:
            if f in row:
                kwargs[f] = row[f]
        records.append(record_cls(**kwargs))
    return records


def load_master_manifest(
    manifest_path: Path,
    modifications_path: Optional[Path] = None,
    entities_path: Optional[Path] = None,
    factors_path: Optional[Path] = None,
    exclusivity_path: Optional[Path] = None,
) -> MasterManifest:
    """Load a complete master manifest from CSV files.

    Parameters
    ----------
    manifest_path : Path
        Path to ``master_condition_manifest.csv``.
    modifications_path : Path, optional
        Path to ``condition_modifications.csv``.
    entities_path : Path, optional
        Path to ``condition_entities.csv``.
    factors_path : Path, optional
        Path to ``condition_factors.csv``.
    exclusivity_path : Path, optional
        Path to ``exclusivity_rules.csv``.

    Returns
    -------
    MasterManifest
        Loaded manifest with all junction data.
    """
    manifest = MasterManifest()

    # Load conditions
    if not manifest_path.exists():
        raise FileNotFoundError(f"Master manifest not found: {manifest_path}")
    for rec in _load_rows_as_records(manifest_path, ConditionRecord):
        manifest.conditions[rec.condition_id] = rec

    # Load junction tables
    for rec in _load_rows_as_records(modifications_path or manifest_path.parent / "condition_modifications.csv", ConditionModificationRecord):
        key = f"{rec.condition_id}:{rec.modification_id}:{rec.sequence_position}"
        manifest.modifications[key] = rec

    for rec in _load_rows_as_records(entities_path or manifest_path.parent / "condition_entities.csv", ConditionEntityRecord):
        key = f"{rec.condition_id}:{rec.entity_type}:{rec.entity_id}"
        manifest.entities[key] = rec

    for rec in _load_rows_as_records(factors_path or manifest_path.parent / "condition_factors.csv", ConditionFactorRecord):
        key = f"{rec.condition_id}:{rec.factor_name}"
        manifest.factors[key] = rec

    # Load exclusivity rules (optional)
    if exclusivity_path and exclusivity_path.exists():
        for rec in _load_rows_as_records(exclusivity_path, ExclusivityRule):
            manifest.exclusivity_rules[rec.rule_id] = rec

    return manifest


# ---------------------------------------------------------------------------
# Convenience loaders (match the CSV filenames in the spec)
# ---------------------------------------------------------------------------

def load_condition_modifications(path: Path) -> List[ConditionModificationRecord]:
    return _load_rows_as_records(path, ConditionModificationRecord)


def load_condition_entities(path: Path) -> List[ConditionEntityRecord]:
    return _load_rows_as_records(path, ConditionEntityRecord)


def load_condition_factors(path: Path) -> List[ConditionFactorRecord]:
    return _load_rows_as_records(path, ConditionFactorRecord)
