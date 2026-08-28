"""
Builder integration — resolves manifest conditions into AF3 jobs.

This is the bridge between the biological specification (manifest)
and the computational representation (AF3 JSON).

Usage::

    from af3_builder.condition_manifest import MasterManifest, build_job

    manifest = MasterManifest.load_from_csvs("registries/")
    job = build_job(manifest, "pou_tpo101", seeds=[1, 2, 3])
    # job is a fully populated JobBuilder ready to serialize

Validation order enforced in build_job:
    1. resolve condition → ConditionSpec
    2. reject UNSUPPORTED representations
    3. reject UNCERTAIN representations (unless allow_uncertain=True)
    4. validate modification positions against construct sequence length
    5. validate custom CCD requirements
    6. validate covalent bond integrity
    7. construct JobBuilder with AF3-compatible modification format
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
)


# ---------------------------------------------------------------------------
# Resolved condition specification
# ---------------------------------------------------------------------------

@dataclass
class ResolvedEntity:
    """A single resolved entity ready for JobBuilder consumption.

    This is an intermediate representation between the manifest's
    biological specification and the builder's AF3 entity classes.
    """
    entity_type: str          # "protein", "dna", "rna", "ligand", "ion"
    entity_id: str            # registry ID
    chain_id: str             # assigned chain letter (A, B, C, ...)
    stoichiometry: int = 1
    sequence: str = ""
    ccd_code: str = ""
    smiles: str = ""
    modifications: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    role: str = ""


@dataclass
class ConditionSpec:
    """Fully resolved specification for a single experimental condition.

    Contains everything needed to build one or more AF3 jobs.
    """
    condition_id: str
    condition_name: str
    proteins: List[ResolvedEntity] = field(default_factory=list)
    dna: List[ResolvedEntity] = field(default_factory=list)
    rna: List[ResolvedEntity] = field(default_factory=list)
    ligands: List[ResolvedEntity] = field(default_factory=list)
    ions: List[ResolvedEntity] = field(default_factory=list)
    bonds: List[List[Any]] = field(default_factory=list)
    representation_warnings: List[str] = field(default_factory=list)

    @property
    def all_entities(self) -> List[ResolvedEntity]:
        """All entities in chain-order."""
        return self.proteins + self.dna + self.rna + self.ligands + self.ions

    @property
    def has_unsupported_representations(self) -> bool:
        """True if any entity has an unsupported AF3 representation."""
        return any("UNSUPPORTED" in w for w in self.representation_warnings)

    @property
    def has_uncertain_representations(self) -> bool:
        """True if any entity has an uncertain AF3 representation."""
        return any("UNCERTAIN" in w or "POSSIBLE" in w
                    for w in self.representation_warnings)


# ---------------------------------------------------------------------------
# AF3 representation lookup
# ---------------------------------------------------------------------------

def _find_af3_representation(
    modification_id: str,
    af3_registry: Optional[Dict[str, AF3CompatibilityRecord]],
) -> Optional[AF3CompatibilityRecord]:
    """Find the AF3 compatibility record for a modification.

    Returns the first matching record, or None if not found.
    Multiple records for the same modification_id are allowed
    (e.g. different representation strategies); the first is used.
    """
    if af3_registry is None:
        return None
    for rec in af3_registry.values():
        if rec.modification_id == modification_id:
            return rec
    return None


# ---------------------------------------------------------------------------
# Modification format conversion
# ---------------------------------------------------------------------------

def _mods_to_af3_format(mods: List[Dict[str, Any]], entity_type: str = "protein") -> List[Dict[str, Any]]:
    """Convert internal modification dicts to AF3-compatible format.

    Internal format (ConditionSpec) contains metadata fields:
        modification_id, ccd_code, position, af3_status, needs_custom_ccd, ...

    AF3 format fields vary by entity type:
        protein: ptmType, ptmPosition
        rna:     modificationType, basePosition
        dna:     modificationType, basePosition

    This ensures the generated JSON is valid for AF3 input.
    """
    # AF3 schema field names by entity type
    _AF3_MOD_FIELDS = {
        "protein": ("ptmType", "ptmPosition"),
        "rna":     ("modificationType", "basePosition"),
        "dna":     ("modificationType", "basePosition"),
    }
    type_key, pos_key = _AF3_MOD_FIELDS.get(entity_type, _AF3_MOD_FIELDS["protein"])

    af3_mods = []
    for mod in mods:
        ccd = mod.get("ccd_code", "")
        pos = mod.get("position")
        if not ccd or pos is None:
            continue  # Skip incomplete modifications
        af3_mod = {type_key: ccd, pos_key: pos}
        af3_mods.append(af3_mod)
    return af3_mods if af3_mods else []


# ---------------------------------------------------------------------------
# Resolution: manifest → ConditionSpec
# ---------------------------------------------------------------------------

def resolve_condition(
    manifest: MasterManifest,
    condition_id: str,
    *,
    protein_registry: Optional[Dict[str, ProteinRecord]] = None,
    construct_registry: Optional[Dict[str, ConstructRecord]] = None,
    modification_registry: Optional[Dict[str, ModificationRecord]] = None,
    nucleic_acid_registry: Optional[Dict[str, NucleicAcidRecord]] = None,
    ligand_registry: Optional[Dict[str, LigandRecord]] = None,
    ion_registry: Optional[Dict[str, IonRecord]] = None,
    af3_compatibility_registry: Optional[Dict[str, AF3CompatibilityRecord]] = None,
    covalent_bond_registry: Optional[Dict[str, CovalentBondRecord]] = None,
) -> ConditionSpec:
    """Resolve a condition into a fully specified ConditionSpec.

    This is the core function that translates biological intent into
    computational specification.  It looks up all registries and resolves
    references into concrete entities with sequences, CCD codes, and
    modifications.

    Protein resolution order:
        1. Explicit ``protein`` entity records in condition_entities
        2. ``construct_id`` references from condition_modifications
        3. If neither provides a protein reference → error

    Parameters
    ----------
    manifest : MasterManifest
        The loaded manifest.
    condition_id : str
        The condition to resolve.
    *_registry : dict, optional
        Loaded registries.  If None, some entities may be unresolved
        (useful for testing or when only partial resolution is needed).

    Returns
    -------
    ConditionSpec
        Fully resolved condition ready for job building.

    Raises
    ------
    ValueError
        If the condition_id is not found, or if the condition has
        no protein references.
    """
    if condition_id not in manifest.conditions:
        raise ValueError(f"Condition '{condition_id}' not found in manifest")

    cond = manifest.conditions[condition_id]
    spec = ConditionSpec(
        condition_id=condition_id,
        condition_name=cond.condition_name,
    )

    # Track chain assignment
    chain_counter = 0

    def next_chain() -> str:
        nonlocal chain_counter
        chain_id = _int_to_chain(chain_counter)
        chain_counter += 1
        return chain_id

    # --- Resolve modifications for this condition ---
    cond_mods = manifest.get_modifications_for_condition(condition_id)
    mod_by_construct: Dict[str, List[Dict[str, Any]]] = {}
    for cm in cond_mods:
        # Look up the modification definition (biological layer)
        mod_def = None
        if modification_registry and cm.modification_id in modification_registry:
            mod_def = modification_registry[cm.modification_id]

        # Look up AF3 representation (technical layer)
        af3_rec = _find_af3_representation(
            cm.modification_id, af3_compatibility_registry
        )

        # Determine CCD code with proper fallback chain:
        # 1. AF3 compatibility registry (authoritative)
        # 2. ModificationRecord.modified_residue (hint, for backward compat)
        # 3. Empty (will fail at job-build time)
        af3_ccd = ""
        af3_status = AF3RepresentationStatus.REPRESENTATION_UNCERTAIN
        if af3_rec:
            af3_ccd = af3_rec.ccd_code
            af3_status = af3_rec.status
            if af3_rec.status == AF3RepresentationStatus.UNSUPPORTED:
                spec.representation_warnings.append(
                    f"UNSUPPORTED: modification '{cm.modification_id}' "
                    f"has no AF3 representation"
                )
            elif af3_rec.status == AF3RepresentationStatus.REPRESENTATION_UNCERTAIN:
                spec.representation_warnings.append(
                    f"UNCERTAIN: modification '{cm.modification_id}' "
                    f"AF3 representation not verified"
                )
        elif mod_def and mod_def.modified_residue:
            af3_ccd = mod_def.modified_residue
            af3_status = AF3RepresentationStatus.REPRESENTATION_POSSIBLE
            spec.representation_warnings.append(
                f"POSSIBLE: modification '{cm.modification_id}' "
                f"using fallback CCD '{af3_ccd}' from modification registry"
            )

        mod_entry = {
            "modification_id": cm.modification_id,
            "ccd_code": af3_ccd,
            "position": int(cm.sequence_position) if cm.sequence_position else None,
            "af3_status": af3_status.value if af3_status else "",
            "needs_custom_ccd": af3_rec.needs_custom_ccd if af3_rec else False,
            "custom_ccd_id": af3_rec.custom_ccd_id if af3_rec else "",
        }
        construct_id = cm.construct_id or "_default"
        mod_by_construct.setdefault(construct_id, []).append(mod_entry)

    # --- Resolve entities for this condition ---
    cond_entities = manifest.get_entities_for_condition(condition_id)

    # Group entities by type
    entities_by_type: Dict[str, List[Any]] = {}
    for ent in cond_entities:
        entities_by_type.setdefault(ent.entity_type, []).append(ent)

    # --- Resolve proteins ---
    # Collect protein references from TWO sources:
    #   1. Explicit "protein" entity records in condition_entities
    #   2. construct_id references from condition_modifications
    constructs_from_entities: Set[str] = set()
    for ent in entities_by_type.get("protein", []):
        constructs_from_entities.add(ent.entity_id)

    constructs_from_mods: Set[str] = set(
        cid for cid in mod_by_construct.keys() if cid != "_default"
    )

    all_constructs = constructs_from_entities | constructs_from_mods

    # Validate: at least one protein reference must exist
    if not all_constructs:
        raise ValueError(
            f"Condition '{condition_id}' has no protein references "
            f"(no protein entities and no modification construct references). "
            f"Add a protein entity record or modification with construct_id."
        )

    if construct_registry:
        for construct_id in sorted(all_constructs):
            construct = construct_registry.get(construct_id)
            if not construct:
                raise ValueError(
                    f"Condition '{condition_id}' references unknown "
                    f"construct '{construct_id}' not in construct registry"
                )

            protein = None
            if protein_registry:
                protein = protein_registry.get(construct.protein_id)

            chain_id = next_chain()
            modifications = mod_by_construct.get(construct_id, [])

            # Validate modification positions against construct sequence
            if construct.construct_sequence:
                seq_len = len(construct.construct_sequence)
                for mod in modifications:
                    pos = mod.get("position")
                    if pos is not None and (pos < 1 or pos > seq_len):
                        raise ValueError(
                            f"Condition '{condition_id}': modification "
                            f"'{mod.get('modification_id', '?')}' at position "
                            f"{pos} is outside construct '{construct_id}' "
                            f"sequence length ({seq_len})"
                        )

            spec.proteins.append(ResolvedEntity(
                entity_type="protein",
                entity_id=construct_id,
                chain_id=chain_id,
                sequence=construct.construct_sequence,
                modifications=modifications,
                description=(
                    f"{construct.construct_name} "
                    f"({protein.protein_name if protein else construct.protein_id})"
                ),
            ))

    # --- Resolve DNA ---
    for ent in entities_by_type.get("dna", []):
        nuc = None
        if nucleic_acid_registry:
            nuc = nucleic_acid_registry.get(ent.entity_id)

        chain_id = next_chain()
        spec.dna.append(ResolvedEntity(
            entity_type="dna",
            entity_id=ent.entity_id,
            chain_id=chain_id,
            stoichiometry=int(ent.stoichiometry) if ent.stoichiometry else 1,
            sequence=nuc.sequence if nuc else "",
            description=nuc.name if nuc else ent.entity_id,
            role=ent.role,
        ))

    # --- Resolve RNA ---
    for ent in entities_by_type.get("rna", []):
        nuc = None
        if nucleic_acid_registry:
            nuc = nucleic_acid_registry.get(ent.entity_id)

        chain_id = next_chain()
        spec.rna.append(ResolvedEntity(
            entity_type="rna",
            entity_id=ent.entity_id,
            chain_id=chain_id,
            stoichiometry=int(ent.stoichiometry) if ent.stoichiometry else 1,
            sequence=nuc.sequence if nuc else "",
            description=nuc.name if nuc else ent.entity_id,
            role=ent.role,
        ))

    # --- Resolve ligands ---
    for ent in entities_by_type.get("ligand", []):
        lig = None
        if ligand_registry:
            lig = ligand_registry.get(ent.entity_id)

        chain_id = next_chain()
        spec.ligands.append(ResolvedEntity(
            entity_type="ligand",
            entity_id=ent.entity_id,
            chain_id=chain_id,
            stoichiometry=int(ent.stoichiometry) if ent.stoichiometry else 1,
            ccd_code=lig.ccd_code if lig else "",
            smiles=lig.smiles if lig and lig.smiles else "",
            description=lig.ligand_name if lig else ent.entity_id,
            role=ent.role,
        ))

    # --- Resolve ions ---
    for ent in entities_by_type.get("ion", []):
        ion = None
        if ion_registry:
            ion = ion_registry.get(ent.entity_id)

        # Check AF3 status — always check, not just when af3_status string is set
        if ion and ion.status == AF3RepresentationStatus.UNSUPPORTED:
            spec.representation_warnings.append(
                f"UNSUPPORTED: ion '{ent.entity_id}' has no AF3 representation"
            )

        chain_id = next_chain()
        spec.ions.append(ResolvedEntity(
            entity_type="ion",
            entity_id=ent.entity_id,
            chain_id=chain_id,
            stoichiometry=int(ent.stoichiometry) if ent.stoichiometry else 1,
            ccd_code=ion.ccd_code if ion else "",
            description=ion.ion_name if ion else ent.entity_id,
            role=ent.role,
        ))

    # --- Resolve covalent bonds ---
    if covalent_bond_registry:
        entity_chain_map: Dict[str, str] = {}
        for ent in spec.all_entities:
            entity_chain_map[ent.entity_id] = ent.chain_id

        for bond_id, bond in covalent_bond_registry.items():
            e1_id = bond.entity_1_id
            e2_id = bond.entity_2_id

            if e1_id not in entity_chain_map or e2_id not in entity_chain_map:
                continue  # Bond doesn't apply to this condition

            if bond.status == AF3RepresentationStatus.UNSUPPORTED:
                spec.representation_warnings.append(
                    f"UNSUPPORTED: covalent bond '{bond_id}' "
                    f"cannot be represented in AF3"
                )
                continue

            chain_1 = entity_chain_map[e1_id]
            chain_2 = entity_chain_map[e2_id]

            try:
                bond_pair = [
                    [chain_1, int(bond.residue_1), bond.atom_1],
                    [chain_2, int(bond.residue_2), bond.atom_2],
                ]
                spec.bonds.append(bond_pair)
            except (ValueError, TypeError) as e:
                spec.representation_warnings.append(
                    f"ERROR: covalent bond '{bond_id}' "
                    f"has invalid atom spec: {e}"
                )

    return spec


# ---------------------------------------------------------------------------
# Pre-build validation
# ---------------------------------------------------------------------------

def _validate_spec_for_build(
    spec: ConditionSpec,
    *,
    allow_uncertain: bool = False,
) -> None:
    """Validate a ConditionSpec before building the JobBuilder.

    Enforces:
        1. No UNSUPPORTED representations
        2. No UNCERTAIN representations (unless allow_uncertain=True)
        3. All modifications have valid CCD codes
        4. No pending custom CCD requirements without definitions
        5. All covalent bonds have valid entity references

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    # 1. Check for UNSUPPORTED representations
    unsupported = [w for w in spec.representation_warnings if "UNSUPPORTED" in w]
    if unsupported:
        raise ValueError(
            f"Condition '{spec.condition_id}' has UNSUPPORTED AF3 "
            f"representations and cannot be built:\n"
            + "\n".join(f"  {w}" for w in unsupported)
        )

    # 2. Check for UNCERTAIN representations
    if not allow_uncertain:
        uncertain = [
            w for w in spec.representation_warnings
            if "UNCERTAIN" in w or "POSSIBLE" in w
        ]
        if uncertain:
            raise ValueError(
                f"Condition '{spec.condition_id}' has uncertain AF3 "
                f"representations (use allow_uncertain=True to override):\n"
                + "\n".join(f"  {w}" for w in uncertain)
            )

    # 3. Validate modification CCD codes
    #    Separate-entity modifications (e.g. SUMO, Ubiquitin) do NOT have
    #    a single CCD code — they are represented as distinct protein
    #    entities with covalent bonds.  Skip the CCD check for those.
    for protein in spec.proteins:
        for mod in protein.modifications:
            ccd = mod.get("ccd_code", "")
            af3_status = mod.get("af3_status", "")
            # Skip CCD validation for separate-entity modifications
            # (they use bonds, not CCD codes)
            if not ccd:
                # Check if bonds exist for this modification position
                has_bond = False
                if spec.bonds and mod.get("position") is not None:
                    for bond in spec.bonds:
                        # bond = [[chain, residue, atom], [chain, residue, atom]]
                        if any(b[1] == mod["position"] for b in bond):
                            has_bond = True
                            break
                if not has_bond:
                    raise ValueError(
                        f"Condition '{spec.condition_id}': modification "
                        f"'{mod.get('modification_id', '?')}' on protein "
                        f"'{protein.entity_id}' has no CCD code and no "
                        f"covalent bond — cannot be represented"
                    )

    # 4. Check custom CCD requirements
    for protein in spec.proteins:
        for mod in protein.modifications:
            if mod.get("needs_custom_ccd"):
                custom_id = mod.get("custom_ccd_id", "")
                if not custom_id:
                    raise ValueError(
                        f"Condition '{spec.condition_id}': modification "
                        f"'{mod.get('modification_id', '?')}' requires "
                        f"custom CCD but no custom_ccd_id is defined"
                    )

    # 5. Validate ligands have CCD codes or SMILES
    for lig in spec.ligands:
        if not lig.ccd_code and not lig.smiles:
            raise ValueError(
                f"Condition '{spec.condition_id}': ligand "
                f"'{lig.entity_id}' has neither CCD code nor SMILES"
            )

    # 6. Validate ions have CCD codes
    for ion in spec.ions:
        if not ion.ccd_code:
            raise ValueError(
                f"Condition '{spec.condition_id}': ion "
                f"'{ion.entity_id}' has no CCD code"
            )

    # 7. Validate protein sequences are non-empty
    for protein in spec.proteins:
        if not protein.sequence:
            raise ValueError(
                f"Condition '{spec.condition_id}': protein "
                f"'{protein.entity_id}' has no sequence"
            )


# ---------------------------------------------------------------------------
# Job building: ConditionSpec → JobBuilder
# ---------------------------------------------------------------------------

def build_job(
    manifest: MasterManifest,
    condition_id: str,
    *,
    seeds: Optional[List[int]] = None,
    name: Optional[str] = None,
    allow_uncertain: bool = False,
    protein_registry: Optional[Dict[str, ProteinRecord]] = None,
    construct_registry: Optional[Dict[str, ConstructRecord]] = None,
    modification_registry: Optional[Dict[str, ModificationRecord]] = None,
    nucleic_acid_registry: Optional[Dict[str, NucleicAcidRecord]] = None,
    ligand_registry: Optional[Dict[str, LigandRecord]] = None,
    ion_registry: Optional[Dict[str, IonRecord]] = None,
    af3_compatibility_registry: Optional[Dict[str, AF3CompatibilityRecord]] = None,
    covalent_bond_registry: Optional[Dict[str, CovalentBondRecord]] = None,
) -> "JobBuilder":
    """Build a single AF3 job from a manifest condition.

    This is the main entry point for converting a biological condition
    into an AF3 job specification.

    Validation order:
        1. resolve_condition → ConditionSpec
        2. _validate_spec_for_build (UNSUPPORTED, UNCERTAIN, CCD, bonds)
        3. construct JobBuilder with AF3-compatible modification format

    Parameters
    ----------
    manifest : MasterManifest
        The loaded manifest.
    condition_id : str
        The condition to build.
    seeds : list of int, optional
        Model seeds.  If None, a random seed is used.
    name : str, optional
        Job name.  Defaults to condition_name.
    allow_uncertain : bool
        If True, allow conditions with UNCERTAIN/POSSIBLE representations.
        Default False (reject by default).
    *_registry : dict, optional
        Loaded registries for resolution.

    Returns
    -------
    JobBuilder
        Fully populated job builder ready to serialize.

    Raises
    ------
    ValueError
        If the condition cannot be resolved or validated.
    """
    spec = resolve_condition(
        manifest, condition_id,
        protein_registry=protein_registry,
        construct_registry=construct_registry,
        modification_registry=modification_registry,
        nucleic_acid_registry=nucleic_acid_registry,
        ligand_registry=ligand_registry,
        ion_registry=ion_registry,
        af3_compatibility_registry=af3_compatibility_registry,
        covalent_bond_registry=covalent_bond_registry,
    )

    # Validate before building
    _validate_spec_for_build(spec, allow_uncertain=allow_uncertain)

    return _build_from_spec(spec, name=name, seeds=seeds)


def build_all_jobs(
    manifest: MasterManifest,
    *,
    seeds: Optional[List[int]] = None,
    allow_uncertain: bool = False,
    **registry_kwargs,
) -> Dict[str, Any]:
    """Build jobs for ALL conditions in the manifest.

    When ``allow_uncertain`` is ``False`` (default), conditions with
    uncertain/possible representations are *skipped* rather than raised.
    Set ``allow_uncertain=True`` to include them.

    Returns
    -------
    dict
        Mapping condition_id → JobBuilder.

    Raises
    ------
    ValueError
        If any condition has UNSUPPORTED representations.
    """
    jobs = {}
    skipped = []
    for cid in manifest.condition_ids:
        spec = resolve_condition(
            manifest, cid,
            **registry_kwargs,
        )
        try:
            _validate_spec_for_build(spec, allow_uncertain=allow_uncertain)
        except ValueError as e:
            if not allow_uncertain and (
                "UNCERTAIN" in str(e) or "POSSIBLE" in str(e)
            ):
                skipped.append(cid)
                continue
            raise
        jobs[cid] = _build_from_spec(spec, seeds=seeds)
    if skipped:
        import logging
        logging.getLogger(__name__).info(
            f"build_all_jobs: skipped {len(skipped)} condition(s) "
            f"with uncertain representations: {skipped}"
        )
    return jobs


def _build_from_spec(
    spec: ConditionSpec,
    *,
    name: Optional[str] = None,
    seeds: Optional[List[int]] = None,
) -> "JobBuilder":
    """Build a JobBuilder from an already-resolved ConditionSpec.

    Converts internal modification metadata into AF3-compatible format
    (ptmType, ptmPosition) before passing to ProteinEntity.
    """
    from af3_builder.core.job import JobBuilder
    from af3_builder.core.entities import (
        ProteinEntity, DNAEntity, RNAEntity, LigandEntity,
    )

    jb = JobBuilder()
    jb.set_name(name or spec.condition_name)

    if seeds:
        jb.set_model_seeds(seeds)

    # Add proteins — convert modifications to AF3 format
    for ent in spec.proteins:
        af3_mods = _mods_to_af3_format(ent.modifications) or None
        jb.add_protein(ProteinEntity(
            id=ent.chain_id,
            sequence=ent.sequence,
            modifications=af3_mods,

        ))

    # Add DNA
    for ent in spec.dna:
        for i in range(ent.stoichiometry):
            chain_id = (
                ent.chain_id if ent.stoichiometry == 1
                else f"{ent.chain_id}{_int_to_chain(i)}"
            )
            jb.add_dna(DNAEntity(
                id=chain_id,
                sequence=ent.sequence,
    
            ))

    # Add RNA
    for ent in spec.rna:
        for i in range(ent.stoichiometry):
            chain_id = (
                ent.chain_id if ent.stoichiometry == 1
                else f"{ent.chain_id}{_int_to_chain(i)}"
            )
            jb.add_rna(RNAEntity(
                id=chain_id,
                sequence=ent.sequence,
    
            ))

    # Add ligands
    for ent in spec.ligands:
        if ent.ccd_code:
            for copy in range(ent.stoichiometry):
                chain_id = (
                    ent.chain_id if ent.stoichiometry == 1
                    else _int_to_chain(_chain_to_int(ent.chain_id) + copy)
                )
                jb.add_ligand(LigandEntity(
                    id=chain_id,
                    ccdCodes=[ent.ccd_code],
        
                ))
        elif ent.smiles:
            jb.add_ligand(LigandEntity(
                id=ent.chain_id,
                smiles=ent.smiles,
    
            ))

    # Add ions (ions are represented as ligands in AF3)
    for ent in spec.ions:
        if ent.ccd_code:
            for copy in range(ent.stoichiometry):
                chain_id = (
                    ent.chain_id if ent.stoichiometry == 1
                    else _int_to_chain(_chain_to_int(ent.chain_id) + copy)
                )
                jb.add_ligand(LigandEntity(
                    id=chain_id,
                    ccdCodes=[ent.ccd_code],
        
                ))

    # Add covalent bonds
    for bond_pair in spec.bonds:
        jb.add_bonded_pair(bond_pair[0], bond_pair[1])

    return jb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _int_to_chain(n: int) -> str:
    """Convert integer to chain letter: 0→A, 1→B, ..., 25→Z, 26→AA, ..."""
    result = ""
    while True:
        result = chr(65 + n % 26) + result
        n = n // 26 - 1
        if n < 0:
            break
    return result


def _chain_to_int(chain: str) -> int:
    """Convert chain letter to integer: A→0, B→1, ..., Z→25, AA→26, ..."""
    result = 0
    for c in chain.upper():
        result = result * 26 + (ord(c) - 64)
    return result - 1
