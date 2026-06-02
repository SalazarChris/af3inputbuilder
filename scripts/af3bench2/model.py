"""
Data containers for the AF3 analysis pipeline.

ConditionModel       — the AF3 rank-1 representative model for one condition.
EnsembleModel        — all seed-*_sample-* replicate models for one condition,
                       used to estimate structural sampling noise.
ExperimentStructure  — parsed factorial layout (PTM group x ion tier, DNA).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-residue record key: (chain_id, residue_number)
# ---------------------------------------------------------------------------
ResidueKey = tuple  # (str, int)


class ConditionModel:
    """
    Representative (AF3 rank-1) model for one condition.

    Protein Calpha and nucleic C4' atoms are stored separately so that
    alignment is always performed on protein backbone only, regardless of
    whether DNA/RNA is present.

    Coordinates are paired with explicit (chain_id, residue_number) keys so
    that downstream alignment can match residues by identity rather than by
    array position.
    """

    def __init__(self, name: str, cif_path: Path) -> None:
        self.name = name
        self.cif_path = cif_path

        # Global confidence scores
        self.ptm: float = float("nan")
        self.iptm: float = float("nan")
        self.ranking_score: float = float("nan")

        # Protein Calpha
        self.ca_coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.ca_plddts: np.ndarray = np.empty(0, dtype=np.float64)
        self.ca_chain_ids: List[str] = []
        self.ca_res_indices: List[int] = []

        # Nucleic C4'
        self.na_coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.na_plddts: np.ndarray = np.empty(0, dtype=np.float64)
        self.na_chain_ids: List[str] = []
        self.na_res_indices: List[int] = []

        # Full-model confidence
        self.pae_matrix: Optional[np.ndarray] = None
        self.atom_plddts: Optional[np.ndarray] = None
        # token -> chain id mapping for PAE block extraction (optional)
        self.token_chain_ids: Optional[List[str]] = None

        # Parsed from data.json
        self.protein_chain_ids_from_json: List[str] = []
        self.nucleic_chain_ids_from_json: List[str] = []
        self.description: str = ""

        # Structured metadata (populated by factors.parse_condition_factors)
        self.ptm_labels: List[str] = []
        self.n_na: int = 0               # explicit Na+ count
        self.n_cl: int = 0               # explicit Cl- count
        self.n_water: int = 0            # explicit water count
        self.n_smiles: int = 0           # entities given as a SMILES string (smilesxN)
        self.has_real_ligand: bool = False

        # Short display label (populated by factors.build_experiment_structure /
        # an optional labels.csv).  Used for every plot axis label and table.
        self.label_short: str = ""

        # Ensemble attached later (optional)
        self.ensemble: Optional["EnsembleModel"] = None

    # ------------------------------------------------------------------
    @property
    def n_protein_residues(self) -> int:
        return len(self.ca_coords)

    @property
    def n_nucleic_residues(self) -> int:
        return len(self.na_coords)

    @property
    def ca_keys(self) -> List[ResidueKey]:
        """(chain_id, residue_number) for each protein Calpha, in array order."""
        return list(zip(self.ca_chain_ids, self.ca_res_indices))

    @property
    def na_keys(self) -> List[ResidueKey]:
        return list(zip(self.na_chain_ids, self.na_res_indices))

    @property
    def mean_plddt(self) -> float:
        if len(self.ca_plddts) > 0:
            return float(np.mean(self.ca_plddts))
        return float("nan")

    @property
    def mean_pae(self) -> float:
        if self.pae_matrix is not None:
            return float(np.mean(self.pae_matrix))
        return float("nan")


@dataclass
class EnsembleModel:
    """
    All successfully-parsed seed-*_sample-* replicate models for one condition.

    Stores protein Calpha coordinates as a stack and the matching residue keys
    so that ensemble statistics (RMSF, per-residue displacement CIs) can be
    computed against a common residue ordering.

    Attributes
    ----------
    name          condition name
    ca_coords     (S, N, 3) stacked Calpha coordinates (S samples, N residues)
    ca_plddts     (S, N)    per-sample per-residue pLDDT
    ca_keys       list[(chain_id, residue_number)] length N (shared ordering)
    ptm           (S,) per-sample global pTM   (may be empty)
    iptm          (S,) per-sample global ipTM
    plddt_mean    (S,) per-sample mean pLDDT
    sample_paths  list[Path] source CIFs, parallel to axis 0
    """

    name: str
    ca_coords: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 3)))
    ca_plddts: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    ca_keys: List[ResidueKey] = field(default_factory=list)
    ptm: np.ndarray = field(default_factory=lambda: np.empty(0))
    iptm: np.ndarray = field(default_factory=lambda: np.empty(0))
    plddt_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    sample_paths: List[Path] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.ca_coords.shape[0] if self.ca_coords.ndim == 3 else 0

    @property
    def n_residues(self) -> int:
        return self.ca_coords.shape[1] if self.ca_coords.ndim == 3 and self.ca_coords.size else 0

    @property
    def has_structural_ensemble(self) -> bool:
        return self.n_samples >= 2 and self.n_residues > 0


@dataclass
class ExperimentStructure:
    """
    Parsed factorial structure of the loaded conditions.

    ion_tier         {name: label}   e.g. "0x", "1x", "10x", "100x"
    ptm_group        {name: label}   e.g. "none", "SEP102", "TPO101", "DNA"
    has_dna          {name: bool}
    has_real_ligand  {name: bool}
    tier_order       sorted unique ion tiers (ascending)
    ptm_order        sorted unique PTM groups ("none" first)
    panel_conditions set of condition names that fit the ion x PTM grid
    ligand_mult      {name: int}     smilesxN multiplier per condition
    ligand_to_salt   {name: float}   ligand_count / salt_count (NaN if no salt)
    label_short      {name: str}     short display label per condition
    confound         dict            ligand/salt co-variation diagnostics
    """
    ion_tier: Dict[str, str]
    ptm_group: Dict[str, str]
    has_dna: Dict[str, bool]
    has_real_ligand: Dict[str, bool]
    tier_order: List[str]
    ptm_order: List[str]
    panel_conditions: set
    ligand_mult: Dict[str, int] = field(default_factory=dict)
    ligand_to_salt: Dict[str, float] = field(default_factory=dict)
    label_short: Dict[str, str] = field(default_factory=dict)
    confound: Dict[str, object] = field(default_factory=dict)
