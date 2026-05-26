# af3_builder/job.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from .entities import ProteinEntity, RNAEntity, DNAEntity, LigandEntity
from .id_manager import IDManager
from .seeds import SeedsHelper

class JobBuilder:
    """Assemble alphafold3 job dictionaries incrementally."""

    def __init__(self):
        self.name: str = ""
        self.modelSeeds: List[int] = []
        self.sequences: List[Dict[str, Any]] = []
        self.bondedAtomPairs: List[List[Any]] = []
        self.userCCD: Optional[str] = None
        self.userCCDPath: Optional[str] = None
        self.dialect: str = "alphafold3"
        self.version: int = 1
        self._id_manager = IDManager()

    # basic setters
    def set_name(self, name: str):
        self.name = str(name)
        return self
    
    def _require_version(self, min_version: int):
        if self.version < min_version:
            self.version = min_version

    def set_version(self, version: int):
        if not isinstance(version, int) or not (1 <= version <= 3):
            raise ValueError("version must be integer between 1 and 3")
        self.version = version
        return self

    def set_model_seeds(self, seeds: List[int]):
        SeedsHelper.validate_seeds(seeds)
        self.modelSeeds = list(seeds)
        return self

    def add_protein(self, prot: ProteinEntity):
        # ensure id unique if string; if None or '', allocate
        self._ensure_entity_id_unique(prot.id)
        # Version upgrades based on used fields (per input.md)
        if prot.unpairedMsaPath or prot.pairedMsaPath:
            self._require_version(2)
        if prot.templates:
            # templates exist; mmcifPath usage specifically requires v2,
            # but we can safely bump to v2 whenever templates are present
            self._require_version(2)
        if prot.description:
            self._require_version(3)
        self.sequences.append(prot.to_dict())
        return self

    def add_rna(self, rna: RNAEntity):
        self._ensure_entity_id_unique(rna.id)
        if rna.unpairedMsaPath:
            self._require_version(2)
        if rna.description:
            self._require_version(3)
        self.sequences.append(rna.to_dict())
        return self

    def add_dna(self, dna: DNAEntity):
        self._ensure_entity_id_unique(dna.id)
        if dna.description:
            self._require_version(3)
        self.sequences.append(dna.to_dict())
        return self

    def add_ligand(self, ligand: LigandEntity):
        self._ensure_entity_id_unique(ligand.id)
        if ligand.description:
            self._require_version(3)
        if ligand.smiles:
            self._require_version(3)
        self.sequences.append(ligand.to_dict())
        return self

    def add_bonded_pair(self, atom1: list, atom2: list):
        # no deep checking here; validator module will check format and existence later
        self.bondedAtomPairs.append([atom1, atom2])
        return self

    def set_userCCD(self, ccd_block: str):
        if self.userCCDPath is not None:
            raise ValueError("userCCDPath already set; userCCD and userCCDPath are mutually exclusive")
        # Ensure any mistakenly double-escaped newlines from JSON strings are cleaned
        self.userCCD = ccd_block.replace('\\n', '\n')
        self._require_version(3)  # userCCDPath requires version >= 3
        return self

    def set_userCCDPath(self, path: str):
        if self.userCCD is not None:
            raise ValueError("userCCD already set inline; mutually exclusive with userCCDPath")
        self.userCCDPath = path
        return self

    def _ensure_entity_id_unique(self, id_val):
        """Check strings or lists for duplicates and auto-assign if None."""
        current_ids = self._current_ids()
        if id_val is None or id_val == "":
            new_id = self._id_manager.next_id(existing_ids=current_ids)
            # callers pass in entity with id property; but we don't mutate entity here
            raise ValueError("Entity id must be provided (string or list). Use id_manager.next_id(...) if you want auto-assignment.")
        # if list, ensure all not in current_ids
        if isinstance(id_val, list):
            for v in id_val:
                if v in current_ids:
                    raise ValueError(f"Duplicate id detected: {v}")
        else:
            if id_val in current_ids:
                raise ValueError(f"Duplicate id detected: {id_val}")

    def _current_ids(self) -> List[str]:
        ids = []
        for ent in self.sequences:
            # ent is like {"protein": {...}} or {"ligand": {...}}
            for key, val in ent.items():
                eid = val.get("id")
                if isinstance(eid, list):
                    ids.extend([str(x) for x in eid])
                elif eid is not None:
                    ids.append(str(eid))
        return ids

    def to_dict(self) -> Dict[str, Any]:
        if not self.modelSeeds:
            import random
            self.modelSeeds = [random.randint(1, 9999)]
        base: Dict[str, Any] = {
            "name": self.name,
            "modelSeeds": list(self.modelSeeds),
            "sequences": list(self.sequences),
        }
        if self.bondedAtomPairs:
            base["bondedAtomPairs"] = list(self.bondedAtomPairs)
        if self.userCCD is not None:
            base["userCCD"] = self.userCCD
        if self.userCCDPath is not None:
            base["userCCDPath"] = self.userCCDPath
            
        # dialect/version usually come at the very end in AF3 JSON format
        base["dialect"] = self.dialect
        base["version"] = int(self.version)
        
        # Ensure all paths are normalized to POSIX style (forward slashes)
        # for compatibility with Linux-based AlphaFold 3 environments.
        return self._normalize_any_paths(base)

    def _normalize_any_paths(self, obj: Any) -> Any:
        """
        Recursively traverse the dictionary and normalize any string field
        ending in 'Path' to use POSIX-style forward slashes (/).
        """
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.endswith("Path") and isinstance(v, str):
                    new_dict[k] = v.replace("\\", "/")
                else:
                    new_dict[k] = self._normalize_any_paths(v)
            return new_dict
        elif isinstance(obj, list):
            return [self._normalize_any_paths(item) for item in obj]
        return obj

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobBuilder":
        """
        Reconstruct a JobBuilder from an alphafold3-dialect job dict.
        Does not auto-validate; caller can validate with AF3Validator.
        """
        if not isinstance(data, dict):
            raise ValueError("Job data must be a dict/object.")
        if data.get("dialect") != "alphafold3":
            raise ValueError("Only dialect='alphafold3' jobs are supported.")

        jb = cls()
        jb.name = data.get("name", "")
        jb.modelSeeds = list(data.get("modelSeeds", []))
        
        # Load and normalize sequences
        raw_seqs = list(data.get("sequences", []))
        LEGACY = {"proteinChain": "protein", "rnaSequence": "rna", "dnaSequence": "dna"}
        normalized = []
        for ent in raw_seqs:
            if isinstance(ent, dict) and len(ent) == 1:
                key = next(iter(ent.keys()))
                if key in LEGACY:
                    ent[LEGACY[key]] = ent.pop(key)
            normalized.append(ent)
        
        jb.sequences = normalized
        jb.bondedAtomPairs = list(data.get("bondedAtomPairs", []))
        jb.userCCD = data.get("userCCD", None)
        jb.userCCDPath = data.get("userCCDPath", None)
        jb.version = int(data.get("version", 1))
        return jb
