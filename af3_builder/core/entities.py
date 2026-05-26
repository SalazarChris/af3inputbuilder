from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from .id_manager import IDManager


# ---------------------------------------------------------
# Helper validation functions
# ---------------------------------------------------------

def _require_nonempty_string(value: str, field: str):
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{field} must be a non-empty string")


def _validate_id(id_val: Union[str, List[str]]):
    if isinstance(id_val, str):
        if id_val.strip() == "":
            raise ValueError("id cannot be empty string")
    elif isinstance(id_val, list):
        if len(id_val) == 0:
            raise ValueError("id list cannot be empty")
        for v in id_val:
            if not isinstance(v, str) or v.strip() == "":
                raise ValueError("id entries in list must be non-empty strings")
    else:
        raise ValueError("id must be a string or list of strings")


def _validate_sequence(seq: str, alphabet: str, label: str):
    if not isinstance(seq, str) or len(seq) == 0:
        raise ValueError(f"{label} must be a non-empty string")

    seq_set = set(seq.upper())
    allowed_set = set(alphabet)

    if not seq_set.issubset(allowed_set):
        invalid = seq_set - allowed_set
        raise ValueError(f"{label} contains invalid characters: {invalid}")


# Protein / RNA / DNA alphabets
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"   # X allowed for unknown
RNA_ALPHABET     = "ACGUNX"                  # N/X allowed
DNA_ALPHABET     = "ACGTNX"


def reverse_complement(seq: str, is_dna: bool = True) -> str:
    """Return the 5'->3' reverse complement of a nucleotide sequence."""
    mapping = {
        "A": "T" if is_dna else "U",
        "T": "A",
        "U": "A",
        "G": "C",
        "C": "G",
        "N": "N",
        "X": "X"
    }
    return "".join(mapping.get(b.upper(), b) for b in reversed(seq))


def slugify(text: str) -> str:
    """Convert a label into a filesystem-safe slug."""
    import re
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text[:48]


def next_spreadsheet_id(used: List[str]) -> str:
    """Return the next unused chain ID (A, B, ..., Z, AA, AB, ...)."""
    from .id_manager import IDManager
    # Find the high-water mark or first hole
    # For simplicity, we'll just increment from the last one or start at 0
    # Actually, let's just find the first i >= 0 that isn't in used
    i = 0
    while True:
        try:
            candidate = IDManager.to_spreadsheet_style(i)
            if candidate not in used:
                return candidate
        except Exception:
            break
        i += 1
    return "X" # Fallback



# ---------------------------------------------------------
# Base validation for MSAs and templates
# ---------------------------------------------------------

def _validate_msa(unpaired: Optional[str], unpaired_path: Optional[str],
                  paired: Optional[str], paired_path: Optional[str]):
    """Validate MSA field combinations.
    
    AF3 semantics:
      - Field absent (None)  → AF3 calculates MSA automatically
      - Field = ""           → explicitly skip MSA search
      - Field = content/path → use the provided MSA
    
    Rules:
      - unpairedMsa and unpairedMsaPath are mutually exclusive (both non-None is invalid)
      - pairedMsa and pairedMsaPath are mutually exclusive
      - Setting msa="" (skip) alongside a path is contradictory
    """
    # Check mutual exclusivity: if EITHER is not None, the other must be None
    if unpaired is not None and unpaired_path is not None:
        raise ValueError("Provide either unpairedMsa OR unpairedMsaPath, not both")
    if paired is not None and paired_path is not None:
        raise ValueError("Provide either pairedMsa OR pairedMsaPath, not both")


def _validate_templates(templates: Optional[List[Dict[str, Any]]]):
    if templates is None:
        return

    if not isinstance(templates, list):
        raise ValueError("templates must be a list")

    for tmpl in templates:
        if not isinstance(tmpl, dict):
            raise ValueError("each template entry must be a dictionary")
        # minimal structural check
        if "mmcif" not in tmpl and "mmcifPath" not in tmpl:
            raise ValueError("template entry must contain mmcif or mmcifPath")


# ---------------------------------------------------------
# ENTITY CLASSES
# ---------------------------------------------------------

@dataclass
class ProteinEntity:
    id: Union[str, List[str]]
    sequence: str
    pairedMsaPath: Optional[str] = None
    unpairedMsaPath: Optional[str] = None
    pairedMsa: Optional[str] = None
    unpairedMsa: Optional[str] = None
    modifications: Optional[List[Dict[str, Any]]] = None
    templates: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None

    def __post_init__(self):
        _validate_id(self.id)
        _validate_sequence(self.sequence, PROTEIN_ALPHABET, "protein sequence")
        _validate_msa(self.unpairedMsa, self.unpairedMsaPath,
                      self.pairedMsa, self.pairedMsaPath)
        _validate_templates(self.templates)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "sequence": self.sequence,
        }
        # Canonical field order: unpaired → paired → modifications → templates → description
        if self.unpairedMsaPath is not None: d["unpairedMsaPath"] = self.unpairedMsaPath
        if self.unpairedMsa is not None:     d["unpairedMsa"] = self.unpairedMsa
        if self.pairedMsaPath is not None:   d["pairedMsaPath"] = self.pairedMsaPath
        if self.pairedMsa is not None:       d["pairedMsa"] = self.pairedMsa
        if self.modifications:               d["modifications"] = self.modifications
        if self.templates is not None:       d["templates"] = self.templates
        if self.description:                 d["description"] = self.description
        return {"protein": d}


@dataclass
class RNAEntity:
    id: Union[str, List[str]]
    sequence: str
    unpairedMsaPath: Optional[str] = None
    unpairedMsa: Optional[str] = None
    modifications: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None

    def __post_init__(self):
        _validate_id(self.id)
        _validate_sequence(self.sequence, RNA_ALPHABET, "RNA sequence")
        _validate_msa(self.unpairedMsa, self.unpairedMsaPath,
                      None, None)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "sequence": self.sequence,
        }
        # Canonical field order: unpaired → modifications → description
        if self.unpairedMsaPath is not None: d["unpairedMsaPath"] = self.unpairedMsaPath
        if self.unpairedMsa is not None:     d["unpairedMsa"] = self.unpairedMsa
        if self.modifications:               d["modifications"] = self.modifications
        if self.description:                 d["description"] = self.description
        return {"rna": d}


@dataclass
class DNAEntity:
    id: Union[str, List[str]]
    sequence: str
    modifications: Optional[List[Dict[str, Any]]] = None
    description: Optional[str] = None
    unpairedMsa: Optional[str] = None
    unpairedMsaPath: Optional[str] = None

    def __post_init__(self):
        _validate_id(self.id)
        _validate_sequence(self.sequence, DNA_ALPHABET, "DNA sequence")
        _validate_msa(self.unpairedMsa, self.unpairedMsaPath,
                      None, None)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id, "sequence": self.sequence}
        if self.unpairedMsaPath is not None: d["unpairedMsaPath"] = self.unpairedMsaPath
        if self.unpairedMsa is not None:     d["unpairedMsa"] = self.unpairedMsa
        if self.modifications:               d["modifications"] = self.modifications
        if self.description:                 d["description"] = self.description
        return {"dna": d}


@dataclass
class LigandEntity:
    id: Union[str, List[str]]
    ccdCodes: Optional[List[str]] = None
    smiles: Optional[str] = None
    description: Optional[str] = None
    copies: Optional[int] = None

    def __post_init__(self):
        # If user requested multiple copies and provided a single string id,
        # expand it into a list of unique IDs.
        if self.copies is not None:
            if not isinstance(self.copies, int) or self.copies <= 0:
                raise ValueError("copies must be a positive integer")

            if isinstance(self.id, str):
                base = self.id.strip()
                if base == "":
                    raise ValueError("id cannot be empty string")

                # If copies==1 keep as single string; otherwise expand
                if self.copies == 1:
                    self.id = base
                else:
                    # Generate IDs: base, nxt, nxt+1... (spreadsheet style)
                    # This ensures only uppercase letters are used
                    try:
                        start_idx = IDManager.from_spreadsheet_style(base)
                        self.id = [IDManager.to_spreadsheet_style(i) 
                                   for i in range(start_idx, start_idx + self.copies)]
                    except ValueError:
                        # Fallback for non-alphabetic base (though _validate_id will catch it later)
                        self.id = [f"{base}{i}" for i in range(1, self.copies + 1)]

            elif isinstance(self.id, list):
                # If id already a list, don't auto-expand; optionally sanity-check length
                # (We won't force it, to avoid breaking existing scripts.)
                pass
            else:
                raise ValueError("id must be a string or list of strings")

        _validate_id(self.id)

        # XOR rule: exactly one must be provided
        if (self.ccdCodes is None and self.smiles is None) or \
           (self.ccdCodes is not None and self.smiles is not None):
            raise ValueError("Ligand must provide exactly one of: ccdCodes OR smiles")

        # Validate CCD codes
        if self.ccdCodes is not None:
            if not isinstance(self.ccdCodes, list) or len(self.ccdCodes) != 1:
                raise ValueError("ccdCodes must be a list containing exactly one CCD code")
            code = self.ccdCodes[0]
            if not isinstance(code, str) or code.strip() == "" or not (1 <= len(code.strip()) <= 4):
                raise ValueError("CCD code must be a 1–4 character non-empty string")
            self.ccdCodes = [code.strip()]


        # SMILES validation left intentionally minimal
        if self.smiles is not None:
            _require_nonempty_string(self.smiles, "smiles")

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id}
        if self.ccdCodes:    d["ccdCodes"] = self.ccdCodes
        if self.smiles:      d["smiles"] = self.smiles
        if self.description: d["description"] = self.description
        return {"ligand": d}
