# af3_builder/bonds.py

"""
Bond-related utilities for AlphaFold 3 JSON building.

Includes:
- atom_descriptor(): create a validated AF3 atom descriptor.
- bonded_pair(): construct a bonded pair between two atoms.
- is_valid_atom_descriptor(): check if a descriptor has correct shape.
- BondList: simple manager for collecting bonded pairs.

"""

from typing import List, Any


# -------------------------------------------------------------
#  Core atom descriptor constructor
# -------------------------------------------------------------
def atom_descriptor(entity_id: str, residue_index: int, atom_name: str) -> List[Any]:
    """
    Create a valid AF3 atom descriptor: [entity_id, residue_index, atom_name].

    Parameters
    ----------
    entity_id : str
    residue_index : int
    atom_name : str

    Returns
    -------
    List[Any]
    """
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError("entity_id must be a non-empty string")

    if not isinstance(residue_index, int) or residue_index < 1:
        raise ValueError("residue_index must be an integer >= 1")

    if not isinstance(atom_name, str) or not atom_name.strip():
        raise ValueError("atom_name must be a non-empty string")

    return [entity_id, residue_index, atom_name]


# -------------------------------------------------------------
#  Validators
# -------------------------------------------------------------
def is_valid_atom_descriptor(desc: Any) -> bool:
    """
    Lightweight check to verify the descriptor shape only.

    Returns
    -------
    bool
    """
    if not isinstance(desc, list):
        return False
    if len(desc) != 3:
        return False

    entity_id, residue_index, atom_name = desc

    return (
        isinstance(entity_id, str)
        and isinstance(residue_index, int) and residue_index >= 1
        and isinstance(atom_name, str)
    )


# -------------------------------------------------------------
#  Bond creation
# -------------------------------------------------------------
def bonded_pair(atom1: List[Any], atom2: List[Any]) -> List[List[Any]]:
    """
    Construct a bonded pair for AF3 JSON.

    Parameters
    ----------
    atom1 : List[Any]
    atom2 : List[Any]

    Returns
    -------
    List[List[Any]]
    """
    # Do NOT raise error here — allow AF3 to handle deep validation
    # Only minimal structural check
    if not is_valid_atom_descriptor(atom1):
        raise ValueError("atom1 is not a valid atom descriptor")
    if not is_valid_atom_descriptor(atom2):
        raise ValueError("atom2 is not a valid atom descriptor")

    return [atom1, atom2]


# -------------------------------------------------------------
#  BondList Manager
# -------------------------------------------------------------
class BondList:
    """
    Minimal manager for accumulating bonded pairs.

    Example:
        bonds = BondList()
        a1 = atom_descriptor("protein_1", 5, "CA")
        a2 = atom_descriptor("protein_1", 5, "CB")
        bonds.add(a1, a2)
        json_data["bonded_pairs"] = bonds.to_list()
    """

    def __init__(self):
        self._bonds: List[List[Any]] = []

    def add(self, atom1: List[Any], atom2: List[Any]):
        """
        Add a bonded pair to the internal list.
        """
        pair = bonded_pair(atom1, atom2)
        self._bonds.append(pair)

    def extend(self, pairs: List[List[Any]]):
        """
        Append multiple bonded pairs (already validated).
        """
        for p in pairs:
            if not (isinstance(p, list) and len(p) == 2):
                raise ValueError("Invalid bonded pair format")
            if not is_valid_atom_descriptor(p[0]) or not is_valid_atom_descriptor(p[1]):
                raise ValueError("Invalid atom descriptor in bonded pair")
            self._bonds.append(p)


    def to_list(self) -> List[List[Any]]:
        """
        Return a JSON-ready list of bonded pairs.
        """
        return self._bonds
