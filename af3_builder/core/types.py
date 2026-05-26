# af3_builder/types.py
from typing import Dict, Any, List, Union, Optional, TypedDict

EntityDict = Dict[str, Any]  # general mapping for entity entries
JobDict = Dict[str, Any]

AtomDescriptor = List[Union[str, int]]  # [entityId, residueIndex (1-based int), atomName]
BondPair = List[AtomDescriptor]  # [[e1,res1,atom1], [e2,res2,atom2]]
