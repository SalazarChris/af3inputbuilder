# af3_builder/validator.py
import os
import json
from copy import deepcopy
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import re
from ..core.types import JobDict, AtomDescriptor
from ..core.seeds import SeedsHelper

def _is_file_path(p: str) -> bool:
    return isinstance(p, str) and p.strip() != "" and os.path.isfile(p) and not os.path.isdir(p)

def _validate_msa_fields(entity: Dict[str, Any], errors: List[str], prefix: str):
    # mutually exclusive pairs
    if entity.get("unpairedMsa") is not None and entity.get("unpairedMsaPath") is not None:
        errors.append(f"{prefix}: unpairedMsa and unpairedMsaPath are mutually exclusive.")
    if entity.get("pairedMsa") is not None and entity.get("pairedMsaPath") is not None:
        errors.append(f"{prefix}: pairedMsa and pairedMsaPath are mutually exclusive.")

    # if a path is provided, it must be a file
    if entity.get("unpairedMsaPath") is not None and not _is_file_path(entity["unpairedMsaPath"]):
        errors.append(f"{prefix}: unpairedMsaPath must be an existing file path (not a directory).")
    if entity.get("pairedMsaPath") is not None and not _is_file_path(entity["pairedMsaPath"]):
        errors.append(f"{prefix}: pairedMsaPath must be an existing file path (not a directory).")

def _validate_templates(entity: Dict[str, Any], errors: List[str], prefix: str):
    templates = entity.get("templates")
    if templates is None:
        return
    if not isinstance(templates, list):
        errors.append(f"{prefix}: templates must be a list.")
        return

    for i, t in enumerate(templates):
        if not isinstance(t, dict):
            errors.append(f"{prefix}: templates[{i}] must be an object.")
            continue

        has_mmcif = "mmcif" in t and t["mmcif"] is not None
        has_path = "mmcifPath" in t and t["mmcifPath"] is not None
        if has_mmcif and has_path:
            errors.append(f"{prefix}: templates[{i}] mmcif and mmcifPath are mutually exclusive.")
        if not has_mmcif and not has_path:
            errors.append(f"{prefix}: templates[{i}] must provide mmcif or mmcifPath.")

        if has_path and not _is_file_path(t["mmcifPath"]):
            errors.append(f"{prefix}: templates[{i}] mmcifPath must be an existing file path (not a directory).")

        qi = t.get("queryIndices")
        ti = t.get("templateIndices")
        if not isinstance(qi, list) or not all(isinstance(x, int) for x in qi):
            errors.append(f"{prefix}: templates[{i}] queryIndices must be list[int].")
        if not isinstance(ti, list) or not all(isinstance(x, int) for x in ti):
            errors.append(f"{prefix}: templates[{i}] templateIndices must be list[int].")
        if isinstance(qi, list) and isinstance(ti, list) and len(qi) != len(ti):
            errors.append(f"{prefix}: templates[{i}] queryIndices and templateIndices must have same length.")


class ValidationError(Exception):
    def __init__(self, messages: List[str]):
        super().__init__("\n".join(messages))
        self.messages = messages

class AF3Validator:
    ENTITY_KEYS = {"protein", "rna", "dna", "ligand"}

    @staticmethod
    def validate_job(job: JobDict, *, require_files: bool = True) -> None:
        errors: List[str] = []
        user_ccd = job.get("userCCD")
        user_ccd_path = job.get("userCCDPath")

        if user_ccd is not None and user_ccd_path is not None:
            errors.append("userCCD and userCCDPath are mutually exclusive.")
               
               
        # Optional strict file checks (for local testing you can turn this off)
        if require_files:
            # userCCDPath
            if job.get("userCCDPath") is not None and not _is_file_path(job.get("userCCDPath")):
                errors.append("userCCDPath must be an existing file path (not a directory).")

            # MSA + template paths per entity
            for i, ent in enumerate(job.get("sequences", []) or []):
                if not isinstance(ent, dict) or len(ent) != 1:
                    continue
                key = next(iter(ent.keys()))
                data = ent[key]
                if not isinstance(data, dict):
                    continue

                # protein MSA paths
                if key == "protein":
                    if data.get("unpairedMsaPath") is not None and not _is_file_path(data.get("unpairedMsaPath")):
                        errors.append(f"sequences[{i}]['protein'].unpairedMsaPath must be an existing file path.")
                    if data.get("pairedMsaPath") is not None and not _is_file_path(data.get("pairedMsaPath")):
                        errors.append(f"sequences[{i}]['protein'].pairedMsaPath must be an existing file path.")

                    # templates mmcifPath
                    tmpls = data.get("templates")
                    if isinstance(tmpls, list):
                        for j, t in enumerate(tmpls):
                            if isinstance(t, dict) and t.get("mmcifPath") is not None:
                                if not _is_file_path(t.get("mmcifPath")):
                                    errors.append(f"sequences[{i}]['protein'].templates[{j}].mmcifPath must be an existing file path.")

                # rna MSA paths
                if key == "rna":
                    if data.get("unpairedMsaPath") is not None and not _is_file_path(data.get("unpairedMsaPath")):
                        errors.append(f"sequences[{i}]['rna'].unpairedMsaPath must be an existing file path.")
                    if data.get("pairedMsaPath") is not None and not _is_file_path(data.get("pairedMsaPath")):
                        errors.append(f"sequences[{i}]['rna'].pairedMsaPath must be an existing file path.")

        # 1. Top-level required fields
        for field, expected_type in [("name", str), ("dialect", str), ("version", int), ("modelSeeds", list), ("sequences", list)]:
            val = job.get(field)
            if val is None:
                errors.append(f"Top-level: missing required field '{field}'.")
            elif not isinstance(val, expected_type):
                errors.append(f"Top-level: '{field}' must be {expected_type.__name__}, got {type(val).__name__}.")

        if job.get("name") == "":
            errors.append("Top-level: 'name' cannot be an empty string.")

        if job.get("dialect") != "alphafold3":
            errors.append(f"Top-level: 'dialect' must be 'alphafold3', got {job.get('dialect')!r}.")

        # 2. Mutually exclusive userCCD
        if job.get("userCCD") is not None and job.get("userCCDPath") is not None:
            errors.append("Top-level: 'userCCD' and 'userCCDPath' are mutually exclusive.")

        # 3. Sequences
        seqs = job.get("sequences", [])
        if isinstance(seqs, list):
            if not seqs:
                errors.append("Top-level: 'sequences' list is empty.")
            
            ids = []
            for i, ent in enumerate(seqs):
                if not isinstance(ent, dict):
                    errors.append(f"sequences[{i}]: must be an object.")
                    continue
                if len(ent) != 1:
                    errors.append(f"sequences[{i}]: must have exactly one key (protein/rna/dna/ligand).")
                    continue
                
                kind = next(iter(ent.keys()))
                data = ent[kind]
                
                # Check for legacy keys
                LEGACY = {"proteinChain": "protein", "rnaSequence": "rna", "dnaSequence": "dna"}
                if kind in LEGACY:
                    errors.append(f"sequences[{i}]: deprecated key '{kind}'. Use '{LEGACY[kind]}' instead.")
                    kind = LEGACY[kind] # treat as modern for further checks

                if kind not in AF3Validator.ENTITY_KEYS:
                    errors.append(f"sequences[{i}]: unsupported entity type '{kind}'.")
                    continue

                if not isinstance(data, dict):
                    errors.append(f"sequences[{i}]['{kind}']: must be an object.")
                    continue

                # ID check
                eid = data.get("id")
                if eid is None:
                    errors.append(f"sequences[{i}]['{kind}']: missing 'id'.")
                else:
                    if isinstance(eid, list):
                        ids.extend(str(x) for x in eid)
                    else:
                        ids.append(str(eid))

                # MSA Swaps and Exclusivity
                prefix = f"sequences[{i}]['{kind}']"
                _validate_msa_fields(data, errors, prefix)
                
                # Heuristic for swapped MSA paths
                upath = data.get("unpairedMsaPath", "")
                ppath = data.get("pairedMsaPath", "")
                if isinstance(upath, str) and isinstance(ppath, str) and upath and ppath:
                    u_low, p_low = upath.lower(), ppath.lower()
                    if "paired" in u_low and "unpaired" not in u_low and "unpaired" in p_low:
                        errors.append(f"{prefix}: POSSIBLE SWAP — 'unpairedMsaPath' looks like a paired MSA, and vice-versa.")

                # Protein specific
                if kind == "protein":
                    _validate_templates(data, errors, prefix)
                
                # DNA should not have MSAs
                if kind == "dna":
                    for f in ["unpairedMsa", "unpairedMsaPath", "pairedMsa", "pairedMsaPath"]:
                        if f in data:
                            errors.append(f"{prefix}: DNA does not support MSA field '{f}'.")

            # Duplicate IDs
            dupes = [k for k, v in Counter(ids).items() if v > 1]
            if dupes:
                errors.append(f"Duplicate chain IDs found: {sorted(dupes)}")

        # 4. Bonded atom pairs
        baps = job.get("bondedAtomPairs", [])
        if isinstance(baps, list):
            for i, pair in enumerate(baps):
                if not isinstance(pair, list) or len(pair) != 2:
                    errors.append(f"bondedAtomPairs[{i}]: must be a list of 2 atom descriptors.")
                    continue
                for j, atom in enumerate(pair):
                    if not isinstance(atom, list) or len(atom) != 3:
                        errors.append(f"bondedAtomPairs[{i}][{j}]: must be [entityId, residueIndex, atomName].")

        if errors:
            raise ValidationError(errors)


