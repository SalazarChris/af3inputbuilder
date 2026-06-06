"""
Experimental factor parsing — single source of truth.

All factor information (PTM identity, ion counts, water count, DNA presence)
is derived from the AF3 ``*_data.json`` input file, never re-derived from the
folder name.  This replaces the original code's two competing sources
(parse_experiment_structure from JSON and _parse_condition_metadata from the
folder name regex).

Key corrections over the original:

  * Accepts both ``ccdCode`` (string) and ``ccdCodes`` (list) — the real AF3
    inputs use the plural list form, which the original code never read.
  * Separates Na+/Cl- ions from water molecules, so the "ion concentration"
    axis is no longer confounded by solvent scaling.  Water is reported as its
    own covariate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

log = logging.getLogger("af3bench.factors")

# CCD codes treated as monatomic ions (extend as needed)
_ION_CCDS = {
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "FE2", "CU", "CU1",
    "NI", "CO", "CD", "HG", "BA", "SR", "LI", "RB", "CS", "BR", "IOD", "F",
}
# CCD codes / SMILES treated as water
_WATER_CCDS = {"HOH", "WAT", "DOD"}
_WATER_SMILES = {"O", "[OH2]", "OO"}  # AF3 uses bare "O" for water in these inputs

_NUCLEIC = {"dna", "rna"}

# CCD monomer codes for nucleotides.  A DNA/RNA molecule supplied as a
# *nonpolymer ligand* entity (rather than a proper ``dna``/``rna`` polymer
# entity) is identified by these codes so it is treated as nucleic acid, not as
# a small-molecule ligand.  DNA monomers are prefixed 'D'.
_DNA_CCDS = {"DA", "DC", "DG", "DT", "DU", "DI", "DN"}
_RNA_CCDS = {"A", "C", "G", "U", "I", "N"}
_NUCLEIC_CCDS = _DNA_CCDS | _RNA_CCDS


def _entity_ids(v: dict) -> List[str]:
    raw = v.get("id", "")
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return [str(raw)] if raw != "" else []


def _ccd_codes(v: dict) -> List[str]:
    """Return CCD codes from either ``ccdCode`` (str) or ``ccdCodes`` (list)."""
    out: List[str] = []
    single = v.get("ccdCode")
    if isinstance(single, str) and single:
        out.append(single)
    plural = v.get("ccdCodes")
    if isinstance(plural, list):
        out.extend(str(c) for c in plural if c)
    elif isinstance(plural, str) and plural:
        out.append(plural)
    return [c.strip().upper() for c in out]


def classify_nonpolymer(v: dict) -> str:
    """
    Classify a ligand/ion/solvent entity as one of:
    'ion', 'water', 'nucleic', 'ligand'.

    A DNA/RNA molecule is sometimes supplied as a nonpolymer ligand entity
    (its monomers given as nucleotide CCD codes, e.g. ``ccdCodes: [DA, DT, DG]``)
    rather than as a proper ``dna``/``rna`` polymer entity.  Such an entity is
    classified as ``'nucleic'`` so it is routed to the nucleic-acid bucket and
    never counted as a small-molecule ligand.

    Decision order: nucleotide CCDs -> CCD ion code -> CCD/SMILES water ->
    SMILES heavy-atom ligand -> fallback 'ligand'.
    """
    ccds = _ccd_codes(v)
    smiles = (v.get("smiles") or "").strip()

    if ccds:
        # A run of nucleotide monomers (≥2) is a nucleic-acid oligomer supplied
        # as a ligand entity.  A single nucleotide CCD is genuinely ambiguous
        # (e.g. a mononucleotide cofactor) and is left as a ligand.
        if all(c in _NUCLEIC_CCDS for c in ccds) and len(ccds) >= 2:
            return "nucleic"
        # DNA-specific codes are unambiguous even as a single monomer.
        if all(c in _DNA_CCDS for c in ccds):
            return "nucleic"
        if all(c in _ION_CCDS for c in ccds):
            return "ion"
        if all(c in _WATER_CCDS for c in ccds):
            return "water"
        # Mixed or unknown CCD -> treat as ligand
        return "ligand"

    if smiles:
        if smiles in _WATER_SMILES:
            return "water"
        if _is_nucleic_smiles(smiles):
            return "nucleic"
        # crude heavy-atom proxy: SMILES longer than a couple of chars
        if len(smiles) > 3:
            return "ligand"
        # single/double char SMILES that is not water -> small molecule/ion-like
        return "ligand"

    return "ligand"


def _is_nucleic_smiles(smiles: str) -> bool:
    """
    Heuristic: does a SMILES string describe a nucleic-acid oligonucleotide?

    A DNA/RNA strand given as SMILES contains a repeating sugar-phosphate
    backbone.  We require (a) at least two phosphate groups (``P(=O)``/``OP``)
    and (b) recognisable nucleobase ring systems, to avoid flagging a single
    phosphorylated small molecule (e.g. a nucleotide cofactor) as a strand.
    """
    s = smiles.upper()
    n_phosphate = s.count("P(=O)") + s.count("OP(") + s.count("P(O)")
    has_base = ("N1C=NC2" in s or "NC=NC" in s or "C1=CN" in s
                or "N=CN" in s or "NC(=O)N" in s)
    return n_phosphate >= 2 and has_base


def parse_condition_factors(data_json: Path) -> dict:
    """
    Parse one ``*_data.json`` and return a factor dict:

        {
          "protein_chain_ids": [...],
          "nucleic_chain_ids": [...],
          "ptm_labels":  ["TPO101", ...],
          "n_na": int, "n_cl": int, "n_ion_other": int,
          "n_water": int, "n_ligand": int,
          "has_dna": bool, "has_real_ligand": bool,
          "description": str,
          "ion_count_legacy": int,   # all non-protein/nucleic entities (old metric)
        }
    """
    try:
        with open(data_json, "r", encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not parse %s: %s", data_json.name, exc)
        return _empty_factors()

    protein_ids: List[str] = []
    nucleic_ids: List[str] = []
    ptm_labels: List[str] = []
    parts: List[str] = []

    n_na = n_cl = n_ion_other = n_water = n_ligand = 0
    has_real_ligand = False

    for entity in d.get("sequences", []):
        for etype, v in entity.items():
            if not isinstance(v, dict):
                continue
            t = etype.lower()
            ids = _entity_ids(v)
            count = len(ids)

            if t == "protein":
                protein_ids.extend(ids)
                seq_len = len(v.get("sequence", ""))
                mods = v.get("modifications", []) or []
                local_labels = []
                for m in mods:
                    label = f"{m.get('ptmType', '?')}{m.get('ptmPosition', '')}"
                    local_labels.append(label)
                ptm_labels.extend(local_labels)
                mod_str = (" +" + ",".join(local_labels)) if local_labels else ""
                parts.append(
                    f"protein(chain{'s' if count > 1 else ''}="
                    f"{','.join(ids)} {seq_len}aa{mod_str})"
                )

            elif t in _NUCLEIC:
                nucleic_ids.extend(ids)
                seq_len = len(v.get("sequence", ""))
                parts.append(
                    f"{t}(chain{'s' if count > 1 else ''}="
                    f"{','.join(ids)} {seq_len}nt)"
                )

            else:
                kind = classify_nonpolymer(v)
                ccds = _ccd_codes(v)
                if kind == "nucleic":
                    # DNA/RNA supplied as a nonpolymer ligand entity: treat as
                    # nucleic acid, not a small-molecule ligand.  Its chains are
                    # added to the nucleic set so has_dna is set correctly and
                    # the ligand factor stays clean.
                    nucleic_ids.extend(ids)
                    smiles = (v.get("smiles") or "")
                    label = "/".join(ccds) if ccds else (
                        smiles[:12] + ("..." if len(smiles) > 12 else ""))
                    parts.append(
                        f"dna_ligand(chain{'s' if count > 1 else ''}="
                        f"{','.join(ids) if ids else label} x{count})"
                    )
                    continue
                if kind == "ion":
                    code = ccds[0] if ccds else "ION"
                    if code == "NA":
                        n_na += count
                    elif code == "CL":
                        n_cl += count
                    else:
                        n_ion_other += count
                    label = "/".join(ccds) if ccds else "ion"
                    parts.append(f"ion({label} x{count})")
                elif kind == "water":
                    n_water += count
                    parts.append(f"water(x{count})")
                else:
                    n_ligand += count
                    has_real_ligand = True
                    smiles = (v.get("smiles") or "")
                    label = "/".join(ccds) if ccds else (smiles[:12] + ("..." if len(smiles) > 12 else ""))
                    parts.append(f"ligand({label} x{count})")

    return {
        "protein_chain_ids": protein_ids,
        "nucleic_chain_ids": nucleic_ids,
        "ptm_labels": ptm_labels,
        "n_na": n_na,
        "n_cl": n_cl,
        "n_ion_other": n_ion_other,
        "n_water": n_water,
        "n_ligand": n_ligand,
        "has_dna": len(nucleic_ids) > 0,
        "has_real_ligand": has_real_ligand,
        "description": "  |  ".join(parts) if parts else "unknown composition",
        "ion_count_legacy": n_na + n_cl + n_ion_other + n_water + n_ligand,
    }


def _empty_factors() -> dict:
    return {
        "protein_chain_ids": [],
        "nucleic_chain_ids": [],
        "ptm_labels": [],
        "n_na": 0,
        "n_cl": 0,
        "n_ion_other": 0,
        "n_water": 0,
        "n_ligand": 0,
        "has_dna": False,
        "has_real_ligand": False,
        "description": "unknown composition",
        "ion_count_legacy": 0,
    }


# ---------------------------------------------------------------------------
# Experiment-level structure (factorial grid)
# ---------------------------------------------------------------------------

def _ion_tier_label(n_ions: int, min_nonzero: int) -> str:
    """Map a salt-ion count to a tier label relative to the minimum observed."""
    if n_ions == 0:
        return "0x"
    ratio = n_ions / min_nonzero if min_nonzero else 1.0
    for target, lab in ((1, "1x"), (10, "10x"), (100, "100x"), (1000, "1000x")):
        if abs(ratio - target) <= target * 0.5:
            return lab
    return f"{n_ions}x"


def build_experiment_structure(conditions: Dict[str, "object"]):
    """
    Build an ExperimentStructure from loaded ConditionModels.

    The ion tier is derived from **salt ions only** (Na + Cl + other), NOT
    from water — fixing the original solvent confound.
    """
    from .model import ExperimentStructure  # local import to avoid cycle

    ion_tier: Dict[str, str] = {}
    ptm_group: Dict[str, str] = {}
    has_dna: Dict[str, bool] = {}
    has_real_ligand: Dict[str, bool] = {}

    salt_counts = {
        n: (c.n_na + c.n_cl) for n, c in conditions.items()
    }
    nonzero = [v for v in salt_counts.values() if v > 0]
    min_nonzero = min(nonzero) if nonzero else 1

    for name, cond in conditions.items():
        ion_tier[name] = _ion_tier_label(salt_counts[name], min_nonzero)

        parts: List[str] = []
        if cond.n_nucleic_residues > 0:
            parts.append("DNA")
        parts.extend(cond.ptm_labels)
        ptm_group[name] = "+".join(parts) if parts else "none"

        has_dna[name] = cond.n_nucleic_residues > 0
        has_real_ligand[name] = cond.has_real_ligand

    panel_conditions = {n for n in conditions if not has_real_ligand[n]}
    # If every condition has a "real ligand" (e.g. water mis-flagged elsewhere),
    # fall back to including all so the grid is never empty.
    if not panel_conditions:
        panel_conditions = set(conditions)

    def _tier_key(t: str) -> float:
        if t == "0x":
            return 0.0
        try:
            return float(t.rstrip("x"))
        except ValueError:
            return 9_999.0

    tier_order = sorted({ion_tier[n] for n in panel_conditions}, key=_tier_key)

    def _group_key(g: str) -> Tuple[int, str]:
        if g == "none":
            return (0, "")
        if g.startswith("DNA") and "+" not in g:
            return (1, g)
        if g.startswith("DNA"):
            return (2, g)
        return (3, g)

    ptm_order = sorted({ptm_group[n] for n in panel_conditions}, key=_group_key)

    return ExperimentStructure(
        ion_tier=ion_tier,
        ptm_group=ptm_group,
        has_dna=has_dna,
        has_real_ligand=has_real_ligand,
        tier_order=tier_order,
        ptm_order=ptm_order,
        panel_conditions=panel_conditions,
    )
