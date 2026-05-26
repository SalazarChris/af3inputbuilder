#!/usr/bin/env python3
"""
add_ions.py — General-purpose AlphaFold 3 ligand/ion concentration sweep generator.

Usage
-----
  python add_ions.py <input.json> [options]

  --ligands  CA,MG,ZN           Comma-separated CCD codes to add (default: CA,MG,ZN,NA,CL)
  --counts   1,2,5,10           Number of copies per concentration step (default: 1,2,5,10)
  --outdir   tests/ions         Output directory (default: same directory as input file)
  --smiles   "CCO"              Optional: use a SMILES string instead of CCD code (single ligand)
  --seed     1327730449         Model seed override (default: keep from input file)
  --separate                    One file per ligand (default: all ligands combined per file)

Examples
--------
  # Sweep CA, MG, ZN combined at copies 1, 2, 5, 10 for a single JSON:
  python add_ions.py tests/oct4_seg_chain_A.json --ligands CA,MG,ZN --counts 1,2,5,10

  # One file per ligand instead of combined:
  python add_ions.py tests/oct4_seg_chain_A.json --ligands CA,MG,ZN --counts 1,2,5 --separate

  # Add a custom SMILES molecule at varying copies:
  python add_ions.py tests/oct4_seg_chain_A.json --smiles "O=C(O)CC(N)C(=O)O" --counts 1,3,5

  # Process all oct4_seg PTM files in tests/, sweep common ions:
  python add_ions.py tests/ --ligands CA,MG,NA,CL --counts 1,5,10
"""

import sys
import os
import argparse
import copy
import json
import glob
import string
from typing import Union, Optional, List, Tuple

# ── Ensure project root on path ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from af3_builder import load_json, save_json

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_LIGANDS = ["CA", "MG", "ZN", "NA", "CL", "HOH"]
DEFAULT_COUNTS  = [1, 2, 5, 10]

# ── Chain ID helpers ──────────────────────────────────────────────────────────

def _all_ids_from_sequences(sequences: list) -> set:
    """Return every chain ID currently used in a sequences list."""
    used = set()
    for ent in sequences:
        # Check modern keys
        for key in ["protein", "dna", "rna", "ligand"]:
            val = ent.get(key)
            if not val: continue
            
            eid = val.get("id")
            if isinstance(eid, list):
                used.update(str(x) for x in eid)
            elif eid is not None:
                used.add(str(eid))
    return used


def _next_ids(used: set, count: int) -> list:
    """
    Return `count` fresh chain IDs not in `used`.
    Cycles A-Z then AA, AB, … as per the AF3 convention.
    """
    pool = list(string.ascii_uppercase)
    pool += [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    pool += [a + b + c
             for a in string.ascii_uppercase
             for b in string.ascii_uppercase
             for c in string.ascii_uppercase]
    result = []
    for candidate in pool:
        if candidate not in used:
            result.append(candidate)
            used.add(candidate)
            if len(result) == count:
                break
    if len(result) < count:
        raise ValueError(f"Ran out of chain IDs (needed {count} more).")
    return result


# ── Core builder ──────────────────────────────────────────────────────────────

def _make_ligand_entry(ids: Union[list, str], ccd: Optional[str], smiles: Optional[str]) -> dict:
    """Build a single ligand sequence entry."""
    if ccd:
        spec = {"ccdCodes": [ccd]}
    elif smiles:
        spec = {"smiles": smiles}
    else:
        raise ValueError("Must supply either a CCD code or a SMILES string.")

    return {"ligand": {"id": ids, **spec}}


def generate_copies(
    input_data: dict,
    ccd: Optional[str],
    smiles: Optional[str],
    count: int,
    base_name_override: Optional[str] = None,
) -> dict:
    """
    Clone *input_data* and inject `count` copies of a SINGLE ligand.

    In AlphaFold 3, "concentration" is modelled by the number of ligand
    copies included in the input JSON.  Each copy gets its own unique
    chain ID.  A list of IDs on a single ligand entry means multiple
    identical copies are requested.

    Returns the new job dict (does not write to disk).
    """
    job = copy.deepcopy(input_data)

    # Ensure dialect is always present
    job.setdefault("dialect", "alphafold3")
    job.setdefault("version", 1)

    used = _all_ids_from_sequences(job.get("sequences", []))
    new_ids = _next_ids(used, count)

    # Use a list of IDs (AF3 multi-copy convention) for count > 1
    id_value = new_ids[0] if count == 1 else new_ids

    ligand_entry = _make_ligand_entry(id_value, ccd, smiles)
    job.setdefault("sequences", []).append(ligand_entry)

    # Build a descriptive name
    label = ccd if ccd else "SMILES"
    base = base_name_override or job.get("name", "job")
    job["name"] = f"{base}_{label}x{count}"

    # userCCD/SMILES require version 3
    if smiles:
        job["version"] = max(job.get("version", 1), 3)

    return job


def generate_combined(
    input_data: dict,
    ligand_specs: List[Tuple[Optional[str], Optional[str]]],
    count: int,
    base_name_override: Optional[str] = None,
) -> dict:
    """
    Clone *input_data* and inject ALL ligands together, each at `count` copies.

    *ligand_specs* is a list of (ccd_or_None, smiles_or_None) tuples.
    Each ligand type gets its own entry with `count` unique chain IDs.

    Returns the new job dict (does not write to disk).
    """
    job = copy.deepcopy(input_data)
    job.setdefault("dialect", "alphafold3")
    job.setdefault("version", 1)
    used = _all_ids_from_sequences(job.get("sequences", []))

    labels = []
    has_smiles = False

    for ccd, smiles in ligand_specs:
        new_ids = _next_ids(used, count)
        id_value = new_ids[0] if count == 1 else new_ids
        ligand_entry = _make_ligand_entry(id_value, ccd, smiles)
        job.setdefault("sequences", []).append(ligand_entry)
        labels.append(ccd if ccd else "SMILES")
        if smiles:
            has_smiles = True

    # Build a descriptive name
    base = base_name_override or job.get("name", "job")
    tag = "_".join(labels)
    job["name"] = f"{base}_{tag}x{count}"

    if has_smiles:
        job["version"] = max(job.get("version", 1), 3)

    return job


def generate_library_sweep(
    input_data: dict,
    library: List[str],
    kind: str = "smiles", # "smiles" or "protein"
    count_per_entry: int = 1,
    base_name_override: Optional[str] = None,
) -> List[dict]:
    """
    Generate a list of jobs, one for each entry in the library list.
    Entries are either SMILES strings or Protein sequences.
    Uses entity classes to ensure proper field handling.
    """
    from af3_builder import ProteinEntity, LigandEntity

    results = []
    for i, entry in enumerate(library):
        entry = entry.strip()
        if not entry: continue
        
        job = copy.deepcopy(input_data)
        job.setdefault("dialect", "alphafold3")
        job.setdefault("version", 1)
        used = _all_ids_from_sequences(job.get("sequences", []))
        new_ids = _next_ids(used, count_per_entry)
        id_value = new_ids[0] if count_per_entry == 1 else new_ids
        
        if kind == "smiles":
            ent = LigandEntity(id=id_value, smiles=entry)
            job.setdefault("sequences", []).append(ent.to_dict())
            label = "lib_smi"
        else:
            ent = ProteinEntity(id=id_value, sequence=entry.upper())
            job.setdefault("sequences", []).append(ent.to_dict())
            label = "lib_pep"
            
        # Build name
        base = base_name_override or job.get("name", "job")
        job["name"] = f"{base}_{label}_{i+1}"
        
        # SMILES requires version 3
        if kind == "smiles":
            job["version"] = max(job.get("version", 1), 3)
            
        results.append(job)
    return results


# ── File-level processing ─────────────────────────────────────────────────────

def process_file(
    json_path: str,
    ligands: list,
    smiles: Optional[str],
    counts: list,
    outdir: str,
    seed: Optional[int],
    separate: bool = False,
):
    data = load_json(json_path)
    if not data:
        print(f"  [SKIP] Could not load: {json_path}")
        return

    if seed is not None:
        data["modelSeeds"] = [seed]
    elif not data.get("modelSeeds"):
        import random
        s = random.randint(1, 9999)
        data["modelSeeds"] = [s]
        print(f"  [INFO] No seed in source; using random: {s}")

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(outdir, exist_ok=True)

    # Build the full list of (ccd_or_None, smiles_or_None) specs
    specs = []
    if smiles:
        specs.append((None, smiles))
    specs.extend([(lig, None) for lig in ligands])

    created = 0

    if separate:
        # ── One file per ligand per count ──
        for ccd, smi in specs:
            for count in counts:
                job = generate_copies(data, ccd, smi, count, base_name_override=base_name)
                label = ccd if ccd else "SMILES"
                out_name = f"{job['name']}.json"
                out_path = os.path.join(outdir, out_name)
                save_json(out_path, job)
                print(f"  [OK] {out_path}  ({label} × {count} copies)")
                created += 1
    else:
        # ── All ligands combined into one file per count ──
        for count in counts:
            job = generate_combined(data, specs, count, base_name_override=base_name)
            out_name = f"{job['name']}.json"
            out_path = os.path.join(outdir, out_name)
            save_json(out_path, job)
            labels_str = ", ".join(c if c else "SMILES" for c, s in specs)
            print(f"  [OK] {out_path}  ({labels_str} × {count} each)")
            created += 1

    print(f"  ↳ {created} file(s) created from {os.path.basename(json_path)}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate AF3 JSON copies with ions/ligands at varying copy counts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Input .json file OR directory of .json files.")
    p.add_argument("--ligands", default=",".join(DEFAULT_LIGANDS),
                   help="Comma-separated CCD codes (default: CA,MG,ZN,NA,CL)")
    p.add_argument("--counts", default=",".join(str(c) for c in DEFAULT_COUNTS),
                   help="Comma-separated copy counts (default: 1,2,5,10)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: same dir as input)")
    p.add_argument("--smiles", default=None,
                   help="SMILES string for a custom molecule (added alongside --ligands)")
    p.add_argument("--seed", default=None, type=int,
                   help="Override model seed in all outputs")
    p.add_argument("--separate", action="store_true",
                   help="One file per ligand instead of combining all ligands per file")
    p.add_argument("--pattern", default="*.json",
                   help="Glob pattern when input is a directory (default: *.json)")
    return p.parse_args()


def main():
    args = parse_args()

    ligands = [x.strip().upper() for x in args.ligands.split(",") if x.strip()]
    counts  = [int(x.strip()) for x in args.counts.split(",") if x.strip()]

    # Resolve input files
    input_path = args.input
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, args.pattern)))
        if not files:
            print(f"No files matching '{args.pattern}' in {input_path}")
            sys.exit(1)
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)

    mode = "separate" if args.separate else "combined"
    print(f"\nAlphaFold 3 — Ion/Ligand Concentration Sweep  ({mode} mode)")
    print(f"  Input files : {len(files)}")
    print(f"  Ligands     : {ligands}")
    if args.smiles:
        print(f"  SMILES      : {args.smiles}")
    print(f"  Copy counts : {counts}")
    print()

    for fpath in files:
        outdir = args.outdir or os.path.dirname(fpath) or "."
        print(f"Processing: {fpath}")
        process_file(
            json_path=fpath,
            ligands=ligands,
            smiles=args.smiles,
            counts=counts,
            outdir=outdir,
            seed=args.seed,
            separate=args.separate,
        )

    print("Done.")


if __name__ == "__main__":
    main()
