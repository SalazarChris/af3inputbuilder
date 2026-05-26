#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
af3_json_validator.py
=====================
Validates AlphaFold 3 job JSON files for correct field usage across all
entity types (protein, RNA, DNA, ligand).

Key checks:
  - Required fields present for each entity type
  - Unknown / misplaced fields (e.g. pairedMsaPath on RNA/DNA)
  - Swapped paired / unpaired MSA fields (detects & optionally auto-fixes)
  - Mutually exclusive field pairs (inline vs path MSAs, ccdCodes vs smiles)
  - Sequence character validity
  - Template structure
  - Top-level fields (name, dialect, version, modelSeeds)

Usage:
  python af3_json_validator.py <file.json> [file2.json ...]
  python af3_json_validator.py <file.json> --fix          # auto-fix & save
  python af3_json_validator.py <file.json> --fix-out <out.json>
"""

import json
import sys
import os
import re
import argparse
from copy import deepcopy
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add project root to path, then pull everything from af3_builder
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from af3_builder import (
        PROTEIN_ALPHABET, RNA_ALPHABET, DNA_ALPHABET,
        RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN,
        _rule, _banner, _section, _ok, _warn, _err, _info, _tip, _divider,
        _ask, _ask_yn, _choose, _pause, _ask_file,
    )
    from af3_builder.validation.validator import AF3Validator, ValidationError
    _BUILDER_OK = True
except ImportError:
    _BUILDER_OK = False
    # ... fallback ...
    AF3Validator = None

def validate_job(job: dict) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Returns (errors, warnings, fixes_applied, fixed_job_dict).
    Uses the library validator.
    """
    errors:   List[str] = []
    warnings: List[str] = []
    fixes:    List[str] = []
    job_out = deepcopy(job)

    if AF3Validator is None:
        errors.append("Could not import AF3Validator from af3_builder.")
        return errors, warnings, fixes, job_out

    try:
        AF3Validator.validate_job(job, require_files=False) # skip file checks for speed in CLI
    except ValidationError as e:
        errors.extend(e.messages)

    # Heuristic for MSA swaps (also in validator but we can do auto-fix here)
    for idx, ent in enumerate(job.get("sequences", []) or []):
        if not isinstance(ent, dict) or len(ent) != 1: continue
        kind = next(iter(ent.keys()))
        data = ent[kind]
        if not isinstance(data, dict): continue
        
        upath = data.get("unpairedMsaPath", "")
        ppath = data.get("pairedMsaPath", "")
        if isinstance(upath, str) and isinstance(ppath, str) and upath and ppath:
            u_low, p_low = upath.lower(), ppath.lower()
            if "paired" in u_low and "unpaired" not in u_low and "unpaired" in p_low:
                fixes.append(f"sequences[{idx}]['{kind}']: swapping 'unpairedMsaPath' ↔ 'pairedMsaPath' based on filenames.")
                job_out["sequences"][idx][kind]["unpairedMsaPath"], job_out["sequences"][idx][kind]["pairedMsaPath"] = ppath, upath

        # Legacy keys auto-fix
        LEGACY = {"proteinChain": "protein", "rnaSequence": "rna", "dnaSequence": "dna"}
        if kind in LEGACY:
            fixes.append(f"sequences[{idx}]: renaming legacy key '{kind}' -> '{LEGACY[kind]}'")
            modern = LEGACY[kind]
            job_out["sequences"][idx][modern] = job_out["sequences"][idx].pop(kind)

    return errors, warnings, fixes, job_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation(filepath: str, auto_fix: bool, fix_out: Optional[str]) -> Tuple[bool, int, int, int]:
    """Returns (is_ok, num_errors, num_warnings, num_fixes)."""
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Validating: {CYAN}{filepath}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    # Load JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            job = json.load(f)
    except json.JSONDecodeError as e:
        _err(f"Invalid JSON: {e}")
        return False, 1, 0, 0
    except FileNotFoundError:
        _err(f"File not found: {filepath}")
        return False, 1, 0, 0
    except Exception as e:
        _err(f"Unexpected error loading file: {e}")
        return False, 1, 0, 0

    errors, warnings, fixes, job_fixed = validate_job(job)

    # -- Report --
    print()
    if errors:
        print(f"{BOLD}{RED}  {len(errors)} ERROR(S) DETECTED:{RESET}")
        for e in errors:
            print(f"    {RED}\u2717{RESET} {e}")
    else:
        print(f"    {GREEN}\u2713{RESET} No structural errors found.")

    if warnings:
        print()
        print(f"{BOLD}{YELLOW}  {len(warnings)} WARNING(S):{RESET}")
        for w in warnings:
            print(f"    {YELLOW}\u26A0{RESET} {w}")

    if fixes:
        print()
        print(f"{BOLD}{GREEN}  {len(fixes)} AUTO-FIX(ES) AVAILABLE:{RESET}")
        for fx in fixes:
            print(f"    {GREEN}\u21BB{RESET} {fx}")

    # -- Summary --
    print()
    print(f"  {BOLD}Status:{RESET}    " + (f"{RED}FAILED{RESET}" if errors else f"{GREEN}PASSED{RESET}"))
    print(f"  {BOLD}Details:{RESET}   {len(errors)} errors, {len(warnings)} warnings, {len(fixes)} fixes")

    # -- Auto-fix --
    if fixes:
        if auto_fix or fix_out:
            out_path = fix_out or filepath
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(job_fixed, f, indent=2, ensure_ascii=False)
                print()
                _ok(f"Applied {len(fixes)} auto-fix(es). Saved to: {out_path}")
            except Exception as ex:
                _err(f"Could not write fixed file: {ex}")
        else:
            print()
            _tip(f"Run with {BOLD}--fix{RESET} to automatically resolve the {len(fixes)} issues listed above.")

    print()
    return (len(errors) == 0), len(errors), len(warnings), len(fixes)


def main():
    parser = argparse.ArgumentParser(
        description="AlphaFold 3 JSON Validator - High performance schema enforcement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("files", nargs="*", help="One or more .json files to validate.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply auto-fixes and overwrite the input file(s).",
    )
    parser.add_argument(
        "--fix-out",
        metavar="OUT",
        help="Apply auto-fixes and save to this path (single file only).",
    )
    args = parser.parse_args()

    files = args.files
    if not files:
        # Interactive mode
        print()
        _tip("Press Ctrl+C at any time to easily exit.")
        print()
        f = _ask_file("Select a JSON file to validate", required=False)
        if f:
            files = [f]
        else:
            print(f"{YELLOW}No files selected. Exiting.{RESET}")
            return

    if args.fix_out and len(files) > 1:
        print(f"{RED}[ERR] --fix-out can only be used with a single input file.{RESET}")
        sys.exit(1)

    all_results = []
    for filepath in files:
        res = run_validation(
            filepath,
            auto_fix=args.fix,
            fix_out=args.fix_out if len(files) == 1 else None,
        )
        all_results.append((filepath, *res))

    # Final Summary Table
    if len(args.files) > 1:
        _banner("Validation Results Summary")
        print(f"  {BOLD}{'File':<40} | {'Status':<8} | {'E':>2} | {'W':>2} | {'F':>2}{RESET}")
        print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*2}-+-{'-'*2}-+-{'-'*2}")
        for fp, ok, e, w, f in all_results:
            status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            name = (os.path.basename(fp)[:37] + "...") if len(os.path.basename(fp)) > 40 else os.path.basename(fp)
            print(f"  {name:<40} | {status:<17} | {e:2} | {w:2} | {f:2}")
        print()

    all_ok = all(r[1] for r in all_results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
