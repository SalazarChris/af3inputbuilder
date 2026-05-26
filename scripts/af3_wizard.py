#!/usr/bin/env python3
"""
af3_wizard.py  -  Beginner-friendly AlphaFold 3 JSON Builder Wizard
=====================================================================
A guided, interactive terminal tool for building AlphaFold 3 job JSON files
without needing to know any programming.  Uses a mix of wizard-style prompts
and a simple numbered menu so new users feel at home immediately.

Run with:
    python af3_wizard.py
"""

from __future__ import annotations
import os
import sys
import json
import string
import copy
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

# ---------------------------------------------------------------------------
# Bootstrap: add project root to path, then pull everything from af3_builder
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from af3_builder import (
        JobBuilder, ProteinEntity, RNAEntity, DNAEntity, LigandEntity,
        AF3Validator, ValidationError, SeedsHelper, save_json, load_json, autosave_json,
        InlineArrayEncoder,
        PROTEIN_ALPHABET, RNA_ALPHABET, DNA_ALPHABET, reverse_complement as _reverse_complement,
        RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN, BLUE, MAG, TW,
        _rule, _banner, _section, _ok, _warn, _err, _info, _tip, _divider,
        _ask, _ask_yn, _choose, _pause, _ask_file,
        add_protein_wizard, add_rna_wizard, add_dna_wizard, add_ligand_wizard, add_common_ions_wizard,
        edit_protein_wizard, edit_rna_wizard, edit_dna_wizard, edit_ligand_wizard,
        manage_modifications_wizard, manage_templates_wizard,
        quick_delete_entity_wizard, manage_bonded_atom_pairs_wizard,
        manage_user_ccd_wizard, strip_entities_wizard,
        show_job_summary, show_help_text,
        _current_ids, _next_letter
    )
    _BUILDER_OK = True
except ImportError as _ie:
    _BUILDER_OK = False
    _IMPORT_ERROR = str(_ie)
    # Minimal fallback so the error message can be printed
    RED = "\033[91m"
    RESET = "\033[0m"

BUS_STOPS = 5  # total stops in the Quick-Start bus-line wizard

# ---------------------------------------------------------------------------
# Sequence validators
# (wizard-specific: return an error *string* rather than raising an exception)
# ---------------------------------------------------------------------------
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
RNA_ALPHABET     = set("ACGUNX")
DNA_ALPHABET     = set("ACGTNX")

def _validate_seq(seq: str, alphabet: set, kind: str) -> str:
    """Return an error string, or an empty string if the sequence is valid."""
    if not seq:
        return f"{kind} sequence cannot be empty."
    bad = set(seq.upper()) - alphabet
    if bad:
        return (
            f"{kind} sequence contains invalid characters: {', '.join(sorted(bad))}\n"
            f"     Allowed characters: {''.join(sorted(alphabet))}"
        )
    return ""


# ===========================================================================
# WIZARD SECTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1.  WELCOME  & STARTUP
# ---------------------------------------------------------------------------

def _welcome():
    os.system("cls" if sys.platform == "win32" else "clear")
    _banner(
        "AlphaFold 3  -  Job Builder Wizard",
        "Guided step-by-step JSON creator  |  No coding required"
    )
    print()
    _tip("This tool helps you build a JSON file for AlphaFold 3 prediction jobs.\n"
         "     You will be guided through each step.  Type a number and press Enter\n"
         "     to choose options.  Press Ctrl+C at any time to safely exit.")
    print()

    if not _BUILDER_OK:
        _err(f"Could not import af3_builder: {_IMPORT_ERROR}")
        _err("Make sure you run this script from the thesis folder (next to af3_builder/).")
        sys.exit(1)


# ---------------------------------------------------------------------------
# 2.  JOB SETTINGS WIZARD
# ---------------------------------------------------------------------------

def wizard_job_settings(jb: "JobBuilder"):
    while True:
        _section("Job Settings")
        print(f"  {BOLD}Job Name{RESET}:    {jb.name or '(none)'}")
        print(f"  {BOLD}Schema Ver{RESET}:  {jb.version}")
        print(f"  {BOLD}Model Seeds{RESET}: {jb.modelSeeds or '(none)'}")
        print()

        choice = _choose("Which setting would you like to change?", [
            ("name",    "Edit Job Name"),
            ("version", "Edit Schema Version"),
            ("seeds",   "Edit Model Seeds"),
        ], allow_back=True, back_label="Done / Back")

        if choice == "BACK":
            break

        if choice == "name":
            _tip("The job name helps identify this prediction later.")
            current_name = jb.name or ""
            name = _ask("Job name", default=current_name or f"af3_job_{os.urandom(2).hex()}")
            if not name:
                name = f"af3_job_{os.urandom(2).hex()}"
                _info(f"No name provided. Auto-generated: {name}")
            jb.set_name(name)
            _ok(f"Job name set to: {name}")

        elif choice == "version":
            _tip("Schema version controls which AF3 features are available.")
            ver_choice = _choose("AlphaFold 3 schema version", [
                ("1", "Version 1  -  basic protein/DNA/RNA/ligand  (default)"),
                ("2", "Version 2  -  + MSA files and structure templates"),
                ("3", "Version 3  -  + custom CCD definitions and chain descriptions"),
            ], allow_back=True, back_label="Cancel")
            if ver_choice != "BACK":
                try:
                    jb.set_version(int(ver_choice))
                    _ok(f"Schema version set to {ver_choice}.")
                except Exception as e:
                    _err(f"Could not set version: {e}")

        elif choice == "seeds":
            _tip("Seeds control the randomness of the prediction.")
            seed_choice = _choose("How do you want to set model seeds?", [
                ("auto",   "Generate seeds automatically (recommended)"),
                ("manual", "Enter seeds manually"),
            ], allow_back=True, back_label="Cancel")
            
            if seed_choice == "auto":
                while True:
                    raw = _ask("How many seeds would you like? (1-10)", default="1")
                    try:
                        n = int(raw)
                        if 1 <= n <= 10:
                            seeds = SeedsHelper.generate_default_seeds(n)
                            jb.set_model_seeds(seeds)
                            _ok(f"Using {n} seed(s): {seeds}")
                            break
                        else:
                            _err("Please enter a number between 1 and 10.")
                    except ValueError:
                        _err("Please enter a whole number.")
            
            elif seed_choice == "manual":
                _tip("Enter space-separated integers, e.g.: 42 100 7")
                while True:
                    raw = _ask("Seeds (space-separated numbers)")
                    if not raw:
                        break
                    try:
                        seeds = [int(x) for x in raw.split()]
                        SeedsHelper.validate_seeds(seeds)
                        jb.set_model_seeds(seeds)
                        _ok(f"Seeds set to: {seeds}")
                        break
                    except ValueError as e:
                        _err(str(e))
        _pause()


# ---------------------------------------------------------------------------
# 8c.  CUSTOM CCD (userCCD / userCCDPath)
# ---------------------------------------------------------------------------

def _manage_user_ccd_wizard(jb: "JobBuilder"):
    manage_user_ccd_wizard(jb)


# ---------------------------------------------------------------------------
# 8a.  ADD MOLECULE SUBMENU
# ---------------------------------------------------------------------------

def _add_molecule_menu(jb: "JobBuilder"):
    """Submenu: pick a molecule type to add."""
    while True:
        _section("Add Protein / DNA / RNA / Ligand")
        seqs = jb.sequences
        n_prot = sum(1 for s in seqs if "protein" in s)
        n_rna  = sum(1 for s in seqs if "rna"     in s)
        n_dna  = sum(1 for s in seqs if "dna"     in s)
        n_lig  = sum(1 for s in seqs if "ligand"  in s)
        print(f"  {DIM}Currently: {n_prot} protein(s), {n_rna} RNA, {n_dna} DNA, {n_lig} ligand(s){RESET}")
        print()

        choice = _choose("What would you like to add?", [
            ("protein", f"Protein chain          (amino-acid sequence)   [{n_prot} added]"),
            ("dna",     f"DNA strand             (A, C, G, T)            [{n_dna} added]"),
            ("rna",     f"RNA chain              (A, C, G, U)            [{n_rna} added]"),
            ("ligand",  f"Ligand / small molecule (CCD or SMILES)         [{n_lig} added]"),
            ("ions",    "Common ions / water      (shortcut)"),
            ("sep1",    "-" * 40),
            ("quickdel","Quick-delete an entity   (remove something)"),
        ], allow_back=True, back_label="Back to main menu")

        if choice == "BACK":
            return
        elif choice == "quickdel":
            quick_delete_entity_wizard(jb)
        elif choice == "protein":
            add_protein_wizard(jb)
        elif choice == "rna":
            add_rna_wizard(jb)
        elif choice == "dna":
            add_dna_wizard(jb)
        elif choice == "ligand":
            add_ligand_wizard(jb)
        elif choice == "ligand":
            add_ligand_wizard(jb)
        elif choice == "ions":
            add_common_ions_wizard(jb)


# ---------------------------------------------------------------------------
# 8b.  EDIT MOLECULE SUBMENU
# ---------------------------------------------------------------------------

def _edit_molecule_menu(jb: "JobBuilder"):
    """Submenu: pick an existing entity and edit its fields."""
    _section("Edit Protein / DNA / RNA / Ligand")

    seqs = jb.sequences
    if not seqs:
        _warn("No entities added yet.  Use 'Add' first.")
        _pause()
        return

    # Build a labelled menu of existing entities
    print()
    print(f"  {BOLD}Select an entity to edit (or delete):{RESET}")
    _divider()
    entries = []
    for i, s in enumerate(seqs):
        kind = list(s.keys())[0]
        data = s[kind]
        eid  = data.get("id", "?")
        seq  = data.get("sequence", "")
        extra = f"  ({len(seq)} residues/bases)" if seq else ""
        ccd = data.get("ccdCodes")
        smi = data.get("smiles")
        if ccd:  extra = f"  CCD={ccd[0]}"
        elif smi: extra = "  SMILES"
        label = f"[{BOLD}{kind.upper():<7}{RESET}]  ID={BOLD}{eid}{RESET}{extra}"
        entries.append((str(i), label))

    entries.append(("quickdel", f"{RED}[DEL] Quick-delete an entity{RESET}"))
    _divider()
    print()

    choice = _choose("Pick an action", entries, allow_back=True,
                     back_label="Back to main menu")
    if choice == "BACK":
        return

    if choice == "quickdel":
        quick_delete_entity_wizard(jb)
        return

    idx = int(choice)
    kind = list(seqs[idx].keys())[0]

    if kind == "protein":
        edit_protein_wizard(jb, idx)
    elif kind == "rna":
        edit_rna_wizard(jb, idx)
    elif kind == "dna":
        edit_dna_wizard(jb, idx)
    elif kind == "ligand":
        edit_ligand_wizard(jb, idx)
    _pause()


# ---------------------------------------------------------------------------
# 9.  SEQUENCE MANAGER  (list / remove entities already added)
# ---------------------------------------------------------------------------

def _manage_sequences(jb: "JobBuilder"):
    _section("Manage Sequences / Entities")

    seqs = jb.sequences
    if not seqs:
        _warn("No sequences added yet.")
        _pause()
        return

    print()
    print(f"  {BOLD}Currently added entities:{RESET}")
    _divider()
    for i, s in enumerate(seqs, start=1):
        kind = list(s.keys())[0]
        data = s[kind]
        eid  = data.get("id", "?")
        seq  = data.get("sequence", "")
        extra = ""
        if seq:
            extra = f"  ({len(seq)} residues/bases)"
        ccd = data.get("ccdCodes")
        smi = data.get("smiles")
        if ccd:
            extra = f"  CCD={ccd[0]}"
        elif smi:
            extra = f"  SMILES"
        print(f"  {CYAN}{i:>2}{RESET}. [{BOLD}{kind.upper():<7}{RESET}]  ID={BOLD}{eid}{RESET}{extra}")
    _divider()
    print()

    action = _choose("What do you want to do?", [
        ("remove", "Remove an entity"),
        ("view",   "View entity details"),
    ], allow_back=True, back_label="Back to main menu")

    if action == "BACK":
        return

    if action in ("remove", "view"):
        while True:
            raw = _ask("Enter the number of the entity (from the list above)")
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(seqs):
                    break
                _err(f"Please enter a number between 1 and {len(seqs)}.")
            except ValueError:
                _err("Please enter a number.")

        ent = seqs[idx]
        kind = list(ent.keys())[0]
        data = ent[kind]

        if action == "view":
            print()
            print(BOLD + f"  === {kind.upper()} details ===" + RESET)
            print(json.dumps(data, indent=4))
            _pause()

        elif action == "remove":
            eid = data.get("id", "?")
            if _ask_yn(f"Remove {kind} '{eid}'? This cannot be undone.", default=False):
                seqs.pop(idx)
                _ok(f"{kind} '{eid}' removed.")
            else:
                _info("Removal cancelled.")
            _pause()


# ---------------------------------------------------------------------------
# 9b.  STRIP ENTITY TYPES
# ---------------------------------------------------------------------------

def _strip_entities_wizard(jb: "JobBuilder"):
    """
    Strip one or more entity types from the current job OR from a loaded JSON
    file.  After stripping the result can be:
      - loaded back into the current session (so you can keep editing it), and/or
      - saved to a new JSON file.
    Returns the new JobBuilder if the user chose to load into memory, else None.
    """
    import copy

    _section("Strip Entity Types")
    _tip("Remove entire entity types (protein / rna / dna / ligand) from a job.\n"
         "     After stripping you can:\n"
         "       \u2022 Load the result back into the wizard to keep editing it\n"
         "       \u2022 Save it to a new JSON file (original is never touched)\n"
         "       \u2022 Both")
    print()

    ENTITY_TYPES = ["protein", "rna", "dna", "ligand"]

    # \u2500\u2500 Source \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    src = _choose("Work on which job?", [
        ("mem",  "Current job (in memory)"),
        ("file", "Load a JSON file from disk"),
    ], allow_back=True, back_label="Back to main menu")
    if src == "BACK":
        return None

    if src == "mem":
        data = jb.to_dict()
        source_label = "current job (in memory)"
    else:
        path = _ask_file("Path to the JSON file", required=True)
        if not path:
            return None
        data = load_json(path)
        if data is None:
            _err("Could not read that file.  Check the path and try again.")
            _pause()
            return None
        source_label = path

    # \u2500\u2500 What\u2019s present? \u2500
    seqs   = data.get("sequences", [])
    counts = {et: sum(1 for s in seqs if et in s) for et in ENTITY_TYPES}
    present = [et for et in ENTITY_TYPES if counts[et] > 0]

    if not present:
        _warn("No sequences found in the source \u2013 nothing to strip.")
        _pause()
        return None

    print()
    _info(f"Source: {source_label}")
    print(f"  {BOLD}Entity types present:{RESET}")
    _divider()
    for et in ENTITY_TYPES:
        if counts[et] > 0:
            n_ptm  = sum(1 for s in seqs
                         if et in s and s[et].get("modifications"))
            n_tmpl = sum(1 for s in seqs
                         if et in s and s[et].get("templates"))
            extras = []
            if n_ptm:  extras.append(f"{n_ptm} with mods")
            if n_tmpl: extras.append(f"{n_tmpl} with templates")
            extra_str = f"  {DIM}({', '.join(extras)}){RESET}" if extras else ""
            print(f"  {CYAN}\u2022{RESET}  {BOLD}{et:<8}{RESET}  {counts[et]} chain(s){extra_str}")
    _divider()
    print()

    # \u2500\u2500 Toggle selection \u2500
    _tip("Toggle which types to REMOVE.  Types marked [\u2713] will be stripped.\n"
         "     Select 0 when you're happy with the selection.")
    to_remove = set()

    while True:
        print()
        print(f"  {BOLD}Types to remove (toggle on/off):{RESET}")
        toggle_opts = []
        for et in present:
            check = f"{GREEN}[\u2713]{RESET}" if et in to_remove else f"{DIM}[ ]{RESET}"
            toggle_opts.append((et, f"{check}  {et}  ({counts[et]} chain(s))"))

        action = _choose("Toggle a type (select 0 when done)",
                         toggle_opts,
                         allow_back=True,
                         back_label="Done \u2013 proceed")
        if action == "BACK":
            break
        if action in to_remove:
            to_remove.discard(action)
        else:
            to_remove.add(action)

    if not to_remove:
        _warn("Nothing selected \u2013 no changes made.")
        _pause()
        return None

    # \u2500\u2500 Preview & confirm \u2500
    removed_counts = {et: counts[et] for et in to_remove}
    kept_count = len(seqs) - sum(removed_counts.values())
    print()
    print(f"  {BOLD}Summary of changes:{RESET}")
    for et, n in sorted(removed_counts.items()):
        print(f"  {RED}\u2716{RESET}  Remove {BOLD}{et}{RESET}: {n} chain(s)")
    print(f"  {GREEN}\u2714{RESET}  Remaining sequences: {kept_count}")
    print()

    if not _ask_yn("Proceed with stripping?", default=False):
        _info("Cancelled.")
        _pause()
        return None

    # \u2500\u2500 Strip \u2500
    stripped = copy.deepcopy(data)
    stripped["sequences"] = [
        s for s in stripped.get("sequences", [])
        if not any(et in s for et in to_remove)
    ]
    removed_total = sum(removed_counts.values())
    _ok(f"Stripped {removed_total} entity/entities. "
        f"{len(stripped['sequences'])} sequence(s) remaining.")
    print()

    # \u2500\u2500 What to do with the result? \u2500
    dest = _choose("What would you like to do with the stripped job?", [
        ("mem",      "Load into wizard memory  (keep editing \u2013 add more chains, save later)"),
        ("file",     "Save to a new JSON file  (don't change current session)"),
        ("both",     "Both  \u2013 load into memory AND save to file"),
    ], allow_back=False, back_label="Cancel")

    if dest == "BACK":
        return None

    new_jb = None

    # \u2500\u2500 Load into memory \u2500
    if dest in ("mem", "both"):
        try:
            new_jb = JobBuilder.from_dict(stripped)
            _ok(f"Stripped job loaded into memory.  "
                f"{len(new_jb.sequences)} sequence(s) ready to edit.")
            _info("You can now add more Protein / DNA / RNA / Ligand chains from the main menu.")
        except Exception as e:
            _err(f"Could not load stripped job: {e}")
            new_jb = None

    # \u2500\u2500 Save to file \u2500
    if dest in ("file", "both"):
        default_name = (data.get("name") or "job").replace(" ", "_")
        tag = "_stripped_" + "_".join(sorted(to_remove))
        default_out  = default_name + tag + ".json"
        fname = _ask("Output filename (leave blank to skip)", default=default_out)
        if not fname:
            _warn("Not saved to disk.")
        else:
            if not fname.lower().endswith(".json"):
                fname += ".json"
            save_json(fname, stripped)
            _ok(f"Saved to: {os.path.abspath(fname)}")

    _pause()
    return new_jb   # None if user chose 'file' only



# ---------------------------------------------------------------------------
# 9.  VIEW / VALIDATE / SAVE
# ---------------------------------------------------------------------------

def _show_json(jb: "JobBuilder"):
    _section("Current JSON Preview")
    print()
    out = json.dumps(jb.to_dict(), cls=InlineArrayEncoder, indent=2)
    # Colorise keys slightly for readability
    for line in out.splitlines():
        if ":" in line and line.strip().startswith('"'):
            key, _, val = line.partition(":")
            print(CYAN + key + RESET + ":" + val)
        else:
            print(line)
    print()
    _pause()


def _validate_job(jb: "JobBuilder", strict: bool = False) -> bool:
    _section("Validation")
    try:
        AF3Validator.validate_job(jb.to_dict(), require_files=strict)
        _ok("All checks passed!  Your JSON looks good.")
        return True
    except ValidationError as ve:
        _err("Validation found problems:")
        for msg in ve.messages:
            print(f"     {RED}-{RESET}  {msg}")
        return False
    finally:
        _pause()


def _save_json_wizard(jb: "JobBuilder"):
    _section("Save JSON File")

    # Quick validation first (non-strict - don't require file paths to exist)
    try:
        AF3Validator.validate_job(jb.to_dict(), require_files=False)
        _ok("Pre-save validation passed.")
    except ValidationError as ve:
        _warn("Validation failed. Please fix the following warnings before saving:")
        for msg in ve.messages:
            print(f"     {YELLOW}-{RESET}  {msg}")
        _pause()
        return

    print()
    default_name = jb.name if jb.name else "af3_job"
    default_name = default_name.replace(" ", "_") + ".json"
    while True:
        fname = _ask("Output filename", default=default_name)
        if not fname:
            _warn("Filename is required.")
            continue
        if not fname.lower().endswith(".json"):
            fname += ".json"
        
        # Sync job name with filename (without extension)
        new_job_name = os.path.splitext(os.path.basename(fname))[0]
        jb.set_name(new_job_name)
        
        save_json(fname, jb.to_dict())
        _ok(f"Saved to: {os.path.abspath(fname)}")
        
        # Delete autosaves on successful save
        import glob
        for af in glob.glob("autosave_*.json"):
            try:
                os.remove(af)
            except OSError:
                pass
        
        break

    _pause()


# ---------------------------------------------------------------------------
# 10.  LOAD / RESUME
# ---------------------------------------------------------------------------

def _load_job_wizard(jb: "JobBuilder") -> "JobBuilder":
    _section("Load Existing JSON")
    _tip("You can load a previously created AlphaFold 3 JSON file to continue\n"
         "     editing it.  This will REPLACE the current job in memory.")
    print()

    path = _ask_file("Path to JSON file", required=True)
    if not path:
        return jb

    data = load_json(path)
    if data is None:
        _err("Could not read that file.  Check the path and try again.")
        _pause()
        return jb

    try:
        new_jb = JobBuilder.from_dict(data)
        _ok(f"Loaded '{new_jb.name}' from {path}")
        _info(f"Sequences: {len(new_jb.sequences)}   Seeds: {new_jb.modelSeeds}")
        _pause()
        return new_jb
    except Exception as e:
        _err(f"Failed to load job: {e}")
        _pause()
        return jb


def _resume_autosave(jb: "JobBuilder") -> "JobBuilder":
    _section("Resume from Autosave")

    autosaves = sorted(
        [f for f in os.listdir(".") if f.startswith("autosave_") and f.endswith(".json")],
        reverse=True
    )
    if not autosaves:
        _warn("No autosave files found in the current folder.")
        _pause()
        return jb

    print(f"\n  {BOLD}Available autosaves (newest first):{RESET}")
    for i, fn in enumerate(autosaves[:10], start=1):
        size = os.path.getsize(fn)
        print(f"  {CYAN}{i:>2}{RESET})  {fn}  {DIM}({size} bytes){RESET}")
    print(f"  {DIM}   0)  Cancel{RESET}")
    print()

    while True:
        raw = _ask("Select autosave")
        if raw == "0":
            return jb
        try:
            idx = int(raw) - 1
            if 0 <= idx < min(len(autosaves), 10):
                break
            _err("Invalid selection.")
        except ValueError:
            _err("Please enter a number.")

    data = load_json(autosaves[idx])
    if data is None:
        _err("Could not read autosave.")
        _pause()
        return jb

    try:
        new_jb = JobBuilder.from_dict(data)
        _ok(f"Resumed from {autosaves[idx]}")
        _pause()
        return new_jb
    except Exception as e:
        _err(f"Failed to load autosave: {e}")
        _pause()
        return jb


# ---------------------------------------------------------------------------
# 11.  QUICK-START WIZARD  (bus-line edition)
#      -------------------------------------
#      A strictly linear 5-stop journey that replaces the old branching wizard.
#      Stop 1 -> name  |  Stop 2 -> seeds  |  Stop 3 -> add molecules
#      Stop 4 -> review  |  Stop 5 -> save
# ---------------------------------------------------------------------------

# -- Bus-line helpers ------------------------------------------------------

def _bus_banner(stop: int, title: str):
    """Clear screen and show a progress bar for the bus-line wizard."""
    os.system("cls" if sys.platform == "win32" else "clear")
    _rule("=", CYAN)
    print(BOLD + CYAN + "AlphaFold 3  -  Quick-Start Wizard".center(TW) + RESET)
    # Progress bar
    filled = "#" * stop
    empty  = "." * (BUS_STOPS - stop)
    bar    = f"{GREEN}{filled}{DIM}{empty}{RESET}  Stop {stop}/{BUS_STOPS}  -  {BOLD}{title}{RESET}"
    print("  " + bar)
    _rule("=", CYAN)
    print()


MOLECULE_MENU = [
    ("protein", "Protein chain  (amino-acid sequence)"),
    ("rna",     "RNA chain      (A, C, G, U)"),
    ("dna",     "DNA strand     (A, C, G, T)"),
    ("ligand",  "Ligand / small molecule  (CCD or SMILES)"),
]

_ADD_FNS = {
    "protein": add_protein_wizard,
    "rna":     add_rna_wizard,
    "dna":     add_dna_wizard,
    "ligand":  add_ligand_wizard,
}


def _molecule_status_line(jb) -> str:
    parts = []
    for kind in ("protein", "rna", "dna", "ligand"):
        n = sum(1 for s in jb.sequences if kind in s)
        if n:
            parts.append(f"{n} {kind}(s)")
    return "Added so far: " + (", ".join(parts) if parts else "(nothing yet)")


# -- The 5 stops -----------------------------------------------------------

def _qs_stop1_name(jb):
    _bus_banner(1, "Job Name")
    _tip("Give this prediction a short name - just a label for your own reference.\n"
         "     Example: my_protein  or  kinase_atp_complex")
    print()
    name = _ask("Job name", default="my_job")
    jb.set_name(name)
    _ok(f"Name set to: {name}")
    _pause()


def _qs_stop2_seeds(jb):
    _bus_banner(2, "Model Seeds")
    _tip("Seeds control the randomness of the AlphaFold 3 prediction.\n"
         "     Press Enter to auto-generate 1 seed (recommended for first runs).\n"
         "     More seeds = more prediction attempts.")
    print()
    raw = _ask("How many seeds would you like? (1-10)", default="1")
    try:
        n = int(raw)
        if n < 1 or n > 10:
            raise ValueError
        # Ask if user wants custom seed values
        custom = _ask("Enter custom seed values? (leave blank to auto-generate)", default="")
        if custom.strip():
            # Parse custom seeds from input
            try:
                seeds = [int(x) for x in custom.strip().replace(",", " ").split()]
                if len(seeds) != n:
                    _warn(f"You entered {len(seeds)} seed(s) but requested {n}. Using what you entered.")
                SeedsHelper.validate_seeds(seeds)
                jb.set_model_seeds(seeds)
                _ok(f"Using custom seed(s): {seeds}")
            except ValueError as e:
                _warn(f"Invalid seeds: {e}. Auto-generating instead.")
                seeds = SeedsHelper.generate_default_seeds(n)
                jb.set_model_seeds(seeds)
                _ok(f"Using {n} auto-generated seed(s): {seeds}")
        else:
            seeds = SeedsHelper.generate_default_seeds(n)
            jb.set_model_seeds(seeds)
            _ok(f"Using {n} seed(s): {seeds}")
    except ValueError:
        _warn("Invalid number. Defaulting to 1 seed.")
        jb.set_model_seeds([1])
    _pause()


def _qs_stop3_molecules(jb):
    """Loop: pick molecule type -> fill in details -> add more?"""
    while True:
        _bus_banner(3, "Add Molecules")
        print(f"  {GREEN}{_molecule_status_line(jb)}{RESET}\n")

        if jb.sequences:
            _tip("Enter 0 when you are done adding molecules and are ready to continue.")
        else:
            _tip("Add at least one molecule to build a valid prediction job.")
        print()

        # Print molecule menu inline (no _choose so we can show 0 = Done)
        for i, (k, label) in enumerate(MOLECULE_MENU, 1):
            print(f"  {BOLD}{CYAN}{i:>2}{RESET})  {label}")
        done_label = "Done - move to review" if jb.sequences else "Cancel Quick-Start"
        print(f"  {BOLD}{DIM}   0{RESET})  {DIM}{done_label}{RESET}")
        print()

        raw = _ask("What would you like to add?")

        if raw == "0":
            if not jb.sequences:
                _warn("You need at least one molecule to continue.")
                _pause()
            else:
                break
            continue

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(MOLECULE_MENU):
                kind = MOLECULE_MENU[idx][0]
                _ADD_FNS[kind](jb)
            else:
                _err("Please enter a number from the list.")
        except ValueError:
            _err("Please enter a number.")


def _qs_stop4_review(jb):
    _bus_banner(4, "Review")
    show_job_summary(jb)
    _pause()


def _qs_stop5_save(jb):
    _bus_banner(5, "Save JSON File")
    _save_json_wizard(jb)


# -- Entry point -----------------------------------------------------------

def _quick_start_wizard(jb: Optional["JobBuilder"] = None):
    """Walk the user through a complete job from scratch in a linear bus-line flow."""
    os.system("cls" if sys.platform == "win32" else "clear")
    _banner(
        "Quick-Start Wizard  [bus]",
        "5 stops - Name -> Seeds -> Molecules -> Review -> Save"
    )
    _tip("You will be guided through 5 short stops to build your AlphaFold 3 job.\n"
         "     Press Enter to accept defaults shown in [brackets].\n"
         "     Press Ctrl+C at any time to cancel and return to the main menu.")
    print()

    jb = JobBuilder()

    _qs_stop1_name(jb)
    _qs_stop2_seeds(jb)
    _qs_stop3_molecules(jb)
    _qs_stop4_review(jb)
    _qs_stop5_save(jb)

    print()
    _ok("All stops complete!  Your job JSON is ready.")
    _pause()


# Helpers and Summaries moved to af3_builder.interactive


# ===========================================================================
# MAIN MENU LOOP
# ===========================================================================

def run_wizard():
    _welcome()

    jb = JobBuilder()

    while True:
        os.system("cls" if sys.platform == "win32" else "clear")
        # Mandatory checks
        has_name = bool(jb.name)
        has_seeds = bool(jb.modelSeeds)
        has_seqs = bool(jb.sequences)
        is_ready = has_name and has_seeds and has_seqs
        
        status_text = f"{GREEN}READY TO SAVE{RESET}" if is_ready else f"{RED}INCOMPLETE{RESET} (needs: " + ", ".join([p for c, p in [(has_name, "name"), (has_seeds, "seeds"), (has_seqs, "molecules")] if not c]) + ")"

        _banner(
            "AlphaFold 3  -  Job Builder Wizard",
            f"Job: {BOLD}{jb.name or '(unnamed)'}{RESET}  |  "
            f"Entities: {BOLD}{len(jb.sequences)}{RESET}  |  "
            f"Status: {status_text}"
        )

        # Dynamic status bar
        seqs = jb.sequences
        n_prot  = sum(1 for s in seqs if "protein" in s)
        n_rna   = sum(1 for s in seqs if "rna"     in s)
        n_dna   = sum(1 for s in seqs if "dna"     in s)
        n_lig   = sum(1 for s in seqs if "ligand"  in s)
        n_bonds = len(jb.bondedAtomPairs)
        has_ccd = bool(jb.userCCD or jb.userCCDPath)
        status_parts = []
        if n_prot:  status_parts.append(f"{n_prot} protein(s)")
        if n_rna:   status_parts.append(f"{n_rna} RNA")
        if n_dna:   status_parts.append(f"{n_dna} DNA")
        if n_lig:   status_parts.append(f"{n_lig} ligand(s)")
        if n_bonds: status_parts.append(f"{n_bonds} bond(s)")
        if has_ccd: status_parts.append("custom CCD")
        if status_parts:
            print(f"  {GREEN}Added so far:{RESET} " + ", ".join(status_parts))
        else:
            print(f"  {DIM}Nothing added yet - use the menu below to get started.{RESET}")
        print()

        MENU = [
            ("settings",   "[*] Job Settings  (name / seeds / model version)"),
            ("sep2",       "-" * 40),
            ("add",        "[+] Add Protein / DNA / RNA / Ligand"),
            ("edit",       "[~] Edit Protein / DNA / RNA / Ligand"),
            ("sep3",       "-" * 40),
            ("userccd",    f"[C] Custom CCD components  "
                           + (f"[{GREEN}set{RESET}]" if has_ccd else f"[{DIM}none{RESET}]")),
            ("bonds",      f"[B] Bonded atom pairs       "
                           + (f"[{GREEN}{n_bonds} pair(s){RESET}]" if n_bonds else f"[{DIM}none{RESET}]")),
            ("sep4",       "-" * 40),
            ("manage",     "[=] View / remove added sequences"),
            ("reset",      "[!] Reset entire job"),
            ("strip",      "[x] Strip entity types  (export filtered copy)"),
            ("review",     "[?] Review Job (summary / raw JSON)"),
            ("save",       "[S] Save JSON file"),
            ("sep5",       "-" * 40),
            ("load",       "[L] Open / Resume  (Load file / Autosave)"),
            ("exit",       "[Q] Exit"),
        ]

        # Print menu with separators
        print()
        num = 0
        key_map = {}
        for item in MENU:
            key, label = item
            if key.startswith("sep"):
                print(f"  {DIM}{label}{RESET}")
            else:
                num += 1
                key_map[str(num)] = key
                print(f"  {BOLD}{CYAN}{num:>2}{RESET})  {label}")
        print()
        print(f"  {DIM}(Tip: Press Ctrl+C at any time to easily exit){RESET}")
        print()

        try:
            raw = _ask("Choose an option")
        except KeyboardInterrupt:
            print()
            _warn("Keyboard interrupt detected.")
            _autosave_on_exit(jb)
            break

        choice = key_map.get(raw, "")
        
        # Allow 0 as exit shortcut (consistent with other menus)
        if raw == "0":
            choice = "exit"

        try:
            if choice == "quickstart":
                _quick_start_wizard(jb)

            elif choice == "settings":
                wizard_job_settings(jb)

            elif choice == "add":
                _add_molecule_menu(jb)

            elif choice == "edit":
                _edit_molecule_menu(jb)

            elif choice == "userccd":
                manage_user_ccd_wizard(jb)

            elif choice == "bonds":
                manage_bonded_atom_pairs_wizard(jb)

            elif choice == "manage":
                _manage_sequences(jb)

            elif choice == "reset":
                if _ask_yn("Are you sure you want to RESET the entire job? This will clear everything.", default=False):
                    jb = JobBuilder()
                    _ok("Job reset.")
                    _pause()

            elif choice == "strip":
                result = strip_entities_wizard(jb)
                if result is not None:
                    jb = result
                    _ok("Session updated with stripped job.")

            elif choice == "review":
                show_job_summary(jb)
                _divider()
                _show_json(jb)

            elif choice == "save":
                _save_json_wizard(jb)

            elif choice == "load":
                m = _choose("Open / Resume", [
                    ("file", "Load from JSON file"),
                    ("auto", "Resume from last autosave"),
                ], allow_back=True)
                if m == "file":
                    jb = _load_job_wizard(jb)
                elif m == "auto":
                    jb = _resume_autosave(jb)

            elif choice == "exit":
                if _ask_yn("Exit the wizard?"):
                    _autosave_on_exit(jb)
                    break

            else:
                _err("Unrecognised option.  Please type a number from the list.")

        except KeyboardInterrupt:
            print()
            _warn("Cancelled - returning to the main menu.")
            try:
                autosave_json(jb.to_dict(), prefix="autosave_interrupt")
            except Exception:
                pass

        except Exception as exc:
            _err(f"An unexpected error occurred: {exc}")
            try:
                p = autosave_json(jb.to_dict(), prefix="autosave_error")
                _warn(f"Current job autosaved to: {p}")
            except Exception:
                pass
            _pause()


def _autosave_on_exit(jb: "JobBuilder"):
    if jb.sequences or jb.name:
        if _ask_yn("Save your current progress before exiting?"):
            try:
                p = autosave_json(jb.to_dict(), prefix="autosave_exit")
                _info(f"Job saved to: {p}")
            except Exception:
                pass
    print()
    _ok("Goodbye!")
    print()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_wizard()
