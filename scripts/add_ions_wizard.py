#!/usr/bin/env python3
"""
add_ions_wizard.py -- Interactive wizard for the AF3 ion/ligand concentration sweep.

Run:   python add_ions_wizard.py
"""
from __future__ import annotations

import sys
import os
import glob
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from af3_builder import load_json, save_json
from af3_builder.ui.ui import (
    _banner, _section, _ok, _warn, _err, _info, _tip,
    _ask, _ask_yn, _ask_file, _ask_dir, _pick_file, _pick_dir, _choose, _pause, _divider,
    _is_gui_available, BOLD, RESET, DIM, CYAN, GREEN, RED, YELLOW, MAG,
)
from add_ions import (
    generate_copies, generate_combined, DEFAULT_LIGANDS, DEFAULT_COUNTS
)

# -- Common ions quick-reference -----------------------------------------------
from af3_builder.core.reference import COMMON_IONS, COMMON_COFACTORS, COMMON_SMALL_MOLECULES

# Build the picker menu for the wizard (ions + cofactors + custom options)
COMMON_IONS_MENU = (
    [(code, desc) for code, desc in COMMON_IONS[:12]]  # Top 12 ions
    + [("---", "--- Cofactors ---")]
    + [(code, desc) for code, desc in COMMON_COFACTORS[:8]]  # Top 8 cofactors
    + [("custom", "Custom CCD code..."), ("smiles", "Custom SMILES string...")]
)


# -- Wizard helpers -------------------------------------------------------------

def _ask_dir_or_file(prompt: str) -> list[str]:
    """
    Ask for a file OR directory and return a list of .json paths.
    """
    gui = _is_gui_available()
    suffix = " (Type '?' to pick)" if gui else ""
    while True:
        raw = _ask(f"{prompt}{suffix}").strip().strip('"').strip("'")
        if raw == "?":
            if not gui:
                _err("GUI picker unavailable. Please type the path manually.")
                continue
            # Show a menu to pick file vs folder
            choice = _choose("What would you like to pick?", [
                ("file", "Pick a single JSON file"),
                ("dir",  "Pick a directory of JSON files"),
            ], allow_back=True)
            if choice == "BACK": continue
            if choice == "file":
                raw = _pick_file("Select AF3 JSON")
            else:
                raw = _pick_dir("Select Folder containing JSONs")
            
            if not raw: continue
            _ok(f"Selected: {raw}")

        if not raw:
            _err("Path cannot be empty.")
            continue
        if os.path.isfile(raw):
            if not raw.endswith(".json"):
                _warn("Not a .json file -- proceeding anyway.")
            return [raw]
        if os.path.isdir(raw):
            files = sorted(glob.glob(os.path.join(raw, "*.json")))
            if not files:
                _err(f"No .json files found in: {raw}")
                continue
            _info(f"Found {len(files)} JSON file(s) in directory.")
            return files
        _err(f"Not found: {raw}")


def _ask_ligands() -> tuple[list[str], Optional[str]]:
    """
    Interactively build a list of CCD codes to sweep, plus an optional SMILES.
    Returns (ccd_list, smiles_or_None).
    """
    _section("Ligands / Ions to Add")
    _tip("Select ions/ligands to sweep. You can pick multiple.\n"
         "     CCD codes are 2-3 letter identifiers (e.g. CA, MG, ATP).")

    selected_ccds: list[str] = []
    smiles_str: Optional[str] = None

    while True:
        print()
        if selected_ccds or smiles_str:
            print(f"  {BOLD}Selected so far:{RESET}")
            for c in selected_ccds:
                print(f"    {GREEN}[ok]{RESET}  CCD: {BOLD}{c}{RESET}")
            if smiles_str:
                print(f"    {GREEN}[ok]{RESET}  SMILES: {BOLD}{smiles_str[:60]}...{RESET}")
            _divider()

        action = _choose(
            "Add ligand",
            [("pick",   "Pick from common ions list"),
             ("ccd",    "Enter a custom CCD code"),
             ("smiles", "Enter a SMILES string"),
             ("remove", "Remove a selection") if (selected_ccds or smiles_str) else ("remove", f"{DIM}Remove (nothing selected){RESET}"),
             ("done",   f"{GREEN}Done -- proceed with current selection{RESET}"),
            ],
            allow_back=False,
            back_label="Cancel",
        )

        if action == "BACK":
            return [], None

        if action == "pick":
            choice = _choose("Select ion", COMMON_IONS_MENU, allow_back=True, back_label="Cancel")
            if choice == "BACK":
                continue
            if choice == "custom":
                code = _ask("Enter CCD code").strip().upper()
                if code and code not in selected_ccds:
                    selected_ccds.append(code)
                    _ok(f"Added: {code}")
                elif code in selected_ccds:
                    _warn(f"{code} already selected.")
            elif choice == "smiles":
                smi = _ask("SMILES string").strip()
                if smi:
                    smiles_str = smi
                    _ok("SMILES saved.")
            else:
                if choice not in selected_ccds:
                    selected_ccds.append(choice)
                    _ok(f"Added: {choice}")
                else:
                    _warn(f"{choice} already in the list.")

        elif action == "ccd":
            code = _ask("CCD code (e.g. ATP)").strip().upper()
            if code and code not in selected_ccds:
                selected_ccds.append(code)
                _ok(f"Added: {code}")
            elif code in selected_ccds:
                _warn(f"{code} already selected.")

        elif action == "smiles":
            smi = _ask("SMILES string").strip()
            if smi:
                smiles_str = smi
                _ok("SMILES saved.")

        elif action == "remove":
            opts = [(c, c) for c in selected_ccds]
            if smiles_str:
                opts.append(("__smiles__", f"SMILES: {smiles_str[:40]}..."))
            if not opts:
                _warn("Nothing to remove.")
                continue
            rem = _choose("Remove which?", opts, allow_back=True, back_label="Cancel")
            if rem == "BACK":
                continue
            if rem == "__smiles__":
                smiles_str = None
                _ok("SMILES removed.")
            elif rem in selected_ccds:
                selected_ccds.remove(rem)
                _ok(f"Removed: {rem}")

        elif action == "done":
            if not selected_ccds and not smiles_str:
                _err("Please select at least one ligand/ion.")
                continue
            break

    return selected_ccds, smiles_str


def _ask_concentrations(ccds: list[str], smiles_str: Optional[str]) -> list[list[int]]:
    """
    Ask the user for a ratio and a geometric multiplier series.

    For N ligands the user enters a ratio  e.g.  1:2:2
    then a base multiplier and how many multiples to generate.

    Returns a list of rows, each row being a list of per-ligand copy
    counts for that sweep step:
        e.g.  [[1, 2, 2], [10, 20, 20], [100, 200, 200]]
    """
    all_labels = list(ccds) + (["SMILES"] if smiles_str else [])
    n = len(all_labels)

    _section("Concentration Sweep -- Ratio & Multiples")

    # -- Single-ligand shortcut ------------------------------------------------
    if n == 1:
        _tip("Only one ligand selected.  Enter copy counts as comma-separated\n"
             "     integers, e.g.:  1,2,5,10  -- each value makes one job file.")
        presets = [
            ("1,2,5,10",  "Low sweep   -- 1, 2, 5, 10 copies"),
            ("1,5,10,20", "Medium sweep -- 1, 5, 10, 20 copies"),
            ("1,10,50",   "High sweep   -- 1, 10, 50 copies"),
            ("1",         "Single copy  -- 1 only"),
            ("custom",    "Enter custom counts..."),
        ]
        choice = _choose("Preset", presets, allow_back=False, back_label="Cancel")
        if choice == "BACK":
            return []
        if choice == "custom":
            while True:
                raw = _ask("Counts (comma-separated integers)", "1,2,5,10")
                try:
                    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    if not vals:
                        raise ValueError
                    break
                except ValueError:
                    _err("Enter integers separated by commas, e.g. 1,2,5,10")
        else:
            vals = [int(x.strip()) for x in choice.split(",")]
        rows = [[v] for v in vals]
        _ok(f"Sweep rows: {rows}")
        return rows

    # -- Multi-ligand: ratio + multiples ---------------------------------------
    _tip(
        f"You have {n} ligand(s): {BOLD}{', '.join(all_labels)}{RESET}\n"
        "\n"
        "     Step A -- Enter their ratio, e.g.  1:2:2\n"
        "     Step B -- Enter the base multiplier and number of multiples\n"
        "\n"
        "     Example:  ratio 1:2:2  -  base 10  -  3 multiples\n"
        f"       Row 1 (x 1)   -> {BOLD}{all_labels[0]}{RESET}x1  {BOLD}{all_labels[1] if n>1 else ''}{RESET}{'x2  ' if n>1 else ''}{BOLD}{all_labels[2] if n>2 else ''}{RESET}{'x2' if n>2 else ''}\n"
        f"       Row 2 (x10)   -> multiplied by 10\n"
        f"       Row 3 (x100)  -> multiplied by 100"
    )

    # -- Step A: ratio ---------------------------------------------------------
    while True:
        raw_ratio = _ask(
            f"Ratio for [{', '.join(all_labels)}] (colon-separated, e.g. 1:2:2)"
        ).strip()
        parts = [p.strip() for p in raw_ratio.split(":") if p.strip()]
        try:
            ratio = [int(p) for p in parts]
        except ValueError:
            _err("Ratio must be integers separated by colons, e.g. 1:2:2")
            continue
        if len(ratio) != n:
            _err(f"Ratio must have exactly {n} value(s) -- one per ligand ({', '.join(all_labels)}).")
            continue
        if any(v <= 0 for v in ratio):
            _err("All ratio values must be positive integers.")
            continue
        break

    _ok(f"Ratio: {':'.join(str(r) for r in ratio)}")

    # -- Step B: base multiplier + number of multiples -------------------------
    while True:
        try:
            base = int(_ask("Base multiplier (e.g. 10)", "10").strip())
            if base <= 0:
                raise ValueError
            break
        except ValueError:
            _err("Enter a positive integer.")

    while True:
        try:
            num_multiples = int(_ask("How many multiples to generate (e.g. 3 -> x1, x10, x100)", "3").strip())
            if num_multiples <= 0:
                raise ValueError
            break
        except ValueError:
            _err("Enter a positive integer.")

    # Build rows
    rows = []
    for m in range(num_multiples):
        multiplier = base ** m
        rows.append([r * multiplier for r in ratio])

    _ok(f"Sweep rows ({num_multiples} steps, base={base}):")
    for i, row in enumerate(rows):
        multiplier = base ** i
        pairs = "  ".join(f"{lbl}x{cnt}" for lbl, cnt in zip(all_labels, row))
        print(f"    {DIM}Row {i+1} (x{multiplier:>6}){RESET}  {pairs}")

    return rows


def _ask_outdir(first_input_path: str) -> str:
    """Ask where to save results."""
    _section("Output Directory")
    default = os.path.join(os.path.dirname(first_input_path) or ".", "ion_sweep")
    _info(f"Default: {default}")
    raw = _ask_dir("Output directory", required=False)
    return raw or default


def _ask_seed() -> Optional[int]:
    """Optionally override the model seed."""
    if _ask_yn("Override model seed in all generated files?", default=False):
        raw = _ask("Seed (integer, or leave blank for random)")
        if not raw.strip():
            import random
            s = random.randint(1, 9999)
            _info(f"Using random seed: {s}")
            return s
        try:
            return int(raw)
        except ValueError:
            _err("Invalid integer. Keeping original seed.")
    return None


# -- Main wizard ----------------------------------------------------------------

def run_wizard():
    _banner(
        "AlphaFold 3 -- Ion/Ligand Toolset Wizard",
        "Concentration Sweeps, Library Mode & Standalone Generation",
    )
    print()
    _tip("Press Ctrl+C at any time to easily exit.")
    print()

    mode = _choose("Select Mode", [
        ("sweep",      "Concentration Sweep -- vary copy counts of specific ions"),
        ("library",    "Ligand Library Sweep -- one JSON per entry from a text file"),
        ("standalone", "Standalone JSON Generator -- generate jobs from a sequence list (no base file)"),
    ])

    if mode == "sweep":
        run_sweep_wizard()
    elif mode == "library":
        run_library_wizard()
    else:
        run_standalone_wizard()


def run_sweep_wizard():
    # -- Step 1: Input ----------------------------------------------------------
    _section("Step 1 -- Input Files")
    _tip("Enter a single .json file path OR a directory containing .json files.")
    input_files = _ask_dir_or_file("File or directory path")

    # Preview
    _section("Files to Process")
    for f in input_files[:10]:
        print(f"  {DIM}{f}{RESET}")
    if len(input_files) > 10:
        print(f"  {DIM}... and {len(input_files) - 10} more{RESET}")

    if not _ask_yn(f"Process these {len(input_files)} file(s)?"):
        _warn("Cancelled.")
        return

    # -- Step 2: Ligands --------------------------------------------------------
    ccds, smiles_str = _ask_ligands()

    # -- Step 3: Concentration sweep (ratio + multiples) ------------------------
    sweep_rows = _ask_concentrations(ccds, smiles_str)

    # -- Step 4: Output dir -----------------------------------------------------
    outdir = _ask_outdir(input_files[0])

    # -- Step 5: Seed -----------------------------------------------------------
    seed = _ask_seed()

    # -- Step 6: Summary --------------------------------------------------------
    _section("Summary -- Confirm")
    all_labels = list(ccds) + (["SMILES"] if smiles_str else [])
    print(f"  {BOLD}Input files :{RESET}  {len(input_files)}")
    if ccds:
        print(f"  {BOLD}CCD ligands :{RESET}  {', '.join(ccds)}")
    if smiles_str:
        print(f"  {BOLD}SMILES      :{RESET}  {smiles_str[:60]}{'...' if len(smiles_str) > 60 else ''}")
    print(f"  {BOLD}Sweep rows  :{RESET}")
    for i, row in enumerate(sweep_rows):
        pairs = "  ".join(f"{lbl}x{cnt}" for lbl, cnt in zip(all_labels, row))
        print(f"      Row {i+1}: {pairs}")
    print(f"  {BOLD}Output dir  :{RESET}  {outdir}")
    if seed is not None:
        print(f"  {BOLD}Seed        :{RESET}  {seed}")

    total = len(sweep_rows) * len(input_files)
    print(f"\n  {BOLD}Total files to generate:{RESET}  {total}  ({len(sweep_rows)} row(s) x {len(input_files)} input(s))")

    if not _ask_yn("Proceed?"):
        _warn("Cancelled.")
        return

    # -- Step 7: Generate -------------------------------------------------------
    _section("Generating Files...")
    os.makedirs(outdir, exist_ok=True)
    created = 0
    errors  = 0

    specs = [(c, None) for c in ccds]
    if smiles_str:
        specs.append((None, smiles_str))

    for fpath in input_files:
        data = load_json(fpath)
        if not data:
            _err(f"Could not load: {fpath}")
            errors += 1
            continue
        if seed is not None:
            data["modelSeeds"] = [seed]

        base_name = os.path.splitext(os.path.basename(fpath))[0]
        print(f"\n  {CYAN}{base_name}{RESET}")

        for row_idx, row_counts in enumerate(sweep_rows):
            try:
                import copy as _copy
                job = _copy.deepcopy(data)
                label_parts = []
                for (ccd, smi), cnt in zip(specs, row_counts):
                    if cnt <= 0: continue
                    job = generate_copies(job, ccd, smi, cnt, base_name_override=base_name)
                    label_parts.append(f"{ccd if ccd else 'SMILES'}x{cnt}")
                
                if not label_parts: continue

                row_label = "_".join(label_parts)
                job["name"] = f"{base_name}_{row_label}"
                fname = f"{base_name}_{row_label}.json"
                out = os.path.join(outdir, fname)
                save_json(out, job)
                print(f"    {GREEN}[ok]{RESET}  {fname}")
                created += 1
            except Exception as e:
                _err(f"Row {row_idx+1} failed: {e}")
                errors += 1

    _section("Complete")
    _ok(f"{created} file(s) written to: {outdir}")
    _pause()


def run_library_wizard():
    _section("Step 1 -- Input Files (Base Structure)")
    _tip("Select the JSON file(s) that will serve as the base for your library.")
    input_files = _ask_dir_or_file("File or directory path")
    
    _section("Step 2 -- Library Selection")
    _tip("Select a text file (.txt or .csv) containing your SMILES or protein sequences.\n"
         "     Each line should contain exactly one sequence.")
    lib_path = _ask_file("Library file path")
    
    try:
        with open(lib_path, "r", encoding="utf-8") as f:
            library = [line.strip() for line in f if line.strip()]
    except Exception as e:
        _err(f"Could not read file: {e}")
        return
        
    if not library:
        _err("Library file is empty.")
        return
        
    _info(f"Loaded {len(library)} entries from: {os.path.basename(lib_path)}")
    
    kind = _choose("What type of library is this?", [
        ("smiles",  "Small Molecule SMILES (ligands)"),
        ("protein", "Protein / Peptide Sequences (protein)"),
    ])
    
    count = 1
    if _ask_yn(f"Add multiple copies of each {kind} per file?", default=False):
        while True:
            try:
                count = int(_ask("Copy count (e.g. 5)", "5"))
                if count > 0: break
            except ValueError:
                _err("Enter a positive integer.")

    outdir = _ask_outdir(input_files[0])
    seed = _ask_seed()
    
    _section("Summary")
    print(f"  Base files : {len(input_files)}")
    print(f"  Lib entries: {len(library)}")
    print(f"  Lib type   : {kind}")
    print(f"  Copies     : {count}")
    print(f"  Output dir : {outdir}")
    
    if not _ask_yn("Proceed?"):
        _warn("Cancelled.")
        return
        
    _section("Generating Files...")
    os.makedirs(outdir, exist_ok=True)
    created = 0
    errors = 0
    
    from add_ions import generate_library_sweep
    
    for fpath in input_files:
        data = load_json(fpath)
        if not data:
            errors += 1
            continue
        if seed is not None:
            data["modelSeeds"] = [seed]
            
        base_name = os.path.splitext(os.path.basename(fpath))[0]
        print(f"\n  {CYAN}{base_name}{RESET}")
        
        try:
            jobs = generate_library_sweep(data, library, kind=kind, count_per_entry=count, base_name_override=base_name)
            for job in jobs:
                fname = f"{job['name']}.json"
                out = os.path.join(outdir, fname)
                save_json(out, job)
                print(f"    {GREEN}[ok]{RESET}  {fname}")
                created += 1
        except Exception as e:
            _err(f"Failed to generate library: {e}")
            errors += 1
            
    _section("Complete")
    _ok(f"{created} file(s) written to: {outdir}")
    _pause()


def run_standalone_wizard():
    """Generate standalone AF3 job JSONs from a sequence list -- no base file needed."""
    _section("Step 1 -- Sequence File")
    _tip("Provide a text file with one sequence per line.\n"
         "     Supported types: Protein, DNA, RNA, or SMILES.")

    lib_path = _ask_file("Sequence file path (.txt or .csv)")
    try:
        with open(lib_path, "r", encoding="utf-8") as f:
            entries = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except Exception as e:
        _err(f"Could not read file: {e}")
        return

    if not entries:
        _err("File is empty (or all lines are comments).")
        return

    _info(f"Loaded {len(entries)} entries from: {os.path.basename(lib_path)}")

    # -- Step 2: Sequence type --------------------------------------------------
    _section("Step 2 -- Sequence Type")
    kind = _choose("What type of sequences are these?", [
        ("protein", "Protein / Peptide"),
        ("dna",     "DNA"),
        ("rna",     "RNA"),
        ("smiles",  "Small Molecule (SMILES)"),
    ])

    # -- Step 3: Job name prefix ------------------------------------------------
    _section("Step 3 -- Naming")
    prefix = _ask("Job name prefix", default="af3_job")

    # -- Step 4: Model seed -----------------------------------------------------
    seed = _ask_seed()
    if seed is None:
        import random
        seed = random.randint(1, 9999)
        _info(f"Using random seed: {seed}")

    # -- Step 5: MSA/Template strategy ------------------------------------------
    search_strategy = "calc"
    if kind in ("protein", "dna", "rna"):
        search_strategy = _choose("MSA & Template search strategy", [
            ("ignore", "Ignore (Skip search - sets empty [])"),
            ("calc",   "Calculate (AF3 default search)"),
        ], default="ignore")

    # -- Step 6: Output directory -----------------------------------------------
    default_outdir = os.path.join(os.path.dirname(lib_path) or ".", "standalone_jobs")
    raw_outdir = _ask_dir("Output directory", required=False)
    outdir = raw_outdir or default_outdir

    # -- Step 7: Summary --------------------------------------------------------
    _section("Summary -- Confirm")
    print(f"  {BOLD}Entries     :{RESET}  {len(entries)}")
    print(f"  {BOLD}Type        :{RESET}  {kind}")
    print(f"  {BOLD}Name prefix :{RESET}  {prefix}")
    print(f"  {BOLD}Seed        :{RESET}  {seed}")
    print(f"  {BOLD}Search Strat:{RESET}  {search_strategy}")
    print(f"  {BOLD}Output dir  :{RESET}  {outdir}")
    print(f"  {BOLD}Total files :{RESET}  {len(entries)}")

    if not _ask_yn("Proceed?"):
        _warn("Cancelled.")
        return

    # -- Step 8: Generate -------------------------------------------------------
    _section("Generating Files...")
    os.makedirs(outdir, exist_ok=True)
    created = 0
    errors = 0

    from af3_builder import JobBuilder, ProteinEntity, DNAEntity, RNAEntity, LigandEntity

    for i, entry in enumerate(entries, 1):
        try:
            jb = JobBuilder()
            jb.set_name(f"{prefix}_{i}")
            jb.set_model_seeds([seed])

            if kind == "protein":
                jb.add_protein(ProteinEntity(
                    id="A",
                    sequence=entry.upper(),
                    unpairedMsa="" if search_strategy == "ignore" else None,
                    pairedMsa="" if search_strategy == "ignore" else None,
                    templates=[] if search_strategy == "ignore" else None,
                ))

            elif kind == "dna":
                jb.add_dna(DNAEntity(
                    id="A",
                    sequence=entry.upper(),
                    unpairedMsa="" if search_strategy == "ignore" else None,
                ))

            elif kind == "rna":
                jb.add_rna(RNAEntity(
                    id="A",
                    sequence=entry.upper(),
                    unpairedMsa="" if search_strategy == "ignore" else None,
                ))

            elif kind == "smiles":
                jb.add_ligand(LigandEntity(id="A", smiles=entry))

            fname = f"{prefix}_{i}.json"
            out_path = os.path.join(outdir, fname)
            save_json(out_path, jb.to_dict())
            print(f"    {GREEN}\u2714{RESET}  {fname}")
            created += 1

        except Exception as e:
            _err(f"Entry {i} failed: {e}")
            errors += 1

    _section("Complete")
    if errors:
        _warn(f"{errors} error(s) encountered.")
    _ok(f"{created} file(s) written to: {outdir}")
    _pause()


if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        print()
        _warn("Exited.")
