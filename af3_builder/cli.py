# af3_builder/cli.py
import os
import json
from typing import List, Dict, Any

from .core.job import JobBuilder
from .core.seeds import SeedsHelper
from .core.entities import (
    ProteinEntity, RNAEntity, DNAEntity, LigandEntity,
    PROTEIN_ALPHABET, RNA_ALPHABET, DNA_ALPHABET
)
from .validation.validator import AF3Validator, ValidationError
from .utils.io import save_json, autosave_json, load_json
from .ui.ui import (
    RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN, BLUE, MAG, TW,
    _rule, _banner, _section, _ok, _warn, _err, _info, _tip, _divider,
    _ask, _ask_yn, _choose, _pause, _ask_file as _prompt_file_path_ui,
)
from .ui.interactive import (
    add_protein_wizard, add_rna_wizard, add_dna_wizard, add_ligand_wizard,
    edit_protein_wizard, edit_rna_wizard, edit_dna_wizard, edit_ligand_wizard,
    quick_delete_entity_wizard, manage_bonded_atom_pairs_wizard,
    manage_user_ccd_wizard, strip_entities_wizard,
    show_job_summary, show_help_text,
    _current_ids, _next_letter
)


def _load_job_into_builder(jb: JobBuilder, data: Dict[str, Any]) -> None:
    """Load dictionary into the current JobBuilder instance."""
    try:
        new_jb = JobBuilder.from_dict(data)
        jb.name = new_jb.name
        jb.version = new_jb.version
        jb.modelSeeds = new_jb.modelSeeds
        jb.sequences = new_jb.sequences
        jb.bondedAtomPairs = new_jb.bondedAtomPairs
        jb.userCCD = new_jb.userCCD
        jb.userCCDPath = new_jb.userCCDPath
        _ok("Job loaded successfully.")
    except Exception as e:
        _err(f"Failed to load job: {e}")


# ============================================================
# ====================== MAIN CLI ============================
# ============================================================

def run_cli():
    strict_files = True
    jb = JobBuilder()

    while True:
        try:
            _rule("═", CYAN)
            print(BOLD + CYAN + " AlphaFold 3 JSON Builder  ·  Expert CLI ".center(TW) + RESET)
            _rule("═", CYAN)
            print(f"  {DIM}Strict file checks: {'ON' if strict_files else 'OFF'}{RESET}  "
                  f"  Sequences: {len(jb.sequences)}  "
                  f"  Seeds: {jb.modelSeeds or '(none)'}")
            _rule()
            print(f"  {BOLD}{CYAN} 1{RESET})  Job settings")
            print(f"  {BOLD}{CYAN} 2{RESET})  Proteins")
            print(f"  {BOLD}{CYAN} 3{RESET})  RNA")
            print(f"  {BOLD}{CYAN} 4{RESET})  DNA")
            print(f"  {BOLD}{CYAN} 5{RESET})  Ligands / Glycans")
            print(f"  {BOLD}{CYAN} 6{RESET})  Bonded atom pairs")
            print(f"  {BOLD}{CYAN} 7{RESET})  CCD / Custom components")
            print(f"  {BOLD}{CYAN} 8{RESET})  Show JSON")
            print(f"  {BOLD}{CYAN} 9{RESET})  Validate JSON")
            print(f"  {BOLD}{CYAN}10{RESET})  Save JSON")
            print(f"  {BOLD}{CYAN}11{RESET})  Load JSON (replace current job)")
            print(f"  {BOLD}{CYAN}12{RESET})  Resume from autosave")
            print(f"  {BOLD}{CYAN}13{RESET})  Preflight summary")
            print(f"  {BOLD}{CYAN}14{RESET})  Toggle strict file checks")
            print(f"  {BOLD}{CYAN}15{RESET})  Generate example jobs")
            print(f"  {BOLD}{CYAN}16{RESET})  Strip entity types from JSON")
            print(f"  {DIM}  0)  Exit{RESET}")
            _rule()

            choice = input(f"{CYAN}  ▶  Select: {RESET}").strip()

            if choice == "1":
                job_settings_menu(jb)

            elif choice == "2":
                protein_menu(jb)

            elif choice == "3":
                rna_menu(jb)

            elif choice == "4":
                dna_menu(jb)

            elif choice == "5":
                ligand_menu(jb)

            elif choice == "6":
                manage_bonded_atom_pairs_wizard(jb)

            elif choice == "7":
                manage_user_ccd_wizard(jb)

            elif choice == "8":
                _rule()
                print(json.dumps(jb.to_dict(), indent=2))
                _rule()
                _pause()

            elif choice == "9":
                try:
                    AF3Validator.validate_job(jb.to_dict(), require_files=strict_files)
                    _ok("Validation passed.")
                except ValidationError as e:
                    _err("Validation failed:")
                    for msg in e.messages:
                        print(f"     {RED}•{RESET}  {msg}")
                _pause()

            elif choice == "10":
                out = _ask("Output filename", default=f"{jb.name or 'af3_job'}.json")
                if not out.lower().endswith(".json"): out += ".json"
                try:
                    save_json(out, jb.to_dict())
                    _ok(f"Saved: {out}")
                except Exception as e:
                    _err(f"Save failed: {e}")
                _pause()

            elif choice == "11":
                path = _prompt_file_path_ui("JSON file to load", required=True)
                if path:
                    data = load_json(path)
                    if data:
                        _load_job_into_builder(jb, data)
                _pause()

            elif choice == "12":
                from af3_wizard import _resume_autosave
                _resume_autosave(jb)

            elif choice == "13":
                show_job_summary(jb)
                _pause()

            elif choice == "14":
                strict_files = not strict_files
                _info(f"Strict file checks: {'ON' if strict_files else 'OFF'}")
                _pause()

            elif choice == "15":
                _generate_example_jobs()
                _pause()

            elif choice == "16":
                result = strip_entities_wizard(jb)
                if result: jb = result

            elif choice == "0":
                _ok("Goodbye.")
                return

            else:
                _err("Invalid choice.")

        except KeyboardInterrupt:
            print()
            _warn("Cancelled by user.")
            try:
                p = autosave_json(jb.to_dict(), prefix="autosave_interrupt")
                _info(f"Autosaved to: {p}")
            except Exception:
                pass
            return

        except Exception as e:
            _err(f"Unexpected error: {e}")
            try:
                p = autosave_json(jb.to_dict(), prefix="autosave_error")
                _warn(f"Current job autosaved to: {p}")
            except Exception as se:
                _warn(f"Autosave also failed: {se}")



# ============================================================
# ====================== JOB SETTINGS =========================
# ============================================================

def job_settings_menu(jb: JobBuilder):
    while True:
        print("\n=== Job Settings ===")
        print(f"Current name: {jb.name}")
        print(f"Version: {jb.version}")
        print(f"Model seeds: {jb.modelSeeds}")
        print("1) Set name")
        print("2) Set version")
        print("3) Set model seeds (manual)")
        print("4) Generate random model seeds")   # NEW
        print("0) Back")

        ch = input("Select: ").strip()

        if ch == "1":
            jb.set_name(input("Job name: ").strip())

        elif ch == "2":
            jb.set_version(int(input("Version (1-4): ").strip()))

        elif ch == "3":
            try:
                raw = input("Enter seeds (space-separated integers): ").strip()
                seeds = [int(x) for x in raw.split()] if raw else []
                SeedsHelper.validate_seeds(seeds)
                jb.set_model_seeds(seeds)
                print("Seeds updated.")
            except ValueError as e:
                print("Error:", e)

        elif ch == "4":
            try:
                n = int(input("How many seeds? ").strip())
                seeds = SeedsHelper.generate_default_seeds(n)
                jb.set_model_seeds(seeds)
                print(f"Using {n} seed(s): {seeds}")
            except ValueError as e:
                print("Error:", e)

        elif ch == "0":
            return
        else:
            print("Invalid choice.")


# ============================================================
# ====================== ENTITY MENUS ========================
# ============================================================

def _entity_menu(jb: JobBuilder, kind: str, add_fn, edit_fn):
    """Generic menu for managing a specific entity type."""
    while True:
        _section(f"{kind.capitalize()} Menu")
        ents = [s[kind] for s in jb.sequences if kind in s]
        print(f"  Currently: {len(ents)} added")
        
        ch = _choose("Pick action:", [
            ("add",    f"Add {kind}"),
            ("edit",   f"Edit {kind}")   if ents else ("edit", f"{DIM}Edit (none){RESET}"),
            ("delete", f"Delete {kind}") if ents else ("delete", f"{DIM}Delete (none){RESET}"),
            ("list",   f"List {kind}")   if ents else ("list", f"{DIM}List (none){RESET}"),
        ], allow_back=True)

        if ch == "BACK": break
        elif ch == "add": add_fn(jb)
        elif ch == "edit":
            print(f"\n  {BOLD}Select {kind} to edit:{RESET}")
            for i, e in enumerate(ents, 1):
                print(f"  {i}) ID={e.get('id','?')}")
            try:
                idx_val = int(_ask("Choice")) - 1
                if 0 <= idx_val < len(ents):
                    # Find actual sequence index
                    actual_idx = [i for i, s in enumerate(jb.sequences) if kind in s][idx_val]
                    edit_fn(jb, actual_idx)
            except ValueError: pass
        elif ch == "delete": quick_delete_entity_wizard(jb)
        elif ch == "list":
            for e in ents: print(f"    • ID={BOLD}{e.get('id','?')}{RESET}  len={len(e.get('sequence',''))}")
            _pause()

def protein_menu(jb: JobBuilder): _entity_menu(jb, "protein", add_protein_wizard, edit_protein_wizard)
def rna_menu(jb: JobBuilder):     _entity_menu(jb, "rna",     add_rna_wizard,     edit_rna_wizard)
def dna_menu(jb: JobBuilder):     _entity_menu(jb, "dna",     add_dna_wizard,     edit_dna_wizard)
def ligand_menu(jb: JobBuilder):  _entity_menu(jb, "ligand",  add_ligand_wizard,  edit_ligand_wizard)


# ============================================================
# =================== STRIP ENTITY TYPES =====================
# ============================================================

# Local strip menu removed - now using shared strip_entities_wizard


# ============================================================
# ======================== EXAMPLE GENERATOR ===========================
# ============================================================
def _generate_example_jobs():
    os.makedirs("examples", exist_ok=True)

    examples = []

    # Example 1: minimal protein
    jb1 = JobBuilder()
    jb1.set_name("example_minimal_protein")
    jb1.set_model_seeds([1])
    jb1.add_protein(ProteinEntity(id="A", sequence="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRT"))
    examples.append(("examples/01_minimal_protein.json", jb1.to_dict()))

    # Example 2: protein + CCD ligand (one CCD code, multiple copies via list IDs)
    jb2 = JobBuilder()
    jb2.set_name("example_protein_ccd_ligand_multicopy")
    jb2.set_model_seeds([1, 2])
    jb2.add_protein(ProteinEntity(id="A", sequence="MTTGLSTAGAQDIGRSSVRPYLEECTRRFQEMFDRHVVTRPTKVELTDAEL"))
    jb2.add_ligand(LigandEntity(id=["B", "C", "D"], ccdCodes=["MG"]))
    examples.append(("examples/02_ccd_ligand_multicopy.json", jb2.to_dict()))

    # Example 3: protein with MSA paths + template path (paths are placeholders)
    jb3 = JobBuilder()
    jb3.set_name("example_protein_with_paths")
    jb3.set_model_seeds([7])
    jb3.add_protein(ProteinEntity(
        id="A",
        sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVLTASQSTGQ",
        unpairedMsaPath="fixtures/data/dummy_unpaired.a3m",
        pairedMsaPath="fixtures/data/dummy_paired.a3m",
        templates=[{
            "mmcifPath": "fixtures/templates/template_A.cif",
            "queryIndices": [0,1,2,3,4,5,6,7,8,9],
            "templateIndices": [0,1,2,3,4,5,6,7,8,9],
        }]
    ))
    examples.append(("examples/03_msa_and_template_paths.json", jb3.to_dict()))

    # Example 4: inline userCCD + custom CCD ligand
    jb4 = JobBuilder()
    jb4.set_name("example_userccd_inline")
    jb4.set_model_seeds([3])
    jb4.add_protein(ProteinEntity(
        id="A",
        sequence="MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQ",
        description="Toy protein with custom CCD ligand"
    ))
    jb4.add_ligand(LigandEntity(id="L", ccdCodes=["LIG"], description="Custom ligand"))
    jb4.set_userCCD(
        "data_LIG\n#\n_chem_comp.id LIG\n_chem_comp.name 'ToyLigand'\n_chem_comp.type non-polymer\n#\n"
    )
    examples.append(("examples/04_userccd_inline.json", jb4.to_dict()))

    for path, data in examples:
        save_json(path, data)

    print(f"Wrote {len(examples)} example jobs to ./examples/")
