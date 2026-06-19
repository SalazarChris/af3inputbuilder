import os
import sys
import string
import json
from typing import Optional, List, Dict, Any, Union, Set

from ..core.job import JobBuilder
from ..core.entities import (
    ProteinEntity, RNAEntity, DNAEntity, LigandEntity,
    PROTEIN_ALPHABET, RNA_ALPHABET, DNA_ALPHABET,
    reverse_complement
)
from ..core.reference import (
    PROTEIN_PTMS, RNA_MODIFICATIONS, DNA_MODIFICATIONS,
    COMMON_IONS, COMMON_COFACTORS, COMMON_SMALL_MOLECULES,
)
from .ui import (
    BOLD, RESET, DIM, CYAN, GREEN, RED, YELLOW,
    _section, _tip, _ok, _warn, _err, _info, _ask, _ask_yn, _choose, _pause, _ask_file, _divider
)
from ..core.seeds import SeedsHelper

# ---------------------------------------------------------------------------
# Constants / Menu Definitions
# ---------------------------------------------------------------------------

MOD_RESIDUE_MAPPING = {
    # Proteins
    "SEP": {"S"},    "TPO": {"T"},    "PTR": {"Y"},    "HIP": {"H"},
    "MLY": {"K"},    "MLZ": {"K"},    "M3L": {"K"},
    "ALY": {"K"},    "HYP": {"P"},    "HY3": {"P"},
    "CSO": {"C"},    "CSD": {"C"},    "OCS": {"C"},    "MSE": {"M"},
    "TYS": {"Y"},    "CIR": {"R"},    "MAR": {"R"},    "SMC": {"R"},
    "DAL": {"A"},    "DVA": {"V"},
    # RNA
    "5MC": {"C"},    "OMG": {"G"},    "OMC": {"C"},    "OMU": {"U"},
    "MA6": {"A"},    "1MA": {"A"},    "PSU": {"U"},    "H2U": {"U"},
    "4SU": {"U"},    "I": {"A"},      "7MG": {"G"},    "2MG": {"G"},
    "5MU": {"U"},
    # DNA
    "5CM": {"C"},    "5HM": {"C"},    "5FC": {"C"},    "5CC": {"C"},
    "6MA": {"A"},    "8OG": {"G"},    "DHU": {"T", "U"},
}

# Build menu lists from reference (add "other" option at the end)
PROTEIN_PTM_COMMON = [(code, desc) for code, desc in PROTEIN_PTMS] + [("other", "Other — enter CCD code manually")]
RNA_MOD_COMMON = [(code, desc) for code, desc in RNA_MODIFICATIONS] + [("other", "Other — enter CCD code manually")]
DNA_MOD_COMMON = [(code, desc) for code, desc in DNA_MODIFICATIONS] + [("other", "Other — enter CCD code manually")]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _current_ids(jb: JobBuilder) -> Set[str]:
    """Get all chain IDs currently in the job."""
    ids = set()
    for ent in jb.sequences:
        for val in ent.values():
            eid = val.get("id")
            if isinstance(eid, list):
                ids.update(str(x) for x in eid)
            elif eid is not None:
                ids.add(str(eid))
    return ids

def _next_letter(used: Set[str]) -> str:
    """Return the next unused single-letter chain ID (A--'Z, then AA, AB---)."""
    for c in string.ascii_uppercase:
        if c not in used:
            return c
    for c1 in string.ascii_uppercase:
        for c2 in string.ascii_uppercase:
            name = c1 + c2
            if name not in used:
                return name
    return "X"

def _validate_seq_interactive(seq: str, alphabet: Union[str, Set[str]], kind: str) -> bool:
    """Validate sequence and print error tip if invalid."""
    if not seq:
        _err(f"{kind} sequence cannot be empty.")
        return False
    
    alpha_set = set(alphabet)
    bad = set(seq.upper()) - alpha_set
    if bad:
        _err(f"{kind} sequence contains invalid characters: {', '.join(sorted(bad))}")
        _tip(f"Allowed characters: {''.join(sorted(alpha_set))}")
        return False
    return True

# ---------------------------------------------------------------------------
# Shared Workflows
# ---------------------------------------------------------------------------

def manage_modifications_wizard(kind: str, sequence: str, existing: list) -> list:
    """
    Interactive loop to build/manage modifications (PTMs / base mods).
    """
    mods = list(existing or [])
    seq_len = len(sequence)

    if kind == "protein":
        label, type_field, pos_field = "PTM", "ptmType", "ptmPosition"
        common_menu = PROTEIN_PTM_COMMON
        tip = ("PTMs are chemical modifications on amino-acid residues.\n"
               "     You need the CCD code and the 1-based position.")
    else:
        label, type_field, pos_field = "base modification", "modificationType", "basePosition"
        common_menu = RNA_MOD_COMMON if kind == "rna" else DNA_MOD_COMMON
        tip = ("Base modifications are chemical changes to individual nucleotides.\n"
               "     You need the CCD code and the 1-based position.")

    _tip(tip)

    while True:
        print()
        print(f"  {BOLD}Current {label}s:{RESET}  " + (f"{len(mods)} defined" if mods else "(none)"))
        if mods:
            for mi, m in enumerate(mods, 1):
                print(f"    {CYAN}{mi}{RESET})  type={BOLD}{m.get(type_field,'?')}{RESET}  pos={m.get(pos_field,'?')}")

        action = _choose(f"Manage {label}s", [
            ("add",    f"Add a {label}"),
            ("remove", "Remove one") if mods else ("remove", f"{DIM}Remove one  (none yet){RESET}"),
            ("clear",  "Clear all")  if mods else ("clear",  f"{DIM}Clear all   (none yet){RESET}"),
        ], allow_back=True, back_label="Done")

        if action == "BACK":
            break
        elif action == "add":
            choice = _choose(f"Select {label} type", common_menu, allow_back=True, back_label="Cancel")
            if choice == "BACK": continue
            if choice == "other":
                choice = _ask("Enter CCD code").upper().strip()
                if not choice: continue
            
            while True:
                try:
                    pos = int(_ask(f"Position (1---{seq_len})"))
                    if 1 <= pos <= seq_len:
                        # Residue check
                        actual_res = sequence[pos-1].upper()
                        allowed_res = MOD_RESIDUE_MAPPING.get(choice)
                        if allowed_res and actual_res not in allowed_res:
                            _warn(f"Residue mismatch: {choice} usually applies to {', '.join(allowed_res)}, but found {BOLD}{actual_res}{RESET} at pos {pos}.")
                            if not _ask_yn("Add it anyway?"):
                                continue
                        break
                    _err(f"Out of range: 1---{seq_len}")
                except ValueError: _err("Enter a number.")
            
            mods.append({type_field: choice, pos_field: pos})
            _ok(f"Added {choice} at pos {pos}")
            
        elif action == "remove" and mods:
            try:
                ri = int(_ask("Number to remove")) - 1
                if 0 <= ri < len(mods):
                    removed = mods.pop(ri)
                    _ok(f"Removed {removed.get(type_field)}")
                else: _err("Invalid number.")
            except ValueError: _err("Enter a number.")
            
        elif action == "clear" and mods:
            if _ask_yn("Clear ALL?"): mods.clear(); _ok("Cleared.")
    
    return mods

def manage_templates_wizard(seq: str, existing: list) -> list:
    """Interactive loop for structure templates."""
    templates = list(existing or [])
    _tip("Templates need an mmCIF file and residue indices.\n"
         "     Indices can be auto-computed if BioPython is installed.")

    while True:
        print()
        print(f"  {BOLD}Current templates:{RESET}  " + (f"{len(templates)}" if templates else "(none)"))
        if templates:
            for ti, t in enumerate(templates, 1):
                p = t.get("mmcifPath") or "(inline)"
                nq = len(t.get("queryIndices", []))
                print(f"    {CYAN}{ti}{RESET})  {p}  ({nq} residues)")

        action = _choose("Manage templates", [
            ("add",    "Add template"),
            ("remove", "Remove one") if templates else ("remove", f"{DIM}Remove (none){RESET}"),
            ("clear",  "Clear all")  if templates else ("clear",  f"{DIM}Clear (none){RESET}"),
        ], allow_back=True, back_label="Done")

        if action == "BACK": break
        elif action == "add":
            path = _ask_file("mmCIF (.cif) file path", required=True)
            if not path: continue
            chid = _ask("Chain ID in template", default="A").strip().upper()

            try:
                from ..utils.templater import run_templater
                _info("Computing alignment...")
                block = run_templater(path, seq, chid)
                block["mmcifPath"] = path
                templates.append(block)
                _ok("Template added via auto-alignment.")
            except Exception as e:
                _warn(f"Auto-alignment failed ({e}). Manual entry needed.")
                # Minimal manual fallback
                _err("Manual template entry currently requires direct JSON editing or BioPython.")
        
        elif action == "remove" and templates:
            try:
                ri = int(_ask("Number to remove")) - 1
                if 0 <= ri < len(templates): templates.pop(ri); _ok("Removed.")
            except ValueError: _err("Enter a number.")
    
    return templates

def add_protein_wizard(jb: JobBuilder):
    while True:
        raw = _ask("Sequence")
        seq = "".join(c for c in raw.upper() if c.isalpha())
        if _validate_seq_interactive(seq, PROTEIN_ALPHABET, "Protein"): break

    used = _current_ids(jb)
    pid = _ask("Chain ID", default=_next_letter(used)).upper()
    if pid in used: return _err(f"ID {pid} used.")

    # 1. MSAs (Unpaired & Paired)
    msa_choice = _choose("MSA Strategy (unpaired/paired)", [
        ("calc",    "Calculate (AF3 server search - default)"),
        ("provide", "Provide local A3M file(s)"),
        ("ignore",  "Ignore (Skip search)"),
    ], default="calc")

    unpaired_msa, paired_msa = None, None
    ump, pmp = None, None

    if msa_choice == "ignore":
        unpaired_msa, paired_msa = "", ""
    elif msa_choice == "provide":
        ump = _ask_file("unpairedMsa Path")
        if ump:
            from ..utils.msa import validate_msa_sequence
            if not validate_msa_sequence(ump, seq):
                _warn("unpairedMsa sequence does not match provided protein sequence.")
            # Path preserved as supplied (Bug 1 fix)
        
        pmp = _ask_file("pairedMsa Path")
        if pmp:
            from ..utils.msa import validate_msa_sequence
            if not validate_msa_sequence(pmp, seq):
                _warn("pairedMsa sequence does not match provided protein sequence.")
            # Path preserved as supplied (Bug 1 fix)

    # 2. Templates
    t_choice = _choose("Template Strategy", [
        ("ignore",  "Ignore (Skip search - default)"),
        ("calc",    "Calculate (AF3 server search)"),
        ("provide", "Provide local mmCIF file(s)"),
    ], default="ignore")

    tmpls = None
    if t_choice == "ignore":
        tmpls = []
    elif t_choice == "provide":
        tmpls = manage_templates_wizard(seq, [])

    mods = []
    if _ask_yn("Add PTMs?"): mods = manage_modifications_wizard("protein", seq, [])

    try:
        jb.add_protein(ProteinEntity(
            id=pid, sequence=seq,
            unpairedMsaPath=ump or None,
            pairedMsaPath=pmp or None,
            unpairedMsa=unpaired_msa,
            pairedMsa=paired_msa,
            modifications=mods or None,
            templates=tmpls,
        ))
        _ok(f"Protein {pid} added.")
    except Exception as e: _err(f"Failed: {e}")
    _pause()

def add_rna_wizard(jb: JobBuilder):
    while True:
        raw = _ask("Sequence")
        seq = "".join(c for c in raw.upper() if c.isalpha())
        if _validate_seq_interactive(seq, RNA_ALPHABET, "RNA"): break

    used = _current_ids(jb)
    rid = _ask("Chain ID", default=_next_letter(used)).upper()
    if rid in used: return _err(f"ID {rid} used.")

    # MSA handling
    msa_choice = _choose("MSA Strategy", [
        ("calc",    "Calculate (AF3 server search - default)"),
        ("provide", "Provide local A3M file"),
        ("ignore",  "Ignore (Skip search)"),
    ], default="calc")

    unpaired_msa = None
    ump = None
    if msa_choice == "ignore":
        unpaired_msa = ""
    elif msa_choice == "provide":
        ump = _ask_file("unpairedMsa Path")
        # Path preserved as supplied (Bug 1 fix)

    mods = []
    if _ask_yn("Add modifications?"): mods = manage_modifications_wizard("rna", seq, [])

    try:
        jb.add_rna(RNAEntity(id=rid, sequence=seq, unpairedMsa=unpaired_msa,
                             unpairedMsaPath=ump or None,
                             modifications=mods or None))
        _ok(f"RNA {rid} added.")
    except Exception as e: _err(f"Failed: {e}")
    _pause()

def add_dna_wizard(jb: JobBuilder):
    while True:
        raw = _ask("Sequence")
        seq = "".join(c for c in raw.upper() if c.isalpha())
        if _validate_seq_interactive(seq, DNA_ALPHABET, "DNA"): break

    used = _current_ids(jb)
    did = _ask("Chain ID", default=_next_letter(used)).upper()
    if did in used: return _err(f"ID {did} used.")

    # MSA handling
    msa_choice = _choose("MSA Strategy", [
        ("calc",    "Calculate (AF3 server search - default)"),
        ("provide", "Provide local A3M file"),
        ("ignore",  "Ignore (Skip search)"),
    ], default="calc")

    unpaired_msa = None
    ump = None
    if msa_choice == "ignore":
        unpaired_msa = ""
    elif msa_choice == "provide":
        ump = _ask_file("unpairedMsa Path")
        # Path preserved as supplied (Bug 1 fix)

    mods = []
    if _ask_yn("Add modifications?"): mods = manage_modifications_wizard("dna", seq, [])

    try:
        jb.add_dna(DNAEntity(id=did, sequence=seq,
                             unpairedMsa=unpaired_msa,
                             unpairedMsaPath=ump or None,
                             modifications=mods or None))
        _ok(f"DNA {did} added.")
        if _ask_yn("Add complementary strand?"):
            cseq = reverse_complement(seq, is_dna=True)
            jb.add_dna(DNAEntity(id=_next_letter(_current_ids(jb)), sequence=cseq,
                                 unpairedMsa=unpaired_msa,
                                 unpairedMsaPath=ump or None))
            _ok("Complementary strand added.")
    except Exception as e: _err(f"Failed: {e}")
    _pause()

def add_ligand_wizard(jb: JobBuilder):
    _section("Add Ligand")
    used = _current_ids(jb)
    lid_raw = _ask("Chain ID(s)", default=_next_letter(used))
    lids = [x.strip().upper() for x in lid_raw.split(",")] if "," in lid_raw else lid_raw.upper()
    
    mode = _choose("Specify via", [("ccd", "CCD Code"), ("smiles", "SMILES")])
    
    count = 1
    if isinstance(lids, str):
        # If user only gave one ID, ask if they want multiple copies
        try:
            c_raw = _ask(f"How many copies of this ligand would you like to add?", default="1")
            count = int(c_raw)
        except ValueError:
            count = 1
            
        if count > 1:
            added_ids = [lids]
            tmp_used = set(used)
            tmp_used.add(lids)
            for _ in range(count - 1):
                nid = _next_letter(tmp_used)
                added_ids.append(nid)
                tmp_used.add(nid)
            lids = added_ids

    if mode == "ccd":
        code = _ask("CCD Code (e.g. ATP)").upper().strip()
        try:
            jb.add_ligand(LigandEntity(id=lids, ccdCodes=[code]))
            _ok(f"Ligand {code} added ({count} copies).")
        except Exception as e: _err(f"Failed: {e}")
    else:
        smi = _ask("SMILES string")
        try:
            jb.add_ligand(LigandEntity(id=lids, smiles=smi))
            _ok(f"Ligand added ({count} copies).")
        except Exception as e: _err(f"Failed: {e}")
    _pause()

def quick_delete_entity_wizard(jb: JobBuilder):
    _section("Quick Delete")
    if not jb.sequences: return _warn("No entities.")

    for i, s in enumerate(jb.sequences, 1):
        kind = list(s.keys())[0]
        eid = s[kind].get("id", "?")
        print(f"  {CYAN}{i:>2}{RESET}) {BOLD}{kind.upper():<8}{RESET} ID={eid}")
    
    try:
        idx = int(_ask("Number to delete (0=cancel)", "0")) - 1
        if 0 <= idx < len(jb.sequences):
            jb.sequences.pop(idx)
            _ok("Deleted.")
    except ValueError: pass
    _pause()

# ---------------------------------------------------------------------------
# Summary & Help
# ---------------------------------------------------------------------------

def show_job_summary(jb: JobBuilder):
    """Display a human-readable summary of the current JobBuilder state."""
    import textwrap
    d = jb.to_dict()
    seqs = d.get("sequences", [])
    
    _section("Job Summary")
    print(f"  {BOLD}Name:{RESET}         {d.get('name', '(unnamed)')}")
    print(f"  {BOLD}Version:{RESET}      {d.get('version', 1)}")
    print(f"  {BOLD}Seeds:{RESET}        {d.get('modelSeeds', [])}")
    print()
    
    counts = {"protein": 0, "rna": 0, "dna": 0, "ligand": 0}
    for s in seqs:
        for k in counts:
            if k in s:
                counts[k] += 1
                data = s[k]
                eid = data.get("id", "?")
                seq = data.get("sequence", "")
                extra = f"len={len(seq)}" if seq else ""
                if k == "ligand":
                    ccd = data.get("ccdCodes")
                    spec = f"CCD={ccd[0]}" if ccd else "SMILES"
                    extra = spec
                print(f"  {BOLD}{k.capitalize():<8}:{RESET} ID={eid}  {extra}")

    bonds = d.get("bondedAtomPairs", [])
    if bonds: print(f"  {BOLD}Bonds:{RESET}        {len(bonds)} pair(s)")
    if d.get("userCCD"): print(f"  {BOLD}Custom CCD:{RESET}   inline ({len(d['userCCD'])} chars)")
    elif d.get("userCCDPath"): print(f"  {BOLD}Custom CCD:{RESET}   path={d['userCCDPath']}")

def show_help_text():
    """Display helpful information about AlphaFold 3 and this tool."""
    import textwrap
    _section("AlphaFold 3 Help")
    help_text = textwrap.dedent(f"""
        AlphaFold 3 predicts the 3D structure of proteins, DNA, RNA, and ligands.
        
        {BOLD}Core Concepts:{RESET}
        - {CYAN}Entities{RESET}: Individual chains or molecules in your system.
        - {CYAN}PTMs{RESET}: Post-translational modifications for proteins.
        - {CYAN}CCD Codes{RESET}: Standard 1–4 character codes for ligands (e.g., ATP, MG, K).
        - {CYAN}SMILES{RESET}: Text-based chemical formulas for custom ligands.
        - {CYAN}Seeds{RESET}: Random numbers for prediction diversity.
        
        {BOLD}Workflow:{RESET}
        1. Set job name and seeds.
        2. Add all molecular entities.
        3. (Optional) Add custom bonds or CCD data.
        4. Validate and Save JSON.
    """)
    print(help_text)

def manage_bonded_atom_pairs_wizard(jb: JobBuilder):
    _section("Bonded Atom Pairs")
    while True:
        pairs = jb.bondedAtomPairs
        print(f"  {BOLD}Current pairs:{RESET}  {len(pairs)}")
        if pairs:
            _divider()
            for i, p in enumerate(pairs, 1):
                print(f"  {i}) {p[0]} <-> {p[1]}")
            _divider()

        act = _choose("Action", [("add", "Add Bond"), ("remove", "Remove"), ("clear", "Clear")], allow_back=True)
        if act == "BACK": break
        elif act == "add":
            def get_atom():
                cid = _ask("Chain ID").upper()
                idx = int(_ask("Residue Index (1-based)"))
                atom = _ask("Atom Name").upper()
                return [cid, idx, atom]
            try:
                a1 = get_atom(); a2 = get_atom()
                jb.add_bonded_pair(a1, a2)
                _ok("Bond added.")
            except Exception as e: _err(f"Error: {e}")
        elif act == "remove" and pairs:
            try:
                ri = int(_ask("Number to remove")) - 1
                if 0 <= ri < len(pairs): pairs.pop(ri); _ok("Removed.")
            except ValueError: pass
    _pause()

def manage_user_ccd_wizard(jb: JobBuilder):
    """Wizard to manage job-level custom CCD definitions."""
    _section("Custom CCD Components")
    _tip("Provide mmCIF data for novel ligands (userCCD/userCCDPath).\n"
         "     This automatically bumps the job to Version 3.")
    
    if jb.userCCD: _info(f"Currently: Inline block ({len(jb.userCCD)} chars)")
    elif jb.userCCDPath: _info(f"Currently: File path --' {jb.userCCDPath}")
    else: _info("Currently: none")

    action = _choose("Action", [
        ("inline", "Paste mmCIF text (Inline)"),
        ("path",   "Point to .cif file (Path)"),
        ("clear",  "Clear custom CCD"),
    ], allow_back=True)

    if action == "inline":
        print(f"  {DIM}(Paste block, then type 'END' on a new line){RESET}")
        lines = []
        while True:
            try: l = input()
            except EOFError: break
            if l.strip().upper() == "END": break
            lines.append(l)
        block = "\n".join(lines).strip()
        # Fix double escaped newlines (e.g. if user pastes raw JSON strings)
        block = block.replace('\\n', '\n')
        
        if block:
            jb.userCCDPath = None; jb.set_userCCD(block); _ok("Saved.")
    elif action == "path":
        p = _ask_file("Path to CCD CIF file", required=True)
        if p:
            jb.userCCD = None; jb.set_userCCDPath(p); jb._require_version(3); _ok("Saved.")
    elif action == "clear":
        if _ask_yn("Clear custom CCD?"): jb.userCCD = None; jb.userCCDPath = None; _ok("Cleared.")
    _pause()

def edit_protein_wizard(jb: JobBuilder, idx: int):
    """Edit fields of an existing protein entity."""
    data = jb.sequences[idx]["protein"]
    _section(f"Edit Protein - ID: {data.get('id', '?')}")
    _tip("Press Enter to keep the current value in [brackets].")

    while True:
        n_ptm = len(data.get("modifications", []) or [])
        n_tmpl = len(data.get("templates", []) or [])
        
        # Determine status for UI
        def get_status(p_key, s_key, empty_val):
            if data.get(p_key): return f"Path: {data[p_key]}"
            if data.get(s_key) == empty_val: return "Ignore (Skip)"
            return "Calculate (Search)"

        umsa_status = get_status("unpairedMsaPath", "unpairedMsa", "")
        pmsa_status = get_status("pairedMsaPath", "pairedMsa", "")
        tmpl_status = "Path(s) defined" if data.get("templates") else ("Ignore (Skip)" if data.get("templates") == [] else "Calculate (Search)")

        field = _choose("Which field?", [
            ("seq",  f"Sequence      [{len(data.get('sequence',''))} residues]"),
            ("desc", f"Description   [{data.get('description', '(none)')}]"),
            ("umsa", f"Unpaired MSA  [{umsa_status}]"),
            ("pmsa", f"Paired MSA    [{pmsa_status}]"),
            ("ptm",  f"PTMs          [{n_ptm} defined]"),
            ("tmpl", f"Templates     [{tmpl_status}]"),
        ], allow_back=True, back_label="Done")
        
        if field == "BACK": break
        if field == "seq":
            raw = _ask("New sequence", default=data.get("sequence", ""))
            seq = "".join(c for c in raw.upper() if c.isalpha())
            if _validate_seq_interactive(seq, PROTEIN_ALPHABET, "Protein"):
                data["sequence"] = seq
                _ok("Updated.")
        elif field == "desc":
            val = _ask("Description", default=data.get("description", ""))
            if val: data["description"] = val; jb._require_version(3)
            else: data.pop("description", None)
        elif field == "umsa":
            strat = _choose("Strategy?", [
                ("calc",    "Calculate (AF3 server search - default)"),
                ("provide", "Provide local A3M file"),
                ("ignore",  "Ignore (Skip search)"),
            ], default="calc")
            if strat == "ignore":
                data["unpairedMsa"] = ""; data.pop("unpairedMsaPath", None)
            elif strat == "calc":
                data.pop("unpairedMsa", None); data.pop("unpairedMsaPath", None)
            else:
                p = _ask_file("Unpaired MSA Path")
                if p:
                    data["unpairedMsaPath"] = p
                    data.pop("unpairedMsa", None)
            jb._require_version(2)
        elif field == "pmsa":
            strat = _choose("Strategy?", [
                ("calc",    "Calculate (AF3 server search - default)"),
                ("provide", "Provide local A3M file"),
                ("ignore",  "Ignore (Skip search)"),
            ], default="calc")
            if strat == "ignore":
                data["pairedMsa"] = ""; data.pop("pairedMsaPath", None)
            elif strat == "calc":
                data.pop("pairedMsa", None); data.pop("pairedMsaPath", None)
            else:
                p = _ask_file("Paired MSA Path")
                if p:
                    data["pairedMsaPath"] = p
                    data.pop("pairedMsa", None)
            jb._require_version(2)
        elif field == "ptm":
            mods = manage_modifications_wizard("protein", data["sequence"], data.get("modifications") or [])
            if mods:
                data["modifications"] = mods
            elif "modifications" in data:
                del data["modifications"]
        elif field == "tmpl":
            strat = _choose("Strategy?", [
                ("ignore",  "Ignore (Skip search - default)"),
                ("calc",    "Calculate (AF3 server search)"),
                ("provide", "Provide local mmCIF file(s)"),
            ], default="ignore")
            if strat == "ignore":
                data["templates"] = []; jb._require_version(2)
            elif strat == "calc":
                data.pop("templates", None)
            else:
                tmpls = manage_templates_wizard(data["sequence"], data.get("templates") or [])
                if tmpls: data["templates"] = tmpls; jb._require_version(2)
                else: data.pop("templates", None)
    _pause()

def edit_rna_wizard(jb: JobBuilder, idx: int):
    """Edit fields of an existing RNA entity."""
    data = jb.sequences[idx]["rna"]
    _section(f"Edit RNA - ID: {data.get('id', '?')}")
    while True:
        n_mod = len(data.get("modifications", []) or [])
        msa_status = f"Path: {data['unpairedMsaPath']}" if data.get("unpairedMsaPath") else ("Ignore (Skip)" if data.get("unpairedMsa") == "" else "Calculate (Search)")
        field = _choose("Which field?", [
            ("seq",  f"Sequence      [{len(data.get('sequence',''))} bases]"),
            ("desc", f"Description   [{data.get('description', '(none)')}]"),
            ("umsa", f"Unpaired MSA  [{msa_status}]"),
            ("mod",  f"Modifications [{n_mod} defined]"),
        ], allow_back=True, back_label="Done")
        if field == "BACK": break
        if field == "seq":
            raw = _ask("New sequence", default=data.get("sequence", ""))
            seq = "".join(c for c in raw.upper() if c.isalpha())
            if _validate_seq_interactive(seq, RNA_ALPHABET, "RNA"):
                data["sequence"] = seq; _ok("Updated.")
        elif field == "desc":
            val = _ask("Description", default=data.get("description", ""))
            if val: data["description"] = val; jb._require_version(3)
            else: data.pop("description", None)
        elif field == "umsa":
            strat = _choose("Strategy?", [
                ("calc",    "Calculate (AF3 server search - default)"),
                ("provide", "Provide local A3M file"),
                ("ignore",  "Ignore (Skip search)"),
            ], default="calc")
            if strat == "ignore":
                data["unpairedMsa"] = ""; data.pop("unpairedMsaPath", None)
            elif strat == "calc":
                data.pop("unpairedMsa", None); data.pop("unpairedMsaPath", None)
            else:
                p = _ask_file("Unpaired MSA Path")
                if p:
                    data["unpairedMsaPath"] = p
                    data.pop("unpairedMsa", None)
            jb._require_version(2)
        elif field == "mod":
            mods = manage_modifications_wizard("rna", data["sequence"], data.get("modifications") or [])
            if mods:
                data["modifications"] = mods
            elif "modifications" in data:
                del data["modifications"]
    _pause()

def edit_dna_wizard(jb: JobBuilder, idx: int):
    """Edit fields of an existing DNA entity."""
    data = jb.sequences[idx]["dna"]
    _section(f"Edit DNA - ID: {data.get('id', '?')}")
    while True:
        n_mod = len(data.get("modifications", []) or [])
        msa_status = f"Path: {data['unpairedMsaPath']}" if data.get("unpairedMsaPath") else ("Ignore (Skip)" if data.get("unpairedMsa") == "" else "Calculate (Search)")
        field = _choose("Which field?", [
            ("seq",  f"Sequence      [{len(data.get('sequence',''))} bases]"),
            ("desc", f"Description   [{data.get('description', '(none)')}]"),
            ("umsa", f"Unpaired MSA  [{msa_status}]"),
            ("mod",  f"Modifications [{n_mod} defined]"),
        ], allow_back=True, back_label="Done")
        if field == "BACK": break
        if field == "seq":
            raw = _ask("New sequence", default=data.get("sequence", ""))
            seq = "".join(c for c in raw.upper() if c.isalpha())
            if _validate_seq_interactive(seq, DNA_ALPHABET, "DNA"):
                data["sequence"] = seq; _ok("Updated.")
        elif field == "desc":
            val = _ask("Description", default=data.get("description", ""))
            if val: data["description"] = val; jb._require_version(3)
            else: data.pop("description", None)
        elif field == "umsa":
            strat = _choose("Strategy?", [
                ("calc",    "Calculate (AF3 server search - default)"),
                ("provide", "Provide local A3M file"),
                ("ignore",  "Ignore (Skip search)"),
            ], default="calc")
            if strat == "ignore":
                data["unpairedMsa"] = ""; data.pop("unpairedMsaPath", None)
            elif strat == "calc":
                data.pop("unpairedMsa", None); data.pop("unpairedMsaPath", None)
            else:
                p = _ask_file("Unpaired MSA Path")
                if p:
                    data["unpairedMsaPath"] = p
                    data.pop("unpairedMsa", None)
            jb._require_version(2)
        elif field == "mod":
            mods = manage_modifications_wizard("dna", data["sequence"], data.get("modifications") or [])
            if mods:
                data["modifications"] = mods
            elif "modifications" in data:
                del data["modifications"]
    _pause()

def edit_ligand_wizard(jb: JobBuilder, idx: int):
    """Edit fields of an existing ligand entity."""
    data = jb.sequences[idx]["ligand"]
    _section(f"Edit Ligand - ID: {data.get('id', '?')}")
    while True:
        ccd = data.get("ccdCodes", [None])[0] or "(none)"
        smi = data.get("smiles", "(none)")
        field = _choose("Which field?", [
            ("spec", f"CCD/SMILES    [{'CCD='+ccd if data.get('ccdCodes') else 'SMILES='+smi[:20]+'...'}]"),
            ("desc", f"Description   [{data.get('description', '(none)')}]"),
        ], allow_back=True, back_label="Done")
        if field == "BACK": break
        if field == "spec":
            m = _choose("Mode", [("ccd", "CCD"), ("smi", "SMILES")])
            if m == "ccd":
                c = _ask("CCD Code").upper().strip()
                if c: data.pop("smiles", None); data["ccdCodes"] = [c]
            else:
                s = _ask("SMILES")
                if s: data.pop("ccdCodes", None); data["smiles"] = s
        elif field == "desc":
            val = _ask("Description", default=data.get("description", ""))
            if val: data["description"] = val; jb._require_version(3)
            else: data.pop("description", None)
    _pause()

def strip_entities_wizard(jb: JobBuilder) -> Optional[JobBuilder]:
    """
    Strip entity types (protein/rna/dna/ligand) from the job.
    Returns the new JobBuilder if loaded into memory, else None.
    """
    import copy
    _section("Strip Entity Types")
    _tip("Remove entire entity types. Result can be loaded or saved.")

    src = _choose("Source", [("mem", "Current job"), ("file", "Load JSON file")], allow_back=True)
    if src == "BACK": return None

    if src == "mem":
        data = jb.to_dict(); label = "current job"
    else:
        path = _ask_file("JSON path", required=True)
        if not path: return None
        data = load_json(path)
        if not data: return None
        label = path

    seqs = data.get("sequences", [])
    present = [et for et in ["protein", "rna", "dna", "ligand"] if any(et in s for s in seqs)]
    if not present: _warn("No sequences!"); _pause(); return None

    to_remove = set()
    while True:
        print(f"\n  {BOLD}Strip selection (toggle):{RESET}")
        opts = []
        for et in present:
            check = f"{GREEN}[\u2713]{RESET}" if et in to_remove else "[ ]"
            opts.append((et, f"{check} {et}"))
        
        act = _choose("Toggle types (Done=0)", opts, allow_back=True, back_label="Done \u2192 Proceed")
        if act == "BACK": break
        if act in to_remove: to_remove.discard(act)
        else: to_remove.add(act)

    if not to_remove: return None

    if not _ask_yn("Proceed with stripping?"): return None

    stripped = copy.deepcopy(data)
    stripped["sequences"] = [s for s in stripped.get("sequences", []) if not any(et in s for et in to_remove)]
    _ok(f"Stripped types: {', '.join(to_remove)}. {len(stripped['sequences'])} left.")

    dest = _choose("Action", [("mem", "Load into memory"), ("file", "Save to file"), ("both", "Both")])
    new_jb = None
    if dest in ("mem", "both"):
        new_jb = JobBuilder.from_dict(stripped); _ok("Loaded.")
    if dest in ("file", "both"):
        fname = _ask("Filename", default="stripped_job.json")
        if not fname.endswith(".json"): fname += ".json"
        save_json(fname, stripped); _ok(f"Saved to {fname}")
    
    _pause()
    return new_jb
def add_common_ions_wizard(jb: JobBuilder):
    """Guided shortcut to add common ions (Na, Cl, Mg, etc.) or water."""
    _section("Add Common Ions / Water")
    _tip("Quickly add multiple copies of standard ions or water molecules.")
    
    options = [
        ("NA", "Na+ (Sodium Ion)"),
        ("CL", "Cl- (Chloride Ion)"),
        ("MG", "Mg2+ (Magnesium Ion)"),
        ("ZN", "Zn2+ (Zinc Ion)"),
        ("K",  "K+  (Potassium Ion)"),
        ("HOH","H2O (Water)"),
    ]
    
    choice = _choose("Select item to add", options, allow_back=True)
    if choice == "BACK":
        return
    
    while True:
        raw = _ask(f"How many copies of {choice} would you like to add?", default="1")
        try:
            count = int(raw)
            if count > 0:
                break
            _err("Count must be at least 1.")
        except ValueError:
            _err("Please enter a whole number.")
            
    # Implementation
    from ..core.entities import LigandEntity
    used = _current_ids(jb)
    added_ids = []
    for _ in range(count):
        new_id = _next_letter(used)
        used.add(new_id)
        added_ids.append(new_id)
    
    # Create a SINGLE LigandEntity with a list of IDs
    id_val = added_ids[0] if count == 1 else added_ids
    lig = LigandEntity(id=id_val, ccdCodes=[choice])
    jb.add_ligand(lig)
        
    _ok(f"Added {count} copies of {choice} (Chains: {', '.join(added_ids)})")
    _pause()
