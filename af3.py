#!/usr/bin/env python3
"""
AlphaFold 3 Master Toolkit
==========================
Unified CLI for building and analysing AlphaFold 3 experiments.

Author: Chris (Master's Thesis)
Version: 2.1.0
Date: May 2026

Usage:
    python af3.py
"""

import sys
import os
import importlib
import subprocess

# ---------------------------------------------------------------------------
# Bootstrap: add the project root to path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Scripts directory
_SCRIPTS = os.path.join(_HERE, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# ---------------------------------------------------------------------------
# Dependency check -- auto-install missing packages on first run
# ---------------------------------------------------------------------------
_REQUIRED_PACKAGES = [
    ("numpy",      "numpy>=1.24.0"),
    ("pandas",     "pandas>=2.0.0"),
    ("scipy",      "scipy>=1.10.0"),
    ("matplotlib", "matplotlib>=3.7.0"),
    ("Bio",        "biopython>=1.81"),
    ("tmtools",    "tmtools"),
    ("sklearn",    "scikit-learn"),
]


def _check_and_install_deps():
    """Check required packages on startup; offer to auto-install via pip if missing."""
    import importlib as _il

    # ------------------------------------------------------------------ #
    # 1. Required packages via pip                                         #
    # ------------------------------------------------------------------ #
    missing_pip = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            _il.import_module(import_name)
        except ImportError:
            missing_pip.append((import_name, pip_name))

    if missing_pip:
        print("\n" + "=" * 60)
        print("  AF3 Toolkit -- Missing Dependencies")
        print("=" * 60)
        for name, pkg in missing_pip:
            print(f"  \033[91m✖\033[0m  {name}  ({pkg})")
        print(f"\n  {len(missing_pip)} package(s) need to be installed.\n")

        try:
            answer = input("  Install via pip now? (Y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            answer = "n"

        if answer in ("", "y", "yes"):
            pip_pkgs = [pkg for _, pkg in missing_pip]
            cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pip_pkgs
            print(f"\n  Installing: {', '.join(pip_pkgs)}...")
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print("  \033[92m✔  Installed.\033[0m\n")
            else:
                print("  \033[91m✖  pip install failed.\033[0m")
                print(f"  Try manually:  pip install {' '.join(pip_pkgs)}\n")
            for import_name, _ in missing_pip:
                try:
                    _il.import_module(import_name)
                except ImportError:
                    pass
        else:
            print("\n  Skipped. Some tools may not work.\n")


_check_and_install_deps()

# ---------------------------------------------------------------------------
# UI helpers (inline fallback -- no external deps needed for the menu)
# ---------------------------------------------------------------------------
try:
    from af3_builder import (
        RESET, BOLD, DIM, RED, GREEN, CYAN, YELLOW,
        _rule, _banner, _ok, _err, _pause,
    )
except ImportError:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    def _rule(char="=", color=""): print(char * 60)
    def _banner(t, s=""): _rule(); print(t.center(60)); _rule()
    def _ok(m):  print("  ✔  " + m)
    def _err(m): print("  ✖  " + m)
    def _pause(): input("\n  Press Enter to continue...")


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

def print_menu():
    os.system("cls" if os.name == "nt" else "clear")
    _banner(
        "AlphaFold 3  -  Master Toolkit",
        "Build jobs, run analysis, extract MSAs"
    )
    print()
    print(f"  {BOLD}{CYAN}1{RESET})  {BOLD}Job Builder{RESET}")
    print(f"      {DIM}Guided multi-entity setup for AF3 input JSONs.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}2{RESET})  {BOLD}MSA Extractor{RESET}")
    print(f"      {DIM}Extract MSAs from AF3 result JSONs for re-runs.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}3{RESET})  {BOLD}Ion / Ligand Sweep{RESET}")
    print(f"      {DIM}Generate concentration-sweep or library-screen job files.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}4{RESET})  {BOLD}JSON Validator{RESET}")
    print(f"      {DIM}Check job files for schema errors and compatibility issues.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}5{RESET})  {BOLD}Analysis Pipeline{RESET}")
    print(f"      {DIM}Structural comparison with smart defaults + optional customization.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}0{RESET})  {BOLD}Exit{RESET}")
    print()
    print(f"  {DIM}(Ctrl+C to exit at any time){RESET}")
    print()


# ---------------------------------------------------------------------------
# Analysis wizard (inline -- no separate wizard file needed on the server)
# ---------------------------------------------------------------------------

def _ask_input(prompt: str, default: str = "") -> str:
    """Simple input with default value display."""
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    return val if val else default


def _ask_number(prompt: str, default: float, min_val: float = 0.0, max_val: float = None) -> float:
    """Ask for a number with validation."""
    while True:
        try:
            val = _ask_input(prompt, str(default))
            if val == "":
                return default
            num = float(val)
            if num < min_val:
                print(f"  {RED}Value must be ≥ {min_val}{RESET}")
                continue
            if max_val is not None and num > max_val:
                print(f"  {RED}Value must be ≤ {max_val}{RESET}")
                continue
            return num
        except ValueError:
            print(f"  {RED}Please enter a valid number{RESET}")


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        val = input(f"  {prompt} ({hint}): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return default
    if val in ("y", "yes"):
        return True
    if val in ("n", "no"):
        return False
    return default


def _check_tmtools() -> bool:
    """Check if tmtools is available for TM-score computation."""
    try:
        import importlib as _il
        _il.import_module("tmtools")
        return True
    except ImportError:
        return False


def _show_analysis_summary(models: str, baseline: str, output: str, options: dict):
    """Display what will be generated before running."""
    print()
    _rule("-")
    print(f"  {BOLD}Analysis Summary{RESET}")
    _rule("-")
    print(f"  Models folder:   {models}")
    print(f"  Baseline:        {baseline if baseline else 'auto-detect'}")
    print(f"  Output folder:   {output}")
    print()
    print(f"  {CYAN}Statistics{RESET}")
    print(f"  • pLDDT cutoff:      {options['plddt_cutoff']}")
    print(f"  • Bootstrap resamples: {options['n_bootstrap']}")
    print(f"  • FDR threshold:    {options['fdr_alpha']}")
    print()
    print(f"  {CYAN}Output{RESET}")
    print(f"  • PyMOL scripts:    {'Yes' if options['pymol'] else 'No'}")
    print(f"  • TM-score:         {'Yes' if options['tm'] else 'No'}")
    print(f"  • Figures DPI:      {options['dpi']}")
    print(f"  • Figure formats:   {', '.join(options['formats'])}")
    print()
    print(f"  {CYAN}What will be generated:{RESET}")
    print(f"  • Structural distances vs baseline table")
    print(f"  • Per-residue displacement profiles")
    print(f"  • Ensemble confidence intervals")
    print(f"  • Factorial design plots (if applicable)")
    print(f"  • Cluster analysis of conditions")
    if options['pymol']:
        print(f"  • PyMOL visualization scripts")
    _rule("-")


def run_analysis():
    """Unified analysis pipeline with smart defaults + optional customization."""
    from af3_builder import _ask_dir as _ask_directory

    print()
    _rule()
    print(f"  {BOLD}Analysis Pipeline{RESET}")
    print(f"  {DIM}Smart defaults with optional parameter customization{RESET}")
    _rule()
    print()

    # Show current working directory for context
    cwd = os.getcwd()
    print(f"  {DIM}Working directory: {cwd}{RESET}")
    subdirs = [d for d in sorted(os.listdir(cwd)) if os.path.isdir(d) and not d.startswith('.')]
    if subdirs:
        print(f"  {DIM}Folders here: {', '.join(subdirs[:10])}{'...' if len(subdirs) > 10 else ''}{RESET}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Models directory (always required)
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 1:{RESET} AF3 output directory")
    print(f"  {DIM}Each immediate subfolder = one condition.{RESET}")
    print(f"  {DIM}Type '?' to browse directories interactively.{RESET}")
    models = _ask_directory("Models folder", required=True)
    if not models:
        return

    # Validate models directory exists
    if not os.path.exists(models):
        print(f"  {RED}Error: Directory '{models}' does not exist.{RESET}")
        return

    condition_dirs = sorted(
        d for d in os.listdir(models)
        if os.path.isdir(os.path.join(models, d)) and not d.startswith('.')
    )
    if not condition_dirs:
        print(f"  {RED}Error: No condition subdirectories found in '{models}'{RESET}")
        return

    preview = ', '.join(condition_dirs[:8])
    suffix  = '...' if len(condition_dirs) > 8 else ''
    print(f"  {DIM}Found {len(condition_dirs)} condition(s): {preview}{suffix}{RESET}")
    print()

    # ------------------------------------------------------------------
    # Set smart defaults (same as old quick mode)
    # ------------------------------------------------------------------
    baseline = ""
    chains = ""
    output = "bench_results"
    pymol = True
    tm = _check_tmtools()  # Use TM-score if available
    plddt_cutoff = 50.0
    n_bootstrap = 2000
    fdr_alpha = 0.05
    max_samples = None
    dpi = 300
    formats = ["png", "pdf"]
    cluster_threshold = 3.0

    # ------------------------------------------------------------------
    # Ask if user wants to customize parameters
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 2:{RESET} Parameter Configuration")
    print(f"  {DIM}The pipeline uses smart defaults for all parameters.{RESET}")
    customize = _ask_yn("Do you want to customize any parameters?", default=False)
    
    if customize:
        print()
        print(f"  {YELLOW}Customizing parameters...{RESET}")
        print()
        
        # ------------------------------------------------------------------
        # Step 2: Baseline condition (optional)
        # ------------------------------------------------------------------
        print(f"  {CYAN}Baseline:{RESET} Condition selection")
        print(f"  {DIM}Available: {', '.join(condition_dirs)}{RESET}")
        print(f"  {DIM}Leave blank to auto-detect (fewest ions/solvent, no PTMs).{RESET}")
        baseline = _ask_input("Baseline condition name", "")
        print()

        # ------------------------------------------------------------------
        # Step 3: Chain filter (optional)
        # ------------------------------------------------------------------
        print(f"  {CYAN}Chain Filter:{RESET} (optional)")
        print(f"  {DIM}Alignment uses protein Cα only. Restrict to specific chains")
        print(f"  when comparing subunits of multi-chain complexes.{RESET}")
        print(f"  {DIM}Example: Enter 'A,B' to compare only chains A and B{RESET}")
        print(f"  {DIM}Leave blank to use all protein chains.{RESET}")
        chains = _ask_input("Chain IDs (comma-separated, or blank for all)", "")
        print()

        # ------------------------------------------------------------------
        # Step 4: Output directory
        # ------------------------------------------------------------------
        print(f"  {CYAN}Output Directory:{RESET}")
        output = _ask_input("Output folder", output)
        print()

        # ------------------------------------------------------------------
        # Step 5: Analysis Options
        # ------------------------------------------------------------------
        print(f"  {CYAN}Analysis Options:{RESET}")
        
        # Check TM-tools availability
        tm_available = _check_tmtools()
        if tm_available:
            tm = _ask_yn("Compute TM-score? (recommended for structural clustering)", default=tm)
        else:
            print(f"  {YELLOW}Note: tmtools not installed. TM-score will be skipped.{RESET}")
            print(f"  {DIM}Install with: pip install tmtools{RESET}")
            tm = False
        
        pymol = _ask_yn("Generate PyMOL .pml scripts?", default=pymol)
        print()

        # ------------------------------------------------------------------
        # Step 6: Statistical Parameters
        # ------------------------------------------------------------------
        print(f"  {CYAN}Statistical Parameters:{RESET}")
        
        plddt_cutoff = _ask_number("pLDDT cutoff for fitting (50-100)", default=plddt_cutoff, min_val=0, max_val=100)
        print(f"  {DIM}Residues with pLDDT < {plddt_cutoff} excluded from alignment{RESET}")
        
        n_bootstrap = int(_ask_number("Bootstrap resamples for CIs", default=n_bootstrap, min_val=100, max_val=10000))
        print(f"  {DIM}Higher = more precise confidence intervals{RESET}")
        
        fdr_alpha = _ask_number("FDR threshold for significance", default=fdr_alpha, min_val=0.001, max_val=0.5)
        print(f"  {DIM}Lower = more conservative significance calls{RESET}")
        
        max_samples_input = _ask_input("Max samples per condition (blank for all)", "")
        max_samples = int(max_samples_input) if max_samples_input.strip() else None
        if max_samples:
            print(f"  {DIM}Will use at most {max_samples} samples per condition{RESET}")
        print()

        # ------------------------------------------------------------------
        # Step 7: Plotting Options
        # ------------------------------------------------------------------
        print(f"  {CYAN}Plotting Options:{RESET}")
        
        dpi = int(_ask_number("Figure DPI (72-600)", default=dpi, min_val=72, max_val=600))
        print(f"  {DIM}Higher DPI = better print quality{RESET}")
        
        formats_input = _ask_input("Figure formats (comma-separated)", ",".join(formats))
        formats = [f.strip() for f in formats_input.split(",") if f.strip()]
        print(f"  {DIM}Will save figures as: {', '.join(formats)}{RESET}")
        
        cluster_threshold = _ask_number("RMSD cut height for clustering (Å)", default=cluster_threshold, min_val=0.5, max_val=10.0)
        print(f"  {DIM}Conditions with RMSD < {cluster_threshold}Å will be grouped{RESET}")
        print()
    else:
        print(f"  {GREEN}Using smart defaults for all parameters.{RESET}")
        print()

    # ------------------------------------------------------------------
    # Show summary and confirm
    # ------------------------------------------------------------------
    options = {
        'pymol': pymol, 'tm': tm, 'plddt_cutoff': plddt_cutoff,
        'n_bootstrap': n_bootstrap, 'fdr_alpha': fdr_alpha,
        'max_samples': max_samples, 'dpi': dpi, 'formats': formats,
        'cluster_threshold': cluster_threshold
    }
    
    _show_analysis_summary(models, baseline, output, options)
    
    if not _ask_yn("Run analysis with these settings?", default=True):
        print(f"  {YELLOW}Cancelled.{RESET}")
        return

    # ------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------
    cmd = [sys.executable, os.path.join(_SCRIPTS, "af3_analysis.py")]
    cmd += ["--models", models]
    cmd += ["--output", output]

    if baseline:
        cmd += ["--baseline", baseline]

    if chains:
        cmd += ["--chains", chains]
    if pymol:
        cmd.append("--pymol")
    if tm:
        cmd.append("--tm")
    
    # Add advanced options
    cmd += ["--plddt-cutoff", str(plddt_cutoff)]
    cmd += ["--n-bootstrap", str(n_bootstrap)]
    cmd += ["--fdr", str(fdr_alpha)]
    cmd += ["--dpi", str(dpi)]
    cmd += ["--formats", ",".join(formats)]
    cmd += ["--cluster-threshold", str(cluster_threshold)]
    
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]

    _rule()
    print(f"  {BOLD}Command:{RESET}")
    print(f"  {DIM}{' '.join(cmd)}{RESET}")
    _rule()
    print()

    if not _ask_yn("Run now?", default=True):
        print(f"  {YELLOW}Cancelled.{RESET}")
        return

    print()
    _ok("Starting pipeline...\n")
    result = subprocess.run(cmd)
    print()
    if result.returncode == 0:
        _ok(f"Done! Results in: {os.path.abspath(output)}")
        print(f"  {DIM}Files generated:{RESET}")
        print(f"  {DIM}• {output}/tables/ - CSV tables with statistics{RESET}")
        print(f"  {DIM}• {output}/plots/ - Publication-quality figures{RESET}")
        if pymol:
            print(f"  {DIM}• {output}/*.pml - PyMOL visualization scripts{RESET}")
        print(f"  {DIM}• {output}/findings.md - Summary report{RESET}")
    else:
        _err(f"Pipeline exited with code {result.returncode}")





# ---------------------------------------------------------------------------
# Module launcher (for tools that live in scripts/)
# ---------------------------------------------------------------------------

def _launch(module_name: str, entry_fn: str):
    """Import module_name from scripts/ and call entry_fn()."""
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        entry = getattr(mod, entry_fn, None)
        if entry:
            entry()
        else:
            _err(f"Module {module_name} has no function '{entry_fn}'.")
    except ImportError as exc:
        _err(f"Could not load {module_name}: {exc}")
        input("\n  Press Enter to return...")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  [!]  Cancelled.{RESET}")
    except Exception as exc:
        import traceback
        _err(f"{module_name} crashed: {exc}")
        print(DIM + traceback.format_exc() + RESET)
        input("\n  Press Enter to return...")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    while True:
        print_menu()
        try:
            choice = input(f"{CYAN}  ▶  Select:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ("0", "exit", "quit", "q"):
            break
        elif choice == "1":
            _launch("af3_wizard", "run_wizard")
            _pause()
        elif choice == "2":
            _launch("msa_wizard", "main")
            _pause()
        elif choice == "3":
            _launch("add_ions_wizard", "run_wizard")
            _pause()
        elif choice == "4":
            _launch("af3_json_validator", "main")
            _pause()
        elif choice == "5":
            run_analysis()
            _pause()
        elif choice:
            _err("Invalid option.")
            import time; time.sleep(0.8)

    print(f"\n{GREEN}  Goodbye!{RESET}\n")


if __name__ == "__main__":
    main()
