#!/usr/bin/env python3
"""
AlphaFold 3 Master Toolkit
==========================
Unified CLI for building and analysing AlphaFold 3 experiments.

Author: Chris (Master's Thesis)
Version: 2.0.0
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
]

_OPTIONAL_PACKAGES = [
    ("pymol2",     "pymol-open-source"),  # conda only
]


def _check_and_install_deps():
    """Check required packages on startup; offer to auto-install if missing."""
    import importlib as _il
    missing = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            _il.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))

    if not missing:
        return True

    print("\n" + "=" * 60)
    print("  AF3 Toolkit -- Missing Dependencies")
    print("=" * 60)
    for name, pkg in missing:
        print(f"  \033[91m✖\033[0m  {name} ({pkg})")
    print(f"\n  {len(missing)} package(s) need to be installed.\n")

    try:
        answer = input("  Install now? (Y/n): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False

    if answer not in ("", "y", "yes"):
        print("\n  Skipped. Some tools may not work.\n")
        return False

    pip_packages = [pkg for _, pkg in missing]
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pip_packages
    print(f"\n  Installing: {', '.join(pip_packages)}...")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("  \033[92m✔  Installed.\033[0m\n")
    else:
        print(f"  \033[91m✖  pip install failed.\033[0m")
        print(f"  Try manually: pip install {' '.join(pip_packages)}\n")

    # Re-import
    for import_name, _ in missing:
        try:
            _il.import_module(import_name)
        except ImportError:
            pass
    return True


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
    print(f"  {BOLD}{CYAN}2{RESET})  {BOLD}Analysis Pipeline{RESET}")
    print(f"      {DIM}Structural comparison across AF3 conditions (baseline or all-vs-all).{RESET}")
    print()
    print(f"  {BOLD}{CYAN}3{RESET})  {BOLD}Ion / Ligand Sweep{RESET}")
    print(f"      {DIM}Generate concentration-sweep or library-screen job files.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}4{RESET})  {BOLD}JSON Validator{RESET}")
    print(f"      {DIM}Check job files for schema errors and compatibility issues.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}5{RESET})  {BOLD}MSA Extractor{RESET}")
    print(f"      {DIM}Extract MSAs from AF3 result JSONs for re-runs.{RESET}")
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


def run_analysis(quick: bool = False):
    """Interactive wizard that collects paths and launches af3_bench.py."""
    from af3_builder import _ask_dir as _ask_directory

    print()
    _rule()
    print(f"  {BOLD}Analysis Pipeline{RESET}")
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
    # Step 1: Models directory
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 1:{RESET} AF3 output directory")
    print(f"  {DIM}Each immediate subfolder = one condition.")
    print(f"  Type '?' to browse directories interactively.{RESET}")
    models = _ask_directory("Models folder", required=True)
    if not models:
        return

    condition_dirs = sorted(
        d for d in os.listdir(models)
        if os.path.isdir(os.path.join(models, d)) and not d.startswith('.')
    )
    if condition_dirs:
        preview = ', '.join(condition_dirs[:8])
        suffix  = '...' if len(condition_dirs) > 8 else ''
        print(f"  {DIM}Found {len(condition_dirs)} condition(s): {preview}{suffix}{RESET}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Baseline condition
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 2:{RESET} Baseline condition")
    if condition_dirs:
        print(f"  {DIM}Available: {', '.join(condition_dirs)}{RESET}")
    print(f"  {DIM}Leave blank to auto-detect (fewest ions/solvent, no PTMs).{RESET}")
    baseline = _ask_input("Baseline condition name", "")
    print()

    # ------------------------------------------------------------------
    # Step 3: Chain filter
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 3:{RESET} Chain filter  {DIM}(optional){RESET}")
    print(f"  {DIM}Alignment uses protein Cα only. Use this to restrict to specific")
    print(f"  chain IDs when comparing a subunit of a multi-chain complex.")
    print(f"  Example: A,B   Leave blank to use all protein chains.{RESET}")
    chains = _ask_input("Chain IDs (comma-separated, or blank for all)", "")
    print()

    # ------------------------------------------------------------------
    # Step 4: Output directory
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 4:{RESET} Output directory")
    output = _ask_input("Output folder", "bench_results")
    print()

    # ------------------------------------------------------------------
    # Step 5: Options
    # ------------------------------------------------------------------
    print(f"  {CYAN}Step 5:{RESET} Options")
    pymol  = _ask_yn("Generate PyMOL .pml scripts?", default=True)
    tm     = _ask_yn("Compute TM-score? (requires tmtools, adds compute time)", default=False)
    print()

    # ------------------------------------------------------------------
    # Build command
    # ------------------------------------------------------------------
    cmd = [sys.executable, os.path.join(_SCRIPTS, "af3_bench.py")]
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
            run_analysis()
            _pause()
        elif choice == "3":
            _launch("add_ions_wizard", "run_wizard")
            _pause()
        elif choice == "4":
            _launch("af3_json_validator", "main")
            _pause()
        elif choice == "5":
            _launch("msa_wizard", "main")
            _pause()
        elif choice:
            _err("Invalid option.")
            import time; time.sleep(0.8)

    print(f"\n{GREEN}  Goodbye!{RESET}\n")


if __name__ == "__main__":
    main()
