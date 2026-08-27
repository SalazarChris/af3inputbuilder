#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from typing import List

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

# Ensure af3_builder package is importable
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# AF3 Condition-Centric Extraction launcher
# ---------------------------------------------------------------------------
def run_complete_analysis_pipeline():
    """Interactive launcher for the complete AF3 analysis pipeline."""
    print()
    _rule()
    print(f"  {BOLD}AF3 Complete Analysis Pipeline{RESET}")
    print(f"  {DIM}Execute the complete workflow from raw AF3 outputs to final analysis results.{RESET}")
    _rule()
    print()
    
    # Show current directory
    cwd = os.getcwd()
    print(f"  {DIM}Working directory: {cwd}{RESET}")
    subdirs = [d for d in sorted(os.listdir(cwd)) if os.path.isdir(d) and not d.startswith('.')]
    if subdirs:
        print(f"  {DIM}Folders here: {', '.join(subdirs[:10])}{'...' if len(subdirs) > 10 else ''}{RESET}")
    print()
    
    # Get input directory
    print(f"  {CYAN}Step 1:{RESET} AF3 predictions directory")
    print(f"  {DIM}Each subfolder = one condition (pou_*, etc){RESET}")
    print(f"  {DIM}Type '?' to browse directories interactively.{RESET}")
    
    try:
        from af3_builder import _ask_dir
        input_dir = _ask_dir("Input folder (AF3 predictions)", required=True)
    except ImportError:
        print(f"  {YELLOW}Note: af3_builder not available - using basic input{RESET}")
        input_dir = _ask_input("Input folder path", "")
    
    if not input_dir:
        print(f"  {RED}Error: Input directory is required.{RESET}")
        input("\n  Press Enter to return...")
        return
    
    if not os.path.exists(input_dir):
        print(f"  {RED}Error: Directory '{input_dir}' does not exist.{RESET}")
        input("\n  Press Enter to return...")
        return
    
    # Get output directory (optional)
    print()
    print(f"  {CYAN}Step 2:{RESET} Output directory")
    print(f"  {DIM}Pipeline outputs will be saved here.{RESET}")
    default_output = os.path.join(os.path.dirname(input_dir), "outputs", os.path.basename(input_dir) + "_analysis")
    
    try:
        from af3_builder import _ask_dir
        output_dir = _ask_dir("Output folder (leave blank for default)", required=False)
    except ImportError:
        output_dir = _ask_input("Output folder (leave blank for default)", default_output)
    
    if not output_dir:
        output_dir = default_output
    
    print()
    print(f"  {CYAN}Step 3:{RESET} Run Identifier")
    run_id = _ask_input("Run ID (leave blank to auto-generate)", "")
    
    print()
    print(f"  {CYAN}Step 4:{RESET} JSON Output Options")
    save_raw_json = _ask_yn("Create raw JSON files for individual predictions?", default=False)
    save_summary_json = _ask_yn("Create extraction_summary.json and validation_report.json?", default=False)
    
    print()
    if not _ask_yn("Run complete pipeline now?", default=True):
        print(f"  {YELLOW}Cancelled.{RESET}")
        input("\n  Press Enter to return...")
        return
        
    print()
    _ok("Starting analysis pipeline...\n")
    
    # Try importing af3_analysis
    # The af3_analysis package is a sibling repository.  We locate it by
    # walking up from af3inputbuilder/ to the shared parent directory that
    # contains both af3inputbuilder/ and af3_analysis/.
    af3analysis_pkg = None  # resolved to the af3_analysis package dir
    try:
        parent_dir = os.path.dirname(_HERE)  # shared parent containing af3_analysis/
        af3analysis_pkg = os.path.normpath(
            os.path.join(parent_dir, "af3_analysis")
        )
        if os.path.isdir(af3analysis_pkg) and parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)

        from af3_analysis.config import create_config_interactive
        from af3_analysis.pipeline import run_pipeline

        config = create_config_interactive(
            raw_af3_root=input_dir,
            output_dir=output_dir,
            run_id=run_id if run_id else None,
        )

        result = run_pipeline(
            config=config,
            save_raw_json=save_raw_json,
            save_summary_json=save_summary_json,
        )

        print()
        if result.success:
            _ok(f"Pipeline completed successfully in {result.elapsed_s:.1f}s")
            print(f"  {DIM}Conditions processed: {result.n_conditions}{RESET}")
            print(f"  {DIM}Replicates processed: {result.n_replicates}{RESET}")
            print(f"  {DIM}Outputs saved to: {result.output_dir}{RESET}")
        else:
            _err(f"Pipeline failed after {result.elapsed_s:.1f}s")
            if result.errors:
                print(f"  {RED}Errors:{RESET}")
                for e in result.errors:
                    print(f"  - {e}")

    except ImportError as e:
        _err(f"Could not load af3_analysis package: {e}")
        if af3analysis_pkg:
            print(f"  {DIM}Searched in: {af3analysis_pkg}{RESET}")
        print(f"  {DIM}Ensure af3_analysis is installed: pip install -e <path>/af3_analysis{RESET}")
    except Exception as e:
        import traceback
        _err(f"Pipeline crashed: {e}")
        print(f"  {DIM}{traceback.format_exc()}{RESET}")
        
    input("\n  Press Enter to return...")


# ---------------------------------------------------------------------------
# Preferred Python for analysis sub-processes (af3_thesis conda env).
# This ensures analysis functions always use the environment that has all
# required packages (pyarrow, gemmi, scipy, etc.) regardless of which
# Python was used to launch af3.py itself.
# ---------------------------------------------------------------------------
def _find_analysis_python() -> str:
    """Locate the af3_thesis conda environment Python, or fall back to sys.executable."""
    import platform
    if platform.system() == "Windows":
        candidates = [
            r"C:\Users\Chris\.conda\envs\af3_thesis\python.exe",
        ]
    else:
        # Linux server: common conda/miniconda paths
        home = os.path.expanduser("~")
        candidates = [
            os.path.join(home, ".conda", "envs", "af3_thesis", "bin", "python"),
            os.path.join(home, "miniconda3", "envs", "af3_thesis", "bin", "python"),
            os.path.join(home, "anaconda3", "envs", "af3_thesis", "bin", "python"),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return sys.executable

_ANALYSIS_PYTHON = _find_analysis_python()

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
    print(f"      {DIM}Fix one entity (e.g. protein) and vary a second entity (sequence, ligand, or ion) across N output JSONs.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}4{RESET})  {BOLD}JSON Validator{RESET}")
    print(f"      {DIM}Check job files for schema errors and compatibility issues.{RESET}")
    print()
    print(f"  {BOLD}{CYAN}5{RESET})  {BOLD}Analysis Pipeline{RESET}")
    print(f"      {DIM}Run the complete AF3 analysis workflow (extraction, QC, visualization, and statistical analysis).{RESET}")
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



# ---------------------------------------------------------------------------
# Module launcher (for tools that live in scripts/)
# ---------------------------------------------------------------------------

def _launch(module_name: str, entry_fn: str, interactive_args: List[str] = None):
    """Import module_name from scripts/ and call entry_fn()."""
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        entry = getattr(mod, entry_fn, None)
        if entry:
            if interactive_args:
                # For interactive mode, we need to modify sys.argv
                import sys
                old_argv = sys.argv
                sys.argv = [module_name] + interactive_args
                entry()
                sys.argv = old_argv
            else:
                entry()
        else:
            _err(f"Module {module_name} has no function '{entry_fn}'.")
    except ImportError as exc:
        _err(f"Could not load {module_name}: {exc}")
        input("\n  Press Enter to return...")
    except KeyboardInterrupt:
        print(f"\n{YELLOW}  [!]  Cancelled.{RESET}")
    except SystemExit:
        # Don't print traceback for normal exits
        pass
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
            run_complete_analysis_pipeline()
            _pause()
        elif choice:
            _err("Invalid option.")
            import time; time.sleep(0.8)

    print(f"\n{GREEN}  Goodbye!{RESET}\n")


if __name__ == "__main__":
    main()
