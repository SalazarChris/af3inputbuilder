#!/usr/bin/env python3
"""
msa_wizard.py
=============
Interactive wizard for extracting MSAs from AlphaFold 3 results.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: shared UI from af3_builder
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from af3_builder import (
        RESET, BOLD, DIM, RED, GREEN, YELLOW, CYAN, BLUE, MAG, TW,
        _rule, _banner, _section, _ok, _warn, _err, _info, _tip, _divider,
        _ask, _ask_yn, _choose, _pause, _ask_dir,
    )
except ImportError:
    # Minimal fallback
    RESET = BOLD = DIM = RED = GREEN = YELLOW = CYAN = BLUE = MAG = ""
    TW = 80
    def _rule(c="=", col=""): print(c * TW)
    def _banner(t, s=""): _rule(); print(t.center(TW)); _rule()
    def _section(t): print(f"\n{BOLD}=== {t} ==={RESET}")
    def _ok(m): print(f"  [OK] {m}")
    def _warn(m): print(f"  [!] {m}")
    def _err(m): print(f"  [ERR] {m}")
    def _info(m): print(f"  [i] {m}")
    def _tip(m): print(f"  (Tip: {m})")
    def _divider(): print("-" * TW)
    def _ask(p, default=""): return input(f"{p} [{default}]: ") or default
    def _ask_yn(p, default=True): return input(f"{p} (y/n) [{'y' if default else 'n'}]: ").lower().startswith('y')
    def _choose(p, opts, **kwargs):
        for k, v in opts: print(f"  {k}) {v}")
        return input(f"{p}: ")
    def _pause(): input("Press Enter...")
    def _ask_dir(p, **kwargs): return input(f"{p}: ")

from msa_extractor import extract_msas

def welcome():
    os.system("cls" if sys.platform == "win32" else "clear")
    _banner(
        "AlphaFold 3  ·  MSA Extractor Wizard",
        "Extract paired and unpaired MSAs from your AF3 results"
    )
    print()
    _tip(
        "This tool scans your AlphaFold 3 output folders for JSON result files\n"
        "     and saves the embedded MSAs as .a3m files for downstream use.\n"
        "     Press Ctrl+C at any time to exit."
    )
    print()

def main():
    welcome()

    _section("Step 1: Results Source")
    _tip("Select the folder containing your AlphaFold 3 results (look for _data.json files).")
    input_dir = _ask_dir("Folder to scan", required=True)
    if not os.path.isdir(input_dir):
        _err(f"Directory not found: {input_dir}")
        _pause()
        return

    _section("Step 2: Output Destination")
    _tip("Where should the extracted .a3m files be saved?")
    output_dir = _ask("Output folder name", default="extracted_msas")

    _section("Execution")
    if _ask_yn(f"Extract MSAs from '{input_dir}' into '{output_dir}'?", default=True):
        print()
        _info("Scanning for results and extracting MSAs...")
        
        try:
            results = extract_msas(input_dir, output_dir)
            
            if results:
                _ok(f"Successfully extracted {len(results)} MSA files.")
                _divider()
                # Show a small summary
                max_show = 10
                for job_id, msa_type in results[:max_show]:
                    print(f"    {GREEN}•{RESET}  {job_id:<20} | {msa_type}")
                if len(results) > max_show:
                    print(f"    ... and {len(results) - max_show} more.")
                _divider()
                _info(f"Files are located in: {os.path.abspath(output_dir)}")
            else:
                _warn("No MSA data found in the specified folder.")
                _info("Make sure the folder contains '*_data.json' files from an AF3 run.")
                
        except Exception as e:
            _err(f"An unexpected error occurred: {e}")
    else:
        _info("Extraction cancelled.")

    _pause()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}  ⚠  Wizard interrupted.{RESET}")
        sys.exit(0)
