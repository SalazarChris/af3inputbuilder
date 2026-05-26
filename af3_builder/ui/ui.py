import os
import sys
import shutil
import textwrap
from typing import Optional, List, Tuple, Union

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------
try:
    import colorama
    colorama.init(autoreset=True)
    _COLORS = True
except ImportError:
    _COLORS = False

# Try to enable ANSI on Windows even without colorama
if sys.platform == "win32" and not _COLORS:
    import ctypes
    try:
        k32 = ctypes.windll.kernel32
        k32.SetConsoleMode(k32.GetStdHandle(-11), 7)
        _COLORS = True
    except Exception:
        pass

RESET  = "\033[0m"   if _COLORS else ""
BOLD   = "\033[1m"   if _COLORS else ""
DIM    = "\033[2m"   if _COLORS else ""
RED    = "\033[91m"  if _COLORS else ""
GREEN  = "\033[92m"  if _COLORS else ""
YELLOW = "\033[93m"  if _COLORS else ""
CYAN   = "\033[96m"  if _COLORS else ""
BLUE   = "\033[94m"  if _COLORS else ""
MAG    = "\033[95m"  if _COLORS else ""

# Width of the terminal (capped so it doesn't go wild on huge screens)
TW = min(shutil.get_terminal_size((80, 24)).columns, 100)

def _is_gui_available() -> bool:
    """Check if we can likely open a GUI window (file picker, etc)."""
    if sys.platform == "win32" or sys.platform == "darwin":
        return True # Windows/macOS usually have a display
    # On Linux/Unix, check for DISPLAY environment variable
    return os.environ.get("DISPLAY") is not None

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _rule(char="-", color=CYAN):
    print(color + char * TW + RESET)

def _banner(title: str, subtitle: str = ""):
    _rule("=", CYAN)
    print(BOLD + CYAN + title.center(TW) + RESET)
    if subtitle:
        print(DIM + subtitle.center(TW) + RESET)
    _rule("=", CYAN)

def _section(title: str):
    print()
    _rule("-", BLUE)
    print(BOLD + BLUE + f"  {title}" + RESET)
    _rule("-", BLUE)

def _ok(msg: str):
    print(GREEN + "  ✔  " + msg + RESET)

def _warn(msg: str):
    print(YELLOW + "  ⚠  " + msg + RESET)

def _err(msg: str):
    print(RED + "  ✖  " + msg + RESET)

def _info(msg: str):
    print(CYAN + "  [i]  " + msg + RESET)

def _tip(msg: str):
    """Print a highlighted tip in a box."""
    prefix = "  [*] TIP: "
    wrapped = textwrap.fill(msg, width=TW - len(prefix) - 2)
    print(MAG + prefix + wrapped + RESET)

def _divider():
    print(DIM + "  " + ". " * ((TW - 4) // 2) + RESET)

def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user, showing default in brackets, return stripped response."""
    hint = f" [{default}]" if default else ""
    full_prompt = CYAN + f"  ▶  {prompt}{hint}: " + RESET
    try:
        val = input(full_prompt).strip()
    except EOFError:
        return default
    
    if val.lower() in ("exit", "quit"):
        raise KeyboardInterrupt()

    return val if val else default

def _ask_yn(prompt: str, default: bool = True) -> bool:
    """Yes/No prompt, returns True for yes."""
    hint = "(Y/n)" if default else "(y/N)"
    raw = _ask(f"{prompt} {hint}", "y" if default else "n").lower()
    return raw in ("y", "yes", "")

def _choose(prompt: str, options: list, allow_back: bool = True, back_label: str = "Back / Cancel", default: str = None) -> str:
    """
    Show a numbered menu from 'options' (list of (key, label) tuples or just strings).
    Returns the key (or string) of the selection, or 'BACK' if user picks 0.
    0 is ALWAYS available as an escape -- even when allow_back=False.
    """
    print()
    default_idx = None
    for i, opt in enumerate(options, start=1):
        key = opt[0] if isinstance(opt, tuple) else opt
        label = opt[1] if isinstance(opt, tuple) else opt
        if default and key == default:
            default_idx = i
            print(f"  {BOLD}{CYAN}{i:>2}{RESET})  {BOLD}{label}{RESET} (default)")
        else:
            print(f"  {BOLD}{CYAN}{i:>2}{RESET})  {label}")
    
    print(f"  {BOLD}{DIM}  0{RESET})  {DIM}{back_label}{RESET}")
    print()

    while True:
        raw = _ask(prompt)
        if not raw and default_idx is not None:
            raw = str(default_idx)
            
        if raw == "0":
            return "BACK"
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx][0] if isinstance(options[idx], tuple) else options[idx]
        except ValueError:
            pass
        _err("Please enter a number from the list above.")

def _pause():
    try:
        input(DIM + "\n  Press Enter to continue..." + RESET)
    except EOFError:
        pass

def _pick_file(title: str = "Select File") -> str:
    """Open a native file picker and return the selection."""
    if not _is_gui_available():
        _err("GUI file picker is not available in this environment (headless/no DISPLAY).")
        return ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title)
        root.destroy()
        return (path or "").strip()
    except Exception as e:
        _err(f"Could not open file picker: {e}")
        return ""

def _pick_dir(title: str = "Select Directory") -> str:
    """Open a native directory picker and return the selection."""
    if not _is_gui_available():
        _err("GUI directory picker is not available in this environment (headless/no DISPLAY).")
        return ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        return (path or "").strip()
    except Exception as e:
        _err(f"Could not open directory picker: {e}")
        return ""

def _browse_text(prompt: str, start_dir: str = ".", pick_dir: bool = False) -> str:
    """A simple text-based directory explorer for headless environments."""
    curr = os.path.abspath(start_dir)
    while True:
        try:
            items = os.listdir(curr)
        except Exception as e:
            _err(f"Could not list {curr}: {e}")
            return ""
        
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(curr, d))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(curr, f))])
        
        _banner("Path Browser", os.path.relpath(curr))
        opts = [("..", f"{BOLD}{BLUE}.. (Up){RESET}")]
        for d in dirs: opts.append((os.path.join(curr, d), f"{BLUE}[DIR]  {d}{RESET}"))
        if not pick_dir:
            for f in files: opts.append((os.path.join(curr, f), f"[FILE] {f}"))
        
        choice = _choose(f"Select item to navigate/pick", opts, back_label="USE THIS DIRECTORY" if pick_dir else "Cancel")
        if choice == "BACK":
            return curr if pick_dir else ""
        if choice == "..":
            curr = os.path.dirname(curr)
            continue
        
        if os.path.isdir(choice):
            curr = choice
        else:
            return choice

def _ask_file(prompt: str, required: bool = True) -> str:
    """Prompt for an existing file path.  Type '?' to open a file picker."""
    gui = _is_gui_available()
    suffix = " (Type '?' to browse)"
    while True:
        p = _ask(f"{prompt}{suffix}").strip()
        if p == "?":
            if gui:
                p = _pick_file(prompt)
            else:
                p = _browse_text(prompt, pick_dir=False)
            
            if p: _ok(f"Selected: {p}")
            else: continue

        if not p:
            if required:
                _err("A file path is required.  Please try again.")
                continue
            return ""

        # Remove surrounding quotes
        if p.startswith('"') and p.endswith('"'): p = p[1:-1]
        elif p.startswith("'") and p.endswith("'"): p = p[1:-1]
        
        if not p: continue

        if os.path.isdir(p):
            _err("That is a folder, not a file.  Please give the full file path.")
        elif not os.path.isfile(p):
            if required:
                _err(f"File not found: {p}")
            else:
                _warn(f"File not found: {p} - skipping.")
                return ""
        else:
            return p

def _ask_dir(prompt: str, required: bool = True) -> str:
    """Prompt for an existing directory. Type '?' to open a picker."""
    gui = _is_gui_available()
    suffix = " (Type '?' to browse)"
    while True:
        p = _ask(f"{prompt}{suffix}").strip()
        if p == "?":
            if gui:
                p = _pick_dir(prompt)
            else:
                p = _browse_text(prompt, pick_dir=True)
            
            if p: _ok(f"Selected: {p}")
            else: continue

        if not p:
            if required:
                _err("A directory path is required.")
                continue
            return ""

        if p.startswith('"') and p.endswith('"'): p = p[1:-1]
        elif p.startswith("'") and p.endswith("'"): p = p[1:-1]
        
        if not p: continue

        if os.path.isfile(p):
            _err("That is a file, not a folder.")
        elif not os.path.isdir(p):
            if required:
                _err(f"Directory not found: {p}")
            else:
                _warn(f"Directory not found: {p}")
                return ""
        else:
            return p

def _ask_file_or_dir(prompt: str, required: bool = True) -> str:
    """Prompt for either a file or a directory. Type '?' to pick via GUI."""
    gui = _is_gui_available()
    suffix = " (Type '?' to browse)"
    while True:
        p = _ask(f"{prompt}{suffix}").strip()
        if p == "?":
            if gui:
                choice = _choose("What would you like to pick?", [
                    ("file", "Pick a single File"),
                    ("dir",  "Pick a Directory/Folder"),
                ], allow_back=True)
                if choice == "BACK": continue
                p = _pick_file(prompt) if choice == "file" else _pick_dir(prompt)
            else:
                p = _browse_text(prompt, pick_dir=False) # Browse for both
            
            if p: _ok(f"Selected: {p}")
            else: continue

        if not p:
            if required:
                _err("A path is required.")
                continue
            return ""

        if p.startswith('"') and p.endswith('"'): p = p[1:-1]
        elif p.startswith("'") and p.endswith("'"): p = p[1:-1]

        if not p: continue

        if not os.path.exists(p):
            if required:
                _err(f"Path not found: {p}")
            else:
                _warn(f"Path not found: {p}")
                return ""
        else:
            return p
