"""
Shared plotting style — one theme for the whole package.

Centralises DPI, output formats, fonts, the colourblind-safe palette, and a
small set of helpers so every figure looks consistent and is print-ready
(>=300 DPI plus an optional vector format).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette
PALETTE: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Semantic colours
C_DISP = "#D55E00"      # displacement curve
C_NOISE = "#56B4E9"     # baseline noise envelope
C_REF = "#0072B2"       # reference pLDDT
C_COND = "#E69F00"      # condition pLDDT
C_SIG = "#009E73"       # significant residues
C_PTM = "#CC79A7"       # PTM site marker
C_FAIL = "#D55E00"      # failed condition

# PTM-group colours (stable across factorial plots)
PTM_COLORS: Dict[str, str] = {
    "none": "#0072B2",
    "SEP102": "#CC79A7",
    "SEP": "#CC79A7",
    "TPO101": "#009E73",
    "TPO": "#009E73",
    "DNA": "#E69F00",
}

_DEFAULT_DPI = 300
_DEFAULT_FORMATS = ("png",)

_settings = {"dpi": _DEFAULT_DPI, "formats": list(_DEFAULT_FORMATS)}


def configure(dpi: int = _DEFAULT_DPI, formats: Sequence[str] = _DEFAULT_FORMATS) -> None:
    """Set global DPI and output formats, and apply rcParams."""
    _settings["dpi"] = int(dpi)
    _settings["formats"] = list(formats)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": int(dpi),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.constrained_layout.use": False,
    })


def ptm_color(group: str) -> str:
    return PTM_COLORS.get(group, "#7F7F7F")


def save(fig, plots_dir: Path, stem: str) -> List[Path]:
    """Save a figure in every configured format; returns written paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for fmt in _settings["formats"]:
        path = plots_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=_settings["dpi"], bbox_inches="tight")
        out.append(path)
    plt.close(fig)
    return out


def condition_labels(names: Sequence[str], max_len: int = 22) -> List[str]:
    """Short display labels: strip a long common prefix, then truncate."""
    names = list(names)
    if len(names) < 2:
        return [n[:max_len] for n in names]
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    if len(prefix) > 8 and all(len(n) > len(prefix) for n in names):
        labels = [n[len(prefix):] for n in names]
    else:
        labels = list(names)
    return [l[:max_len] + "..." if len(l) > max_len else l for l in labels]
