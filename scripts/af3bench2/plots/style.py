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

# --- Two-tier confidence styling (af3bench2, plan 0.4) -------------------
# Confidence tiers classify how trustworthy a prediction is:
#   ok               normal confidence
#   low_confidence   ipTM below LOW_CONF_IPTM — interpret with caution
#   likely_artifact  pTM/ipTM collapsed — exclude from grids/conc-response
LOW_CONF_IPTM = 0.40
# NOTE: the plan's literal artifact thresholds (pTM<0.15 AND ipTM<0.10) do not
# capture this dataset's collapsed cluster (pTM≈0.20, ipTM≈0.14), which sits in
# a clear gap below the healthy floor (pTM≥0.59, ipTM≥0.62).  We widen them to
# 0.30/0.20 so the obvious collapse is flagged, while staying well clear of the
# healthy group.  Tunable here in one place.
ARTIFACT_PTM = 0.30
ARTIFACT_IPTM = 0.20

TIER_HATCH = {"ok": "", "low_confidence": "//", "likely_artifact": "xxxx"}
TIER_LABEL = {
    "ok": "ok",
    "low_confidence": "low confidence (ipTM<0.40)",
    "likely_artifact": "likely artifact (collapsed)",
}


def classify_tier(ptm: float, iptm: float) -> str:
    """Return the confidence tier for one condition (plan 0.4)."""
    import math as _m
    p = ptm if (ptm is not None and _m.isfinite(ptm)) else float("nan")
    i = iptm if (iptm is not None and _m.isfinite(iptm)) else float("nan")
    if _m.isfinite(p) and _m.isfinite(i) and p < ARTIFACT_PTM and i < ARTIFACT_IPTM:
        return "likely_artifact"
    if _m.isfinite(i) and i < LOW_CONF_IPTM:
        return "low_confidence"
    if _m.isfinite(p) and p < LOW_CONF_IPTM and not _m.isfinite(i):
        return "low_confidence"
    return "ok"


# Heterogeneity tier markers (plan 3.3)
HETERO_MARKER = {"low": "●", "moderate": "◆⚠", "high": "▲⚠⚠"}
HETERO_COLOR = {"low": "#009E73", "moderate": "#E69F00", "high": "#D55E00"}

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


def short_labels(
    names: Sequence[str],
    label_map: "Dict[str, str] | None" = None,
    max_len: int = 22,
) -> List[str]:
    """
    Resolve display labels using an explicit ``label_short`` map when available
    (af3bench2 plan 2.4), falling back to common-prefix stripping.
    """
    if label_map:
        return [label_map.get(n, n) for n in names]
    return condition_labels(names, max_len=max_len)
