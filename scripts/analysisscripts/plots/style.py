"""
Shared plotting style — one theme for the whole package.

Centralises DPI, output formats, fonts, the colourblind-safe palette, and a
small set of helpers so every figure looks consistent and is print-ready
(>=300 DPI plus an optional vector format).
"""

from __future__ import annotations

import re
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

# Import collapsed detection for dynamic checking
try:
    from ..collapsed_detection import detect_collapsed_conditions
    HAS_COLLAPSED_DETECTION = True
except ImportError:
    HAS_COLLAPSED_DETECTION = False

# Display labels come from the per-condition ``label_short`` (built by
# factors.build_experiment_structure) passed in as ``label_map``; this global
# fallback is intentionally empty so no dataset-specific names are hardcoded.
DISPLAY_NAMES: Dict[str, str] = {}

# Artifact/collapse status is determined data-drivenly per run
# (collapsed_detection.detect_collapsed_conditions), not from a hardcoded set.
# Kept as an empty set so legacy references resolve without injecting
# dataset-specific assumptions.
FAILED_CONDITIONS: set = set()

# PTM-group colours (stable across factorial plots) - updated for consistency (§ 9)
PTM_COLORS: Dict[str, str] = {
    "unmodified": "#4C72B0",   # blue
    "SEP102": "#DD8452",       # orange
    "SEP": "#DD8452",          # orange (alias)
    "TPO101": "#55A868",       # green
    "TPO": "#55A868",          # green (alias)
    "DNA": "#C44E52",          # red
    "failed": "#AAAAAA",       # grey (overrides PTM colour)
    # Legacy aliases for backward compatibility
    "none": "#4C72B0",
}

# --- Confidence styling (af3bench2) -------------------------------------
# Confidence tiers classify how trustworthy a prediction is.  They are based on
# MACROMOLECULE-SCOPED observables only — protein mean pLDDT and the
# macromolecule-scoped mean PAE (protein+nucleic tokens) — never full-system
# pTM/ipTM.  In AF3 every ion/water is its own token, so full-system pTM/ipTM
# are deflated purely by solvent count and would mislabel confidently folded,
# heavily solvated conditions as low-confidence (see collapsed_detection.py).
#
# Thresholds are the AlphaFold per-residue pLDDT confidence bands
# (Jumper et al. 2021; AlphaFold-DB guidance):
#   pLDDT > 90 very high · 70–90 confident · 50–70 low · < 50 very low
#   ok               protein mean pLDDT >= 70 (confident fold)
#   low_confidence   protein mean pLDDT < 70 (interpret with caution)
#   likely_artifact  protein mean pLDDT < 50 AND macromolecule PAE > 25 Å
#                    (consistent with detect_collapsed_conditions)
LOW_CONF_PLDDT = 70.0
ARTIFACT_PLDDT = 50.0
ARTIFACT_PAE = 25.0

TIER_HATCH = {"ok": "", "low_confidence": "//", "likely_artifact": "xxxx"}
TIER_LABEL = {
    "ok": "ok (pLDDT≥70)",
    "low_confidence": "low confidence (pLDDT<70)",
    "likely_artifact": "likely artifact (pLDDT<50 & PAE>25Å)",
}


def classify_tier(plddt_mean: float, mean_pae: float) -> str:
    """Return the confidence tier for one condition from macromolecule-scoped
    observables (protein mean pLDDT and macromolecule-scoped mean PAE)."""
    import math as _m
    pl = plddt_mean if (plddt_mean is not None and _m.isfinite(plddt_mean)) else float("nan")
    pae = mean_pae if (mean_pae is not None and _m.isfinite(mean_pae)) else float("nan")
    if not _m.isfinite(pl):
        return "ok"  # no protein confidence available — cannot down-tier
    if pl < ARTIFACT_PLDDT and _m.isfinite(pae) and pae > ARTIFACT_PAE:
        return "likely_artifact"
    if pl < LOW_CONF_PLDDT:
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


# ----------------------------------------------------------------------------
# Minimum font sizes for dense multi-panel layouts (Fix 1).  Centralised so
# every figure references the same legibility floor at high DPI / reduced zoom.
# ----------------------------------------------------------------------------
FS_SUBPLOT_TITLE = 9   # subplot titles
FS_AXIS_LABEL = 8      # axis labels
FS_TICK_LABEL = 7      # tick labels
FS_ANNOTATION = 7      # value annotations on bars / heatmap cells


def ptm_color(group: str) -> str:
    return PTM_COLORS.get(group, "#7F7F7F")


def save(fig, plots_dir: Path, stem: str, metadata: "Dict[str, str] | None" = None) -> List[Path]:
    """Save a figure in every configured format; returns written paths.

    ``metadata`` (when supported by the format, e.g. PNG/SVG/PDF) is embedded so
    information such as the full-condition-name mapping survives in the file
    (Fix 6) without cluttering the axes.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: List[Path] = []
    for fmt in _settings["formats"]:
        path = plots_dir / f"{stem}.{fmt}"
        try:
            fig.savefig(path, dpi=_settings["dpi"], bbox_inches="tight",
                        metadata=metadata if metadata else None)
        except (TypeError, ValueError):
            # Some backends/formats reject the metadata dict; save without it.
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
    # Use the global display names as fallback
    return [DISPLAY_NAMES.get(n, n) for n in names]


# Regex for the factorial salt block: _nax{N}_clx{N}_smilesx{M}
_SALT_BLOCK_RE = re.compile(r"_?nax(\d+)_clx\d+(?:_smilesx\d+)?", re.IGNORECASE)


def short_condition_label(
    name: str,
    ion_tier: "Dict[str, str] | None" = None,
) -> str:
    """
    Compact, axis-friendly condition label (Fix 6).

    Transforms a full condition name such as
    ``oct4_seg_chain_b_sep102_nax10_clx10_smilesx100_no_msa`` into a short
    factor string like ``SEP102_10x``.

    Steps:
      (a) strip the ``oct4_seg_chain_b_`` / ``oct4_seg_chainb_`` prefix,
      (b) strip the ``_no_msa`` suffix,
      (c) replace ``_nax{N}_clx{N}_smilesx{M}`` with a salt-tier label.  When an
          ``ion_tier`` map (the authoritative mapping computed in factors.py) is
          supplied it is used directly; otherwise the tier is derived from the
          ``nax{N}`` multiplier in the name,
      (d) replace the PTM/DNA component with its short form
          (``sep102`` -> ``SEP102``, ``tpo101`` -> ``TPO101``,
           ``dna`` -> ``DNA``, absence -> ``unmod``).

    The full condition name is never altered in the data; this only affects the
    rendered label.
    """
    stem = name
    for prefix in ("oct4_seg_chain_b_", "oct4_seg_chainb_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    if stem.endswith("_no_msa"):
        stem = stem[: -len("_no_msa")]

    # Authoritative salt tier (factors.py) if available, else parse from name.
    tier = ion_tier.get(name) if ion_tier else None
    if tier is None:
        m = _SALT_BLOCK_RE.search(stem)
        if m:
            tier = f"{int(m.group(1))}x"

    # Remove the salt block from the stem so only PTM/DNA tokens remain.
    stem = _SALT_BLOCK_RE.sub("", stem)
    stem = stem.strip("_")

    tokens = [t for t in stem.split("_") if t]
    factors: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low == "sep102":
            factors.append("SEP102")
        elif low == "tpo101":
            factors.append("TPO101")
        elif low == "dna":
            factors.append("DNA")
        elif low == "ions":
            factors.append("ions")
        else:
            factors.append(tok)

    if not factors:
        factors = ["unmod"]
    elif "DNA" not in factors and "SEP102" not in factors and "TPO101" not in factors:
        # No recognised PTM/DNA token -> unmodified background.
        factors = ["unmod"] + factors if factors != ["unmod"] else ["unmod"]

    label = "+".join(factors)
    if tier:
        label = f"{label}_{tier}"
    return label


def short_condition_labels(
    names: Sequence[str],
    ion_tier: "Dict[str, str] | None" = None,
) -> List[str]:
    """Vectorised :func:`short_condition_label` (Fix 6)."""
    return [short_condition_label(n, ion_tier) for n in names]


def is_failed_condition(condition_name: str) -> bool:
    """Check if a condition is in the failed predictions set (§ 0)."""
    # For now, use the hardcoded set; will be updated when detection is integrated
    return condition_name in FAILED_CONDITIONS

def is_collapsed_condition(condition_name: str, confidence_data: dict = None) -> bool:
    """Check if a condition is collapsed using detection criteria."""
    # Use hardcoded set for now; dynamic detection will be added later
    return condition_name in FAILED_CONDITIONS


def split_conditions(names: Sequence[str]) -> tuple[List[str], List[str]]:
    """Split conditions into valid and failed lists (§ 0)."""
    valid = []
    failed = []
    for name in names:
        if is_failed_condition(name):
            failed.append(name)
        else:
            valid.append(name)
    return valid, failed


def get_ptm_group(condition_name: str) -> str:
    """Extract PTM group from condition name."""
    if "sep102" in condition_name.lower():
        return "SEP102"
    elif "tpo101" in condition_name.lower():
        return "TPO101"
    elif "dna" in condition_name.lower():
        return "DNA"
    else:
        return "unmodified"


def draw_failed_marker(ax, x, y, size=None, color="#BBBBBB"):
    """Draw failed condition marker: grey background with red cross (§ 0)."""
    if size is None:
        size = 1
    # Draw grey background
    ax.scatter([x], [y], s=size*100, marker='s', color=color, alpha=0.7, zorder=10)
    # Draw red cross
    ax.plot([x-size*0.3, x+size*0.3], [y-size*0.3, y+size*0.3], 'r-', linewidth=2, zorder=11)
    ax.plot([x-size*0.3, x+size*0.3], [y+size*0.3, y-size*0.3], 'r-', linewidth=2, zorder=11)


def cluster_color(cluster_id: int) -> str:
    """Return a color for a cluster ID using a cycling palette."""
    # Use Okabe-Ito palette cycling through clusters
    palette = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # green
        "#CC79A7",  # reddish purple
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#F0E442",  # yellow
        "#999999",  # grey
    ]
    return palette[cluster_id % len(palette)]
