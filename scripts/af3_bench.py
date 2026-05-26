#!/usr/bin/env python3
"""
af3_bench.py  —  AF3 Condition Comparison Pipeline
====================================================
Compares AF3 structural predictions across conditions against a baseline.

Each condition is represented by AF3's own best-model output (the top-level
*_model.cif + *_confidences.json). The pipeline asks three questions:

  1. Is there a structural difference?
     → Cα RMSD and TM-score between each condition and the baseline.

  2. Where does the difference localise?
     → Per-residue Cα displacement profile with pLDDT overlay and
       reference seed-SD stability band.

  3. Does confidence co-vary with the structural shift?
     → Δ pTM, Δ ipTM, Δ mean-pLDDT, Δ mean-PAE with per-seed spread.

When the dataset has a factorial structure (ion concentration × PTM),
three additional structured plots are generated automatically:
  - panel_per_residue.png   grid: PTM rows × concentration columns
  - concentration_response.png  mean displacement vs concentration per PTM
  - ptm_effect_grid.png     2D heatmap of mean displacement

Baseline auto-detection: the condition with the fewest non-protein entities
and no PTMs is selected as the reference (protein + minimal solvent).
Override with --baseline.

Usage
-----
  python af3_bench.py --models <dir> [--baseline <name>] [options]

Options
-------
  --models   DIR    Root folder; each immediate subdirectory is one condition.
  --baseline NAME   Baseline condition (auto-detected if omitted).
  --output   DIR    Output directory (default: af3_results).
  --chains   A,B    Restrict alignment to these protein chain IDs.
  --pymol           Generate PyMOL .pml scripts.
  --tm              Compute TM-score (requires: pip install tmtools).
"""

import sys
import re
import json
import math
import argparse
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    from Bio.PDB import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import tmtools
    HAS_TMTOOLS = True
except ImportError:
    HAS_TMTOOLS = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("af3_bench")

# ANSI colours (Windows-safe)
_USE_COLOR = False
try:
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7
        )
    _USE_COLOR = True
except Exception:
    pass

R  = "\033[0m"  if _USE_COLOR else ""
B  = "\033[1m"  if _USE_COLOR else ""
GR = "\033[92m" if _USE_COLOR else ""
CY = "\033[96m" if _USE_COLOR else ""
YL = "\033[93m" if _USE_COLOR else ""

# ===========================================================================
# DATA STRUCTURES
# ===========================================================================

class ConditionModel:
    """
    Representative model for one AF3 condition.

    AF3 writes a top-level *_model.cif alongside seed subfolders.
    That top-level file is used as the representative here.

    Protein Cα and nucleic C4' atoms are stored in separate arrays so
    alignment is always performed on protein backbone only, regardless
    of whether DNA/RNA is present in the structure.

    Attributes
    ----------
    name            condition name (= subfolder name)
    cif_path        Path to *_model.cif
    ptm             global pTM  (float or NaN)
    iptm            global ipTM (float or NaN; NaN for monomers)
    ranking_score   AF3 ranking score (float or NaN)

    Protein backbone (used for all alignments)
    ca_coords       (N, 3) float64  Cα coordinates
    ca_plddts       (N,)   float64  per-residue pLDDT
    ca_chain_ids    list[str]       chain ID per residue
    ca_res_indices  list[int]       residue sequence number per residue

    Nucleic backbone (reported separately, never used for alignment)
    na_coords       (M, 3) float64  C4' coordinates  (empty array if absent)
    na_plddts       (M,)   float64  per-residue pLDDT
    na_chain_ids    list[str]
    na_res_indices  list[int]

    Confidence data
    pae_matrix      (K, K) float32 PAE matrix or None
    atom_plddts     (K,)   float32 per-atom pLDDT or None
    """

    def __init__(self, name: str, cif_path: Path) -> None:
        self.name = name
        self.cif_path = cif_path
        self.ptm: float = float("nan")
        self.iptm: float = float("nan")
        self.ranking_score: float = float("nan")

        # Protein Cα
        self.ca_coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.ca_plddts: np.ndarray = np.empty(0, dtype=np.float64)
        self.ca_chain_ids: List[str] = []
        self.ca_res_indices: List[int] = []

        # Nucleic C4'
        self.na_coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.na_plddts: np.ndarray = np.empty(0, dtype=np.float64)
        self.na_chain_ids: List[str] = []
        self.na_res_indices: List[int] = []

        # Full-model confidence
        self.pae_matrix: Optional[np.ndarray] = None
        self.atom_plddts: Optional[np.ndarray] = None

        # Parsed from data.json (populated by load_input_json)
        self.protein_chain_ids_from_json: List[str] = []
        self.nucleic_chain_ids_from_json: List[str] = []
        self.description: str = ""
        # Structured metadata for factorial analysis
        self.ptm_labels: List[str] = []          # e.g. ["SEP102", "TPO235"]
        self.ion_count: int = 0                  # total non-protein, non-nucleic entities
        self.has_real_ligand: bool = False        # SMILES with >5 heavy atoms

    # ------------------------------------------------------------------
    def load_summary_confidences(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.ptm = float(d["ptm"]) if d.get("ptm") is not None else float("nan")
            self.iptm = float(d["iptm"]) if d.get("iptm") is not None else float("nan")
            self.ranking_score = (
                float(d["ranking_score"])
                if d.get("ranking_score") is not None
                else float("nan")
            )
        except Exception as exc:
            log.warning("Could not parse summary confidences %s: %s", path.name, exc)

    # ------------------------------------------------------------------
    def load_full_confidences(self, path: Path) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if "atom_plddts" in d:
                self.atom_plddts = np.array(d["atom_plddts"], dtype=np.float32)
            if "pae" in d:
                self.pae_matrix = np.array(d["pae"], dtype=np.float32)
        except Exception as exc:
            log.warning("Could not parse full confidences %s: %s", path.name, exc)

    # ------------------------------------------------------------------
    def load_input_json(self, path: Path) -> None:
        """
        Parse *_data.json to extract chain identity and build a human-readable
        condition description.

        Populates:
          protein_chain_ids_from_json  — chain IDs of protein entities
          nucleic_chain_ids_from_json  — chain IDs of DNA/RNA entities
          description                  — one-line summary of what's in this condition

        Entity types recognised: protein, dna, rna, ligand, ion, ccdCode.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception as exc:
            log.debug("Could not parse input JSON %s: %s", path.name, exc)
            return

        protein_ids: List[str] = []
        nucleic_ids: List[str] = []
        parts: List[str] = []
        ptm_labels: List[str] = []
        ion_count: int = 0
        has_real_ligand: bool = False

        _NUCLEIC = {"dna", "rna"}
        _LIGAND  = {"ligand", "ion", "ccdcode"}

        for entity in d.get("sequences", []):
            for entity_type, v in entity.items():
                if not isinstance(v, dict):
                    continue
                etype = entity_type.lower()
                raw_id = v.get("id", "")
                if isinstance(raw_id, list):
                    ids = [str(x) for x in raw_id]
                else:
                    ids = [str(raw_id)] if raw_id else []

                if etype == "protein":
                    protein_ids.extend(ids)
                    seq_len = len(v.get("sequence", ""))
                    mods = v.get("modifications", [])
                    mod_str = ""
                    if mods:
                        for m in mods:
                            label = m.get("ptmType", "?") + str(m.get("ptmPosition", ""))
                            ptm_labels.append(label)
                        mod_str = " +" + ",".join(ptm_labels)
                    parts.append(
                        f"protein(chain{'s' if len(ids)>1 else ''}="
                        f"{','.join(ids)} {seq_len}aa{mod_str})"
                    )
                elif etype in _NUCLEIC:
                    nucleic_ids.extend(ids)
                    seq_len = len(v.get("sequence", ""))
                    parts.append(
                        f"{etype}(chain{'s' if len(ids)>1 else ''}="
                        f"{','.join(ids)} {seq_len}nt)"
                    )
                else:
                    # Ligand / ion / solvent
                    smiles = v.get("smiles", "")
                    ccd   = v.get("ccdCode", "")
                    count = len(ids)
                    ion_count += count
                    # Real ligand: SMILES with more than 5 characters (not water/ion)
                    if smiles and len(smiles) > 5:
                        has_real_ligand = True
                    label = ccd if ccd else (smiles[:12] + "..." if len(smiles) > 12 else smiles)
                    parts.append(
                        f"{etype}(chain{'s' if count>1 else ''}="
                        f"{','.join(ids[:3])}{'...' if count>3 else ''}"
                        + (f" {label}" if label else "") + ")"
                    )

        self.protein_chain_ids_from_json = protein_ids
        self.nucleic_chain_ids_from_json = nucleic_ids
        self.description = "  |  ".join(parts) if parts else "unknown composition"
        self.ptm_labels = ptm_labels
        self.ion_count = ion_count
        self.has_real_ligand = has_real_ligand

    # ------------------------------------------------------------------
    def load_structure(self, chain_filter: Optional[List[str]] = None) -> bool:
        """
        Parse CIF and populate ca_* (protein Cα) and na_* (nucleic C4') arrays.

        chain_filter restricts which protein chains are included in ca_*.
        Nucleic chains are always collected regardless of chain_filter
        (they are never used for alignment but may be reported).
        """
        if not HAS_BIOPYTHON:
            log.error("Biopython not installed — cannot parse CIF files.")
            return False
        try:
            parser = MMCIFParser(QUIET=True)
            struct = parser.get_structure(self.name, str(self.cif_path))
            model = struct[0]

            ca_c, ca_p, ca_ch, ca_ri = [], [], [], []
            na_c, na_p, na_ch, na_ri = [], [], [], []

            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        # Protein residue
                        if chain_filter and chain.id not in chain_filter:
                            continue
                        atom = residue["CA"]
                        ca_c.append(atom.get_coord())
                        ca_p.append(atom.bfactor)
                        ca_ch.append(chain.id)
                        ca_ri.append(residue.get_id()[1])
                    elif "C4'" in residue:
                        # Nucleic acid residue — always collect, never filter
                        atom = residue["C4'"]
                        na_c.append(atom.get_coord())
                        na_p.append(atom.bfactor)
                        na_ch.append(chain.id)
                        na_ri.append(residue.get_id()[1])

            if not ca_c and not na_c:
                log.warning("No Cα or C4' atoms found in %s", self.cif_path.name)
                return False

            if ca_c:
                self.ca_coords = np.array(ca_c, dtype=np.float64)
                self.ca_plddts = np.array(ca_p, dtype=np.float64)
                self.ca_chain_ids = ca_ch
                self.ca_res_indices = ca_ri
            if na_c:
                self.na_coords = np.array(na_c, dtype=np.float64)
                self.na_plddts = np.array(na_p, dtype=np.float64)
                self.na_chain_ids = na_ch
                self.na_res_indices = na_ri

            return True

        except Exception as exc:
            log.warning("Error loading structure %s: %s", self.cif_path.name, exc)
            return False

    # ------------------------------------------------------------------
    @property
    def n_protein_residues(self) -> int:
        return len(self.ca_coords)

    @property
    def n_nucleic_residues(self) -> int:
        return len(self.na_coords)

    @property
    def mean_plddt(self) -> float:
        if len(self.ca_plddts) > 0:
            return float(np.mean(self.ca_plddts))
        return float("nan")

    @property
    def mean_pae(self) -> float:
        if self.pae_matrix is not None:
            return float(np.mean(self.pae_matrix))
        return float("nan")

# ===========================================================================
# DISCOVERY
# ===========================================================================

def discover_conditions(
    models_dir: Path,
    chain_filter: Optional[List[str]],
) -> Dict[str, ConditionModel]:
    """
    Scan models_dir for AF3 condition subdirectories.

    Each immediate subdirectory containing a top-level *_model.cif is one
    condition. Seed subfolders are ignored.

    Returns {condition_name: ConditionModel} sorted by name.
    """
    if not HAS_BIOPYTHON:
        log.error("Biopython is required.  pip install biopython")
        sys.exit(1)

    conditions: Dict[str, ConditionModel] = {}
    subdirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if not subdirs:
        log.error("No subdirectories found in %s", models_dir)
        sys.exit(1)

    for cond_dir in subdirs:
        name = cond_dir.name

        # Primary: {name}_model.cif
        cif_path = cond_dir / f"{name}_model.cif"
        if not cif_path.exists():
            # Fallback: any top-level *_model.cif (not inside a seed subfolder)
            candidates = [
                f for f in cond_dir.glob("*_model.cif")
                if f.parent == cond_dir
                and not re.search(r"seed-\d+_sample-\d+", f.name)
            ]
            if not candidates:
                log.debug("Skipping %s — no top-level *_model.cif", name)
                continue
            cif_path = candidates[0]

        model = ConditionModel(name, cif_path)

        summary_json = cond_dir / f"{name}_summary_confidences.json"
        if summary_json.exists():
            model.load_summary_confidences(summary_json)
        else:
            log.warning("No summary_confidences.json for '%s'", name)

        full_json = cond_dir / f"{name}_confidences.json"
        if full_json.exists():
            model.load_full_confidences(full_json)

        # Load input JSON to get authoritative chain identity
        input_json = cond_dir / f"{name}_data.json"
        if input_json.exists():
            model.load_input_json(input_json)

        # Determine effective protein chain filter:
        #   1. --chains CLI override (explicit, highest priority)
        #   2. protein chains from data.json (authoritative)
        #   3. None → fall back to Cα-detection in load_structure
        if chain_filter:
            effective_filter = chain_filter
        elif model.protein_chain_ids_from_json:
            effective_filter = model.protein_chain_ids_from_json
        else:
            effective_filter = None

        if not model.load_structure(effective_filter):
            log.warning("Skipping '%s' — structure could not be loaded", name)
            continue

        if model.n_protein_residues == 0:
            log.warning(
                "Skipping '%s' — no protein Cα atoms found "
                "(use --chains to specify chain IDs if the structure has non-standard chain naming)",
                name,
            )
            continue

        conditions[name] = model

        # Build info line
        desc = model.description if model.description else "no data.json"
        log.info(
            "Loaded %s%s%s  protein=%d aa  nucleic=%d nt  pTM=%.3f  ipTM=%s\n"
            "         %s",
            B, name, R,
            model.n_protein_residues,
            model.n_nucleic_residues,
            model.ptm,
            f"{model.iptm:.3f}" if math.isfinite(model.iptm) else "N/A",
            desc,
        )

    if not conditions:
        # Help the user understand why nothing loaded
        all_subdirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
        has_nested = any(
            any(
                f.parent == sub and not re.search(r"seed-\d+_sample-\d+", f.name)
                for f in sub.glob("*_model.cif")
            )
            is False and any(d.is_dir() for d in sub.iterdir() if d.is_dir())
            for sub in all_subdirs
        )
        # Check if model CIFs exist one level deeper
        deeper_cifs = list(models_dir.glob("*/*_model.cif"))
        deeper_cifs = [f for f in deeper_cifs if not re.search(r"seed-\d+_sample-\d+", f.name)]
        if deeper_cifs:
            log.error(
                "No valid conditions found in %s\n"
                "  However, AF3 model files were found one level deeper:\n"
                "    %s\n"
                "  Point --models at a directory whose IMMEDIATE subdirectories\n"
                "  are AF3 job output folders (each containing a *_model.cif).\n"
                "  Example: --models %s",
                models_dir,
                deeper_cifs[0].parent.parent,
                deeper_cifs[0].parent.parent,
            )
        else:
            log.error("No valid conditions found in %s", models_dir)
        sys.exit(1)

    return dict(sorted(conditions.items()))


def resolve_baseline(
    conditions: Dict[str, ConditionModel],
    baseline_arg: Optional[str],
) -> str:
    """
    Return baseline name.

    Priority:
    1. Explicit --baseline argument
    2. Name containing 'baseline', 'apo', 'wt', 'ctrl', 'control', 'ref'
    3. Condition with fewest non-protein entities AND no PTMs
       (the "cleanest" reference — protein + minimal solvent)
    4. Condition with fewest non-protein entities (ignoring PTM status)
    5. First alphabetically
    """
    if baseline_arg:
        if baseline_arg not in conditions:
            log.error(
                "Baseline '%s' not found. Available: %s",
                baseline_arg, ", ".join(conditions),
            )
            sys.exit(1)
        return baseline_arg

    # Keyword detection
    for keyword in ("baseline", "apo", "wt", "ctrl", "control", "ref"):
        for name in conditions:
            if keyword in name.lower():
                log.info("Auto-detected baseline (keyword): %s%s%s", B, name, R)
                return name

    # Fewest non-protein entities, no PTMs, no nucleic acid, no real ligand
    no_ptm = {
        n: c for n, c in conditions.items()
        if not c.ptm_labels
        and c.n_nucleic_residues == 0
        and not c.has_real_ligand
    }
    if no_ptm:
        best = min(no_ptm, key=lambda n: no_ptm[n].ion_count)
        log.info(
            "Auto-detected baseline (fewest solvent/ions, no PTM, no DNA): %s%s%s  "
            "(ion_count=%d)",
            B, best, R, conditions[best].ion_count,
        )
        return best

    # Fewest non-protein entities, no nucleic acid (PTMs allowed)
    no_dna = {
        n: c for n, c in conditions.items()
        if c.n_nucleic_residues == 0
        and not c.has_real_ligand
    }
    if no_dna:
        best = min(no_dna, key=lambda n: no_dna[n].ion_count)
        log.info(
            "Auto-detected baseline (fewest solvent/ions, no DNA): %s%s%s  "
            "(ion_count=%d)",
            B, best, R, conditions[best].ion_count,
        )
        return best

    # Last resort: fewest non-protein entities regardless of composition
    best = min(conditions, key=lambda n: conditions[n].ion_count)
    log.info(
        "Auto-detected baseline (fewest non-protein entities): %s%s%s  (ion_count=%d)",
        B, best, R, conditions[best].ion_count,
    )
    return best

# ===========================================================================
# GEOMETRY — KABSCH ALIGNMENT (protein Cα only)
# ===========================================================================

def kabsch_rmsd(
    ref: np.ndarray,
    mob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Kabsch superposition of mob onto ref.  Both (N,3) float64.
    If lengths differ the shorter is used.

    Returns R_mat (3,3), t_vec (3,), rmsd (float).
    Apply as:  mob_aligned = mob @ R_mat.T + t_vec
    """
    k = min(len(ref), len(mob))
    r, m = ref[:k], mob[:k]
    r_mean, m_mean = r.mean(0), m.mean(0)
    rc, mc = r - r_mean, m - m_mean
    H = mc.T @ rc
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1] *= -1
        R_mat = Vt.T @ U.T
    t_vec = r_mean - m_mean @ R_mat.T
    rmsd = float(np.sqrt(np.mean(np.sum((rc - mc @ R_mat.T) ** 2, axis=1))))
    return R_mat, t_vec, rmsd


def _warn_length_mismatch(name_ref: str, name_mob: str, n_ref: int, n_mob: int) -> None:
    """Warn when protein residue counts differ by more than 5%."""
    if n_ref == 0 or n_mob == 0:
        return
    diff_pct = abs(n_ref - n_mob) / max(n_ref, n_mob) * 100
    if diff_pct > 5:
        log.warning(
            "Protein length mismatch: %s=%d aa vs %s=%d aa (%.0f%% difference). "
            "RMSD computed on the first %d shared residues.",
            name_ref, n_ref, name_mob, n_mob, diff_pct, min(n_ref, n_mob),
        )


def align_and_rmsd(
    ref: ConditionModel,
    mob: ConditionModel,
    plddt_cutoff: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Align mob onto ref using high-confidence protein Cα atoms (pLDDT > cutoff).
    Falls back to all Cα if fewer than 3 pass the cutoff.

    Returns R_mat, t_vec, rmsd, n_residues_used.
    """
    _warn_length_mismatch(
        ref.name, mob.name, ref.n_protein_residues, mob.n_protein_residues
    )
    k = min(ref.n_protein_residues, mob.n_protein_residues)
    if k == 0:
        nan = float("nan")
        return np.eye(3), np.zeros(3), nan, 0

    r_coords = ref.ca_coords[:k]
    m_coords = mob.ca_coords[:k]
    r_plddt  = ref.ca_plddts[:k]
    m_plddt  = mob.ca_plddts[:k]

    mask = (r_plddt > plddt_cutoff) & (m_plddt > plddt_cutoff)
    if mask.sum() < 3:
        mask = np.ones(k, dtype=bool)

    R_mat, t_vec, rmsd = kabsch_rmsd(r_coords[mask], m_coords[mask])
    return R_mat, t_vec, rmsd, int(mask.sum())


def per_residue_displacement(
    ref: ConditionModel,
    mob: ConditionModel,
    R_mat: np.ndarray,
    t_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-residue Cα displacement (Å) for protein and nucleic chains separately.

    The protein superposition (R_mat, t_vec) is applied to both protein and
    nucleic coordinates of mob.  This means nucleic displacement is measured
    after the protein has been optimally superimposed — which is the correct
    framing: "given that the protein aligns as well as possible, how much
    does the nucleic acid move?"

    Returns
    -------
    prot_disp : (k,) array  protein Cα displacement
    nuc_disp  : (m,) array  nucleic C4' displacement (empty if absent)
    """
    k = min(ref.n_protein_residues, mob.n_protein_residues)
    prot_disp = np.sqrt(
        np.sum((ref.ca_coords[:k] - (mob.ca_coords[:k] @ R_mat.T + t_vec)) ** 2, axis=1)
    )

    m = min(ref.n_nucleic_residues, mob.n_nucleic_residues)
    if m > 0:
        nuc_disp = np.sqrt(
            np.sum((ref.na_coords[:m] - (mob.na_coords[:m] @ R_mat.T + t_vec)) ** 2, axis=1)
        )
    else:
        nuc_disp = np.empty(0, dtype=np.float64)

    return prot_disp, nuc_disp


def tm_score_pair(
    ref: ConditionModel,
    mob: ConditionModel,
) -> Tuple[float, float]:
    """TM-score (norm to ref, norm to mob) using protein Cα only."""
    if not HAS_TMTOOLS:
        return float("nan"), float("nan")
    try:
        k = min(ref.n_protein_residues, mob.n_protein_residues)
        if k < 5:
            return float("nan"), float("nan")
        c1 = np.asarray(ref.ca_coords[:k], dtype=np.float64)
        c2 = np.asarray(mob.ca_coords[:k], dtype=np.float64)
        seq = "A" * k
        result = tmtools.tm_align(c1, c2, seq, seq)
        return float(result.tm_norm_chain1), float(result.tm_norm_chain2)
    except Exception as exc:
        log.warning("TM-score failed for %s vs %s: %s", mob.name, ref.name, exc)
        return float("nan"), float("nan")

# ===========================================================================
# SHARED HELPERS
# ===========================================================================

def _safe_round(v: float, n: int = 4) -> float:
    return round(v, n) if math.isfinite(v) else float("nan")


def _condition_labels(names: List[str], max_len: int = 22) -> List[str]:
    """
    Return short display labels for a list of condition names.

    If all names share a common prefix longer than 8 characters, strip it
    so the labels show only the distinguishing suffix.  Then truncate to
    max_len if still too long.
    """
    if len(names) < 2:
        return [n[:max_len] for n in names]

    # Find common prefix
    prefix = names[0]
    for n in names[1:]:
        while not n.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break

    # Only strip if the prefix is meaningfully long (> 8 chars) and leaves
    # something behind in every name
    if len(prefix) > 8 and all(len(n) > len(prefix) for n in names):
        labels = [n[len(prefix):] for n in names]
    else:
        labels = list(names)

    return [l[:max_len] + "..." if len(l) > max_len else l for l in labels]


def _build_per_residue_df(
    ref: ConditionModel,
    mob: ConditionModel,
    R_mat: np.ndarray,
    t_vec: np.ndarray,
) -> pd.DataFrame:
    """
    Build a per-residue displacement DataFrame covering protein and nucleic rows.

    Columns: residue_type, chain_id, residue_index,
             displacement_angstrom, ref_plddt, cond_plddt, delta_plddt.
    """
    prot_disp, nuc_disp = per_residue_displacement(ref, mob, R_mat, t_vec)
    rows = []

    k = len(prot_disp)
    for i in range(k):
        ref_p = ref.ca_plddts[i]
        mob_p = mob.ca_plddts[i]
        rows.append({
            "residue_type":         "protein",
            "chain_id":             ref.ca_chain_ids[i],
            "residue_index":        ref.ca_res_indices[i],
            "displacement_angstrom": round(float(prot_disp[i]), 4),
            "ref_plddt":            round(float(ref_p), 2),
            "cond_plddt":           round(float(mob_p), 2),
            "delta_plddt":          round(float(mob_p - ref_p), 2),
        })

    m = len(nuc_disp)
    for i in range(m):
        ref_p = ref.na_plddts[i]
        mob_p = mob.na_plddts[i]
        rows.append({
            "residue_type":         "nucleic",
            "chain_id":             ref.na_chain_ids[i],
            "residue_index":        ref.na_res_indices[i],
            "displacement_angstrom": round(float(nuc_disp[i]), 4),
            "ref_plddt":            round(float(ref_p), 2),
            "cond_plddt":           round(float(mob_p), 2),
            "delta_plddt":          round(float(mob_p - ref_p), 2),
        })

    return pd.DataFrame(rows)


# ===========================================================================
# EXPERIMENT STRUCTURE PARSER
# ===========================================================================

from dataclasses import dataclass, field as dc_field

@dataclass
class ExperimentStructure:
    """
    Parsed factorial structure of the loaded conditions.

    Attributes
    ----------
    ion_tier        {name: label}  e.g. "1x", "10x", "100x", "unknown"
    ptm_group       {name: label}  e.g. "none", "SEP102", "TPO235"
    has_dna         {name: bool}
    has_real_ligand {name: bool}
    tier_order      sorted list of unique ion tiers (ascending by count)
    ptm_order       sorted list of unique PTM groups ("none" first)
    panel_conditions set of condition names that fit the ion×PTM grid
                     (no real ligand, no DNA — pure solvent/PTM variation)
    """
    ion_tier:        Dict[str, str]
    ptm_group:       Dict[str, str]
    has_dna:         Dict[str, bool]
    has_real_ligand: Dict[str, bool]
    tier_order:      List[str]
    ptm_order:       List[str]
    panel_conditions: set


def parse_experiment_structure(
    conditions: Dict[str, ConditionModel],
) -> ExperimentStructure:
    """
    Infer the factorial structure (ion concentration × PTM) from loaded conditions.

    Ion tier is derived from the condition's ion_count relative to the minimum
    observed across all conditions:
      - min count  → "1x"
      - 10× min    → "10x"
      - 100× min   → "100x"
      - 1000× min  → "1000x"
      - other      → "{N}x" (actual count)

    PTM group is the joined PTM label string, or "none".
    """
    ion_tier: Dict[str, str] = {}
    ptm_group: Dict[str, str] = {}
    has_dna: Dict[str, bool] = {}
    has_real_ligand: Dict[str, bool] = {}

    # Find minimum non-zero ion count as the "1x" reference
    counts = [c.ion_count for c in conditions.values() if c.ion_count > 0]
    min_count = min(counts) if counts else 1

    for name, cond in conditions.items():
        # Ion tier
        cnt = cond.ion_count
        if cnt == 0:
            tier = "0x"
        else:
            ratio = cnt / min_count
            if abs(ratio - 1) < 0.5:
                tier = "1x"
            elif abs(ratio - 10) < 2:
                tier = "10x"
            elif abs(ratio - 100) < 20:
                tier = "100x"
            elif abs(ratio - 1000) < 200:
                tier = "1000x"
            else:
                tier = f"{cnt}x"
        ion_tier[name] = tier

        # PTM group
        ptm_group[name] = ",".join(cond.ptm_labels) if cond.ptm_labels else "none"

        # DNA / real ligand
        has_dna[name] = cond.n_nucleic_residues > 0
        has_real_ligand[name] = cond.has_real_ligand

    # Panel conditions: no real ligand, no DNA
    panel_conditions = {
        n for n in conditions
        if not has_dna[n] and not has_real_ligand[n]
    }

    # Ordered tiers (0x first, then numeric ascending)
    def _tier_key(t: str) -> float:
        if t == "0x": return 0.0
        try: return float(t.rstrip("x"))
        except: return 9999.0

    tier_order = sorted({ion_tier[n] for n in panel_conditions}, key=_tier_key)

    # PTM order: "none" first, then alphabetical
    ptm_order = sorted(
        {ptm_group[n] for n in panel_conditions},
        key=lambda p: ("" if p == "none" else p),
    )

    return ExperimentStructure(
        ion_tier=ion_tier,
        ptm_group=ptm_group,
        has_dna=has_dna,
        has_real_ligand=has_real_ligand,
        tier_order=tier_order,
        ptm_order=ptm_order,
        panel_conditions=panel_conditions,
    )


# ===========================================================================
# CONFIDENCE SUMMARY (mode-independent)
# ===========================================================================

def compute_confidence_summary(
    conditions: Dict[str, ConditionModel],
    reference_name: Optional[str],
) -> pd.DataFrame:
    """
    Per-condition confidence metrics.

    When reference_name is given (baseline mode), delta columns are
    computed relative to that condition.  In survey mode pass None —
    delta columns are omitted.
    """
    ref = conditions[reference_name] if reference_name else None
    rows = []
    for name, cond in conditions.items():
        row: dict = {
            "condition":    name,
            "is_reference": name == reference_name,
            "ptm":          _safe_round(cond.ptm),
            "iptm":         _safe_round(cond.iptm),
            "plddt_mean":   _safe_round(cond.mean_plddt),
            "mean_pae":     _safe_round(cond.mean_pae),
            "n_protein_residues": cond.n_protein_residues,
            "n_nucleic_residues": cond.n_nucleic_residues,
        }
        if ref is not None:
            row["delta_ptm"]        = _safe_round(cond.ptm - ref.ptm)
            row["delta_iptm"]       = _safe_round(cond.iptm - ref.iptm)
            row["delta_plddt_mean"] = _safe_round(cond.mean_plddt - ref.mean_plddt)
            row["delta_mean_pae"]   = _safe_round(cond.mean_pae - ref.mean_pae)
        rows.append(row)
    return pd.DataFrame(rows)


def write_representative_selection(
    conditions: Dict[str, ConditionModel],
    reference_name: Optional[str],
    output_dir: Path,
) -> None:
    rows = []
    for name, cond in conditions.items():
        rows.append({
            "condition":          name,
            "is_reference":       name == reference_name,
            "model_file":         str(cond.cif_path),
            "ptm":                _safe_round(cond.ptm),
            "iptm":               _safe_round(cond.iptm),
            "ranking_score":      _safe_round(cond.ranking_score),
            "plddt_mean":         _safe_round(cond.mean_plddt),
            "n_protein_residues": cond.n_protein_residues,
            "n_nucleic_residues": cond.n_nucleic_residues,
        })
    pd.DataFrame(rows)  # kept in memory; no CSV written
    log.info("Representative structures: %d conditions loaded", len(conditions))

# ===========================================================================
# BASELINE MODE — analysis functions
# ===========================================================================

def run_baseline_mode(
    conditions: Dict[str, ConditionModel],
    baseline_name: str,
    compute_tm: bool,
    output_dir: Path,
    pymol: bool,
    models_dir: Path,
    chain_filter: Optional[List[str]],
) -> None:
    """Full baseline-vs-conditions analysis."""

    baseline = conditions[baseline_name]
    non_baseline = [n for n in sorted(conditions) if n != baseline_name]

    # ------------------------------------------------------------------
    # Load seed replicates for SD bands / spread bars
    # ------------------------------------------------------------------
    log.info("Loading seed replicates for spread estimation...")
    seed_plddts: Dict[str, List[np.ndarray]] = {}
    seed_scores: Dict[str, Dict[str, List[float]]] = {}
    for name in conditions:
        cond_dir = models_dir / name
        eff_filter = chain_filter or conditions[name].protein_chain_ids_from_json or None
        seed_plddts[name] = load_seed_replicates(cond_dir, eff_filter)
        seed_scores[name] = load_seed_confidence_scores(cond_dir)
        if seed_plddts[name]:
            log.info("  %s: %d seed replicates", name, len(seed_plddts[name]))

    # Per-residue SD for the baseline (used as stability band)
    baseline_sd = compute_seed_sd(
        seed_plddts.get(baseline_name, []),
        baseline.n_protein_residues,
    )

    # ------------------------------------------------------------------
    # Q1 — Structural distances
    # ------------------------------------------------------------------
    log.info("%s--- Q1: Structural distances ---%s", CY, R)
    dist_rows = []
    for name in non_baseline:
        cond = conditions[name]
        _, _, rmsd, n_aligned = align_and_rmsd(baseline, cond)
        tm_c, tm_b = tm_score_pair(baseline, cond) if compute_tm else (float("nan"), float("nan"))
        dist_rows.append({
            "condition":          name,
            "rmsd_angstrom":      _safe_round(rmsd),
            "tm_score_norm_cond": _safe_round(tm_c),
            "tm_score_norm_base": _safe_round(tm_b),
            "n_residues_aligned": n_aligned,
        })
    df_dist = pd.DataFrame(dist_rows)

    # ------------------------------------------------------------------
    # Q2 — Per-residue profiles (in memory only, written as plots)
    # ------------------------------------------------------------------
    log.info("%s--- Q2: Per-residue displacement profiles ---%s", CY, R)
    profiles: Dict[str, pd.DataFrame] = {}
    for name in non_baseline:
        cond = conditions[name]
        R_mat, t_vec, _, _ = align_and_rmsd(baseline, cond)
        profiles[name] = _build_per_residue_df(baseline, cond, R_mat, t_vec)

    # ------------------------------------------------------------------
    # Q3 — Confidence summary (in memory only)
    # ------------------------------------------------------------------
    log.info("%s--- Q3: Confidence summary ---%s", CY, R)
    df_conf = compute_confidence_summary(conditions, baseline_name)

    # ------------------------------------------------------------------
    # PyMOL
    # ------------------------------------------------------------------
    if pymol:
        write_pymol_baseline(conditions, baseline_name, output_dir)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    log.info("%s--- Generating plots ---%s", CY, R)
    if not df_dist.empty:
        plot_baseline_distances(df_dist, baseline_name, output_dir)
    if profiles:
        # Compute global Y-axis max across all displacement profiles
        global_disp_max = max(
            df["displacement_angstrom"].max()
            for df in profiles.values()
            if not df.empty
        )
        plot_per_residue(
            profiles, output_dir, mode="baseline",
            ref_sd=baseline_sd,
            global_disp_max=global_disp_max,
            ref_name=baseline_name,
        )
    plot_confidence_summary(df_conf, output_dir, seed_scores=seed_scores)
    plot_pae_comparison(df_conf, output_dir)

    # ------------------------------------------------------------------
    # Structured factorial analysis
    # ------------------------------------------------------------------
    log.info("%s--- Structured factorial analysis ---%s", CY, R)
    struct = parse_experiment_structure(conditions)
    log.info(
        "Experiment structure: %d panel conditions  "
        "PTM groups=%s  Ion tiers=%s",
        len(struct.panel_conditions),
        struct.ptm_order,
        struct.tier_order,
    )
    if struct.panel_conditions:
        global_disp_max_panel = max(
            (
                float(np.mean(
                    per_residue_displacement(
                        baseline, conditions[n],
                        *align_and_rmsd(baseline, conditions[n])[:2],
                    )[0]
                ) * 3)  # rough ceiling: 3× mean
                for n in struct.panel_conditions
                if n in conditions and n != baseline_name
            ),
            default=10.0,
        )
        # Use actual max for consistency
        all_disps = []
        for n in struct.panel_conditions:
            if n not in conditions or n == baseline_name:
                continue
            R_mat, t_vec, _, _ = align_and_rmsd(baseline, conditions[n])
            pd_arr, _ = per_residue_displacement(baseline, conditions[n], R_mat, t_vec)
            if len(pd_arr):
                all_disps.append(float(np.max(pd_arr)))
        global_disp_max_panel = max(all_disps) if all_disps else 10.0

        plot_panel_per_residue(
            conditions, struct, baseline_name, baseline_sd,
            global_disp_max_panel, output_dir,
        )
        plot_concentration_response(conditions, struct, baseline_name, output_dir)
        plot_ptm_effect_grid(conditions, struct, baseline_name, output_dir)
    else:
        log.info("No panel conditions detected — skipping structured plots.")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info("\nStructural distances vs baseline (%s):", baseline_name)
    log.info("  %-40s  %8s  %8s", "Condition", "RMSD (Å)", "TM-score")
    log.info("  " + "-" * 62)
    for _, row in df_dist.iterrows():
        tm_str = f"{row['tm_score_norm_cond']:.4f}" if math.isfinite(row["tm_score_norm_cond"]) else "    N/A "
        log.info("  %-40s  %8.4f  %8s", row["condition"], row["rmsd_angstrom"], tm_str)

# ===========================================================================
# SURVEY MODE — all-vs-all analysis
# ===========================================================================

def run_survey_mode(
    conditions: Dict[str, ConditionModel],
    compute_tm: bool,
    output_dir: Path,
    pymol: bool,
    models_dir: Path,
    chain_filter: Optional[List[str]],
) -> None:
    """All-vs-all pairwise structural comparison."""

    names = sorted(conditions)
    pairs = list(itertools.combinations(names, 2))
    n = len(names)

    if n < 2:
        log.error(
            "Survey mode requires at least 2 conditions; only %d loaded. "
            "Check that --models points at a directory whose immediate "
            "subdirectories are AF3 job output folders.",
            n,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load seed confidence scores for spread bars
    # ------------------------------------------------------------------
    log.info("Loading seed replicates for spread estimation...")
    seed_scores: Dict[str, Dict[str, List[float]]] = {}
    for name in names:
        cond_dir = models_dir / name
        seed_scores[name] = load_seed_confidence_scores(cond_dir)

    if n < 2:
        log.error(
            "Survey mode requires at least 2 conditions; only %d loaded. "
            "Check that --models points at a directory whose immediate "
            "subdirectories are AF3 job output folders.",
            n,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pairwise distances
    # ------------------------------------------------------------------
    log.info("%s--- Pairwise structural distances (%d pairs) ---%s", CY, len(pairs), R)
    pair_rows = []
    rmsd_matrix = np.full((n, n), float("nan"))
    np.fill_diagonal(rmsd_matrix, 0.0)
    name_to_idx = {name: i for i, name in enumerate(names)}

    for ci, cj in pairs:
        ref = conditions[ci]
        mob = conditions[cj]
        _, _, rmsd, n_aligned = align_and_rmsd(ref, mob)
        tm_c, tm_b = tm_score_pair(ref, mob) if compute_tm else (float("nan"), float("nan"))
        pair_rows.append({
            "condition_i":        ci,
            "condition_j":        cj,
            "rmsd_angstrom":      _safe_round(rmsd),
            "tm_score_norm_i":    _safe_round(tm_c),
            "tm_score_norm_j":    _safe_round(tm_b),
            "n_residues_aligned": n_aligned,
        })
        if math.isfinite(rmsd):
            i, j = name_to_idx[ci], name_to_idx[cj]
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

    df_pairs = pd.DataFrame(pair_rows)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values("rmsd_angstrom", ascending=False).reset_index(drop=True)

    df_matrix = pd.DataFrame(rmsd_matrix, index=names, columns=names)

    # ------------------------------------------------------------------
    # Per-residue profiles: each condition vs the survey reference
    # Survey reference = condition with highest pTM (most confident
    # prediction), falling back to first alphabetically.
    # ------------------------------------------------------------------
    survey_ref_name = max(
        names,
        key=lambda nm: conditions[nm].ptm if math.isfinite(conditions[nm].ptm) else -1.0,
    )
    survey_ref = conditions[survey_ref_name]
    non_ref = [nm for nm in names if nm != survey_ref_name]

    log.info(
        "%s--- Per-residue profiles vs survey reference (%s) ---%s",
        CY, survey_ref_name, R,
    )

    # Also load seed SD for the survey reference
    survey_ref_dir = models_dir / survey_ref_name
    eff_filter = chain_filter or conditions[survey_ref_name].protein_chain_ids_from_json or None
    survey_ref_seed_plddts = load_seed_replicates(survey_ref_dir, eff_filter)
    survey_ref_sd = compute_seed_sd(survey_ref_seed_plddts, survey_ref.n_protein_residues)

    profiles: Dict[str, pd.DataFrame] = {}
    for nm in non_ref:
        cond = conditions[nm]
        R_mat, t_vec, _, _ = align_and_rmsd(survey_ref, cond)
        profiles[nm] = _build_per_residue_df(survey_ref, cond, R_mat, t_vec)

    # ------------------------------------------------------------------
    # Confidence summary (in memory only)
    # ------------------------------------------------------------------
    log.info("%s--- Confidence summary ---%s", CY, R)
    df_conf = compute_confidence_summary(conditions, reference_name=None)

    # ------------------------------------------------------------------
    # PyMOL
    # ------------------------------------------------------------------
    if pymol:
        write_pymol_survey(conditions, output_dir)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    log.info("%s--- Generating plots ---%s", CY, R)
    plot_pairwise_heatmap(df_matrix, df_conf, output_dir)
    finite_rmsds = df_pairs["rmsd_angstrom"].dropna().tolist() if not df_pairs.empty else []
    median_rmsd = float(np.median(finite_rmsds)) if finite_rmsds else float("nan")
    plot_rmsd_distribution(df_pairs, median_rmsd, output_dir)
    if profiles:
        global_disp_max = max(
            df["displacement_angstrom"].max()
            for df in profiles.values()
            if not df.empty
        )
        plot_per_residue(
            profiles, output_dir, mode="survey",
            ref_sd=survey_ref_sd,
            global_disp_max=global_disp_max,
            ref_name=survey_ref_name,
        )
    plot_confidence_summary(df_conf, output_dir, seed_scores=seed_scores)
    plot_pae_comparison(df_conf, output_dir)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    # Console summary — distances vs survey reference
    # ------------------------------------------------------------------
    log.info("\nStructural distances vs survey reference (%s):", survey_ref_name)
    log.info("  %-40s  %8s", "Condition", "RMSD (Å)")
    log.info("  " + "-" * 52)
    ref_distances = df_pairs[
        (df_pairs["condition_i"] == survey_ref_name) |
        (df_pairs["condition_j"] == survey_ref_name)
    ].copy()
    if not ref_distances.empty:
        ref_distances["other"] = ref_distances.apply(
            lambda r: r["condition_j"] if r["condition_i"] == survey_ref_name else r["condition_i"],
            axis=1,
        )
        for _, row in ref_distances.sort_values("rmsd_angstrom", ascending=False).iterrows():
            log.info("  %-40s  %8.4f", row["other"], row["rmsd_angstrom"])
    log.info("\nTop 10 most different pairs overall:")
    log.info("  %-35s  %-35s  %8s", "Condition i", "Condition j", "RMSD (Å)")
    log.info("  " + "-" * 82)
    if not df_pairs.empty:
        for _, row in df_pairs.head(10).iterrows():
            log.info(
                "  %-35s  %-35s  %8.4f",
                row["condition_i"], row["condition_j"], row["rmsd_angstrom"],
            )

# ===========================================================================
# SEED REPLICATE LOADING (for SD bands and spread bars)
# ===========================================================================

def load_seed_replicates(
    cond_dir: Path,
    chain_filter: Optional[List[str]],
) -> List[np.ndarray]:
    """
    Load per-residue pLDDT arrays from all seed subfolders of a condition.

    Returns a list of (N,) float64 arrays — one per successfully loaded
    seed replicate.  Only protein Cα pLDDT values are returned (same
    residue ordering as ConditionModel.ca_plddts).

    Used to compute intra-condition spread for SD bands and error bars.
    """
    if not HAS_BIOPYTHON:
        return []

    arrays: List[np.ndarray] = []
    seed_dirs = [
        d for d in cond_dir.iterdir()
        if d.is_dir() and re.search(r"seed-\d+_sample-\d+", d.name)
    ]
    if not seed_dirs:
        return []

    parser = MMCIFParser(QUIET=True)
    for sd in seed_dirs:
        cifs = [
            f for f in sd.glob("*_model.cif")
            if not re.search(r"seed-\d+_sample-\d+", f.parent.name + f.name)
            or f.parent == sd
        ]
        if not cifs:
            continue
        cif = cifs[0]
        try:
            struct = parser.get_structure(sd.name, str(cif))
            plddts = []
            for chain in struct[0]:
                if chain_filter and chain.id not in chain_filter:
                    continue
                for residue in chain:
                    if "CA" in residue:
                        plddts.append(residue["CA"].bfactor)
            if plddts:
                arrays.append(np.array(plddts, dtype=np.float64))
        except Exception:
            pass

    return arrays


def compute_seed_sd(
    seed_arrays: List[np.ndarray],
    n_residues: int,
) -> Optional[np.ndarray]:
    """
    Per-residue SD across seed replicates.

    Aligns all arrays to length n_residues (truncates longer, skips shorter).
    Returns (n_residues,) float64 array, or None if fewer than 2 seeds loaded.
    """
    valid = [a[:n_residues] for a in seed_arrays if len(a) >= n_residues]
    if len(valid) < 2:
        return None
    return np.std(np.stack(valid, axis=0), axis=0, ddof=1)


def load_seed_confidence_scores(
    cond_dir: Path,
) -> Dict[str, List[float]]:
    """
    Load ptm, iptm, plddt_mean from all seed summary_confidences.json files.

    Returns {"ptm": [...], "iptm": [...], "plddt_mean": [...]} with one
    value per successfully loaded seed replicate.
    """
    result: Dict[str, List[float]] = {"ptm": [], "iptm": [], "plddt_mean": []}
    seed_dirs = [
        d for d in cond_dir.iterdir()
        if d.is_dir() and re.search(r"seed-\d+_sample-\d+", d.name)
    ]
    for sd in seed_dirs:
        jsons = list(sd.glob("*_summary_confidences.json"))
        if not jsons:
            continue
        try:
            with open(jsons[0], "r", encoding="utf-8") as fh:
                d = json.load(fh)
            ptm  = d.get("ptm")
            iptm = d.get("iptm")
            # plddt_mean from atom_plddts in full confidences if available,
            # otherwise skip — we only need spread, not absolute value
            if ptm  is not None: result["ptm"].append(float(ptm))
            if iptm is not None: result["iptm"].append(float(iptm))
        except Exception:
            pass
    return result

_PALETTE = [
    "marine", "salmon", "forest", "tv_orange", "slate",
    "hotpink", "cyan", "tv_yellow", "limon", "violet",
    "deeppurple", "teal", "firebrick", "olive", "skyblue",
]

_PML_SETTINGS = [
    "reinitialize",
    "bg_color white",
    "set ray_opaque_background, off",
    "set cartoon_fancy_helices, 1",
    "set cartoon_smooth_loops, 1",
    "set cartoon_side_chain_helper, 1",
    "set depth_cue, 0",
    "set specular, 0.2",
    "",
]

_PML_FOOTER = [
    "",
    "hide everything",
    "show cartoon, polymer",
    "# To show ligands:  show sticks, organic",
    "# To show ions:     show spheres, inorganic",
    "",
    "zoom polymer",
    "orient",
    "",
]


def _pml_header(title: str, note: str = "") -> List[str]:
    lines = [f"# {title}"]
    if note:
        lines.append(f"# {note}")
    lines.append("# Generated by af3_bench.py")
    lines.append("#")
    return lines + _PML_SETTINGS


def write_pymol_baseline(
    conditions: Dict[str, ConditionModel],
    baseline_name: str,
    output_dir: Path,
) -> None:
    """
    Three PyMOL scripts for baseline mode.

    01_overlay.pml       — all conditions aligned to baseline, coloured by condition
    02_plddt.pml         — all conditions, pLDDT spectrum
    03_per_condition/    — one script per non-baseline condition
    """
    pymol_dir = output_dir / "pymol"
    pymol_dir.mkdir(parents=True, exist_ok=True)
    per_dir = pymol_dir / "03_per_condition"
    per_dir.mkdir(parents=True, exist_ok=True)

    base_cif = conditions[baseline_name].cif_path.resolve()
    non_baseline = [n for n in sorted(conditions) if n != baseline_name]

    # --- 01_overlay.pml ---
    lines = _pml_header(
        "AF3 Condition Overlay — all conditions vs baseline",
        f"baseline: {baseline_name}",
    )
    lines += [
        f"load {base_cif}, BASE_{baseline_name}",
        f"color grey70, BASE_{baseline_name}",
        f"set cartoon_transparency, 0.35, BASE_{baseline_name}",
        "",
    ]
    for idx, name in enumerate(non_baseline):
        cif = conditions[name].cif_path.resolve()
        obj = f"COND_{name}"
        color = _PALETTE[idx % len(_PALETTE)]
        lines += [
            f"load {cif}, {obj}",
            f"align {obj}, BASE_{baseline_name}",
            f"color {color}, {obj}",
            f"# {name}",
            "",
        ]
    lines += _PML_FOOTER
    lines += [
        "# --- Tips ---",
        "# save overlay.pse",
        f"# super COND_<name>, BASE_{baseline_name}  (more robust for distant structures)",
        "# disable all; enable BASE_*; enable COND_<name>  (isolate one condition)",
    ]
    (pymol_dir / "01_overlay.pml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- 02_plddt.pml ---
    lines2 = _pml_header(
        "AF3 pLDDT Confidence Map — all conditions",
        "B-factor = pLDDT.  blue=high confidence, red=low.",
    )
    lines2 += [f"load {base_cif}, BASE_{baseline_name}", ""]
    for name in non_baseline:
        cif = conditions[name].cif_path.resolve()
        lines2 += [
            f"load {cif}, COND_{name}",
            f"align COND_{name}, BASE_{baseline_name}",
            "",
        ]
    lines2 += _PML_FOOTER
    lines2 += [
        "spectrum b, blue_white_red, polymer, minimum=50, maximum=100",
        "# save plddt.pse",
    ]
    (pymol_dir / "02_plddt.pml").write_text("\n".join(lines2) + "\n", encoding="utf-8")

    # --- 03_per_condition/ ---
    for idx, name in enumerate(non_baseline):
        cif = conditions[name].cif_path.resolve()
        color = _PALETTE[idx % len(_PALETTE)]
        lines3 = _pml_header(
            f"AF3: {name} vs {baseline_name}",
            f"baseline=grey  condition={color} coloured by pLDDT",
        )
        lines3 += [
            f"load {base_cif}, BASE_{baseline_name}",
            f"color grey70, BASE_{baseline_name}",
            f"set cartoon_transparency, 0.35, BASE_{baseline_name}",
            "",
            f"load {cif}, COND_{name}",
            f"align COND_{name}, BASE_{baseline_name}",
            "",
        ]
        lines3 += _PML_FOOTER
        lines3 += [
            f"spectrum b, blue_white_red, COND_{name}, minimum=50, maximum=100",
            f"color grey70, BASE_{baseline_name}",
            "",
            "# --- Tips ---",
            f"# super COND_{name}, BASE_{baseline_name}",
            f"# show sticks, resi 100-120 and COND_{name}",
            f"# save {name}.pse",
        ]
        (per_dir / f"{name}.pml").write_text("\n".join(lines3) + "\n", encoding="utf-8")

    log.info("PyMOL scripts: %s", pymol_dir)


def write_pymol_survey(
    conditions: Dict[str, ConditionModel],
    output_dir: Path,
) -> None:
    """
    Two PyMOL scripts for survey mode.

    01_overlay.pml   — all conditions aligned to the first (alphabetically),
                       each coloured distinctly
    02_plddt.pml     — all conditions aligned, pLDDT spectrum
    """
    pymol_dir = output_dir / "pymol"
    pymol_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(conditions)
    ref_name = names[0]          # first alphabetically as alignment anchor
    ref_cif  = conditions[ref_name].cif_path.resolve()
    others   = names[1:]

    # --- 01_overlay.pml ---
    lines = _pml_header(
        "AF3 Survey Overlay — all conditions",
        f"Alignment reference (first alphabetically): {ref_name}",
    )
    # Load reference
    lines += [
        f"load {ref_cif}, {ref_name}",
        f"color grey70, {ref_name}",
        f"set cartoon_transparency, 0.35, {ref_name}",
        "",
    ]
    # Load and align all others
    for idx, name in enumerate(others):
        cif = conditions[name].cif_path.resolve()
        color = _PALETTE[idx % len(_PALETTE)]
        lines += [
            f"load {cif}, {name}",
            f"align {name}, {ref_name}",
            f"color {color}, {name}",
            "",
        ]
    lines += _PML_FOOTER
    lines += [
        "# --- Tips ---",
        "# To use a different reference:",
        f"#   align <condition_name>, <your_reference>",
        "# super is more robust for distant structures:",
        f"#   super <condition_name>, {ref_name}",
        "# save survey.pse",
    ]
    (pymol_dir / "01_overlay.pml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- 02_plddt.pml ---
    lines2 = _pml_header(
        "AF3 Survey pLDDT Map — all conditions",
        "B-factor = pLDDT.  blue=high, red=low.",
    )
    lines2 += [f"load {ref_cif}, {ref_name}", ""]
    for name in others:
        cif = conditions[name].cif_path.resolve()
        lines2 += [
            f"load {cif}, {name}",
            f"align {name}, {ref_name}",
            "",
        ]
    lines2 += _PML_FOOTER
    lines2 += [
        "spectrum b, blue_white_red, polymer, minimum=50, maximum=100",
        "# save plddt.pse",
    ]
    (pymol_dir / "02_plddt.pml").write_text("\n".join(lines2) + "\n", encoding="utf-8")

    log.info("PyMOL scripts: %s", pymol_dir)

# ===========================================================================
# PLOTS
# ===========================================================================

def plot_baseline_distances(
    df: pd.DataFrame,
    baseline_name: str,
    output_dir: Path,
) -> None:
    """Bar chart(s): RMSD and optionally TM-score per condition vs baseline."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    has_tm = df["tm_score_norm_cond"].notna().any()
    n_panels = 2 if has_tm else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    names  = df["condition"].tolist()
    labels = _condition_labels(names)
    x      = range(len(names))

    ax = axes[0]
    ax.bar(x, df["rmsd_angstrom"], color="#1f77b4", edgecolor="black", alpha=0.85)
    for i, v in enumerate(df["rmsd_angstrom"]):
        if math.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Protein Cα RMSD vs baseline (Å)")
    ax.set_title(f"Structural Distance vs {baseline_name}")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if has_tm:
        ax2 = axes[1]
        tm_vals = df["tm_score_norm_cond"].tolist()
        ax2.bar(x, tm_vals, color="#2ca02c", edgecolor="black", alpha=0.85)
        for i, v in enumerate(tm_vals):
            if math.isfinite(v):
                ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax2.set_ylabel("TM-score (normalised to condition)")
        ax2.set_title("TM-score vs baseline")
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Structural Distances vs Baseline", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "structural_distances.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/structural_distances.png")


def _parse_condition_metadata(name: str) -> Dict[str, str]:
    """
    Extract structured metadata from an AF3 condition name by pattern matching.

    Returns a dict with keys:
      ion_tier   — "1x" / "10x" / "100x" / "1000x" / "unknown"
      ptm        — "SEP" / "TPO" / "none" / "other:<type>"
      has_dna    — "yes" / "no"
    """
    name_l = name.lower()

    # Ion concentration tier from nax{N} pattern
    m = re.search(r"nax(\d+)", name_l)
    if m:
        n = int(m.group(1))
        ion_tier = f"{n}x"
    else:
        ion_tier = "unknown"

    # PTM type
    if "sep" in name_l:
        ptm = "SEP"
    elif "tpo" in name_l:
        ptm = "TPO"
    elif "ptm" in name_l:
        ptm = "PTM"
    else:
        ptm = "none"

    # DNA presence
    has_dna = "yes" if "dna" in name_l else "no"

    return {"ion_tier": ion_tier, "ptm": ptm, "has_dna": has_dna}


def plot_pairwise_heatmap(
    df_matrix: pd.DataFrame,
    df_conf: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    N×N RMSD heatmap with hierarchical clustering.

    Improvements:
    - Metadata color strips on both axes: ion tier, PTM, DNA presence
    - Failed conditions (from df_conf) marked with a border on the strip
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    names = list(df_matrix.index)
    n = len(names)
    mat = df_matrix.values.copy()

    # Hierarchical clustering order
    order = list(range(n))
    if HAS_SCIPY and n > 2:
        try:
            mat_fill = np.where(np.isnan(mat), np.nanmax(mat) * 1.1, mat)
            np.fill_diagonal(mat_fill, 0.0)
            condensed = squareform(mat_fill)
            Z = linkage(condensed, method="average")
            order = list(leaves_list(Z))
        except Exception:
            pass

    reordered_names = [names[i] for i in order]
    reordered_mat   = mat[np.ix_(order, order)]
    labels = _condition_labels(reordered_names, max_len=18)

    # Metadata for each condition
    meta = [_parse_condition_metadata(nm) for nm in reordered_names]

    # Color maps for metadata strips
    ion_tiers = sorted({m["ion_tier"] for m in meta})
    ptm_types = sorted({m["ptm"] for m in meta})
    ion_cmap  = plt.colormaps.get_cmap("Blues").resampled(max(len(ion_tiers), 2))
    ptm_colors = {"none": "#cccccc", "SEP": "#e377c2", "TPO": "#17becf", "PTM": "#bcbd22"}
    dna_colors = {"yes": "#2ca02c", "no": "#dddddd"}

    ion_color  = {t: ion_cmap(i / max(len(ion_tiers) - 1, 1)) for i, t in enumerate(ion_tiers)}

    # Figure layout: main heatmap + 3 metadata strips (ion, ptm, dna)
    strip_h = 0.18   # inches per strip
    heatmap_size = max(6, n * 0.7)
    fig = plt.figure(figsize=(heatmap_size + 1.5, heatmap_size + strip_h * 3 + 0.5))

    # GridSpec: 4 rows (3 strips + heatmap), 2 cols (heatmap + colorbar)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[strip_h, strip_h, strip_h, heatmap_size],
        width_ratios=[heatmap_size, 0.4],
        hspace=0.02, wspace=0.05,
    )

    ax_ion  = fig.add_subplot(gs[0, 0])
    ax_ptm  = fig.add_subplot(gs[1, 0])
    ax_dna  = fig.add_subplot(gs[2, 0])
    ax_main = fig.add_subplot(gs[3, 0])
    ax_cb   = fig.add_subplot(gs[3, 1])

    # Draw metadata strips
    for strip_ax, key, cmap_dict, title in [
        (ax_ion,  "ion_tier", ion_color,  "Ion"),
        (ax_ptm,  "ptm",      ptm_colors, "PTM"),
        (ax_dna,  "has_dna",  dna_colors, "DNA"),
    ]:
        for i, m in enumerate(meta):
            val = m[key]
            c = cmap_dict.get(val, "#aaaaaa")
            strip_ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c, ec="white", lw=0.5))
        strip_ax.set_xlim(0, n)
        strip_ax.set_ylim(0, 1)
        strip_ax.set_yticks([0.5])
        strip_ax.set_yticklabels([title], fontsize=7)
        strip_ax.set_xticks([])
        strip_ax.tick_params(left=False)
        for spine in strip_ax.spines.values():
            spine.set_visible(False)

    # Main heatmap
    masked = np.ma.masked_invalid(reordered_mat)
    im = ax_main.imshow(masked, cmap="YlOrRd", vmin=0, aspect="auto")
    fig.colorbar(im, cax=ax_cb, label="Protein Cα RMSD (Å)")

    ax_main.set_xticks(range(n))
    ax_main.set_yticks(range(n))
    ax_main.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_main.set_yticklabels(labels, fontsize=8)
    ax_main.set_title("Pairwise RMSD Matrix (clustered)", fontweight="bold", pad=4)

    vmax = float(np.nanmax(reordered_mat)) if not np.all(np.isnan(reordered_mat)) else 1.0
    for i in range(n):
        for j in range(n):
            v = reordered_mat[i, j]
            if math.isfinite(v):
                ax_main.text(j, i, f"{v:.1f}", ha="center", va="center",
                             fontsize=6,
                             color="white" if v > vmax * 0.6 else "black")

    # Legend for strips
    legend_lines = []
    from matplotlib.patches import Patch
    for tier in ion_tiers:
        legend_lines.append(Patch(facecolor=ion_color[tier], label=f"ion {tier}"))
    for pt in ptm_types:
        legend_lines.append(Patch(facecolor=ptm_colors.get(pt, "#aaaaaa"), label=f"PTM: {pt}"))
    legend_lines.append(Patch(facecolor=dna_colors["yes"], label="DNA: yes"))
    legend_lines.append(Patch(facecolor=dna_colors["no"],  label="DNA: no"))
    ax_main.legend(
        handles=legend_lines, loc="lower right", fontsize=6,
        framealpha=0.8, ncol=2, title="Metadata", title_fontsize=6,
    )

    plt.savefig(plots_dir / "pairwise_rmsd_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/pairwise_rmsd_heatmap.png")


def plot_rmsd_distribution(
    df_pairs: pd.DataFrame,
    median_rmsd: float,
    output_dir: Path,
) -> None:
    """Histogram of all pairwise RMSDs with median line."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    vals = df_pairs["rmsd_angstrom"].dropna().tolist()
    if not vals:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=max(5, len(vals) // 3), color="#1f77b4", edgecolor="black", alpha=0.8)
    if math.isfinite(median_rmsd):
        ax.axvline(median_rmsd, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Median = {median_rmsd:.2f} Å")
        ax.legend(fontsize=9)
    ax.set_xlabel("Protein Cα RMSD (Å)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Distribution of Pairwise RMSDs", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "rmsd_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/rmsd_distribution.png")


def plot_per_residue(
    profiles: Dict[str, pd.DataFrame],
    output_dir: Path,
    mode: str = "baseline",
    ref_sd: Optional[np.ndarray] = None,
    global_disp_max: Optional[float] = None,
    ref_name: str = "reference",
) -> None:
    """
    One figure per profile: displacement (top) + pLDDT overlay (bottom).

    Improvements:
    - Standardized Y-axis across all panels (global_disp_max)
    - Shaded SD band on displacement panel showing reference intra-condition
      variability (ref_sd), as a positive stability indicator
    - Protein / nucleic boundary marker
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ref_label  = "baseline pLDDT" if mode == "baseline" else "ref pLDDT"
    cond_label = "condition pLDDT" if mode == "baseline" else "cond pLDDT"

    keys = list(profiles.keys())
    short_labels = _condition_labels(keys, max_len=60)
    key_to_short = dict(zip(keys, short_labels))

    # Y-axis ceiling: use global max with 10% headroom, minimum 5 Å
    if global_disp_max is not None and math.isfinite(global_disp_max):
        y_ceil = max(5.0, global_disp_max * 1.10)
    else:
        y_ceil = None  # auto per-plot

    for key, df in profiles.items():
        short = key_to_short[key]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        title = f"Per-residue vs {ref_name}: {short}"
        fig.suptitle(title, fontweight="bold", fontsize=10)

        prot_mask = df["residue_type"] == "protein" if "residue_type" in df.columns else pd.Series([True] * len(df))
        x_all = np.arange(len(df))
        x_prot = x_all[prot_mask.values]
        disp_all = df["displacement_angstrom"].values

        # --- Top panel: displacement ---
        ax1.plot(x_all, disp_all, color="#d62728", linewidth=0.8, alpha=0.9)
        ax1.fill_between(x_all, disp_all, alpha=0.15, color="#d62728")

        # SD band: reference intra-condition variability (protein residues only)
        if ref_sd is not None and len(x_prot) > 0:
            n_prot = min(len(x_prot), len(ref_sd))
            ax1.fill_between(
                x_prot[:n_prot],
                ref_sd[:n_prot],
                alpha=0.25,
                color="#1f77b4",
                label=f"{ref_name} seed SD",
            )
            ax1.legend(fontsize=7, loc="upper right")

        ax1.set_ylabel("Cα / C4' displacement (Å)")
        if y_ceil is not None:
            ax1.set_ylim(0, y_ceil)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)

        # --- Bottom panel: pLDDT ---
        ax2.plot(x_all, df["ref_plddt"].values,  color="#1f77b4", linewidth=0.8, label=ref_label)
        ax2.plot(x_all, df["cond_plddt"].values, color="#ff7f0e", linewidth=0.8, label=cond_label)
        ax2.set_ylabel("pLDDT")
        ax2.set_xlabel("Residue position")
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

        # Chain boundary markers
        if "chain_id" in df.columns:
            chains = df["chain_id"].tolist()
            for i in range(1, len(chains)):
                if chains[i] != chains[i - 1]:
                    ax1.axvline(i, color="gray", linestyle=":", linewidth=0.8)
                    ax2.axvline(i, color="gray", linestyle=":", linewidth=0.8)
                    ax2.text(i + 1, 3, chains[i], fontsize=7, color="gray",
                             rotation=90, va="bottom")

        # Protein / nucleic boundary
        if "residue_type" in df.columns and "nucleic" in df["residue_type"].values:
            split = int(df.index[df["residue_type"] == "nucleic"][0])
            for ax in (ax1, ax2):
                ax.axvline(split, color="#9467bd", linestyle="--", linewidth=1.2)
            ax2.text(split + 1, 95, "nucleic", fontsize=7, color="#9467bd", va="top")

        plt.tight_layout()
        safe_key = re.sub(r"[^\w\-]", "_", key)[:120]
        fname = plots_dir / f"per_residue_{safe_key}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved: plots/per_residue_%s.png", safe_key)


def _detect_failed_conditions(
    df: pd.DataFrame,
    seed_scores: Dict[str, Dict[str, List[float]]],
) -> set:
    """
    Identify conditions that appear to be failed predictions using
    Tukey fences on ipTM (primary) or mean PAE (fallback).

    A condition is flagged if its ipTM is below Q1 - 1.5*IQR of the
    distribution across all conditions, OR if its mean PAE is above
    Q3 + 1.5*IQR.  Returns a set of condition names.
    """
    failed: set = set()

    # ipTM-based detection
    if "iptm" in df.columns:
        vals = df["iptm"].dropna()
        if len(vals) >= 4:
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            low_fence = q1 - 1.5 * iqr
            for _, row in df.iterrows():
                if math.isfinite(row.get("iptm", float("nan"))) and row["iptm"] < low_fence:
                    failed.add(row["condition"])

    # PAE-based detection (catches monomers where ipTM is NaN)
    if "mean_pae" in df.columns:
        vals = df["mean_pae"].dropna()
        if len(vals) >= 4:
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            high_fence = q3 + 1.5 * iqr
            for _, row in df.iterrows():
                if math.isfinite(row.get("mean_pae", float("nan"))) and row["mean_pae"] > high_fence:
                    failed.add(row["condition"])

    return failed


def plot_confidence_summary(
    df: pd.DataFrame,
    output_dir: Path,
    seed_scores: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> None:
    """
    Bar chart: pTM, ipTM, mean pLDDT per condition.

    Improvements:
    - Error bars showing per-seed SD (when seed_scores provided)
    - Failed conditions (Tukey fence outliers) shown in red/orange
    - Reference condition hatched
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("ptm",        "pTM",        "#1f77b4", "ptm"),
        ("iptm",       "ipTM",       "#ff7f0e", "iptm"),
        ("plddt_mean", "Mean pLDDT", "#2ca02c", None),
    ]
    metrics = [(c, l, col, sk) for c, l, col, sk in metrics
               if c in df.columns and df[c].notna().any()]
    if not metrics:
        return

    failed = _detect_failed_conditions(df, seed_scores or {})

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    names  = df["condition"].tolist()
    labels = _condition_labels(names)
    x      = range(len(names))

    for ax, (col, lbl, color, score_key) in zip(axes, metrics):
        vals = df[col].tolist()

        # Per-bar colors: red for failed, normal color otherwise
        bar_colors = [
            "#d62728" if names[i] in failed else color
            for i in range(len(names))
        ]

        # SD error bars from seed replicates
        yerr = None
        if seed_scores and score_key:
            sds = []
            for name in names:
                sc = seed_scores.get(name, {}).get(score_key, [])
                sds.append(float(np.std(sc, ddof=1)) if len(sc) >= 2 else 0.0)
            yerr = sds

        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", alpha=0.85,
                      yerr=yerr, capsize=4, error_kw={"linewidth": 1.2, "ecolor": "black"})

        # Hatch reference
        if "is_reference" in df.columns:
            for bi in df.index[df["is_reference"]].tolist():
                bars[bi].set_hatch("//")
                bars[bi].set_linewidth(2)

        for i, v in enumerate(vals):
            if math.isfinite(v):
                ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel(lbl)
        ax.set_title(lbl)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    notes = []
    if "is_reference" in df.columns and df["is_reference"].any():
        notes.append("hatched = reference")
    if failed:
        notes.append("red = likely failed prediction")
    note_str = f"  ({', '.join(notes)})" if notes else ""
    fig.suptitle(f"Confidence Metrics per Condition{note_str}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "confidence_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/confidence_summary.png")
    if failed:
        log.info("  Flagged as likely failed: %s", ", ".join(sorted(failed)))


def plot_pae_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of mean PAE per condition (skipped if no PAE data)."""
    if "mean_pae" not in df.columns or not df["mean_pae"].notna().any():
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    names  = df["condition"].tolist()
    labels = _condition_labels(names)
    x      = range(len(names))
    vals   = df["mean_pae"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(x, vals, color="#9467bd", edgecolor="black", alpha=0.85)
    if "is_reference" in df.columns:
        for bi in df.index[df["is_reference"]].tolist():
            bars[bi].set_hatch("//")
    for i, v in enumerate(vals):
        if math.isfinite(v):
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean PAE (Å)")
    ax.set_title("Mean PAE per Condition", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "pae_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/pae_comparison.png")


# ===========================================================================
# STRUCTURED FACTORIAL ANALYSIS PLOTS
# ===========================================================================

def plot_panel_per_residue(
    conditions: Dict[str, ConditionModel],
    struct: ExperimentStructure,
    baseline_name: str,
    baseline_sd: Optional[np.ndarray],
    global_disp_max: float,
    output_dir: Path,
) -> None:
    """
    Grid of per-residue displacement profiles.
    Rows = PTM group, Columns = ion concentration tier.
    Each cell = displacement vs baseline (fixed reference).
    Only conditions in struct.panel_conditions are shown.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows_list = struct.ptm_order
    cols_list  = struct.tier_order
    n_rows = len(rows_list)
    n_cols = len(cols_list)

    if n_rows == 0 or n_cols == 0:
        log.info("Structured panel: no panel conditions found, skipping.")
        return

    # Build lookup: (ptm_group, ion_tier) → condition name
    cell: Dict[tuple, str] = {}
    for name in struct.panel_conditions:
        key = (struct.ptm_group[name], struct.ion_tier[name])
        cell[key] = name  # last one wins if duplicates

    ref = conditions[baseline_name]
    y_ceil = max(5.0, global_disp_max * 1.10)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        sharex=True, sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Per-residue displacement vs {baseline_name}\n"
        f"Rows = PTM group  |  Columns = ion concentration",
        fontweight="bold", fontsize=10,
    )

    for ri, ptm in enumerate(rows_list):
        for ci, tier in enumerate(cols_list):
            ax = axes[ri][ci]
            key = (ptm, tier)
            name = cell.get(key)

            # Column / row labels
            if ri == 0:
                ax.set_title(tier, fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ptm if ptm != "none" else "unmodified", fontsize=8)

            if name is None or name not in conditions:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            cond = conditions[name]
            R_mat, t_vec, _, _ = align_and_rmsd(ref, cond)
            prot_disp, _ = per_residue_displacement(ref, cond, R_mat, t_vec)
            x = np.arange(len(prot_disp))

            ax.plot(x, prot_disp, color="#d62728", linewidth=0.7, alpha=0.9)
            ax.fill_between(x, prot_disp, alpha=0.12, color="#d62728")

            # Reference SD band
            if baseline_sd is not None:
                n_prot = min(len(x), len(baseline_sd))
                ax.fill_between(
                    x[:n_prot], baseline_sd[:n_prot],
                    alpha=0.2, color="#1f77b4",
                )

            ax.set_ylim(0, y_ceil)
            ax.grid(axis="y", linestyle="--", alpha=0.25)

            # Annotate mean displacement
            mean_d = float(np.mean(prot_disp))
            ax.text(0.97, 0.95, f"μ={mean_d:.1f}Å",
                    ha="right", va="top", transform=ax.transAxes,
                    fontsize=7, color="#d62728")

    # Shared axis labels
    for ax in axes[-1]:
        ax.set_xlabel("Residue position", fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "panel_per_residue.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/panel_per_residue.png")


def plot_concentration_response(
    conditions: Dict[str, ConditionModel],
    struct: ExperimentStructure,
    baseline_name: str,
    output_dir: Path,
) -> None:
    """
    Mean displacement vs ion concentration tier, one line per PTM group.
    Shows whether the concentration response is monotonic, saturating, or
    non-monotonic.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ref = conditions[baseline_name]

    # Build (ptm_group, tier) → mean_displacement
    data: Dict[str, Dict[str, float]] = {}  # ptm → {tier: mean_disp}
    for name in struct.panel_conditions:
        ptm  = struct.ptm_group[name]
        tier = struct.ion_tier[name]
        if name not in conditions:
            continue
        cond = conditions[name]
        R_mat, t_vec, _, _ = align_and_rmsd(ref, cond)
        prot_disp, _ = per_residue_displacement(ref, cond, R_mat, t_vec)
        mean_d = float(np.mean(prot_disp))
        data.setdefault(ptm, {})[tier] = mean_d

    if not data:
        return

    # Colour per PTM group
    ptm_colors = {
        "none": "#1f77b4",
        "SEP102": "#e377c2", "SEP": "#e377c2",
        "TPO101": "#17becf", "TPO": "#17becf",
        "TPO235": "#17becf",
    }

    def _tier_key(t: str) -> float:
        if t == "0x": return 0.0
        try: return float(t.rstrip("x"))
        except: return 9999.0

    fig, ax = plt.subplots(figsize=(7, 4))

    for ptm in struct.ptm_order:
        if ptm not in data:
            continue
        tier_vals = data[ptm]
        tiers_sorted = sorted(tier_vals.keys(), key=_tier_key)
        x_labels = tiers_sorted
        y_vals   = [tier_vals[t] for t in tiers_sorted]
        color = ptm_colors.get(ptm, "#7f7f7f")
        label = ptm if ptm != "none" else "unmodified"
        ax.plot(x_labels, y_vals, marker="o", linewidth=1.8,
                color=color, label=label, markersize=6)
        for xi, (xl, yv) in enumerate(zip(x_labels, y_vals)):
            ax.text(xi, yv + 0.1, f"{yv:.1f}", ha="center", va="bottom",
                    fontsize=7, color=color)

    ax.set_xlabel("Ion concentration tier")
    ax.set_ylabel("Mean Cα displacement vs reference (Å)")
    ax.set_title(
        f"Concentration–response  |  reference: {baseline_name}",
        fontweight="bold",
    )
    ax.legend(fontsize=9, title="PTM group")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plots_dir / "concentration_response.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/concentration_response.png")


def plot_ptm_effect_grid(
    conditions: Dict[str, ConditionModel],
    struct: ExperimentStructure,
    baseline_name: str,
    output_dir: Path,
) -> None:
    """
    2D heatmap: rows = PTM group, columns = ion tier.
    Cell value = mean Cα displacement vs baseline.
    Makes the factorial structure and PTM×concentration interactions visible.
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ref = conditions[baseline_name]
    rows_list = struct.ptm_order
    cols_list  = struct.tier_order

    if not rows_list or not cols_list:
        return

    grid = np.full((len(rows_list), len(cols_list)), float("nan"))

    for ri, ptm in enumerate(rows_list):
        for ci, tier in enumerate(cols_list):
            # Find matching condition
            matches = [
                n for n in struct.panel_conditions
                if struct.ptm_group[n] == ptm and struct.ion_tier[n] == tier
                and n in conditions
            ]
            if not matches:
                continue
            name = matches[0]
            cond = conditions[name]
            R_mat, t_vec, _, _ = align_and_rmsd(ref, cond)
            prot_disp, _ = per_residue_displacement(ref, cond, R_mat, t_vec)
            grid[ri, ci] = float(np.mean(prot_disp))

    fig, ax = plt.subplots(figsize=(max(4, len(cols_list) * 1.5 + 1),
                                    max(3, len(rows_list) * 1.2 + 1)))
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Cα displacement (Å)", shrink=0.8)

    ax.set_xticks(range(len(cols_list)))
    ax.set_yticks(range(len(rows_list)))
    ax.set_xticklabels(cols_list, fontsize=9)
    ax.set_yticklabels(
        [p if p != "none" else "unmodified" for p in rows_list],
        fontsize=9,
    )
    ax.set_xlabel("Ion concentration tier")
    ax.set_ylabel("PTM group")
    ax.set_title(
        f"PTM × Concentration effect grid\nreference: {baseline_name}",
        fontweight="bold",
    )

    for ri in range(len(rows_list)):
        for ci in range(len(cols_list)):
            v = grid[ri, ci]
            if math.isfinite(v):
                ax.text(ci, ri, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > np.nanmax(grid) * 0.6 else "black")

    plt.tight_layout()
    plt.savefig(plots_dir / "ptm_effect_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved: plots/ptm_effect_grid.png")


# ===========================================================================
# CLI + MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AF3 Condition Comparison Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--models", "-m", required=True, type=Path,
                   help="Root folder; each subdirectory is one AF3 condition.")
    p.add_argument("--baseline", "-b", default=None,
                   help="Baseline condition name (auto-detected if omitted).")
    p.add_argument("--output", "-o", default="af3_results", type=Path,
                   help="Output directory (default: af3_results).")
    p.add_argument("--chains", default=None,
                   help="Restrict alignment to these protein chain IDs, e.g. A,B.")
    p.add_argument("--pymol", action="store_true",
                   help="Generate PyMOL .pml scripts.")
    p.add_argument("--tm", action="store_true",
                   help="Compute TM-score (requires: pip install tmtools).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    models_dir: Path = args.models.resolve()
    output_dir: Path = args.output.resolve()
    chain_filter: Optional[List[str]] = (
        [c.strip() for c in args.chains.split(",")] if args.chains else None
    )
    compute_tm: bool = args.tm and HAS_TMTOOLS

    if args.tm and not HAS_TMTOOLS:
        log.warning("--tm requested but tmtools not installed. pip install tmtools")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("%s=== AF3 Condition Comparison Pipeline ===%s", B, R)
    log.info("Models : %s", models_dir)
    log.info("Output : %s", output_dir)

    # ------------------------------------------------------------------
    # Load conditions
    # ------------------------------------------------------------------
    conditions = discover_conditions(models_dir, chain_filter)
    log.info("Loaded %d conditions", len(conditions))

    # ------------------------------------------------------------------
    # Resolve baseline and run analysis
    # ------------------------------------------------------------------
    baseline_name = resolve_baseline(conditions, args.baseline)
    log.info("Baseline : %s%s%s", B, baseline_name, R)
    write_representative_selection(conditions, baseline_name, output_dir)

    run_baseline_mode(
        conditions, baseline_name, compute_tm, output_dir, args.pymol,
        models_dir, chain_filter,
    )

    log.info("%s=== Done  →  %s ===%s", GR, output_dir, R)


if __name__ == "__main__":
    main()
