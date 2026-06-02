"""
IO — discovery, structure/confidence loading, ensemble loading, table writers.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .factors import parse_condition_factors
from .model import ConditionModel, EnsembleModel

log = logging.getLogger("af3bench.io")

try:
    from Bio.PDB import MMCIFParser
    HAS_BIOPYTHON = True
except ImportError:  # pragma: no cover
    HAS_BIOPYTHON = False

_SEED_RE = re.compile(r"seed-\d+_sample-\d+")

# 3-letter -> 1-letter for TM-score sequences (standard + common PTM parents)
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    "SEP": "S", "TPO": "T", "PTR": "Y", "MSE": "M",
}


# ---------------------------------------------------------------------------
# Confidence loading
# ---------------------------------------------------------------------------

def load_summary_confidences(model: ConditionModel, path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        model.ptm = float(d["ptm"]) if d.get("ptm") is not None else float("nan")
        model.iptm = float(d["iptm"]) if d.get("iptm") is not None else float("nan")
        model.ranking_score = (
            float(d["ranking_score"]) if d.get("ranking_score") is not None else float("nan")
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse summary confidences %s: %s", path.name, exc)


def load_full_confidences(model: ConditionModel, path: Path) -> None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if "atom_plddts" in d:
            model.atom_plddts = np.array(d["atom_plddts"], dtype=np.float32)
        if "pae" in d:
            model.pae_matrix = np.array(d["pae"], dtype=np.float32)
        if "token_chain_ids" in d:
            model.token_chain_ids = [str(c) for c in d["token_chain_ids"]]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse full confidences %s: %s", path.name, exc)


def apply_factors(model: ConditionModel, data_json: Path) -> None:
    f = parse_condition_factors(data_json)
    model.protein_chain_ids_from_json = f["protein_chain_ids"]
    model.nucleic_chain_ids_from_json = f["nucleic_chain_ids"]
    model.description = f["description"]
    model.ptm_labels = f["ptm_labels"]
    model.n_na = f["n_na"]
    model.n_cl = f["n_cl"]
    model.n_water = f["n_water"]
    model.has_real_ligand = f["has_real_ligand"]
    model.ion_count = f["ion_count_legacy"]


# ---------------------------------------------------------------------------
# CIF parsing
# ---------------------------------------------------------------------------

def _parse_cif_backbone(
    cif_path: Path,
    name: str,
    chain_filter: Optional[List[str]],
) -> Optional[dict]:
    """
    Parse a CIF and return backbone arrays, or None on failure.

    Returns dict with ca_coords, ca_plddts, ca_chain_ids, ca_res_indices,
    ca_seq_letters, and the matching na_* arrays.
    """
    if not HAS_BIOPYTHON:
        log.error("Biopython not installed — cannot parse CIF files.")
        return None
    try:
        parser = MMCIFParser(QUIET=True)
        struct = parser.get_structure(name, str(cif_path))
        m = struct[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("Error loading structure %s: %s", cif_path.name, exc)
        return None

    ca_c, ca_p, ca_ch, ca_ri, ca_seq = [], [], [], [], []
    na_c, na_p, na_ch, na_ri = [], [], [], []

    for chain in m:
        for residue in chain:
            if "CA" in residue:
                if chain_filter and chain.id not in chain_filter:
                    continue
                atom = residue["CA"]
                ca_c.append(atom.get_coord())
                ca_p.append(atom.bfactor)
                ca_ch.append(chain.id)
                ca_ri.append(residue.get_id()[1])
                ca_seq.append(_THREE_TO_ONE.get(residue.resname.strip().upper(), "A"))
            elif "C4'" in residue:
                atom = residue["C4'"]
                na_c.append(atom.get_coord())
                na_p.append(atom.bfactor)
                na_ch.append(chain.id)
                na_ri.append(residue.get_id()[1])

    if not ca_c and not na_c:
        return None

    return {
        "ca_coords": np.array(ca_c, dtype=np.float64) if ca_c else np.empty((0, 3)),
        "ca_plddts": np.array(ca_p, dtype=np.float64) if ca_c else np.empty(0),
        "ca_chain_ids": ca_ch,
        "ca_res_indices": ca_ri,
        "ca_seq_letters": ca_seq,
        "na_coords": np.array(na_c, dtype=np.float64) if na_c else np.empty((0, 3)),
        "na_plddts": np.array(na_p, dtype=np.float64) if na_c else np.empty(0),
        "na_chain_ids": na_ch,
        "na_res_indices": na_ri,
    }


def load_structure(model: ConditionModel, chain_filter: Optional[List[str]]) -> bool:
    parsed = _parse_cif_backbone(model.cif_path, model.name, chain_filter)
    if parsed is None:
        return False
    model.ca_coords = parsed["ca_coords"]
    model.ca_plddts = parsed["ca_plddts"]
    model.ca_chain_ids = parsed["ca_chain_ids"]
    model.ca_res_indices = parsed["ca_res_indices"]
    model.ca_seq_letters = parsed["ca_seq_letters"]  # dynamic attr for TM-score
    model.na_coords = parsed["na_coords"]
    model.na_plddts = parsed["na_plddts"]
    model.na_chain_ids = parsed["na_chain_ids"]
    model.na_res_indices = parsed["na_res_indices"]
    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_conditions(
    models_dir: Path,
    chain_filter: Optional[List[str]],
) -> Tuple[Dict[str, ConditionModel], List[str]]:
    """
    Scan ``models_dir``; each immediate subdirectory with a top-level
    ``*_model.cif`` is one condition.

    Returns (conditions sorted by name, skipped_names).
    """
    if not HAS_BIOPYTHON:
        log.error("Biopython is required.  pip install biopython")
        sys.exit(1)

    conditions: Dict[str, ConditionModel] = {}
    skipped: List[str] = []

    subdirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if not subdirs:
        log.error("No subdirectories found in %s", models_dir)
        sys.exit(1)

    for cond_dir in subdirs:
        name = cond_dir.name
        cif_path = cond_dir / f"{name}_model.cif"
        if not cif_path.exists():
            candidates = [
                f for f in cond_dir.glob("*_model.cif")
                if f.parent == cond_dir and not _SEED_RE.search(f.name)
            ]
            if not candidates:
                log.debug("Skipping %s — no top-level *_model.cif", name)
                skipped.append(name)
                continue
            cif_path = candidates[0]

        model = ConditionModel(name, cif_path)

        summary_json = cond_dir / f"{name}_summary_confidences.json"
        if summary_json.exists():
            load_summary_confidences(model, summary_json)
        else:
            log.warning("No summary_confidences.json for '%s'", name)

        full_json = cond_dir / f"{name}_confidences.json"
        if full_json.exists():
            load_full_confidences(model, full_json)

        input_json = cond_dir / f"{name}_data.json"
        if input_json.exists():
            apply_factors(model, input_json)

        if chain_filter:
            eff = chain_filter
        elif model.protein_chain_ids_from_json:
            eff = model.protein_chain_ids_from_json
        else:
            eff = None

        if not load_structure(model, eff):
            log.warning("Skipping '%s' — structure could not be loaded", name)
            skipped.append(name)
            continue
        if model.n_protein_residues == 0:
            log.warning("Skipping '%s' — no protein Calpha atoms found", name)
            skipped.append(name)
            continue

        conditions[name] = model
        log.info(
            "Loaded %s  protein=%d aa  nucleic=%d nt  pTM=%.3f  ipTM=%s | %s",
            name, model.n_protein_residues, model.n_nucleic_residues, model.ptm,
            f"{model.iptm:.3f}" if math.isfinite(model.iptm) else "N/A",
            model.description,
        )

    if not conditions:
        deeper = [f for f in models_dir.glob("*/*_model.cif") if not _SEED_RE.search(f.name)]
        if deeper:
            log.error(
                "No valid conditions in %s, but model files exist one level deeper "
                "(e.g. %s). Point --models at the directory whose immediate "
                "subdirectories are AF3 job folders.",
                models_dir, deeper[0].parent.parent,
            )
        else:
            log.error("No valid conditions found in %s", models_dir)
        sys.exit(1)

    return dict(sorted(conditions.items())), skipped


# ---------------------------------------------------------------------------
# Ensemble loading
# ---------------------------------------------------------------------------

def load_ensemble(
    cond_dir: Path,
    chain_filter: Optional[List[str]],
    reference_keys: Optional[List[tuple]] = None,
    max_samples: Optional[int] = None,
) -> EnsembleModel:
    """
    Load every seed-*_sample-* replicate model for a condition into an
    EnsembleModel.  Coordinates are aligned to ``reference_keys`` (the
    representative's (chain,resnum) ordering) so all samples share one ordering.
    """
    name = cond_dir.name
    ens = EnsembleModel(name=name)
    if not HAS_BIOPYTHON:
        return ens

    seed_dirs = sorted(
        d for d in cond_dir.iterdir()
        if d.is_dir() and _SEED_RE.search(d.name)
    )
    if not seed_dirs:
        return ens

    coords_list: List[np.ndarray] = []
    plddt_list: List[np.ndarray] = []
    ptm_list: List[float] = []
    iptm_list: List[float] = []
    plddtmean_list: List[float] = []
    paths: List[Path] = []
    keys_ref: Optional[List[tuple]] = list(reference_keys) if reference_keys else None

    for sd in seed_dirs:
        cifs = [f for f in sd.glob("*_model.cif")]
        if not cifs:
            continue
        parsed = _parse_cif_backbone(cifs[0], sd.name, chain_filter)
        if parsed is None or parsed["ca_coords"].shape[0] == 0:
            continue

        keys = list(zip(parsed["ca_chain_ids"], parsed["ca_res_indices"]))
        coord = parsed["ca_coords"]
        pl = parsed["ca_plddts"]

        if keys_ref is None:
            keys_ref = keys

        # Reindex onto the reference key ordering; skip residues not present
        lookup = {k: i for i, k in enumerate(keys)}
        sel = [lookup.get(k) for k in keys_ref]
        if any(s is None for s in sel):
            # ragged sample — keep only fully-matching ones for a clean stack
            valid = [s for s in sel if s is not None]
            if len(valid) != len(keys_ref):
                # tolerate by trimming reference to the intersection on first ragged
                continue
        sel_arr = np.array([s for s in sel], dtype=int)
        coords_list.append(coord[sel_arr])
        plddt_list.append(pl[sel_arr])
        paths.append(cifs[0])

        # per-sample scores
        sj = list(sd.glob("*_summary_confidences.json"))
        if sj:
            try:
                with open(sj[0], "r", encoding="utf-8") as fh:
                    sd_json = json.load(fh)
                ptm_list.append(float(sd_json["ptm"]) if sd_json.get("ptm") is not None else np.nan)
                iptm_list.append(float(sd_json["iptm"]) if sd_json.get("iptm") is not None else np.nan)
            except Exception:  # noqa: BLE001
                ptm_list.append(np.nan)
                iptm_list.append(np.nan)
        plddtmean_list.append(float(np.mean(pl)) if pl.size else np.nan)

        if max_samples and len(coords_list) >= max_samples:
            break

    if coords_list and keys_ref is not None:
        ens.ca_coords = np.stack(coords_list, axis=0)
        ens.ca_plddts = np.stack(plddt_list, axis=0)
        ens.ca_keys = keys_ref
        ens.ptm = np.array(ptm_list, dtype=np.float64)
        ens.iptm = np.array(iptm_list, dtype=np.float64)
        ens.plddt_mean = np.array(plddtmean_list, dtype=np.float64)
        ens.sample_paths = paths
    return ens


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    log.info("Saved: %s", path.relative_to(path.parents[1]) if len(path.parents) >= 2 else path.name)


def write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    log.info("Saved: %s", path.name)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
