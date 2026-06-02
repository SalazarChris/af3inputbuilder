"""
Analysis orchestration — baseline-vs-conditions pipeline.

Pulls together IO, geometry, ensemble statistics, factor parsing, plotting, and
reporting.  Every structural measurement is reported with an ensemble-derived
confidence interval, and per-residue displacement is tested against the
baseline's intrinsic structural noise.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import geometry as geom
from . import stats as st
from . import cluster as clust
from .factors import build_experiment_structure
from .io import (
    discover_conditions, load_ensemble, write_csv, write_json, ensure_dir,
)
from .model import ConditionModel, EnsembleModel
from .plots import confidence as plot_conf
from .plots import distances as plot_dist
from .plots import factorial as plot_fac
from .plots import per_residue as plot_pr
from .plots import clusters as plot_clust
from .plots import style
from .pymol import write_pymol_baseline
from .report import write_findings

log = logging.getLogger("af3bench.analysis")

# Sentinel default for the CLI cluster threshold; when left unchanged a
# data-driven cut height is used instead.
_DEFAULT_CLUSTER_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------

def resolve_baseline(
    conditions: Dict[str, ConditionModel],
    baseline_arg: Optional[str],
) -> Tuple[str, str]:
    """Return (baseline_name, human-readable reason)."""
    if baseline_arg:
        if baseline_arg not in conditions:
            raise SystemExit(
                f"Baseline '{baseline_arg}' not found. "
                f"Available: {', '.join(conditions)}"
            )
        return baseline_arg, "explicit --baseline"

    for kw in ("baseline", "apo", "wt", "ctrl", "control", "ref"):
        for name in conditions:
            if kw in name.lower():
                return name, f"keyword '{kw}' in name"

    # cleanest reference: no PTM, no DNA, no real ligand, fewest salt ions
    def salt(n):
        return conditions[n].n_na + conditions[n].n_cl

    clean = {
        n for n, c in conditions.items()
        if not c.ptm_labels and c.n_nucleic_residues == 0 and not c.has_real_ligand
    }
    if clean:
        best = min(clean, key=salt)
        return best, "no PTM/DNA/ligand, fewest salt ions"

    no_dna = {n for n, c in conditions.items() if c.n_nucleic_residues == 0 and not c.has_real_ligand}
    if no_dna:
        best = min(no_dna, key=salt)
        return best, "no DNA/ligand, fewest salt ions"

    best = min(conditions, key=salt)
    return best, "fewest salt ions (fallback)"


# ---------------------------------------------------------------------------
# Main baseline pipeline
# ---------------------------------------------------------------------------

def run(
    models_dir: Path,
    output_dir: Path,
    baseline_arg: Optional[str],
    chain_filter: Optional[List[str]],
    compute_tm: bool,
    pymol: bool,
    plddt_cutoff: float,
    n_bootstrap: int,
    fdr_alpha: float,
    max_samples: Optional[int],
    make_plots: bool,
    n_clusters: Optional[int] = None,
    cluster_threshold: float = 3.0,
) -> None:
    ensure_dir(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    rng = np.random.default_rng(20240531)

    conditions, skipped = discover_conditions(models_dir, chain_filter)
    log.info("Loaded %d conditions (%d skipped)", len(conditions), len(skipped))

    baseline_name, reason = resolve_baseline(conditions, baseline_arg)
    log.info("Baseline: %s  (%s)", baseline_name, reason)
    baseline = conditions[baseline_name]
    others = [n for n in sorted(conditions) if n != baseline_name]

    # ------------------------------------------------------------------
    # Ensembles (the variance engine)
    # ------------------------------------------------------------------
    log.info("Loading sample ensembles for variance estimation...")
    ensembles: Dict[str, EnsembleModel] = {}
    for name, cond in conditions.items():
        eff = chain_filter or cond.protein_chain_ids_from_json or None
        ens = load_ensemble(models_dir / name, eff,
                            reference_keys=cond.ca_keys, max_samples=max_samples)
        ensembles[name] = ens
        cond.ensemble = ens
        if ens.has_structural_ensemble:
            log.info("  %s: %d samples", name, ens.n_samples)

    # Baseline intrinsic per-residue noise (RMSF), keyed on baseline ca_keys
    base_ens = ensembles[baseline_name]
    if base_ens.has_structural_ensemble:
        baseline_rmsf_full = geom.ensemble_rmsf(
            base_ens.ca_coords, base_ens.ca_plddts, plddt_cutoff
        )
        baseline_rmsf_keys = base_ens.ca_keys
    else:
        baseline_rmsf_full = np.full(baseline.n_protein_residues, np.nan)
        baseline_rmsf_keys = baseline.ca_keys
    rmsf_lookup = dict(zip(baseline_rmsf_keys, baseline_rmsf_full))

    # ------------------------------------------------------------------
    # Per-condition structural analysis
    # ------------------------------------------------------------------
    dist_rows = []
    profiles: Dict[str, dict] = {}
    per_res_frames: Dict[str, pd.DataFrame] = {}
    top_residues: Dict[str, list] = {}

    for name in others:
        cond = conditions[name]
        al = geom.align(baseline, cond, plddt_cutoff)
        rmsd = al["rmsd"]

        # ensemble RMSD CI: align each cond sample to baseline rep, collect RMSD
        rmsd_samples = _ensemble_rmsd_samples(baseline, ensembles[name], plddt_cutoff)
        if rmsd_samples.size >= 2:
            r_mean, r_lo, r_hi = st.bootstrap_ci(rmsd_samples, n_bootstrap, rng=rng)
        else:
            r_mean, r_lo, r_hi = rmsd, rmsd, rmsd

        tm_c = tm_b = float("nan")
        if compute_tm:
            tm_c, tm_b = geom.tm_score(baseline, cond)

        # per-sample displacement matrix over shared residues (vs baseline rep)
        disp_mat, shared_keys = _ensemble_displacement_matrix(
            baseline, cond, ensembles[name], plddt_cutoff
        )
        baseline_rmsf_shared = np.array([rmsf_lookup.get(k, np.nan) for k in shared_keys])

        sig = st.displacement_significance(disp_mat, baseline_rmsf_shared, alpha=fdr_alpha)
        n_sig = int(np.nansum(sig["significant"]))

        dist_rows.append({
            "condition": name,
            "rmsd": _r(r_mean), "rmsd_lo": _r(r_lo), "rmsd_hi": _r(r_hi),
            "tm_score": _r(tm_c),
            "n_residues_aligned": al["n_fit"],
            "n_residues_shared": al["n_shared"],
            "n_significant": n_sig,
            "ensemble_n": int(ensembles[name].n_samples),
        })

        # Build profile for plotting/tables
        res_numbers = [k[1] for k in shared_keys]
        chain_ids = [k[0] for k in shared_keys]
        ref_pl, cond_pl = _shared_plddt(baseline, cond, shared_keys)

        profiles[name] = {
            "name": name,
            "res_numbers": res_numbers,
            "chain_ids": chain_ids,
            "disp_mean": sig["mean"],
            "disp_lo": sig["lo"],
            "disp_hi": sig["hi"],
            "baseline_rmsf": baseline_rmsf_shared,
            "significant": sig["significant"],
            "ref_plddt": ref_pl,
            "cond_plddt": cond_pl,
            "ptm_labels": cond.ptm_labels,
            "n_samples": int(ensembles[name].n_samples),
            "n_samples_base": int(base_ens.n_samples),
        }

        per_res_frames[name] = pd.DataFrame({
            "chain_id": chain_ids,
            "residue_number": res_numbers,
            "disp_mean_A": np.round(sig["mean"], 4),
            "disp_ci_lo_A": np.round(sig["lo"], 4),
            "disp_ci_hi_A": np.round(sig["hi"], 4),
            "baseline_rmsf_A": np.round(baseline_rmsf_shared, 4),
            "p_value": sig["pval"],
            "q_value": sig["qval"],
            "significant": sig["significant"],
            "ref_plddt": np.round(ref_pl, 2),
            "cond_plddt": np.round(cond_pl, 2),
        })

        # top movers
        order = np.argsort(-np.nan_to_num(sig["mean"]))
        top = []
        for idx in order[:5]:
            top.append({
                "chain": chain_ids[idx],
                "resnum": int(res_numbers[idx]),
                "mean": float(sig["mean"][idx]),
                "lo": float(sig["lo"][idx]),
                "hi": float(sig["hi"][idx]),
                "rmsf": float(baseline_rmsf_shared[idx]) if np.isfinite(baseline_rmsf_shared[idx]) else float("nan"),
                "significant": bool(sig["significant"][idx]),
            })
        top_residues[name] = top

    df_dist = pd.DataFrame(dist_rows)

    # global y ceiling for per-residue plots
    all_hi = [np.nanmax(p["disp_hi"]) for p in profiles.values() if len(p["disp_hi"])]
    y_ceiling = max(5.0, (max(all_hi) * 1.1) if all_hi else 5.0)
    for p in profiles.values():
        p["y_ceiling"] = y_ceiling

    # ------------------------------------------------------------------
    # Confidence summary (with seed SD)
    # ------------------------------------------------------------------
    df_conf, seed_sd = _confidence_summary(conditions, ensembles, baseline_name)

    # Flag likely-failed predictions (Tukey fences on ipTM / mean PAE) so that
    # large displacements driven by a collapsed model are not misread as real
    # conformational change.
    failed = plot_conf.detect_failed(df_conf)
    for row in dist_rows:
        row["likely_failed"] = row["condition"] in failed
    df_dist = pd.DataFrame(dist_rows)

    # ------------------------------------------------------------------
    # Write tables
    # ------------------------------------------------------------------
    write_csv(df_dist, tables_dir / "structural_distances.csv")
    write_csv(df_conf, tables_dir / "confidence_summary.csv")
    write_csv(_representative_table(conditions, baseline_name),
              tables_dir / "representative_selection.csv")
    per_res_dir = tables_dir / "per_residue"
    for name, frame in per_res_frames.items():
        safe = _safe(name)
        write_csv(frame, per_res_dir / f"{safe}.csv")

    # baseline RMSF table
    write_csv(
        pd.DataFrame({
            "chain_id": [k[0] for k in baseline_rmsf_keys],
            "residue_number": [k[1] for k in baseline_rmsf_keys],
            "rmsf_A": np.round(baseline_rmsf_full, 4),
        }),
        tables_dir / "baseline_rmsf.csv",
    )

    # ------------------------------------------------------------------
    # Factorial structure
    # ------------------------------------------------------------------
    struct = build_experiment_structure(conditions)
    log.info("Experiment structure: %d panel conditions | PTM=%s | tiers=%s",
             len(struct.panel_conditions), struct.ptm_order, struct.tier_order)

    # ------------------------------------------------------------------
    # Structural clustering (all-vs-all RMSD of representatives)
    # ------------------------------------------------------------------
    cluster_names, rmsd_matrix = clust.pairwise_rmsd(conditions, plddt_cutoff)
    # If the user did not request a specific granularity, derive a data-driven
    # cut height from the off-diagonal RMSD distribution (median) so well-folded
    # conditions group together instead of every condition becoming a singleton.
    eff_threshold = cluster_threshold
    if n_clusters is None and cluster_threshold == _DEFAULT_CLUSTER_THRESHOLD:
        finite = rmsd_matrix[np.isfinite(rmsd_matrix) & (rmsd_matrix > 0)]
        if finite.size:
            eff_threshold = float(np.median(finite))
            log.info("Cluster cut height (data-driven median RMSD): %.2f Å", eff_threshold)
    cl = clust.cluster_conditions(
        cluster_names, rmsd_matrix,
        threshold=eff_threshold, n_clusters=n_clusters,
    )
    cluster_labels = cl["labels"]
    cluster_summary = clust.cluster_summary(cluster_labels, conditions)
    log.info("Structural clusters: %d groups", cl["n_clusters"])
    for cs in cluster_summary:
        log.info("  cluster %d (n=%d): %s", cs["cluster"], cs["n_members"],
                 ", ".join(cs["members"]))

    # write cluster tables
    write_csv(
        pd.DataFrame([
            {"condition": nm, "cluster": cluster_labels[nm]}
            for nm in sorted(cluster_labels)
        ]),
        tables_dir / "cluster_assignments.csv",
    )
    write_csv(
        pd.DataFrame(rmsd_matrix, index=cluster_names, columns=cluster_names)
        .reset_index().rename(columns={"index": "condition"}),
        tables_dir / "pairwise_rmsd_matrix.csv",
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    written_plots: List[Path] = []
    if make_plots:
        log.info("Generating plots...")
        written_plots += plot_dist.plot_distances(df_dist, baseline_name, plots_dir)
        for name, prof in profiles.items():
            written_plots += plot_pr.plot_profile(prof, plots_dir, baseline_name)
        written_plots += plot_conf.plot_confidence_summary(df_conf, plots_dir, seed_sd)
        written_plots += plot_conf.plot_pae(df_conf, plots_dir)
        written_plots += plot_clust.plot_cluster_heatmap(
            cluster_names, rmsd_matrix, cl, plots_dir, cut_height=eff_threshold,
        )
        written_plots += _factorial_plots(
            conditions, ensembles, struct, baseline, baseline_name,
            rmsf_lookup, y_ceiling, plddt_cutoff, plots_dir,
        )

    if pymol:
        write_pymol_baseline(
            conditions, baseline_name, output_dir,
            profiles=profiles, cluster_labels=cluster_labels,
            global_disp_max=y_ceiling,
        )

    # ------------------------------------------------------------------
    # Manifest + findings
    # ------------------------------------------------------------------
    manifest = {
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
        "baseline": baseline_name,
        "baseline_reason": reason,
        "n_conditions": len(conditions),
        "conditions": list(conditions),
        "skipped": skipped,
        "parameters": {
            "plddt_cutoff": plddt_cutoff,
            "n_bootstrap": n_bootstrap,
            "fdr_alpha": fdr_alpha,
            "max_samples": max_samples,
            "compute_tm": compute_tm,
        },
        "ensemble_sizes": {n: int(e.n_samples) for n, e in ensembles.items()},
        "clusters": cluster_summary,
    }
    write_json(manifest, output_dir / "run_manifest.json")

    findings = {
        "baseline": baseline_name,
        "baseline_reason": reason,
        "n_conditions": len(conditions),
        "skipped": skipped,
        "likely_failed": sorted(failed),
        "ensemble_note": _ensemble_note(ensembles),
        "distances": dist_rows,
        "clusters": cluster_summary,
        "top_residues": top_residues,
        "caveats": _caveats(conditions, ensembles, baseline_name, failed),
    }
    write_findings(findings, output_dir)

    # console summary
    log.info("\nStructural distances vs baseline (%s):", baseline_name)
    log.info("  %-44s  %12s  %6s", "Condition", "RMSD (Å)", "n_sig")
    log.info("  " + "-" * 70)
    for row in dist_rows:
        log.info("  %-44s  %5.2f [%4.2f,%4.2f]  %5d",
                 row["condition"], row["rmsd"], row["rmsd_lo"], row["rmsd_hi"],
                 row["n_significant"])
    log.info("Done -> %s", output_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(v, n=4):
    return round(float(v), n) if (v is not None and math.isfinite(v)) else float("nan")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]


def _ensemble_rmsd_samples(baseline, ens: EnsembleModel, plddt_cutoff: float) -> np.ndarray:
    """RMSD of each condition sample (after fit) to the baseline representative."""
    if not ens.has_structural_ensemble:
        return np.empty(0)
    # match condition ensemble keys to baseline rep keys
    ref_idx, mob_idx = geom.match_indices(baseline.ca_keys, ens.ca_keys)
    if len(ref_idx) < 3:
        return np.empty(0)
    r = baseline.ca_coords[ref_idx]
    r_pl = baseline.ca_plddts[ref_idx]
    out = []
    for s in range(ens.n_samples):
        m = ens.ca_coords[s][mob_idx]
        m_pl = ens.ca_plddts[s][mob_idx]
        mask = (r_pl > plddt_cutoff) & (m_pl > plddt_cutoff)
        if mask.sum() < 3:
            mask = np.ones(len(r), dtype=bool)
        _, _, rmsd = geom.kabsch(r[mask], m[mask])
        out.append(rmsd)
    return np.asarray(out, dtype=np.float64)


def _ensemble_displacement_matrix(
    baseline, cond, ens: EnsembleModel, plddt_cutoff: float,
) -> Tuple[np.ndarray, list]:
    """
    Per-sample per-residue displacement vs the baseline representative.

    Returns (matrix (S, M), shared residue keys length M).  Falls back to the
    single representative if no ensemble is available.
    """
    if ens.has_structural_ensemble:
        ref_idx, mob_idx = geom.match_indices(baseline.ca_keys, ens.ca_keys)
        if len(ref_idx) < 3:
            return _single_displacement(baseline, cond, plddt_cutoff)
        r = baseline.ca_coords[ref_idx]
        r_pl = baseline.ca_plddts[ref_idx]
        keys = [baseline.ca_keys[i] for i in ref_idx]
        rows = []
        for s in range(ens.n_samples):
            m_all = ens.ca_coords[s]
            m = m_all[mob_idx]
            m_pl = ens.ca_plddts[s][mob_idx]
            mask = (r_pl > plddt_cutoff) & (m_pl > plddt_cutoff)
            if mask.sum() < 3:
                mask = np.ones(len(r), dtype=bool)
            R_mat, t_vec, _ = geom.kabsch(r[mask], m[mask])
            disp = np.sqrt(np.sum((r - (m @ R_mat.T + t_vec)) ** 2, axis=1))
            rows.append(disp)
        return np.stack(rows, axis=0), keys
    return _single_displacement(baseline, cond, plddt_cutoff)


def _single_displacement(baseline, cond, plddt_cutoff: float) -> Tuple[np.ndarray, list]:
    al = geom.align(baseline, cond, plddt_cutoff)
    disp, keys = geom.per_residue_displacement(
        baseline, cond, al["R"], al["t"], al["ref_idx"], al["mob_idx"]
    )
    return disp[None, :], keys


def _shared_plddt(baseline, cond, shared_keys) -> Tuple[np.ndarray, np.ndarray]:
    bmap = {k: baseline.ca_plddts[i] for i, k in enumerate(baseline.ca_keys)}
    cmap = {k: cond.ca_plddts[i] for i, k in enumerate(cond.ca_keys)}
    ref_pl = np.array([bmap.get(k, np.nan) for k in shared_keys])
    cond_pl = np.array([cmap.get(k, np.nan) for k in shared_keys])
    return ref_pl, cond_pl


def _confidence_summary(conditions, ensembles, baseline_name):
    ref = conditions[baseline_name]
    rows = []
    seed_sd: Dict[str, Dict[str, float]] = {}
    for name, cond in conditions.items():
        ens = ensembles[name]
        ptm_mean, ptm_sd, _ = st.summarize_scores(list(ens.ptm))
        iptm_mean, iptm_sd, _ = st.summarize_scores(list(ens.iptm))
        pl_mean, pl_sd, _ = st.summarize_scores(list(ens.plddt_mean))
        seed_sd[name] = {"ptm": ptm_sd, "iptm": iptm_sd, "plddt_mean": pl_sd}
        row = {
            "condition": name,
            "is_reference": name == baseline_name,
            "ptm": _r(cond.ptm), "iptm": _r(cond.iptm),
            "plddt_mean": _r(cond.mean_plddt),
            "mean_pae": _r(cond.mean_pae),
            "ptm_seed_sd": _r(ptm_sd), "iptm_seed_sd": _r(iptm_sd),
            "plddt_seed_sd": _r(pl_sd),
            "n_protein_residues": cond.n_protein_residues,
            "n_nucleic_residues": cond.n_nucleic_residues,
            "n_na": cond.n_na, "n_cl": cond.n_cl, "n_water": cond.n_water,
        }
        row["delta_ptm"] = _r(cond.ptm - ref.ptm)
        row["delta_iptm"] = _r(cond.iptm - ref.iptm)
        row["delta_plddt_mean"] = _r(cond.mean_plddt - ref.mean_plddt)
        row["delta_mean_pae"] = _r(cond.mean_pae - ref.mean_pae)
        rows.append(row)
    return pd.DataFrame(rows), seed_sd


def _representative_table(conditions, baseline_name):
    rows = []
    for name, cond in conditions.items():
        rows.append({
            "condition": name,
            "is_reference": name == baseline_name,
            "model_file": str(cond.cif_path),
            "ptm": _r(cond.ptm), "iptm": _r(cond.iptm),
            "ranking_score": _r(cond.ranking_score),
            "plddt_mean": _r(cond.mean_plddt),
            "n_protein_residues": cond.n_protein_residues,
            "n_nucleic_residues": cond.n_nucleic_residues,
            "ptm_labels": ",".join(cond.ptm_labels),
            "description": cond.description,
        })
    return pd.DataFrame(rows)


def _factorial_plots(conditions, ensembles, struct, baseline, baseline_name,
                     rmsf_lookup, y_ceiling, plddt_cutoff, plots_dir) -> List[Path]:
    written: List[Path] = []
    panel = struct.panel_conditions
    if not panel:
        return written

    # Build per-(ptm,tier) cell profiles and concentration-response data
    cells: Dict[tuple, dict] = {}
    conc: Dict[str, Dict[str, dict]] = {}
    grid_rows = struct.ptm_order
    grid_cols = struct.tier_order
    grid = np.full((len(grid_rows), len(grid_cols)), np.nan)
    ns_mask = np.zeros_like(grid, dtype=bool)

    for name in panel:
        if name == baseline_name or name not in conditions:
            continue
        cond = conditions[name]
        disp_mat, shared_keys = _ensemble_displacement_matrix(
            baseline, cond, ensembles[name], plddt_cutoff
        )
        rmsf_shared = np.array([rmsf_lookup.get(k, np.nan) for k in shared_keys])
        mean_disp = float(np.nanmean(disp_mat))
        per_res_mean = np.nanmean(disp_mat, axis=0)

        ptm = struct.ptm_group[name]
        tier = struct.ion_tier[name]
        cells[(ptm, tier)] = {
            "res_numbers": [k[1] for k in shared_keys],
            "disp_mean": per_res_mean,
            "baseline_rmsf": rmsf_shared,
            "ptm_labels": cond.ptm_labels,
            "mean_disp": mean_disp,
        }

        # concentration response: bootstrap CI of the mean displacement
        flat = disp_mat.reshape(-1)
        m, lo, hi = st.bootstrap_ci(flat)
        conc.setdefault(ptm, {})[tier] = {"mean": m, "lo": lo, "hi": hi}

        ri = grid_rows.index(ptm)
        ci = grid_cols.index(tier)
        grid[ri, ci] = mean_disp
        noise = np.nanmean(rmsf_shared)
        ns_mask[ri, ci] = bool(np.isfinite(noise) and mean_disp <= noise)

    written += plot_fac.plot_panel_per_residue(
        cells, grid_rows, grid_cols, baseline_name, y_ceiling, plots_dir
    )
    written += plot_fac.plot_concentration_response(conc, grid_rows, baseline_name, plots_dir)
    written += plot_fac.plot_ptm_effect_grid(
        grid, ns_mask, grid_rows, grid_cols, baseline_name, plots_dir
    )
    return written


def _ensemble_note(ensembles) -> str:
    sizes = [e.n_samples for e in ensembles.values() if e.n_samples]
    if not sizes:
        return "no sample ensembles found; CIs degenerate to point estimates"
    return (f"per-condition sample ensembles of {min(sizes)}–{max(sizes)} models "
            f"used for displacement CIs and baseline RMSF")


def _caveats(conditions, ensembles, baseline_name, failed=None) -> List[str]:
    out = [
        "Displacement is measured against AF3's own sampling noise (baseline RMSF); "
        "residues whose 95% CI does not clear the noise band are not interpreted as real motion.",
        "Salt-ion tier reflects Na+Cl count only; water scales separately and is reported "
        "as its own covariate, so the concentration axis is not confounded by solvent.",
    ]
    if failed:
        out.append(
            "Likely-failed predictions (low ipTM / high PAE outliers): "
            + ", ".join(sorted(failed))
            + ". Their large displacements reflect model collapse, not conformational change, "
            "and should be excluded from biological interpretation."
        )
    no_ens = [n for n, e in ensembles.items() if not e.has_structural_ensemble]
    if no_ens:
        out.append("No usable sample ensemble for: " + ", ".join(sorted(no_ens))
                    + " (their displacement CIs are point estimates).")
    return out
