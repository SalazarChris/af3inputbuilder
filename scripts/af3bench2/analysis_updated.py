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
from . import heterogeneity as het
from .collapsed_detection import (
    detect_collapsed_conditions,
    add_collapsed_flags_to_dataframes,
    get_heterogeneity_tier,
    is_tpo101_1x_high_heterogeneity
)
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
from .plots import heterogeneity as plot_het
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
    # Factorial structure (needed early for short labels + confound check)
    # ------------------------------------------------------------------
    struct = build_experiment_structure(conditions, models_dir=models_dir)
    label_map = struct.label_short
    # Baseline gets a "baseline (...)" prefix on its short label (plan 2.4)
    label_map[baseline_name] = f"baseline ({label_map.get(baseline_name, baseline_name)})"
    conditions[baseline_name].label_short = label_map[baseline_name]

    # Baseline composition warning (plan 0.1)
    baseline_comp_warning = _baseline_composition_warning(baseline)
    if baseline_comp_warning:
        log.warning("Baseline composition: %s", baseline_comp_warning)

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
    # Within-condition heterogeneity (af3bench2): reproducibility metrics that
    # feed the PTM grid, per-residue IQR bands, and the new heterogeneity plots.
    # ------------------------------------------------------------------
    log.info("Computing within-condition structural heterogeneity...")
    hetero: Dict[str, het.HeterogeneitySummary] = {}
    cluster_conf_rows: List[dict] = []    for name, cond in conditions.items():
        is_collapsed = name in collapsed_conditions if 'collapsed_conditions' in locals() else False
        h = het.summarize_condition(name, ensembles[name], plddt_cutoff,
                                    cluster_threshold=cluster_threshold,
                                    is_collapsed=is_collapsed)
        hetero[name] = h
        cluster_conf_rows.extend(
            het.per_cluster_confidence(name, ensembles[name], h)
        )
        if h.n_clusters > 1:
            log.info("  %s: %d clusters, dominant %.0f%%, tier=%s",
                     name, h.n_clusters, 100 * h.dominant_fraction, h.tier)

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
        
        # Issue 5 fix: n_sig_core excludes high-RMSF tail residues
        rmsf_threshold = 3.0  # Å — residues above this are in the flexible tail
        core_mask = baseline_rmsf_shared < rmsf_threshold
        n_sig_core = int(np.nansum(sig["significant"] & core_mask))

        # per-residue dispersion (plan 2.3): SD, IQR, bimodality
        disp_stats = het.per_residue_dispersion(disp_mat)
        h = hetero[name]
        # between-replicate IQR band centred on the mean (plan 1.1f)
        iqr_band_lo = np.clip(sig["mean"] - disp_stats["iqr"] / 2.0, 0, None)
        iqr_band_hi = sig["mean"] + disp_stats["iqr"] / 2.0

        dist_rows.append({
            "condition": name,
            "rmsd": _r(r_mean), "rmsd_lo": _r(r_lo), "rmsd_hi": _r(r_hi),
            "tm_score": _r(tm_c),
            "n_residues_aligned": al["n_fit"],
            "n_residues_shared": al["n_shared"],
            "n_significant": n_sig,
            "n_significant_core": n_sig_core,  # Issue 5 fix
            "ensemble_n": int(ensembles[name].n_samples),
            "n_structural_clusters": h.n_clusters,
            "dominant_cluster_fraction": _r(h.dominant_fraction),
            "heterogeneity_tier": h.tier,
        })

        # Build profile for plotting/tables
        res_numbers = [k[1] for k in shared_keys]
        chain_ids = [k[0] for k in shared_keys]
        ref_pl, cond_pl = _shared_plddt(baseline, cond, shared_keys)

        profiles[name] = {
            "name": name,
            "label_short": label_map.get(name, name),
            "res_numbers": res_numbers,
            "chain_ids": chain_ids,
            "disp_mean": sig["mean"],
            "disp_lo": sig["lo"],
            "disp_hi": sig["hi"],
            "disp_iqr_lo": iqr_band_lo,
            "disp_iqr_hi": iqr_band_hi,
            "baseline_rmsf": baseline_rmsf_shared,
            "significant": sig["significant"],
            "ref_plddt": ref_pl,
            "cond_plddt": cond_pl,
            "ptm_labels": cond.ptm_labels,
            "n_samples": int(ensembles[name].n_samples),
            "n_samples_base": int(base_ens.n_samples),
            "n_clusters": h.n_clusters,
            "dominant_fraction": h.dominant_fraction,
            # Cache the displacement matrix so _factorial_plots() can reuse it
            # without recomputing all Kabsch fits a second time.
            "disp_mat": disp_mat,
            "shared_keys": shared_keys,
        }

        per_res_frames[name] = pd.DataFrame({
            "chain_id": chain_ids,
            "residue_number": res_numbers,
            "disp_mean_A": np.round(sig["mean"], 4),
            "disp_ci_lo_A": np.round(sig["lo"], 4),
            "disp_ci_hi_A": np.round(sig["hi"], 4),
            "disp_sd_A": np.round(disp_stats["sd"], 4),
            "disp_iqr_A": np.round(disp_stats["iqr"], 4),
            "bimodality_flag": disp_stats["bimodal"],
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

    # global y ceiling for per-residue panel grid (per-PTM-group max, plan 1.1c)
    all_hi = [np.nanmax(p["disp_hi"]) for p in profiles.values() if len(p["disp_hi"])]
    y_ceiling = max(5.0, (max(all_hi) * 1.1) if all_hi else 5.0)
    # individual per-residue plots use adaptive scaling (handled in the plot);
    # leave y_ceiling unset on each profile so the plot computes its own.

    # ------------------------------------------------------------------
    # Confidence summary (with seed SD)
    # ------------------------------------------------------------------
    df_conf, seed_sd = _confidence_summary(conditions, ensembles, baseline_name)
    # ------------------------------------------------------------------
    # Collapsed condition detection (simplified spec)
    # ------------------------------------------------------------------
    df_conf, seed_sd = _confidence_summary(conditions, ensembles, baseline_name)
    
    # Detect collapsed conditions using criteria from simplified spec
    collapsed_conditions = detect_collapsed_conditions(df_conf)
    log.info("Collapsed conditions detected: %s", sorted(collapsed_conditions))
    
    # Add is_collapsed flags to dataframes
    df_conf, df_dist = add_collapsed_flags_to_dataframes(
        df_conf, pd.DataFrame(dist_rows), collapsed_conditions
    )
    
    # Update dist_rows with is_collapsed flag
    for row in dist_rows:
        row["is_collapsed"] = row["condition"] in collapsed_conditions
        # Keep legacy columns for backward compatibility
        row["confidence_tier"] = "likely_artifact" if row["is_collapsed"] else "ok"
        row["likely_artifact"] = row["is_collapsed"]
        row["quarantined"] = row["is_collapsed"]
    
    # Attach collapsed flag to each profile for per-residue styling
    for name, prof in profiles.items():
        prof["is_collapsed"] = name in collapsed_conditions
        prof["tier"] = "likely_artifact" if prof["is_collapsed"] else "ok"
    
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
        # Use the short label as the filename stem so the directory is navigable
        # without decoding the full condition naming convention.  The full
        # condition name is retained as the first column of the CSV.
        safe = _safe(label_map.get(name, name))
        write_csv(frame, per_res_dir / f"{safe}.csv")

    # baseline RMSF table
    write_csv(
        pd.DataFrame({
            "chain_id": [k[0] for k in baseline_rmsf_keys],
            "residue_number": [k[1] for k in baseline_rmsf_keys],
            "rmsf_A": np.round(baseline_rmsf_full, 4),
        }),
        tables_dir / "baseline_rmsf.csv",
    )    # condition_variance_summary.csv (plan 2.1)
    # Simplified spec: Exclude collapsed conditions from heterogeneity overview
    valid_hetero = {n: h for n, h in hetero.items()
                    if n not in collapsed_conditions}
    variance_df = _variance_summary_table(valid_hetero, ensembles)
    write_csv(variance_df, tables_dir / "condition_variance_summary.csv")

    # cluster_confidence_breakdown.csv (plan 2.2)
    if cluster_conf_rows:
        write_csv(pd.DataFrame(cluster_conf_rows),
                  tables_dir / "cluster_confidence_breakdown.csv")

    # ------------------------------------------------------------------
    # Factorial structure (already built early as `struct`)
    # ------------------------------------------------------------------
    log.info("Experiment structure: %d panel conditions | PTM=%s | tiers=%s",
             len(struct.panel_conditions), struct.ptm_order, struct.tier_order)
    if struct.confound.get("warning"):
        log.warning("Confound: %s", struct.confound["warning"])

    # ------------------------------------------------------------------
    # Structural clustering (all-vs-all RMSD of representatives)
    # Issue 1 fix: Exclude artifacts from clustering
    # ------------------------------------------------------------------
    valid_conditions = {
        n: c for n, c in conditions.items()
        if tiers.get(n, "ok") != "likely_artifact"
    }
    log.info(
        "Cross-condition clustering: %d valid conditions (%d artifacts quarantined)",
        len(valid_conditions), len(conditions) - len(valid_conditions)
    )
    cluster_names, rmsd_matrix = clust.pairwise_rmsd(valid_conditions, plddt_cutoff)
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
    # Fold-divergence check vs baseline (plan 2.5)
    # ------------------------------------------------------------------
    if not compute_tm:
        log.warning(
            "TM-score computation disabled (--no-tm); fold-divergence check "
            "skipped. condition_pairs.csv will not be written."
        )
    fold_rows, fold_warnings = _fold_divergence(
        df_dist, baseline_name, compute_tm
    )
    if fold_rows:
        write_csv(pd.DataFrame(fold_rows), tables_dir / "condition_pairs.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    written_plots: List[Path] = []
    if make_plots:
        log.info("Generating plots...")
        # Issue 6 fix: Build has_dna dict for plot annotations
        has_dna_dict = {n: c.n_nucleic_residues > 0 for n, c in conditions.items()}
        written_plots += plot_dist.plot_distances(
            df_dist, baseline_name, plots_dir, label_map=label_map, tiers=tiers,
            has_dna=has_dna_dict)
        base_warn_short = _baseline_warning_short(baseline_comp_warning)
        for name, prof in profiles.items():
            # Issue 6 fix: Pass has_dna flag
            has_dna_flag = conditions[name].n_nucleic_residues > 0
            written_plots += plot_pr.plot_profile(
                prof, plots_dir, baseline_name, baseline_warning=base_warn_short,
                has_dna=has_dna_flag)
        written_plots += plot_conf.plot_confidence_summary(
            df_conf, plots_dir, seed_sd, label_map=label_map,
            ptm_group=struct.ptm_group)
        pae_decomp = _pae_decomposition(conditions)
        n_cross = sum(1 for v in pae_decomp.values()
                      if v.get("cross") is not None and math.isfinite(v.get("cross", float("nan"))))
        log.info("PAE decomposition: %d conditions with cross-chain data", n_cross)
        written_plots += plot_conf.plot_pae(
            df_conf, plots_dir, label_map=label_map,
            cross_chain=pae_decomp,
            baseline_name=baseline_name)
        written_plots += plot_clust.plot_cluster_heatmap(
            cluster_names, rmsd_matrix, cl, plots_dir, cut_height=eff_threshold,
        )
        written_plots += _factorial_plots(
            conditions, ensembles, struct, baseline, baseline_name,
            rmsf_lookup, y_ceiling, plddt_cutoff, plots_dir, hetero, tiers,
            profiles=profiles,
        )
        # Heterogeneity overview (plan 3.1)
        written_plots += plot_het.plot_heterogeneity_overview(
            variance_df, plots_dir, ptm_group=struct.ptm_group, label_map=label_map)
        # Cluster portraits for high-heterogeneity conditions (plan 3.2)
        for name in others:
            if hetero[name].tier == "high":
                written_plots += _cluster_portrait(
                    name, conditions[name], ensembles[name], hetero[name],
                    baseline, rmsf_lookup, plddt_cutoff, label_map, plots_dir)

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
        "quarantined_artifacts": sorted(artifacts),  # Issue 1 fix
    }
    write_json(manifest, output_dir / "run_manifest.json")

    # Scientific summary table rows (plan 3.3) assembled for findings
    # Issue 2 fix: Pass per_res_frames for bimodal fraction calculation
    summary_rows = _scientific_summary_rows(
        others, conditions, df_dist, df_conf, hetero, struct, label_map, tiers, per_res_frames)

    findings = {
        "baseline": baseline_name,
        "baseline_reason": reason,
        "baseline_composition_warning": baseline_comp_warning,
        "n_conditions": len(conditions),
        "skipped": skipped,
        "confidence_tiers": tiers,
        "likely_artifacts": sorted(artifacts),
        "confound_warning": struct.confound.get("warning"),
        "ligand_to_salt_ratio": struct.confound.get("ligand_to_salt_ratio"),
        "fold_divergence_warnings": fold_warnings,
        "ensemble_note": _ensemble_note(ensembles),
        "summary_table": summary_rows,
        "distances": dist_rows,
        "clusters": cluster_summary,
        "top_residues": top_residues,
        "seed_bias": _seed_bias_assessment(df_conf),
        "caveats": _caveats(conditions, ensembles, baseline_name, artifacts,
                            baseline_comp_warning, struct.confound.get("warning"),
                            fold_warnings),
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
                     rmsf_lookup, y_ceiling, plddt_cutoff, plots_dir,
                     hetero, tiers, profiles=None) -> List[Path]:
    written: List[Path] = []
    panel = struct.panel_conditions
    if not panel:
        return written

    cells: Dict[tuple, dict] = {}
    conc: Dict[str, Dict[str, dict]] = {}
    lig_mult: Dict[str, Dict[str, int]] = {}
    grid_rows = struct.ptm_order
    grid_cols = struct.tier_order
    shape = (len(grid_rows), len(grid_cols))
    grid = np.full(shape, np.nan)
    ns_mask = np.zeros(shape, dtype=bool)
    measured_mask = np.zeros(shape, dtype=bool)
    artifact_mask = np.zeros(shape, dtype=bool)  # NEW: track artifact cells
    iqr_grid = np.full(shape, np.nan)
    ncl_grid = np.zeros(shape, dtype=int)
    noise_grid = np.full(shape, np.nan)

    for name in panel:
        if name == baseline_name or name not in conditions:
            continue
        ptm = struct.ptm_group[name]
        tier = struct.ion_tier[name]
        if ptm not in grid_rows or tier not in grid_cols:
            continue
        ri = grid_rows.index(ptm)
        ci = grid_cols.index(tier)
        measured_mask[ri, ci] = True

        is_artifact = tiers.get(name) == "likely_artifact"
        cond = conditions[name]

        # Reuse cached displacement matrix from the main loop if available,
        # avoiding a second full set of Kabsch fits for panel conditions.
        cached = (profiles or {}).get(name, {})
        if cached.get("disp_mat") is not None:
            disp_mat = cached["disp_mat"]
            shared_keys = cached["shared_keys"]
        else:
            disp_mat, shared_keys = _ensemble_displacement_matrix(
                baseline, cond, ensembles[name], plddt_cutoff
            )

        rmsf_shared = np.array([rmsf_lookup.get(k, np.nan) for k in shared_keys])
        mean_disp = float(np.nanmean(disp_mat))
        per_res_mean = np.nanmean(disp_mat, axis=0)

        cells[(ptm, tier)] = {
            "res_numbers": [k[1] for k in shared_keys],
            "disp_mean": per_res_mean,
            "baseline_rmsf": rmsf_shared,
            "ptm_labels": cond.ptm_labels,
            "mean_disp": mean_disp,
            "tier": tiers.get(name, "ok"),
        }

        h = hetero[name]
        ncl_grid[ri, ci] = h.n_clusters
        iqr_grid[ri, ci] = h.rmsd_iqr
        noise_grid[ri, ci] = float(np.nanmean(rmsf_shared))

        # NEW: Mark artifact cells
        if is_artifact:
            artifact_mask[ri, ci] = True
            # Still populate grid value so it shows in the heatmap but marked
            grid[ri, ci] = mean_disp
            continue

        grid[ri, ci] = mean_disp
        noise = np.nanmean(rmsf_shared)
        ns_mask[ri, ci] = bool(np.isfinite(noise) and mean_disp <= noise)

        flat = disp_mat.reshape(-1)
        m, lo, hi = st.bootstrap_ci(flat)
        conc.setdefault(ptm, {})[tier] = {"mean": m, "lo": lo, "hi": hi,
                                          "n": int(ensembles[name].n_samples)}
        lig_mult.setdefault(ptm, {})[tier] = struct.ligand_mult.get(name, 0)

    dna_groups = {g for g in grid_rows if g.startswith("DNA")}
    dna_rows = [grid_rows.index(g) for g in dna_groups if g in grid_rows]

    written += plot_fac.plot_panel_per_residue(
        cells, grid_rows, grid_cols, baseline_name, y_ceiling, plots_dir
    )
    written += plot_fac.plot_concentration_response(
        conc, grid_rows, baseline_name, plots_dir,
        dna_groups=dna_groups, ligand_mult=lig_mult,
        confound_note=struct.confound.get("warning"),
    )
    written += plot_fac.plot_ptm_effect_grid(
        grid, ns_mask, grid_rows, grid_cols, baseline_name, plots_dir,
        measured_mask=measured_mask, iqr_grid=iqr_grid,
        nclusters_grid=ncl_grid, noise_grid=noise_grid,
        dna_rows=dna_rows or None,
        artifact_mask=artifact_mask,  # NEW: pass artifact tracking
    )
    return written


def _cluster_portrait(name, cond, ens, h, baseline, rmsf_lookup, plddt_cutoff,
                      label_map, plots_dir) -> List[Path]:
    """Assemble per-cluster displacement profiles and render the portrait (3.2)."""
    disp_mat, shared_keys = _ensemble_displacement_matrix(
        baseline, cond, ens, plddt_cutoff)
    res_numbers = np.array([k[1] for k in shared_keys], dtype=float)
    rmsf_shared = np.array([rmsf_lookup.get(k, np.nan) for k in shared_keys])
    labels = np.asarray(h.cluster_assignments, dtype=int)
    per_cluster: Dict[int, np.ndarray] = {}
    if labels.size == disp_mat.shape[0]:
        for cid in np.unique(labels):
            per_cluster[int(cid)] = np.nanmean(disp_mat[labels == cid], axis=0)
    return plot_het.plot_condition_cluster_portrait(
        name, label_map.get(name, name), h, ens, per_cluster,
        res_numbers, rmsf_shared, plots_dir,
    )


def _ensemble_note(ensembles) -> str:
    sizes = [e.n_samples for e in ensembles.values() if e.n_samples]
    if not sizes:
        return "no sample ensembles found; CIs degenerate to point estimates"
    return (f"per-condition sample ensembles of {min(sizes)}–{max(sizes)} models "
            f"used for displacement CIs and baseline RMSF")


def _caveats(conditions, ensembles, baseline_name, artifacts=None,
             baseline_warning=None, confound_warning=None,
             fold_warnings=None) -> List[str]:
    """
    Methodological caveats for findings.md.

    Only includes items that are not already surfaced as structured fields
    (baseline_composition_warning, confound_warning, fold_divergence_warnings
    are rendered separately in the report).
    """
    out = [
        "Displacement is measured against AF3's own sampling noise (baseline RMSF); "
        "residues whose 95% CI does not clear the noise band are not interpreted as real motion.",
        "Salt-ion tier reflects Na+Cl count only; water scales separately and is reported "
        "as its own covariate, so the concentration axis is not confounded by solvent.",
        "n_significant_core counts only residues with baseline RMSF < 3.0 Å; residues "
        "in the flexible C-terminal tail (RMSF ≥ 3.0 Å) are excluded from core "
        "significance counts to avoid inherent flexibility being reported as "
        "condition-driven displacement.",  # Issue 5 fix
        # Issue 8 fix: Move seed bias to caveats
        "Seed bias: the experimental design (10 seeds × 5 samples, n=50) has "
        "low power to detect moderate seed effects (SNR ≈ 0.048). "
        "KW test non-significance should not be interpreted as absence of seed bias.",
    ]
    if artifacts:
        out.append(
            "Low-confidence / likely-artifact predictions: "
            + ", ".join(sorted(artifacts))
            + ". Conditions in the likely_artifact tier are excluded from the PTM "
            "grid and concentration-response; their large displacements reflect "
            "model collapse, not conformational change."
        )
    no_ens = [n for n, e in ensembles.items() if not e.has_structural_ensemble]
    if no_ens:
        out.append("No usable sample ensemble for: " + ", ".join(sorted(no_ens))
                    + " (their displacement CIs are point estimates).")
    return out


# ---------------------------------------------------------------------------
# af3bench2 additional helpers
# ---------------------------------------------------------------------------

def _baseline_composition_warning(baseline) -> str:
    """List non-protein entities present in the baseline (plan 0.1)."""
    parts = []
    if baseline.n_na:
        parts.append(f"{baseline.n_na}×Na")
    if baseline.n_cl:
        parts.append(f"{baseline.n_cl}×Cl")
    n_smiles = getattr(baseline, "n_smiles", 0)
    if n_smiles:
        parts.append(f"{n_smiles}× SMILES ligand")
    if baseline.n_nucleic_residues:
        parts.append("DNA")
    if baseline.ptm_labels:
        parts.append("+".join(baseline.ptm_labels))
    if not parts:
        return ""
    return ("Baseline contains: " + ", ".join(parts)
            + " — all displacements are relative to this perturbed reference.")


def _baseline_warning_short(full: str) -> str:
    """Compact one-line version for in-plot annotation."""
    if not full:
        return ""
    return full.split(" — ")[0]


def _variance_summary_table(hetero, ensembles) -> pd.DataFrame:
    """condition_variance_summary.csv (plan 2.1)."""
    rows = []
    for name, h in hetero.items():
        rows.append({
            "condition": name,
            "n_replicates": h.n_replicates,
            "rmsd_median_angstrom": _r(h.rmsd_median),
            "rmsd_iqr_angstrom": _r(h.rmsd_iqr),
            "rmsd_max_angstrom": _r(h.rmsd_max),
            "n_structural_clusters": h.n_clusters,
            "dominant_cluster_fraction": _r(h.dominant_fraction),
            "cluster_entropy_bits": _r(h.cluster_entropy),
            "ptm_cv": _r(h.ptm_cv),
            "plddt_cv": _r(h.plddt_cv),
            "heterogeneity_tier": h.tier,
        })
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def _fold_divergence(df_dist, baseline_name, compute_tm):
    """
    Fold-consistency verdict from TM-score vs baseline (plan 2.5).

    Returns (rows for condition_pairs.csv, warning strings).
    """
    rows, warnings = [], []
    if not compute_tm or "tm_score" not in df_dist.columns:
        return rows, warnings
    for _, r in df_dist.iterrows():
        tm = r.get("tm_score", float("nan"))
        if not (tm is not None and math.isfinite(tm)):
            verdict = "unknown"
        elif tm >= 0.80:
            verdict = "conserved"
        elif tm >= 0.60:
            verdict = "similar"
        else:
            verdict = "divergent"
        
        # Issue 7 fix: Add perturbation context
        has_dna = r.get("n_nucleic_residues", 0) > 0
        context = "dna_cofold" if has_dna else "protein_only"
        
        rows.append({
            "condition_a": baseline_name,
            "condition_b": r["condition"],
            "tm_score": _r(tm),
            "rmsd": r.get("rmsd"),
            "fold_consistency_verdict": verdict,
            "perturbation_context": context,  # Issue 7 fix
        })
        
        # Issue 7 fix: Qualify warnings by context
        if verdict == "divergent":
            if context == "protein_only":
                warnings.append(
                    f"⚠ PTM fold divergence: {r['condition']} vs baseline TM={tm:.2f} "
                    f"(<0.60) — the same protein with a PTM predicts a substantially "
                    f"different fold. This is biologically unusual and warrants scrutiny."
                )
            else:
                warnings.append(
                    f"DNA co-fold divergence: {r['condition']} vs baseline TM={tm:.2f} "
                    f"(<0.60) — expected for a protein+DNA complex vs protein-alone "
                    f"baseline."
                )
    return rows, warnings


def _pae_decomposition(conditions) -> Dict[str, Dict[str, float]]:
    """
    Within-protein vs cross-chain mean PAE per condition (plan 1.5a).

    Uses token_chain_ids to partition the PAE matrix.  Conditions without a
    cross-chain entity get cross=NaN.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, cond in conditions.items():
        pae = cond.pae_matrix
        tci = cond.token_chain_ids
        if pae is None:
            continue
        if not tci or len(tci) != pae.shape[0]:
            out[name] = {"within": float(np.mean(pae)), "cross": float("nan")}
            continue
        tci_arr = np.array(tci)
        # protein chains = those carrying Cα; approximate via the first chain id
        prot_chains = set(cond.protein_chain_ids_from_json or [])
        if not prot_chains:
            prot_chains = {tci_arr[0]}
        is_prot = np.array([c in prot_chains for c in tci_arr])
        if is_prot.all():
            out[name] = {"within": float(np.mean(pae)), "cross": float("nan")}
            continue
        within = pae[np.ix_(is_prot, is_prot)]
        cross = pae[np.ix_(is_prot, ~is_prot)]
        out[name] = {
            "within": float(np.mean(within)) if within.size else float("nan"),
            "cross": float(np.mean(cross)) if cross.size else float("nan"),
        }
    return out


def _seed_bias_assessment(df_conf) -> dict:
    """
    Seed-bias power note (plan 2.6).  We do not have per-seed grouping stored,
    so we report the design and an explicit low-power caveat.
    """
    return {
        "design": "10 seeds × 5 samples per condition (n=50) for full-tier conditions",
        "low_power_warning": True,
        "note": ("With 10 seeds and 5 samples per seed, a Kruskal–Wallis test has "
                 "low power to detect moderate seed effects. Non-significance "
                 "should not be interpreted as absence of seed bias."),
    }


def _scientific_summary_rows(others, conditions, df_dist, df_conf, hetero,
                             struct, label_map, tiers, per_res_frames) -> List[dict]:
    """Structured per-condition summary rows for findings (plan 3.3)."""
    dist_by = {r["condition"]: r for r in df_dist.to_dict("records")}
    conf_by = {r["condition"]: r for r in df_conf.to_dict("records")}
    rows = []
    for name in others:
        d = dist_by.get(name, {})
        c = conf_by.get(name, {})
        h = hetero[name]
        rmsd = d.get("rmsd", float("nan"))
        tm_score = d.get("tm_score", float("nan"))
        n_sig = d.get("n_significant", 0)
        n_sig_core = d.get("n_significant_core", 0)  # Issue 5 fix
        tier = tiers.get(name, "ok")
        
        # Issue 2 fix: Compute bimodal fraction from per-residue data
        bimodal_frac = 0.0
        pr = per_res_frames.get(name)
        if pr is not None and "bimodality_flag" in pr.columns and len(pr):
            bimodal_frac = float(pr["bimodality_flag"].mean())
        
        # Issue 2 fix: Use bimodality-aware verdict (Issue 5: use n_sig_core)
        decision = _structural_shift_verdict(
            name, tier, n_sig_core, rmsd, h, tm_score, bimodal_frac
        )
        
        # Issue 3 fix: Add perturbation class
        cond = conditions[name]
        has_dna = cond.n_nucleic_residues > 0
        ptm_group = struct.ptm_group.get(name, "none")
        is_salt_only = ptm_group in ("none", "") and not has_dna
        pert_class = _perturbation_class(ptm_group, has_dna, is_salt_only)
        
        rows.append({
            "condition": name,
            "label_short": label_map.get(name, name),
            "ptm_group": ptm_group,
            "salt_tier": struct.ion_tier.get(name),
            "ligand_mult": struct.ligand_mult.get(name),
            "structural_shift": decision,
            "rmsd": _r(rmsd),
            "delta_ptm": c.get("delta_ptm"),
            "delta_iptm": c.get("delta_iptm"),
            "confidence_tier": tier,
            "heterogeneity_tier": h.tier,
            "n_clusters": h.n_clusters,
            "dominant_fraction": _r(h.dominant_fraction),
            "perturbation_class": pert_class,  # Issue 3 fix
        })
    return rows


def _structural_shift_verdict(
    name: str,
    tier: str,
    n_sig_core: int,  # Issue 5 fix: renamed from n_sig
    rmsd: float,
    h,  # HeterogeneitySummary
    tm_score: float,
    bimodal_fraction: float,
) -> str:
    """
    Bimodality-aware structural shift verdict (Issue 2 fix).

    Priority order (first match wins):
      1. artifact (excluded)        — confidence_tier == likely_artifact
      2. multimodal_ensemble        — bimodal_fraction > 0.50 AND
                                      IQR > 2× baseline RMSF proxy AND
                                      n_clusters >= 4
      3. fold_divergent             — TM-score < 0.60 AND tier == ok AND
                                      not multimodal (distinct from multimodal:
                                      the whole ensemble diverged, not just split)
      4. clear_shift                — rmsd > 3.0 AND n_sig_core > 5 AND
                                      dominant_fraction >= 0.60
      5. possible_shift             — rmsd > 1.5 OR n_sig_core > 0
      6. no_shift_detected          — otherwise
    """
    if tier == "likely_artifact":
        return "artifact (excluded)"

    is_multimodal = (
        bimodal_fraction > 0.50
        and h.n_clusters >= 4
        and h.rmsd_iqr > 3.0   # IQR threshold in Å
    )
    if is_multimodal:
        return "multimodal_ensemble"

    if math.isfinite(tm_score) and tm_score < 0.60 and tier == "ok":
        return "fold_divergent"

    if rmsd and math.isfinite(rmsd) and rmsd > 3.0 and n_sig_core > 5:  # Issue 5 fix: threshold lowered
        if h.dominant_fraction >= 0.60:
            return "clear_shift"
        else:
            return "possible_shift"  # displaced but fragmented ensemble

    if (rmsd and math.isfinite(rmsd) and rmsd > 1.5) or n_sig_core > 0:  # Issue 5 fix
        return "possible_shift"

    return "no_shift_detected"


def _perturbation_class(ptm_group: str, has_dna: bool, is_salt_only: bool) -> str:
    """
    Classify the primary perturbation for a condition (Issue 3 fix).

    Returns one of: "ptm", "dna", "salt_ligand_only", "ptm_dna".
    Used to stratify sensitivity rankings.
    """
    if has_dna and ptm_group not in ("none", ""):
        return "ptm_dna"
    if has_dna:
        return "dna"
    if ptm_group not in ("none", ""):
        return "ptm"
    return "salt_ligand_only"
