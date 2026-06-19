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

# Try to import scipy (optional for clustering)
try:
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

from . import geometry as geom
from . import stats as st
from . import cluster as clust
from . import heterogeneity as het
from . import __version__ as __pipeline_version__
from .collapsed_detection import (
    detect_collapsed_conditions,
    add_collapsed_flags_to_dataframes,
    get_heterogeneity_tier,
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
from .plots import summary as plot_sum
from .plots import quality as plot_qlt
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
    # Confidence summary + collapsed-condition detection — computed BEFORE the
    # heterogeneity loop, which consumes the collapsed set.  Collapse detection
    # is macromolecule-scoped (protein pLDDT + macromolecule PAE) and does not
    # depend on heterogeneity, so this ordering is safe.
    # ------------------------------------------------------------------
    df_conf, seed_sd = _confidence_summary(conditions, ensembles, baseline_name)
    collapsed_conditions = detect_collapsed_conditions(df_conf)
    log.info("Collapsed conditions detected: %s", sorted(collapsed_conditions))

    # ------------------------------------------------------------------
    # Within-condition heterogeneity (analysisscripts): reproducibility metrics that
    # feed the PTM grid, per-residue IQR bands, and the new heterogeneity plots.
    # ------------------------------------------------------------------
    log.info("Computing within-condition structural heterogeneity...")
    hetero: Dict[str, het.HeterogeneitySummary] = {}
    cluster_conf_rows: List[dict] = []
    for name, cond in conditions.items():
        is_collapsed = name in collapsed_conditions
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
    # Baseline ensemble characterization (for Cycle 11)
    # ------------------------------------------------------------------
    baseline_cluster_info = _baseline_cluster_analysis(
        baseline_name, ensembles, plddt_cutoff, cluster_threshold)
    if baseline_cluster_info:
        log.info("Baseline ensemble: %d clusters, dominant %.0f%%",
                 baseline_cluster_info["n_clusters"],
                 baseline_cluster_info["dominant_fraction"] * 100)

    # ------------------------------------------------------------------
    # Per-condition structural analysis
    # ------------------------------------------------------------------
    dist_rows = []
    profiles: Dict[str, dict] = {}
    per_res_frames: Dict[str, pd.DataFrame] = {}
    top_residues: Dict[str, list] = {}

    # Context-matched references (stratify by DNA presence) so RMSD reflects the
    # studied perturbation, not the constant apo->DNA-bound transition.
    context_refs, refs_by_context = select_context_references(
        conditions, baseline_name, struct)
    # Context reference for each condition, with an explicit label when the
    # reference is the condition itself (no PTM/ion perturbation in that context).
    ctx_ref_labels = {}
    for n, ref in context_refs.items():
        if ref == n:
            ctx_ref_labels[n] = "self"  # no perturbation expected
        else:
            ctx_ref_labels[n] = ref
    # Quantify the constant apo->DNA-bound offset (the confound removed by
    # context-matching): RMSD between the two context references.
    _dna_ref = refs_by_context.get(True)
    _nodna_ref = refs_by_context.get(False)
    context_offset_rmsd = None
    if (_dna_ref and _nodna_ref and _dna_ref in conditions
            and _nodna_ref in conditions and _dna_ref != _nodna_ref):
        context_offset_rmsd = float(
            geom.align(conditions[_nodna_ref], conditions[_dna_ref], plddt_cutoff)["rmsd"])

    # Per-residue RMSF noise envelope for each distinct context reference, so a
    # context-matched condition's displacement is judged against the spread of
    # ITS reference (not always the apo baseline).
    ref_rmsf_lookups = {baseline_name: rmsf_lookup}
    for _ref in set(v for v in refs_by_context.values() if v):
        if _ref == baseline_name or _ref not in ensembles:
            continue
        _re = ensembles[_ref]
        if _re.has_structural_ensemble:
            _rf = geom.ensemble_rmsf(_re.ca_coords, _re.ca_plddts, plddt_cutoff)
            ref_rmsf_lookups[_ref] = dict(zip(_re.ca_keys, _rf))
        else:
            ref_rmsf_lookups[_ref] = {}

    for name in others:
        cond = conditions[name]
        al = geom.align(baseline, cond, plddt_cutoff)
        rmsd = al["rmsd"]

        # ensemble RMSD CI: align each cond sample to baseline rep, collect RMSD
        rmsd_samples = _ensemble_rmsd_samples(baseline, ensembles[name], plddt_cutoff)
        seed_labels = _seed_labels_for(ensembles[name])
        if rmsd_samples.size >= 2:
            r_mean, r_lo, r_hi = st.bootstrap_ci(
                rmsd_samples, n_bootstrap, rng=rng, seed_labels=seed_labels)
        else:
            r_mean, r_lo, r_hi = rmsd, rmsd, rmsd

        # Context-matched RMSD: distance to the same-DNA-context unmodified
        # reference, isolating the studied perturbation from the apo->DNA shift.
        ctx_ref = context_refs.get(name, baseline_name)
        if ctx_ref == baseline_name or ctx_ref not in conditions:
            ctx_ref = baseline_name
            ctx_mean, ctx_lo, ctx_hi = r_mean, r_lo, r_hi
        else:
            ctx_samples = _ensemble_rmsd_samples(
                conditions[ctx_ref], ensembles[name], plddt_cutoff)
            if ctx_samples.size >= 2:
                ctx_mean, ctx_lo, ctx_hi = st.bootstrap_ci(
                    ctx_samples, n_bootstrap, rng=rng, seed_labels=seed_labels)
            else:
                ctx_al = geom.align(conditions[ctx_ref], cond, plddt_cutoff)
                ctx_mean = ctx_lo = ctx_hi = ctx_al["rmsd"]

        tm_c = tm_b = float("nan")
        if compute_tm:
            tm_c, tm_b = geom.tm_score(baseline, cond)

        # per-sample displacement matrix over shared residues, measured against
        # the CONTEXT-MATCHED reference so the residue-level map reflects the
        # studied perturbation rather than the apo→DNA-bound transition.
        pr_ref = ctx_ref if (ctx_ref and ctx_ref in conditions) else baseline_name
        pr_ref_model = conditions[pr_ref]
        disp_mat, shared_keys = _ensemble_displacement_matrix(
            pr_ref_model, cond, ensembles[name], plddt_cutoff
        )
        ref_rmsf_lookup = ref_rmsf_lookups.get(pr_ref, rmsf_lookup)
        baseline_rmsf_shared = np.array([ref_rmsf_lookup.get(k, np.nan) for k in shared_keys])

        # per-residue confidence (pLDDT) over the shared residues, used both to
        # gate interpretable motion and for the per-residue plots.
        ref_pl, cond_pl = _shared_plddt(pr_ref_model, cond, shared_keys)

        sig = st.displacement_significance(
            disp_mat, baseline_rmsf_shared, alpha=fdr_alpha,
            ref_plddt=ref_pl, cond_plddt=cond_pl, seed_labels=seed_labels,
        )
        # Headline count = residues with statistically significant motion that
        # are ALSO confidently placed (pLDDT >= 70 in both states); disordered
        # low-confidence residues are not interpreted as motion.
        n_sig = int(np.nansum(sig["significant"]))
        # Raw statistical count (pre-confidence-gate), retained as provenance.
        n_sig_stat_only = int(np.nansum(sig["significant_stat"]))

        # per-residue dispersion (plan 2.3): SD, IQR, bimodality
        disp_stats = het.per_residue_dispersion(disp_mat)
        h = hetero[name]
        # between-replicate IQR band centred on the mean (plan 1.1f)
        iqr_band_lo = np.clip(sig["mean"] - disp_stats["iqr"] / 2.0, 0, None)
        iqr_band_hi = sig["mean"] + disp_stats["iqr"] / 2.0

        dist_rows.append({
            "condition": name,
            "rmsd": _r(r_mean), "rmsd_lo": _r(r_lo), "rmsd_hi": _r(r_hi),
            "context_ref": ctx_ref,
            "rmsd_vs_context_ref": _r(ctx_mean),
            "rmsd_vs_context_ref_lo": _r(ctx_lo),
            "rmsd_vs_context_ref_hi": _r(ctx_hi),
            "tm_score": _r(tm_c),
            "n_residues_aligned": al["n_fit"],
            "n_residues_shared": al["n_shared"],
            "n_significant": n_sig,
            "n_significant_stat_only": n_sig_stat_only,  # pre-confidence-gate
            "per_residue_ref": pr_ref,
            "ensemble_n": int(ensembles[name].n_samples),
            "n_structural_clusters": h.n_clusters,
            "dominant_cluster_fraction": _r(h.dominant_fraction),
            "heterogeneity_tier": h.tier,
        })

        # Build profile for plotting/tables
        res_numbers = [k[1] for k in shared_keys]
        chain_ids = [k[0] for k in shared_keys]

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
            "per_residue_ref": pr_ref,
            "ptm_labels": cond.ptm_labels,
            "n_samples": int(ensembles[name].n_samples),
            "n_samples_base": int(base_ens.n_samples),
            "n_clusters": h.n_clusters,
            "dominant_fraction": h.dominant_fraction,
            # Cycle 11: Baseline cluster info for per-residue plots
            "baseline_cluster_info": baseline_cluster_info,
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
            "significant_stat": sig["significant_stat"],
            "confident_residue": sig["confident_residue"],
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
    # Collapsed flags propagated to the dataframes (confidence summary and
    # collapsed set were computed earlier, before the heterogeneity loop).
    # ------------------------------------------------------------------
    # Add is_collapsed flags to dataframes
    df_conf, df_dist_temp = add_collapsed_flags_to_dataframes(
        df_conf, pd.DataFrame(dist_rows), collapsed_conditions
    )
    
    # Update dist_rows with collapsed flags
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

    # Confidence-tier map consumed by the plotting layer (artifact differentiation,
    # Fix 2).  The authoritative artifact set is the collapsed-condition set.
    tiers: Dict[str, str] = {
        name: ("likely_artifact" if name in collapsed_conditions else "ok")
        for name in conditions
    }

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
    )

    # condition_variance_summary.csv (plan 2.1)
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
    # Structural clustering (all-vs-all RMSD of representatives)
    # Simplified spec: Exclude collapsed conditions from clustering
    # ------------------------------------------------------------------
    valid_conditions = {
        n: c for n, c in conditions.items()
        if n not in collapsed_conditions
    }
    log.info(
        "Cross-condition clustering: %d valid conditions (%d collapsed excluded)",
        len(valid_conditions), len(collapsed_conditions)
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

    # ------------------------------------------------------------------
    # Cluster separability vs within-condition sampling noise.
    # Between-condition clustering uses one representative per condition, so a
    # cut height finer than the *within-condition* ensemble spread resolves
    # sampling noise rather than condition-driven structure.  We estimate the
    # within-condition pairwise-RMSD noise floor from the heterogeneity
    # summaries (two replicates each at RMSD r from their mean differ by
    # ≈ √2·r on average) and compare it to the between-condition RMSD scale.
    within_r = [h.rmsd_median for h in hetero.values()
                if h is not None and np.isfinite(getattr(h, "rmsd_median", np.nan))]
    noise_floor = float(np.sqrt(2.0) * np.median(within_r)) if within_r else float("nan")
    finite_off = rmsd_matrix[np.isfinite(rmsd_matrix) & (rmsd_matrix > 0)]
    between_median = float(np.median(finite_off)) if finite_off.size else float("nan")
    separation_adequate = bool(
        np.isfinite(noise_floor) and np.isfinite(between_median)
        and between_median > noise_floor
    )
    cluster_separation = {
        "cut_height_A": round(float(eff_threshold), 3),
        "between_condition_median_rmsd_A": round(between_median, 3) if np.isfinite(between_median) else None,
        "within_condition_noise_floor_A": round(noise_floor, 3) if np.isfinite(noise_floor) else None,
        "separation_adequate": separation_adequate,
        "n_clusters": cl["n_clusters"],
    }
    if not separation_adequate:
        log.warning(
            "Between-condition RMSD (median %.2f Å) is within the within-condition "
            "sampling noise floor (%.2f Å); structural clusters are NOT robustly "
            "separable from ensemble noise and are exploratory.",
            between_median, noise_floor,
        )

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
        df_dist, baseline_name, compute_tm, conditions
    )
    if fold_rows:
        write_csv(pd.DataFrame(fold_rows), tables_dir / "condition_pairs.csv")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    written_plots: List[Path] = []
    # Baseline ensemble RMSD spread = the noise floor for "RMSD vs baseline"
    # (the RMSD a baseline replicate shows against the baseline representative
    # even when nothing changed).  Reused by the distance plot and the shift
    # verdict so a single, data-derived floor is shown everywhere.
    _bh = hetero.get(baseline_name)
    baseline_noise_rmsd = (
        float(_bh.rmsd_median)
        if _bh is not None and np.isfinite(getattr(_bh, "rmsd_median", np.nan))
        else float("nan")
    )
    if make_plots:
        log.info("Generating plots...")
        # Issue 6 fix: Build has_dna dict for plot annotations
        has_dna_dict = {n: c.n_nucleic_residues > 0 for n, c in conditions.items()}
        rmsd_max_map = {n: h.rmsd_max for n, h in hetero.items()}
        written_plots += plot_dist.plot_distances(
            df_dist, baseline_name, plots_dir, label_map=label_map, tiers=tiers,
            has_dna=has_dna_dict, ion_tier=struct.ion_tier,
            rmsd_max_map=rmsd_max_map, noise_floor=baseline_noise_rmsd)
        base_warn_short = _baseline_warning_short(baseline_comp_warning)
        for name, prof in profiles.items():
            # Issue 6 fix: Pass has_dna flag
            has_dna_flag = conditions[name].n_nucleic_residues > 0
            written_plots += plot_pr.plot_profile(
                prof, plots_dir, baseline_name, baseline_warning=base_warn_short,
                has_dna=has_dna_flag)
        written_plots += plot_conf.plot_confidence_summary(
            df_conf, plots_dir, seed_sd, label_map=label_map,
            ptm_group=struct.ptm_group, ion_tier=struct.ion_tier)
        # Cycle 2: pTM/ipTM scatter plot
        written_plots += plot_conf.plot_ptm_scatter(
            df_conf, plots_dir, label_map=label_map, ion_tier=struct.ion_tier)
        # Baseline per-seed pTM diagnostics (Fix 7).
        base_ptm_seeds = [float(v) for v in np.asarray(base_ens.ptm, dtype=float)
                          if math.isfinite(v)] if base_ens.n_samples else []
        written_plots += plot_conf.plot_baseline_diagnostics(
            df_conf, plots_dir, baseline_name, per_seed_ptm=base_ptm_seeds)
        # Cycle 11: Baseline ensemble violin plots
        baseline_violin_data = _baseline_violin_data(baseline_name, ensembles)
        written_plots += plot_conf.plot_baseline_violins(baseline_name, baseline_violin_data, plots_dir)
        
        # Cycle 12: New summary visualizations
        # PTM × concentration effect grid
        ptm_grid_data = _build_ptm_grid_data(conditions, df_dist, hetero, struct, baseline_name)
        written_plots += plot_sum.plot_ptm_concentration_effect_grid(
            ptm_grid_data, plots_dir, label_map=label_map, 
            ion_tier=struct.ion_tier, baseline_name=baseline_name)
        
        # Structural distance heatmap with hierarchical clustering
        # Generic: works with any perturbation factors (DNA, protein, small molecule, etc.)
        written_plots += plot_sum.plot_structural_distance_heatmap(
            rmsd_matrix, cluster_names, struct.ptm_group,
            plots_dir, label_map=label_map,
            perturbation_annotations=None)  # None = no additional annotations
        
        # Concentration response curves (faceted by perturbation state)
        # Generic: works for any ligand/modification (DNA, protein, small molecule, etc.)
        written_plots += plot_sum.plot_concentration_response_faceted(
            ptm_grid_data, plots_dir, label_map=label_map,
            perturbation_groups=None)  # None uses default (PTM group as perturbation)
        
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
            label_map=label_map, ion_tier=struct.ion_tier,
            artifact_names=sorted(collapsed_conditions),
            separation=cluster_separation,
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
        
        # Cycle 12: New visualizations - quality dashboard and clustering overview
        written_plots += plot_sum.plot_quality_dashboard(df_conf, df_dist, plots_dir)
        written_plots += plot_qlt.plot_confidence_distributions(df_conf, plots_dir)
        written_plots += plot_qlt.plot_quality_correlations(df_conf, plots_dir)
        
        # Cluster confidence breakdown
        if cluster_conf_rows:
            written_plots += plot_qlt.plot_cluster_confidence_breakdown(
                pd.DataFrame(cluster_conf_rows), plots_dir)
        
        # Per-condition quality assessment
        written_plots += plot_qlt.plot_quality_by_condition(df_conf, df_dist, plots_dir)

    if pymol:
        write_pymol_baseline(
            conditions, baseline_name, output_dir,
            profiles=profiles, cluster_labels=cluster_labels,
            global_disp_max=y_ceiling,
            cluster_separation=cluster_separation,
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
        "collapsed_conditions": sorted(collapsed_conditions),  # Simplified spec
    }
    write_json(manifest, output_dir / "run_manifest.json")

    # Scientific summary table rows (plan 3.3) assembled for findings
    # Issue 2 fix: Pass per_res_frames for bimodal fraction calculation
    # baseline_noise_rmsd (baseline ensemble RMSD spread) computed above is the
    # noise floor for the global-shift verdict.
    summary_rows = _scientific_summary_rows(
        others, conditions, df_dist, df_conf, hetero, struct, label_map,
        collapsed_conditions, per_res_frames, baseline_noise_rmsd=baseline_noise_rmsd,
        baseline_name=baseline_name)

    # Baseline ensemble metrics for Cycle 11 (baseline ensemble characterization)
    baseline_violin_data = _baseline_violin_data(baseline_name, ensembles)
    baseline_pairwise_rmsd = _baseline_pairwise_rmsd_distribution(
        baseline_name, ensembles, plddt_cutoff)
    baseline_confidence_metrics = _baseline_confidence_metrics(baseline_name, ensembles)
    
    findings = {
        "baseline": baseline_name,
        "baseline_reason": reason,
        "baseline_composition_warning": baseline_comp_warning,
        "n_conditions": len(conditions),
        "skipped": skipped,
        "collapsed_conditions": sorted(collapsed_conditions),  # Simplified spec
        "confound_warning": struct.confound.get("warning"),
        "ligand_to_salt_ratio": struct.confound.get("ligand_to_salt_ratio"),
        "fold_divergence_warnings": fold_warnings,
        "cluster_separation": cluster_separation,
        "context_references": {
            "by_context": {("dna" if k else "no_dna"): v for k, v in refs_by_context.items()},
            "ctx_ref_labels": ctx_ref_labels,
            "apo_to_dna_offset_rmsd_A": (round(context_offset_rmsd, 3)
                                         if context_offset_rmsd is not None else None),
        },
        "ensemble_note": _ensemble_note(ensembles),
        "summary_table": summary_rows,
        "distances": dist_rows,
        "clusters": cluster_summary,
        "top_residues": top_residues,
        "seed_bias": _seed_bias_assessment(ensembles),
        "caveats": _caveats(conditions, ensembles, baseline_name, collapsed_conditions,
                            baseline_comp_warning, struct.confound.get("warning"),
                            fold_warnings, cluster_separation,
                            refs_by_context=refs_by_context,
                            context_offset=context_offset_rmsd),
        "pipeline_version": __pipeline_version__,  # Cycle 10: Add pipeline version
        # Cycle 11: Baseline ensemble characterization
        "baseline_violin_data": baseline_violin_data,
        "baseline_cluster_info": baseline_cluster_info,
        "baseline_pairwise_rmsd": baseline_pairwise_rmsd,
        "baseline_confidence_metrics": baseline_confidence_metrics,
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


def _seed_labels_for(ens: EnsembleModel):
    """Per-sample seed id parsed from replicate paths (None if not parseable).

    Drives the hierarchical (seed-aware) bootstrap so within-seed correlation is
    not treated as independent replication.
    """
    import re as _re
    rx = _re.compile(r"seed-(\d+)_sample-")
    paths = getattr(ens, "sample_paths", []) or []
    if ens.n_samples == 0 or len(paths) != ens.n_samples:
        return None
    labs = []
    for p in paths:
        m = rx.search(str(p))
        if m is None:
            return None
        labs.append(int(m.group(1)))
    return labs if len(set(labs)) >= 2 else None


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
            "mean_pae_full": _r(cond.mean_pae_full),
            "pae_scope": (
                "macromolecule"
                if cond.token_chain_ids is not None
                and cond.macromolecular_chain_ids
                else "full_matrix"
            ),
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


def _ref_grid_cell(struct, baseline_name, grid_rows, grid_cols):
    """Return (row, col) of the baseline reference cell in the effect grid, or None.

    Fix 4: the reference cell (unmodified, 1x) must be drawn as an explicit
    "REF" cell rather than left blank.
    """
    ptm = struct.ptm_group.get(baseline_name)
    tier = struct.ion_tier.get(baseline_name)
    if ptm in grid_rows and tier in grid_cols:
        return (grid_rows.index(ptm), grid_cols.index(tier))
    return None


def _build_ptm_grid_data(conditions: Dict[str, ConditionModel], 
                         df_dist: pd.DataFrame,
                         hetero: Dict[str, het.HeterogeneitySummary],
                         struct,
                         baseline_name: str) -> pd.DataFrame:
    """
    Build PTM × concentration grid data for visualization.
    
    Returns DataFrame with columns: condition, ptm_group, ion_tier,
    mean_disp, lo, hi, significant, p_value, rmsd
    """
    rows = []
    for name, cond in conditions.items():
        # Skip baseline
        if name == baseline_name:
            continue
        
        ptm = struct.ptm_group.get(name, "none")
        tier = struct.ion_tier.get(name, "?")
        
        # Get displacement values from df_dist
        dist_row = df_dist[df_dist["condition"] == name]
        if dist_row.empty:
            continue
        
        rows.append({
            "condition": name,
            "ptm_group": ptm,
            "ion_tier": tier,
            "mean_disp": dist_row["rmsd"].iloc[0],
            "lo": dist_row["rmsd_lo"].iloc[0],
            "hi": dist_row["rmsd_hi"].iloc[0],
            "significant": dist_row["n_significant"].iloc[0] > 0,
            "p_value": 0.01 if dist_row["n_significant"].iloc[0] > 0 else 0.5,
            "rmsd": dist_row["rmsd"].iloc[0],
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
    hetero_high_mask = np.zeros(shape, dtype=bool)  # Fix 3: high-heterogeneity cells

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

        if cached.get("baseline_rmsf") is not None:
            rmsf_shared = cached["baseline_rmsf"]
        else:
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
        if getattr(h, "tier", "low") == "high":
            hetero_high_mask[ri, ci] = True

        # NEW: Mark artifact cells
        if is_artifact:
            artifact_mask[ri, ci] = True
            # Still populate grid value so it shows in the heatmap but marked
            grid[ri, ci] = mean_disp
            # Record the artifact point in the concentration-response data so it
            # can be drawn with an open marker (Fix 2), flagged as artifact.
            # Sample-level CI: bootstrap the per-sample mean displacement over
            # the actual replicates (seed-aware), NOT over flattened
            # residue×sample values (which understates the CI ~sqrt(M)-fold).
            per_sample_mean = np.nanmean(disp_mat, axis=1)
            m, lo, hi = st.bootstrap_ci(per_sample_mean,
                                        seed_labels=_seed_labels_for(ensembles[name]))
            conc.setdefault(ptm, {})[tier] = {
                "mean": m, "lo": lo, "hi": hi,
                "n": int(ensembles[name].n_samples),
                "artifact": True,
                "hetero_tier": h.tier,
                "iqr": h.rmsd_iqr,
                "rmsd_max": h.rmsd_max,
            }
            lig_mult.setdefault(ptm, {})[tier] = struct.ligand_mult.get(name, 0)
            continue

        grid[ri, ci] = mean_disp
        noise = np.nanmean(rmsf_shared)
        ns_mask[ri, ci] = bool(np.isfinite(noise) and mean_disp <= noise)

        # Sample-level CI (see artifact branch): bootstrap per-sample mean over
        # replicates (seed-aware), not flattened residue×sample values.
        per_sample_mean = np.nanmean(disp_mat, axis=1)
        m, lo, hi = st.bootstrap_ci(per_sample_mean,
                                    seed_labels=_seed_labels_for(ensembles[name]))
        conc.setdefault(ptm, {})[tier] = {
            "mean": m, "lo": lo, "hi": hi,
            "n": int(ensembles[name].n_samples),
            "artifact": False,
            "hetero_tier": h.tier,
            "iqr": h.rmsd_iqr,
            "rmsd_max": h.rmsd_max,
        }
        lig_mult.setdefault(ptm, {})[tier] = struct.ligand_mult.get(name, 0)

    # Cycle 7 fix: Also check for "DNA" directly (not just prefix)
    dna_groups = {g for g in grid_rows if g.startswith("DNA") or g == "DNA"}
    dna_rows = [grid_rows.index(g) for g in dna_groups if g in grid_rows]

    written += plot_fac.plot_panel_per_residue(
        cells, grid_rows, grid_cols, baseline_name, y_ceiling, plots_dir
    )
    written += plot_fac.plot_concentration_response(
        conc, grid_rows, baseline_name, plots_dir,
        dna_groups=dna_groups, ligand_mult=lig_mult,
        confound_note=struct.confound.get("warning"),
        noise_floor=float(np.nanmean(list(rmsf_lookup.values()))) if rmsf_lookup else float("nan"),
    )
    written += plot_fac.plot_ptm_effect_grid(
        grid, ns_mask, grid_rows, grid_cols, baseline_name, plots_dir,
        measured_mask=measured_mask, iqr_grid=iqr_grid,
        nclusters_grid=ncl_grid, noise_grid=noise_grid,
        dna_rows=dna_rows or None,
        artifact_mask=artifact_mask,  # NEW: pass artifact tracking
        hetero_high_mask=hetero_high_mask,  # Fix 3: high-heterogeneity cells
        ref_cell=_ref_grid_cell(struct, baseline_name, grid_rows, grid_cols),
        ion_tier=struct.ion_tier,
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
    
    # Cycle 6: Check for small ensemble sizes and add warning
    min_size = min(sizes)
    small_warning = ""
    if min_size < 10:
        small_warning = " ⚠ small ensembles (<10 models)"
    elif min_size < 20:
        small_warning = " ⚠ modest ensembles (<20 models)"
    
    return (f"per-condition sample ensembles of {min_size}–{max(sizes)} models "
            f"used for displacement CIs and baseline RMSF{small_warning}")


def _caveats(conditions, ensembles, baseline_name, collapsed_conditions=None,
             baseline_warning=None, confound_warning=None,
             fold_warnings=None, cluster_separation=None,
             refs_by_context=None, context_offset=None) -> List[str]:
    """
    Methodological caveats for findings.md.

    Only includes items that are not already surfaced as structured fields
    (baseline_composition_warning, confound_warning, fold_divergence_warnings
    are rendered separately in the report).
    """
    # Data-driven salt/Cl/water collinearity check.  In these inputs the
    # "concentration" axis co-varies Na, Cl and water; if they are (near-)
    # perfectly correlated, any concentration-response trend cannot be
    # attributed to ionic strength vs hydration — the effects are inseparable.
    salt = [(c.n_na, c.n_cl, c.n_water) for c in conditions.values() if c.n_na > 0]
    out = []
    if len(salt) >= 3:
        na = np.array([s[0] for s in salt], dtype=float)
        cl = np.array([s[1] for s in salt], dtype=float)
        wat = np.array([s[2] for s in salt], dtype=float)
        def _corr(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])
        r_naw = _corr(na, wat)
        r_nacl = _corr(na, cl)
        if np.isfinite(r_naw) and abs(r_naw) > 0.98:
            out.append(
                f"Salt/water confound: across the {len(salt)} salt-containing conditions, "
                f"Na and water counts are essentially collinear (r={r_naw:.3f}; "
                f"Cl–Na r={r_nacl:.3f}). The 'concentration' axis varies Na, Cl and water "
                "in lockstep, so a concentration-response trend CANNOT separate ionic-strength "
                "effects from hydration/solvent effects; do not attribute it to ionic strength alone."
            )
        else:
            out.append(
                f"Salt/water relationship: Na–water r={r_naw:.3f}, Cl–Na r={r_nacl:.3f} across "
                f"{len(salt)} salt conditions (reported so concentration-response trends can be "
                "interpreted with the salt/solvent covariation in mind)."
            )
    out += [
        "Displacement is measured against AF3's own sampling noise (baseline RMSF); "
        "residues whose 95% CI does not clear the noise band are not interpreted as real motion.",
        "n_significant counts residues with statistically significant displacement "
        "(FDR<0.05, 95% CI clears the baseline-RMSF noise band) that are ALSO "
        "confidently placed (per-residue pLDDT \u2265 70 in both baseline and condition). "
        "Low-confidence residues (e.g. disordered termini) are excluded because their "
        "large apparent displacement reflects AF3 placement uncertainty, not motion. "
        "The pre-confidence-gate statistical count is retained as n_significant_stat_only.",
    ]
    if collapsed_conditions:
        out.append(
            "Collapsed predictions (protein mean pLDDT < 50 AND macromolecule PAE > 25 Å): "
            + ", ".join(sorted(collapsed_conditions))
            + ". Collapsed predictions are excluded from the PTM "
            "grid, concentration-response, and clustering; their displacements are not "
            "interpretable as conformational change. Full-system pTM/PAE are not used "
            "for this verdict because free ion/water tokens inflate them independent of fold quality."
        )
    if refs_by_context and refs_by_context.get(True) and refs_by_context.get(False) \
            and refs_by_context.get(True) != refs_by_context.get(False):
        msg = (
            "Context-matched references: RMSD is reported both vs the global baseline "
            f"({baseline_name}) and vs a DNA-context-matched reference "
            f"(no-DNA→{refs_by_context.get(False)}, DNA→{refs_by_context.get(True)}). "
            "Because the baseline is apo, a DNA condition's RMSD-vs-baseline is dominated "
            "by the constant apo→DNA-bound transition")
        if context_offset is not None:
            msg += f" (≈{context_offset:.1f} Å)"
        msg += ("; rmsd_vs_context_ref isolates the PTM/ion perturbation within the "
                "DNA-bound state and is the biologically relevant quantity for those conditions.")
        out.append(msg)
    if cluster_separation and cluster_separation.get("separation_adequate") is False:
        out.append(
            "Between-condition structural clusters are EXPLORATORY: the between-condition "
            f"RMSD scale (median {cluster_separation.get('between_condition_median_rmsd_A')} Å) is "
            f"within the within-condition ensemble noise floor "
            f"({cluster_separation.get('within_condition_noise_floor_A')} Å, ≈√2× median "
            "replicate-to-mean RMSD). Cluster assignments are not robustly separable from AF3 "
            "sampling noise and depend on which representative was selected; do not interpret "
            "them as distinct condition-driven conformational states."
        )
    no_ens = [n for n, e in ensembles.items() if not e.has_structural_ensemble]
    if no_ens:
        out.append("No usable sample ensemble for: " + ", ".join(sorted(no_ens))
                    + " (their displacement CIs are point estimates).")
    return out


# ---------------------------------------------------------------------------
# analysisscripts additional helpers
# ---------------------------------------------------------------------------

def _baseline_violin_data(baseline_name: str, ensembles: Dict[str, EnsembleModel]) -> dict:
    """
    Prepare baseline ensemble metrics for violin plotting (new feature).
    
    Returns dict with pTM, ipTM, pLDDT, PAE distributions across baseline seeds.
    """
    base_ens = ensembles.get(baseline_name)
    if not base_ens or not base_ens.has_structural_ensemble:
        return {}
    
    ptm = np.asarray(base_ens.ptm, dtype=float) if hasattr(base_ens, 'ptm') else np.empty(0)
    iptm = np.asarray(base_ens.iptm, dtype=float) if hasattr(base_ens, 'iptm') else np.empty(0)
    plddt = np.asarray(base_ens.plddt_mean, dtype=float) if hasattr(base_ens, 'plddt_mean') else np.empty(0)
    
    return {
        "n_samples": int(base_ens.n_samples),
        "ptm": ptm[np.isfinite(ptm)].tolist(),
        "iptm": iptm[np.isfinite(iptm)].tolist(),
        "plddt": plddt[np.isfinite(plddt)].tolist(),
    }


def _baseline_cluster_analysis(baseline_name: str, ensembles: Dict[str, EnsembleModel],
                                plddt_cutoff: float, cluster_threshold: float) -> dict:
    """
    Detect structural substates in baseline ensemble (new feature).
    
    Returns cluster assignments, centroid RMSD stats, and dominant fraction
    for the baseline ensemble itself.
    """
    base_ens = ensembles.get(baseline_name)
    if not base_ens or not base_ens.has_structural_ensemble:
        return {}
    
    coords = np.asarray(base_ens.ca_coords, dtype=np.float64)
    plddts = np.asarray(base_ens.ca_plddts, dtype=np.float64)
    S, N, _ = coords.shape
    
    fit_mask = np.all(plddts > plddt_cutoff, axis=0)
    if fit_mask.sum() < 3:
        fit_mask = np.ones(N, dtype=bool)
    
    aligned = geom.superpose_stack_to_mean(coords, plddts, plddt_cutoff)
    mean = aligned.mean(axis=0)
    rmsd_to_mean = np.sqrt(
        np.mean(np.sum((aligned[:, fit_mask] - mean[fit_mask]) ** 2, axis=2), axis=1)
    )
    
    labels = np.ones(S, dtype=int)
    if HAS_SCIPY and S >= 2:
        from scipy.spatial.distance import squareform
        from scipy.cluster.hierarchy import linkage
        
        pw = geom.pairwise_rmsd(aligned, fit_mask)
        condensed = squareform(pw, checks=False)
        if condensed.size and np.any(condensed > 0):
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=cluster_threshold, criterion="distance")
    
    uniq, counts = np.unique(labels, return_counts=True)
    fractions = counts / counts.sum()
    
    return {
        "n_clusters": int(uniq.size),
        "cluster_assignments": [int(c) for c in labels.tolist()],
        "dominant_fraction": float(fractions.max()),
        "rmsd_to_mean": [float(x) for x in rmsd_to_mean.tolist()],
        "rmsd_median": float(np.median(rmsd_to_mean)),
    }


def _baseline_pairwise_rmsd_distribution(baseline_name: str, ensembles: Dict[str, EnsembleModel],
                                          plddt_cutoff: float) -> dict:
    """
    Compute pairwise RMSD distribution within baseline ensemble (new feature).
    
    This quantifies the natural RMSD variability between baseline trajectories
    before interpreting differences in perturbed systems.
    """
    base_ens = ensembles.get(baseline_name)
    if not base_ens or not base_ens.has_structural_ensemble:
        return {}
    
    coords = np.asarray(base_ens.ca_coords, dtype=np.float64)
    plddts = np.asarray(base_ens.ca_plddts, dtype=np.float64)
    S, N, _ = coords.shape
    
    fit_mask = np.all(plddts > plddt_cutoff, axis=0)
    if fit_mask.sum() < 3:
        fit_mask = np.ones(N, dtype=bool)
    
    aligned = geom.superpose_stack_to_mean(coords, plddts, plddt_cutoff)
    pw_rmsd = geom.pairwise_rmsd(aligned, fit_mask)
    
    triu_idx = np.triu_indices(S, k=1)
    pairwise_vals = pw_rmsd[triu_idx]
    pairwise_vals = pairwise_vals[np.isfinite(pairwise_vals)]
    
    return {
        "n_pairs": int(len(pairwise_vals)),
        "mean": float(np.mean(pairwise_vals)) if pairwise_vals.size else float("nan"),
        "median": float(np.median(pairwise_vals)) if pairwise_vals.size else float("nan"),
        "std": float(np.std(pairwise_vals, ddof=1)) if pairwise_vals.size >= 2 else float("nan"),
        "iqr": float(np.percentile(pairwise_vals, 75) - np.percentile(pairwise_vals, 25)) if pairwise_vals.size else float("nan"),
        "max": float(np.max(pairwise_vals)) if pairwise_vals.size else float("nan"),
    }


def _baseline_confidence_metrics(baseline_name: str, ensembles: Dict[str, EnsembleModel]) -> dict:
    """
    Detailed confidence metrics for baseline ensemble (new feature).
    
    Returns per-seed distributions of pTM, ipTM, pLDDT for statistical analysis.
    """
    base_ens = ensembles.get(baseline_name)
    if not base_ens or not base_ens.has_structural_ensemble:
        return {}
    
    ptm = np.asarray(base_ens.ptm, dtype=float) if hasattr(base_ens, 'ptm') else np.empty(0)
    iptm = np.asarray(base_ens.iptm, dtype=float) if hasattr(base_ens, 'iptm') else np.empty(0)
    plddt = np.asarray(base_ens.plddt_mean, dtype=float) if hasattr(base_ens, 'plddt_mean') else np.empty(0)
    
    return {
        "n_samples": int(base_ens.n_samples),
        "ptm_mean": float(np.mean(ptm[np.isfinite(ptm)])) if ptm.size else float("nan"),
        "ptm_std": float(np.std(ptm[np.isfinite(ptm)], ddof=1)) if ptm.size >= 2 else float("nan"),
        "ptm_cv": float(np.std(ptm[np.isfinite(ptm)], ddof=1) / np.mean(ptm[np.isfinite(ptm)])) if ptm.size and np.mean(ptm[np.isfinite(ptm)]) > 0 else float("nan"),
        "iptm_mean": float(np.mean(iptm[np.isfinite(iptm)])) if iptm.size else float("nan"),
        "iptm_std": float(np.std(iptm[np.isfinite(iptm)], ddof=1)) if iptm.size >= 2 else float("nan"),
        "plddt_mean": float(np.mean(plddt[np.isfinite(plddt)])) if plddt.size else float("nan"),
        "plddt_std": float(np.std(plddt[np.isfinite(plddt)], ddof=1)) if plddt.size >= 2 else float("nan"),
    }


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


def _fold_divergence(df_dist, baseline_name, compute_tm, conditions=None):
    """
    Fold-consistency verdict from TM-score vs baseline (plan 2.5).

    Thresholds follow the established TM-score interpretation: a TM-score > 0.5
    indicates the same fold, while < 0.5 indicates a genuinely different fold
    (Zhang & Skolnick 2004, Proteins 57:702; Xu & Zhang 2010, Bioinformatics
    26:889).  The 0.5 cut is length-independent by construction.

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
            verdict = "conserved"        # very high similarity
        elif tm >= 0.50:
            verdict = "similar"          # same fold (TM > 0.5)
        else:
            verdict = "divergent"        # different fold (TM < 0.5)
        
        # Issue 7 fix: Add perturbation context
        has_dna = conditions[r["condition"]].n_nucleic_residues > 0 if conditions and r["condition"] in conditions else False
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
                    f"(<0.50, different fold) — the same protein with a PTM predicts a "
                    f"different fold. This is biologically unusual and warrants scrutiny."
                )
            else:
                warnings.append(
                    f"DNA co-fold divergence: {r['condition']} vs baseline TM={tm:.2f} "
                    f"(<0.50) — expected for a protein+DNA complex vs protein-alone "
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


def _seed_bias_assessment(ensembles) -> dict:
    """
    Seed-bias power note (plan 2.6), derived from the actual ensemble layout.

    AF3 replicates are generated as seeds × samples-per-seed; samples sharing a
    seed are more correlated than samples across seeds, so a small number of
    seeds limits the power to detect seed effects.  Rather than asserting a fixed
    design, we parse the real seed/sample structure from the replicate file
    names (``seed-<S>_sample-<M>``).
    """
    import re as _re
    seed_re = _re.compile(r"seed-(\d+)_sample-(\d+)")
    n_seeds_per_cond: List[int] = []
    n_samples_per_cond: List[int] = []
    parsed_any = False
    for e in ensembles.values():
        seeds = set()
        n_samp = 0
        for p in getattr(e, "sample_paths", []) or []:
            m = seed_re.search(str(p))
            if m:
                seeds.add(int(m.group(1)))
                n_samp += 1
        if seeds:
            parsed_any = True
            n_seeds_per_cond.append(len(seeds))
            n_samples_per_cond.append(n_samp)

    if not parsed_any:
        return {
            "design": "seed/sample structure not recoverable from replicate file names",
            "low_power_warning": True,
            "note": ("Replicate seed grouping could not be parsed, so seed-effect power "
                     "cannot be assessed; treat any per-seed analysis with caution."),
        }

    import numpy as _np
    med_seeds = int(_np.median(n_seeds_per_cond))
    med_n = int(_np.median(n_samples_per_cond))
    sps = (med_n / med_seeds) if med_seeds else float("nan")
    low_power = med_seeds < 20  # few independent seeds -> limited power
    return {
        "design": (f"{med_seeds} seeds × ~{sps:.0f} samples/seed "
                   f"(median n={med_n}) per condition, parsed from replicate names"),
        "n_seeds_median": med_seeds,
        "n_samples_median": med_n,
        "low_power_warning": bool(low_power),
        "note": (f"With a median of {med_seeds} seeds per condition, a between-seed "
                 "test (e.g. Kruskal–Wallis) has limited power to detect moderate seed "
                 "effects. Non-significance should not be read as absence of seed bias; "
                 "replicate-level bootstrap CIs also treat the within-seed samples as "
                 "if independent, so they may be mildly anti-conservative."),
    }


def _tier_key(t: str) -> float:
    if t == "0x":
        return 0.0
    try:
        return float(t.rstrip("x"))
    except ValueError:
        return 9_999.0


def select_context_references(conditions, baseline_name, struct):
    """Choose a context-matched structural reference for each condition.

    Comparing every condition to a single apo baseline conflates the large,
    constant apo→DNA-bound transition with the perturbations actually under
    study (PTM, ionic strength).  We therefore stratify by molecular context
    (DNA present/absent) and, within each context, use the unmodified,
    baseline-ion reference (e.g. apo-noDNA = the baseline; DNA = the
    unmodified ion-free DNA condition).  RMSD to this reference isolates the
    studied perturbation; the DNA-binding transition is held constant.

    Returns (refs_by_condition, refs_by_context) where refs_by_context maps
    has_dna(bool) -> reference condition name.
    """
    base_tier = struct.ion_tier.get(baseline_name)
    # An "unmodified" reference has no PTM.  Note build_experiment_structure
    # encodes ptm_group as "+".join(["DNA"?] + ptm_labels), so an unmodified DNA
    # condition has ptm_group == "DNA" (not "none").
    _none = {"none", "unmodified", "", "DNA"}
    refs_by_context: Dict[bool, Optional[str]] = {}
    for dna_status in (False, True):
        same = [n for n in conditions
                if struct.has_dna.get(n, False) == dna_status
                and struct.ptm_group.get(n, "none") in _none]
        if not same:
            refs_by_context[dna_status] = None
            continue
        # prefer the reference whose ion tier matches the baseline's; else the
        # lowest ion tier in that context.
        exact = [n for n in same if struct.ion_tier.get(n) == base_tier]
        pool = exact if exact else sorted(
            same, key=lambda n: _tier_key(struct.ion_tier.get(n, "")))
        refs_by_context[dna_status] = sorted(pool)[0]

    refs = {}
    for n in conditions:
        ds = struct.has_dna.get(n, False)
        refs[n] = refs_by_context.get(ds) or baseline_name
    return refs, refs_by_context


def _scientific_summary_rows(others, conditions, df_dist, df_conf, hetero,
                             struct, label_map, collapsed_conditions, per_res_frames,
                             baseline_noise_rmsd=float("nan"), baseline_name=None) -> List[dict]:
    """Structured per-condition summary rows for findings (plan 3.3)."""
    dist_by = {r["condition"]: r for r in df_dist.to_dict("records")}
    conf_by = {r["condition"]: r for r in df_conf.to_dict("records")}
    rows = []
    for name in others:
        d = dist_by.get(name, {})
        c = conf_by.get(name, {})
        h = hetero[name]
        rmsd = d.get("rmsd", float("nan"))
        rmsd_lo = d.get("rmsd_lo", float("nan"))
        tm_score = d.get("tm_score", float("nan"))
        # n_significant is now confidence-gated (FDR-significant AND pLDDT>=70 in
        # both states), the grounded successor to the old RMSF<3 "core" count.
        n_sig_core = d.get("n_significant", 0)
        is_collapsed = name in collapsed_conditions
        
        # Issue 2 fix: Compute bimodal fraction from per-residue data
        bimodal_frac = 0.0
        pr = per_res_frames.get(name)
        if pr is not None and "bimodality_flag" in pr.columns and len(pr):
            bimodal_frac = float(pr["bimodality_flag"].mean())
        
        # Judge the studied perturbation: use the context-matched RMSD and the
        # context reference's own ensemble spread as the noise floor, so DNA
        # conditions are not scored on the apo→DNA-bound transition.
        ctx_ref = d.get("context_ref")
        rmsd_ctx = d.get("rmsd_vs_context_ref", rmsd)
        rmsd_ctx_lo = d.get("rmsd_vs_context_ref_lo", rmsd_lo)
        ctx_noise = baseline_noise_rmsd
        if ctx_ref in hetero and np.isfinite(getattr(hetero[ctx_ref], "rmsd_median", np.nan)):
            ctx_noise = float(hetero[ctx_ref].rmsd_median)

        # n_significant is measured vs the global (apo) baseline, so for
        # context-matched (e.g. DNA) conditions it includes the apo→DNA shift
        # and must NOT be used as a perturbation-shift trigger; judge those on
        # the context-matched RMSD alone.
        context_matched = bool(ctx_ref and ctx_ref != baseline_name)
        verdict_nsig = 0 if context_matched else n_sig_core

        # Issue 2 fix: Use bimodality-aware verdict (Issue 5: use n_sig_core)
        decision = _structural_shift_verdict(
            name, is_collapsed, verdict_nsig, rmsd_ctx, h, tm_score, bimodal_frac,
            rmsd_lo=rmsd_ctx_lo, baseline_noise_rmsd=ctx_noise,
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
            "structural_shift_basis": ("context_matched" if ctx_ref and ctx_ref != baseline_name
                                       else "vs_baseline"),
            "context_ref": ctx_ref,
            "rmsd": _r(rmsd),
            "rmsd_vs_context_ref": _r(rmsd_ctx),
            "delta_ptm": c.get("delta_ptm"),
            "delta_iptm": c.get("delta_iptm"),
            "confidence_tier": "likely_artifact" if is_collapsed else "ok",
            "heterogeneity_tier": h.tier,
            "n_clusters": h.n_clusters,
            "dominant_fraction": _r(h.dominant_fraction),
            "perturbation_class": pert_class,  # Issue 3 fix
        })
    return rows


def _structural_shift_verdict(
    name: str,
    is_collapsed: bool,
    n_sig_core: int,  # Issue 5 fix: renamed from n_sig
    rmsd: float,
    h,  # HeterogeneitySummary
    tm_score: float,
    bimodal_fraction: float,
    rmsd_lo: float = float("nan"),
    baseline_noise_rmsd: float = float("nan"),
) -> str:
    """
    Bimodality-aware, noise-grounded structural shift verdict.

    The global RMSD criterion is grounded in the baseline ensemble's *intrinsic*
    RMSD spread (``baseline_noise_rmsd``, the baseline replicate-to-mean RMSD),
    not in fixed absolute Å thresholds.  Comparing a condition to a single
    baseline representative yields an RMSD of order the baseline's own sampling
    spread even when nothing changed, so a global shift is only credible when
    the condition's RMSD (95% CI lower bound) exceeds that noise floor.  When
    the noise floor is unavailable it falls back to a conservative 3.0 Å.
    Per-residue evidence (``n_sig_core``) is already noise- and confidence-gated
    upstream, so it remains a valid independent signal of localised motion.

    Priority order (first match wins):
      1. artifact (excluded)        — is_collapsed == True
      2. multimodal_ensemble        — bimodal_fraction > 0.50 AND
                                      n_clusters >= 4 AND IQR > 3 Å
      3. fold_divergent             — TM-score < 0.50 (different fold;
                                      Zhang & Skolnick 2004; Xu & Zhang 2010)
      4. clear_shift                — rmsd_lo > baseline noise floor AND
                                      n_sig_core > 5 AND dominant_fraction >= 0.60
      5. possible_shift             — rmsd > baseline noise floor OR n_sig_core > 0
      6. no_shift_detected          — otherwise
    """
    if is_collapsed:
        return "artifact (excluded)"

    is_multimodal = (
        bimodal_fraction > 0.50
        and h.n_clusters >= 4
        and h.rmsd_iqr > 3.0   # IQR threshold in Å
    )
    # Cycle 5 improvement: Also flag as multimodal if bimodal_fraction > 0.30
    # AND clusters >= 3 AND IQR > 2.5 Å (less strict but still meaningful)
    is_weak_multimodal = (
        bimodal_fraction > 0.30
        and h.n_clusters >= 3
        and h.rmsd_iqr > 2.5
        and not is_multimodal
    )
    if is_multimodal:
        return "multimodal_ensemble"
    if is_weak_multimodal:
        return "possible_multimodal"  # Suggestive but not definitive

    if math.isfinite(tm_score) and tm_score < 0.50 and not is_collapsed:
        return "fold_divergent"

    floor = baseline_noise_rmsd if math.isfinite(baseline_noise_rmsd) else 3.0
    rmsd_lo_eff = rmsd_lo if math.isfinite(rmsd_lo) else rmsd

    if (rmsd_lo_eff and math.isfinite(rmsd_lo_eff) and rmsd_lo_eff > floor
            and n_sig_core > 5):
        if h.dominant_fraction >= 0.60:
            return "clear_shift"
        else:
            return "possible_shift"  # displaced but fragmented ensemble

    if (rmsd and math.isfinite(rmsd) and rmsd > floor) or n_sig_core > 0:
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
