"""
Reporting — machine- and human-readable findings (af3bench2 overhaul).

Writes:
  findings.json   structured findings (now incl. baseline composition warning,
                  confound warning, confidence tiers, fold-divergence warnings,
                  scientific summary table, seed-bias assessment).
  findings.md     thesis-facing summary led by a structured per-condition table
                  (plan 3.3).
  findings.html   rendered companion with colour badges + heterogeneity
                  thumbnail (plan 3.4).
"""

from __future__ import annotations

import html
import logging
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .io import write_json
from .plots import style

log = logging.getLogger("af3bench2.report")


def write_findings(findings: dict, output_dir: Path) -> None:
    write_json(findings, output_dir / "findings.json")
    _write_markdown(findings, output_dir / "findings.md")
    _write_html(findings, output_dir / "findings.html")


def _fmt_ci(mean, lo, hi, unit="Å"):
    if mean is None:
        return "n/a"
    try:
        return f"{mean:.2f} [{lo:.2f}, {hi:.2f}] {unit}"
    except (TypeError, ValueError):
        return "n/a"


_HETERO_MD = {"low": "low", "moderate": "moderate ⚠", "high": "high ⚠⚠"}


def _write_markdown(f: dict, path: Path) -> None:
    lines: List[str] = []
    lines.append("# AF3 condition-comparison findings (af3bench2)\n")
    # Cycle 10: Add pipeline version to markdown report
    version = f.get('pipeline_version', 'unknown')
    lines.append(f"- Pipeline version: **af3bench2 {version}**  ")
    lines.append(f"- Baseline: **{f.get('baseline')}**  ")
    lines.append(f"- Baseline selection: {f.get('baseline_reason', 'n/a')}  ")
    lines.append(f"- Conditions analysed: {f.get('n_conditions', 0)}  ")
    if f.get("skipped"):
        lines.append(f"- Skipped (no model output): {', '.join(f['skipped'])}  ")
    lines.append(f"- Ensemble: {f.get('ensemble_note', 'n/a')}\n")

    # Correctness warnings up front
    if f.get("baseline_composition_warning"):
        lines.append(f"> ⚠ **Baseline composition.** {f['baseline_composition_warning']}\n")
    if f.get("confound_warning"):
        lines.append(f"> ⚠ **Confound.** {f['confound_warning']}\n")
    for fw in f.get("fold_divergence_warnings", []):
        lines.append(f"> ⚠ **Fold divergence.** {fw}\n")

    # Issue 1 fix: Quarantined artifacts section
    artifacts = f.get("likely_artifacts", [])
    if artifacts:
        lines.append("## Quarantined artifacts (model failure — excluded from clustering)\n")
        lines.append("| Condition | RMSD (Å) | Confidence tier | Signature |")
        lines.append("|---|---|---|---|")
        for art in artifacts:
            # Find the condition in distances
            dist_row = next((r for r in f.get("distances", []) if r.get("condition") == art), None)
            if dist_row:
                rmsd = dist_row.get("rmsd", "n/a")
                tier = dist_row.get("confidence_tier", "n/a")
                lines.append(
                    f"| {art} | {rmsd if isinstance(rmsd, str) else f'{rmsd:.2f}'} | {tier} | "
                    f"Model collapse — excluded from analysis |"
                )
        lines.append("")

    # Cycle 11: Baseline ensemble characterization section
    baseline_violin = f.get("baseline_violin_data", {})
    baseline_cluster = f.get("baseline_cluster_info", {})
    baseline_rmsd = f.get("baseline_pairwise_rmsd", {})
    baseline_conf = f.get("baseline_confidence_metrics", {})
    
    if baseline_violin or baseline_cluster or baseline_rmsd or baseline_conf:
        lines.append("## Baseline ensemble characterization (Cycle 11)\n")
        
        if baseline_conf:
            cv = baseline_conf.get("ptm_cv", float("nan"))
            lines.append(f"- **Baseline confidence variability:** PTM CV = {cv:.3f}" + 
                        (" (high variability)" if cv and cv > 0.05 else ""))
        
        if baseline_cluster:
            n_clusters = baseline_cluster.get("n_clusters", 1)
            dominant = baseline_cluster.get("dominant_fraction", 1.0)
            lines.append(f"- **Baseline structural clusters:** {n_clusters} (dominant fraction = {dominant:.0%})")
        
        if baseline_rmsd:
            n_pairs = baseline_rmsd.get("n_pairs", 0)
            mean_rmsd = baseline_rmsd.get("mean", float("nan"))
            std_rmsd = baseline_rmsd.get("std", float("nan"))
            if n_pairs > 0:
                lines.append(f"- **Baseline pairwise RMSD:** {mean_rmsd:.2f} ± {std_rmsd:.2f} Å (n={n_pairs} pairs)")
        
        lines.append("")
    
    # Scientific summary table (plan 3.3)
    summary = f.get("summary_table", [])
    if summary:
        lines.append("## Scientific summary\n")
        
        # Issue 2 fix: Add multimodal warning if any conditions are multimodal
        multimodal_conds = [r for r in summary if r.get("structural_shift") in ("multimodal_ensemble", "possible_multimodal")]
        if multimodal_conds:
            lines.append("> ⚠ **Multimodal conditions.** The following conditions produced heterogeneous")
            lines.append("> ensembles with bimodal per-residue displacement distributions. The mean")
            lines.append("> displacement is not a reliable summary. Inspect the cluster portrait for")
            lines.append("> the dominant structural populations: "
                        + ", ".join(r.get("label_short", r.get("condition", "")) for r in multimodal_conds))
            lines.append("")
        
        # Issue 3 fix: Stratify by perturbation class
        CLASSES = [
            ("ptm", "PTM conditions"),
            ("dna", "DNA conditions"),
            ("salt_ligand_only", "Salt / ligand-only controls"),
            ("ptm_dna", "PTM + DNA conditions"),
        ]
        for cls_key, cls_label in CLASSES:
            cls_rows = [r for r in summary if r.get("perturbation_class") == cls_key]
            if not cls_rows:
                continue
            lines.append(f"### {cls_label}\n")
            lines.append("| Condition (short) | PTM | Salt | Lig.mult | Structural shift | "
                         "ΔpTM | ΔipTM | Confidence | Reproducibility | n clusters | Dom. frac |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in cls_rows:
                lines.append(
                    f"| {_md(r.get('label_short'))} | {_md(r.get('ptm_group'))} | {_md(r.get('salt_tier'))} | "
                    f"{r.get('ligand_mult')}× | {_md(r.get('structural_shift'))} | "
                    f"{_num(r.get('delta_ptm'))} | {_num(r.get('delta_iptm'))} | "
                    f"{_md(r.get('confidence_tier'))} | {_HETERO_MD.get(r.get('heterogeneity_tier'),'?')} | "
                    f"{r.get('n_clusters')} | {_num(r.get('dominant_fraction'))} |"
                )
            lines.append("")

    # Structural distances
    lines.append("## Structural distance vs baseline\n")
    lines.append("Two RMSDs are reported: **vs baseline** (the apo reference) and "
                 "**vs context-matched reference** (same DNA context), which isolates the "
                 "PTM/ion perturbation from the constant apo→DNA-bound transition.\n")
    lines.append("| Condition | RMSD vs baseline (mean [95% CI]) | RMSD vs context ref [95% CI] (ref) | TM-score | Residues moving (FDR<0.05) | Confidence |")
    lines.append("|---|---|---|---|---|---|")
    for row in f.get("distances", []):
        rmsd = _fmt_ci(row.get("rmsd"), row.get("rmsd_lo"), row.get("rmsd_hi"))
        tm = f"{row['tm_score']:.3f}" if row.get("tm_score") is not None else "n/a"
        tier = row.get("confidence_tier", "ok")
        cref = row.get("context_ref")
        if cref and cref != f.get("baseline") and row.get("rmsd_vs_context_ref") is not None:
            # Use ctx_ref_labels if available (nested under context_references), otherwise show reference name
            ref_label = f.get("context_references", {}).get("ctx_ref_labels", {}).get(cref, cref)
            if ref_label == "self":
                ref_label = "self (no perturbation)"
            ctx = _fmt_ci(row.get("rmsd_vs_context_ref"),
                          row.get("rmsd_vs_context_ref_lo"),
                          row.get("rmsd_vs_context_ref_hi")) + f" ({ref_label})"
        else:
            ctx = "— (= baseline)"
        lines.append(
            f"| {row['condition']} | {rmsd} | {ctx} | {tm} | {row.get('n_significant', 0)} | {tier} |"
        )
    lines.append("")

    # Structural clusters
    clusters = f.get("clusters", [])
    if clusters:
        lines.append("## Structural clusters (between conditions)\n")
        lines.append("| Cluster | n | PTM groups | DNA | Members |")
        lines.append("|---|---|---|---|---|")
        for c in clusters:
            lines.append(
                f"| {c['cluster']} | {c['n_members']} | "
                f"{', '.join(c['ptm_groups'])} | {'yes' if c['any_dna'] else 'no'} | "
                f"{', '.join(c['members'])} |"
            )
        lines.append("")

    # Issue 8 fix: Only render seed bias section when low_power_warning=False
    sb = f.get("seed_bias")
    if sb and not sb.get("low_power_warning"):
        lines.append("## Seed bias assessment\n")
        lines.append(f"- Design: {sb.get('design')}")
        lines.append(f"- Low statistical power: {'yes' if sb.get('low_power_warning') else 'no'}")
        lines.append(f"- {sb.get('note')}\n")

    # Top moving residues
    lines.append("## Top moving residues per condition\n")
    for cond, residues in f.get("top_residues", {}).items():
        if not residues:
            continue
        lines.append(f"**{cond}**  ")
        for r in residues:
            flag = "significant" if r.get("significant") else "within noise"
            lines.append(
                f"- {r['chain']}{r['resnum']}: "
                f"{_fmt_ci(r['mean'], r['lo'], r['hi'])} "
                f"(baseline noise {r.get('rmsf', float('nan')):.2f} Å) — {flag}"
            )
        lines.append("")

    # Caveats
    lines.append("## Caveats\n")
    for c in f.get("caveats", []):
        lines.append(f"- {c}")
    if not f.get("fold_divergence_warnings") and not f.get("confidence_tiers"):
        lines.append("- TM-score computation was disabled; fold-divergence check was skipped.")
    
    # Cycle 10: Pipeline version in footer
    version = f.get('pipeline_version', 'unknown')
    lines.append("")
    lines.append(f"*Report generated with af3bench2 v{version}*")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved: %s", path.name)


def _num(v) -> str:
    try:
        return f"{float(v):+.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _md(v) -> str:
    """Escape a value for safe inclusion in a markdown table cell."""
    if v is None:
        return ""
    return str(v).replace("|", "\\|")


# ---------------------------------------------------------------------------
# HTML rendering (plan 3.4)
# ---------------------------------------------------------------------------

_SHIFT_BADGE = {
    "clear_shift": "#D55E00",
    "possible_shift": "#E69F00",
    "no_shift_detected": "#999999",
    "artifact (excluded)": "#B2182B",
    "multimodal_ensemble": "#7B2D8B",   # Issue 2 fix: purple — ambiguous, not a clean shift
    "fold_divergent": "#CC6600",        # Issue 2 fix: dark orange — distinct from clear_shift
}
_TIER_BADGE = {"low": "#009E73", "moderate": "#E69F00", "high": "#D55E00"}
_CONF_BADGE = {"ok": "#009E73", "low_confidence": "#E69F00", "likely_artifact": "#B2182B"}


def _badge(text, color):
    safe = html.escape(str(text))
    return (f'<span style="background:{color};color:#fff;padding:1px 7px;'
            f'border-radius:8px;font-size:0.82em;white-space:nowrap;">{safe}</span>')


def _write_html(f: dict, path: Path) -> None:
    summary = f.get("summary_table", [])
    rows_html: List[str] = []
    for r in summary:
        cond = html.escape(str(r.get("condition", "")))
        link = f"plots/per_residue_{_safe(cond)}.png"
        shift = r.get("structural_shift", "")
        tier = r.get("heterogeneity_tier", "")
        conf = r.get("confidence_tier", "")
        rows_html.append(
            "<tr>"
            f'<td><a href="{link}">{html.escape(str(r.get("label_short","")))}</a></td>'
            f"<td>{html.escape(str(r.get('ptm_group','')))}</td>"
            f"<td>{html.escape(str(r.get('salt_tier','')))}</td>"
            f"<td>{html.escape(str(r.get('ligand_mult','')))}×</td>"
            f"<td>{_badge(shift, _SHIFT_BADGE.get(shift, '#777'))}</td>"
            f"<td>{_num(r.get('delta_ptm'))}</td>"
            f"<td>{_num(r.get('delta_iptm'))}</td>"
            f"<td>{_badge(conf, _CONF_BADGE.get(conf, '#777'))}</td>"
            f"<td>{_badge(tier, _TIER_BADGE.get(tier, '#777'))}</td>"
            f"<td>{html.escape(str(r.get('n_clusters','')))}</td>"
            f"<td>{_num(r.get('dominant_fraction'))}</td>"
            "</tr>"
        )

    warnings_html = ""
    for key, lbl in (("baseline_composition_warning", "Baseline composition"),
                     ("confound_warning", "Confound")):
        if f.get(key):
            warnings_html += (f'<div class="warn"><b>{lbl}.</b> '
                              f'{html.escape(str(f[key]))}</div>')
    for fw in f.get("fold_divergence_warnings", []):
        warnings_html += f'<div class="warn"><b>Fold divergence.</b> {html.escape(str(fw))}</div>'
    
    # Cycle 11: Baseline ensemble info (HTML)
    baseline_violin = f.get("baseline_violin_data", {})
    baseline_cluster = f.get("baseline_cluster_info", {})
    baseline_rmsd = f.get("baseline_pairwise_rmsd", {})
    baseline_conf = f.get("baseline_confidence_metrics", {})
    if baseline_conf or baseline_cluster or baseline_rmsd:
        baseline_info = []
        if baseline_conf:
            cv = baseline_conf.get("ptm_cv", float("nan"))
            baseline_info.append(f'<p><b>Baseline confidence variability:</b> PTM CV = {cv:.3f}' + 
                                (' (high variability)' if cv and cv > 0.05 else '') + '</p>')
        if baseline_cluster:
            n_clusters = baseline_cluster.get("n_clusters", 1)
            dominant = baseline_cluster.get("dominant_fraction", 1.0)
            baseline_info.append(f'<p><b>Baseline structural clusters:</b> {n_clusters} (dominant = {dominant:.0%})</p>')
        if baseline_rmsd:
            n_pairs = baseline_rmsd.get("n_pairs", 0)
            mean_rmsd = baseline_rmsd.get("mean", float("nan"))
            std_rmsd = baseline_rmsd.get("std", float("nan"))
            if n_pairs > 0:
                baseline_info.append(f'<p><b>Baseline pairwise RMSD:</b> {mean_rmsd:.2f} ± {std_rmsd:.2f} Å (n={n_pairs})</p>')
        warnings_html += ''.join(baseline_info)

    thumb = ""
    if (path.parent / "plots" / "heterogeneity_overview.png").exists():
        thumb = ('<h2>Heterogeneity overview</h2>'
                 '<img src="plots/heterogeneity_overview.png" '
                 'style="max-width:760px;border:1px solid #ddd;">')
    
    # Cycle 11: Baseline ensemble violin plot
    if (path.parent / "plots" / "baseline_violins.png").exists():
        thumb += ('<h2>Baseline ensemble distribution</h2>'
                  '<img src="plots/baseline_violins.png" '
                  'style="max-width:760px;border:1px solid #ddd;">')

    # Cycle 10: Get version from findings, fallback to __version__
    version = f.get('pipeline_version')
    if not version:
        from . import __version__ as _ver
        version = _ver
    else:
        _ver = version

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>af3bench2 findings — {html.escape(str(f.get('baseline','')))}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2em auto;
        max-width: 1100px; color: #222; line-height: 1.45; }}
 h1 {{ font-size: 1.5em; }} h2 {{ font-size: 1.15em; margin-top: 1.4em; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
 th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: left; }}
 th {{ background: #f4f4f4; }}
 .warn {{ background:#FFF6E5; border-left:4px solid #E69F00; padding:8px 12px;
          margin:8px 0; font-size:0.9em; }}
 .meta {{ color:#777; font-size:0.82em; margin-top:2em; border-top:1px solid #eee;
          padding-top:1em; }}
</style></head><body>
<h1>AF3 condition-comparison findings (af3bench2)</h1>
<p><b>Baseline:</b> {html.escape(str(f.get('baseline','')))}<br>
<b>Selection:</b> {html.escape(str(f.get('baseline_reason','')))}<br>
<b>Conditions:</b> {f.get('n_conditions',0)}</p>
{warnings_html}
<h2>Scientific summary</h2>
<table>
<tr><th>Condition</th><th>PTM</th><th>Salt</th><th>Lig.</th><th>Structural shift</th>
<th>ΔpTM</th><th>ΔipTM</th><th>Confidence</th><th>Reproducibility</th>
<th>n cl.</th><th>Dom.frac</th></tr>
{''.join(rows_html)}
</table>
{thumb}
<div class="meta">
 af3bench2 v{html.escape(str(version))} · baseline: {html.escape(str(f.get('baseline','')))} · generated {date.today().isoformat()}
 {('· ' + html.escape(str(f.get('baseline_composition_warning')))) if f.get('baseline_composition_warning') else ''}
</div>
</body></html>"""
    path.write_text(doc, encoding="utf-8")
    log.info("Saved: %s", path.name)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:120]
