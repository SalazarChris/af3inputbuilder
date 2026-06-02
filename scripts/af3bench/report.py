"""
Reporting — machine- and human-readable findings.

Writes findings.json (structured) and findings.md (thesis-facing summary):
baseline choice, per-condition RMSD +/- CI, top moving residues with
significance, concentration / PTM interaction summary, and explicit
within-noise caveats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from .io import write_json

log = logging.getLogger("af3bench.report")


def write_findings(
    findings: dict,
    output_dir: Path,
) -> None:
    write_json(findings, output_dir / "findings.json")
    _write_markdown(findings, output_dir / "findings.md")


def _fmt_ci(mean, lo, hi, unit="Å"):
    if mean is None:
        return "n/a"
    try:
        return f"{mean:.2f} [{lo:.2f}, {hi:.2f}] {unit}"
    except (TypeError, ValueError):
        return "n/a"


def _write_markdown(f: dict, path: Path) -> None:
    lines: List[str] = []
    lines.append("# AF3 condition-comparison findings\n")
    lines.append(f"- Baseline: **{f.get('baseline')}**  ")
    lines.append(f"- Baseline selection: {f.get('baseline_reason', 'n/a')}  ")
    lines.append(f"- Conditions analysed: {f.get('n_conditions', 0)}  ")
    if f.get("skipped"):
        lines.append(f"- Skipped (no model output): {', '.join(f['skipped'])}  ")
    lines.append(f"- Ensemble: {f.get('ensemble_note', 'n/a')}\n")

    # Structural distances
    lines.append("## Structural distance vs baseline\n")
    lines.append("| Condition | RMSD (mean [95% CI]) | TM-score | Residues moving (FDR<0.05) | QC |")
    lines.append("|---|---|---|---|---|")
    for row in f.get("distances", []):
        rmsd = _fmt_ci(row.get("rmsd"), row.get("rmsd_lo"), row.get("rmsd_hi"))
        tm = f"{row['tm_score']:.3f}" if row.get("tm_score") is not None else "n/a"
        qc = "⚠ likely failed" if row.get("likely_failed") else "ok"
        lines.append(
            f"| {row['condition']} | {rmsd} | {tm} | {row.get('n_significant', 0)} | {qc} |"
        )
    lines.append("")
    if f.get("likely_failed"):
        lines.append(
            "> ⚠ Conditions flagged **likely failed** (low ipTM / high PAE) show large "
            "displacements that reflect model collapse, not biology.\n"
        )

    # Structural clusters
    clusters = f.get("clusters", [])
    if clusters:
        lines.append("## Structural clusters\n")
        lines.append("Conditions grouped by overall Cα RMSD similarity "
                     "(see structural_clustering figure and 06_clusters PyMOL scenes).\n")
        lines.append("| Cluster | n | PTM groups | DNA | Members |")
        lines.append("|---|---|---|---|---|")
        for c in clusters:
            lines.append(
                f"| {c['cluster']} | {c['n_members']} | "
                f"{', '.join(c['ptm_groups'])} | {'yes' if c['any_dna'] else 'no'} | "
                f"{', '.join(c['members'])} |"
            )
        lines.append("")

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
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Saved: %s", path.name)
