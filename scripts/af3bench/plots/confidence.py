"""Confidence summary and PAE figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import style


def detect_failed(df: pd.DataFrame) -> set:
    """
    Flag likely-failed predictions.

    Combines two criteria so the result is robust even when failures are a large
    fraction of the conditions (where Tukey fences alone break down because the
    outliers inflate the IQR):

      1. Absolute AF3 confidence thresholds — ipTM/pTM < 0.4 or mean PAE > 20 Å
         indicate a model with little global/inter-chain confidence.
      2. Tukey fences (Q1 - 1.5 IQR on ipTM, Q3 + 1.5 IQR on PAE) — catch
         relative outliers when the batch is otherwise healthy.
    """
    failed: set = set()

    # 1. Absolute thresholds (robust to many failures)
    for _, row in df.iterrows():
        cond = row["condition"]
        iptm = row.get("iptm", float("nan"))
        ptm = row.get("ptm", float("nan"))
        pae = row.get("mean_pae", float("nan"))
        if math.isfinite(iptm) and iptm < 0.4:
            failed.add(cond)
        elif math.isfinite(ptm) and ptm < 0.4:
            failed.add(cond)
        if math.isfinite(pae) and pae > 20.0:
            failed.add(cond)

    # 2. Tukey fences (relative outliers)
    for col, side in (("iptm", "low"), ("mean_pae", "high")):
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 4:
            continue
        q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr = q3 - q1
        for _, row in df.iterrows():
            v = row.get(col, float("nan"))
            if not math.isfinite(v):
                continue
            if side == "low" and v < q1 - 1.5 * iqr:
                failed.add(row["condition"])
            if side == "high" and v > q3 + 1.5 * iqr:
                failed.add(row["condition"])
    return failed


def plot_confidence_summary(
    df: pd.DataFrame,
    plots_dir: Path,
    seed_sd: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Path]:
    """
    Grouped bar chart of pTM / ipTM / mean pLDDT with per-seed SD error bars.

    seed_sd: {condition: {"ptm": sd, "iptm": sd, "plddt_mean": sd}}
    """
    metrics = [
        ("ptm", "pTM", style.PALETTE[0]),
        ("iptm", "ipTM", style.PALETTE[4]),
        ("plddt_mean", "Mean pLDDT", style.PALETTE[2]),
    ]
    metrics = [m for m in metrics if m[0] in df.columns and df[m[0]].notna().any()]
    if not metrics:
        return []

    failed = detect_failed(df)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    names = df["condition"].tolist()
    labels = style.condition_labels(names)
    x = range(len(names))

    for ax, (col, lbl, color) in zip(axes, metrics):
        vals = df[col].tolist()
        bar_colors = [style.C_FAIL if names[i] in failed else color for i in range(len(names))]
        yerr = None
        if seed_sd:
            yerr = [seed_sd.get(n, {}).get(col, 0.0) or 0.0 for n in names]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", alpha=0.9,
                      yerr=yerr, capsize=4, error_kw={"linewidth": 1.1})
        if "is_reference" in df.columns:
            for bi in df.index[df["is_reference"]].tolist():
                bars[bi].set_hatch("//")
                bars[bi].set_linewidth(2)
        for i, v in enumerate(vals):
            if math.isfinite(v):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=40, ha="right")
        ax.set_ylabel(lbl)
        ax.set_title(lbl)

    notes = []
    if "is_reference" in df.columns and df["is_reference"].any():
        notes.append("hatched = reference")
    if failed:
        notes.append("red = likely failed")
    suffix = f"  ({', '.join(notes)})" if notes else ""
    fig.suptitle(f"Confidence metrics per condition{suffix}", fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "confidence_summary")


def plot_pae(df: pd.DataFrame, plots_dir: Path) -> List[Path]:
    if "mean_pae" not in df.columns or not df["mean_pae"].notna().any():
        return []
    names = df["condition"].tolist()
    labels = style.condition_labels(names)
    x = range(len(names))
    vals = df["mean_pae"].tolist()

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.1), 5))
    bars = ax.bar(x, vals, color=style.PALETTE[3], edgecolor="black", alpha=0.9)
    if "is_reference" in df.columns:
        for bi in df.index[df["is_reference"]].tolist():
            bars[bi].set_hatch("//")
    for i, v in enumerate(vals):
        if math.isfinite(v):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Mean PAE (Å)")
    ax.set_title("Mean PAE per condition", fontweight="bold")
    fig.tight_layout()
    return style.save(fig, plots_dir, "pae_comparison")
