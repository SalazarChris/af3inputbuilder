"""
Descriptive inspection of manifest experimental design.

This module produces human-readable summaries of the manifest structure.
It does NOT select statistical models or make biological interpretations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .manifest import MasterManifest


@dataclass
class ManifestInspection:
    """Descriptive summary of a condition manifest."""

    n_conditions: int = 0
    n_factors: int = 0
    factor_names: List[str] = field(default_factory=list)
    factor_levels: Dict[str, Set[str]] = field(default_factory=dict)
    factor_types: Dict[str, str] = field(default_factory=dict)
    n_modification_types: int = 0
    n_entity_types: Set[str] = field(default_factory=set)
    observed_combinations: List[Dict[str, str]] = field(default_factory=list)
    missing_combinations: List[Dict[str, str]] = field(default_factory=list)
    is_complete_factorial: bool = False
    expected_n_conditions: int = 0
    condition_groups: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Conditions: {self.n_conditions}",
            f"Factors: {self.n_factors}",
            "",
            "Factor details:",
        ]
        for name in self.factor_names:
            levels = sorted(self.factor_levels.get(name, set()))
            ftype = self.factor_types.get(name, "unknown")
            lines.append(f"  {name} ({ftype}): {len(levels)} level(s) = {levels}")

        lines.append("")

        # Modification summary
        lines.append(f"Modification types used: {self.n_modification_types}")

        # Entity summary
        if self.n_entity_types:
            lines.append(f"Entity types present: {sorted(self.n_entity_types)}")

        lines.append("")

        # Factorial structure
        if self.n_factors == 0:
            lines.append("No factors defined — conditions are unstructured.")
        elif self.is_complete_factorial:
            level_counts = " × ".join(
                str(len(self.factor_levels[a])) for a in self.factor_names
            )
            lines.append(f"Complete factorial: {level_counts} "
                        f"({self.n_conditions} conditions)")
        else:
            lines.append(f"Incomplete design: {self.n_conditions} observed "
                        f"out of {self.expected_n_conditions} possible combinations")
            if self.missing_combinations:
                lines.append(f"Missing combinations: {len(self.missing_combinations)}")

        # Condition groups
        if self.condition_groups:
            lines.append("")
            lines.append("Condition groups:")
            for group, count in sorted(self.condition_groups.items()):
                lines.append(f"  {group}: {count} conditions")

        return "\n".join(lines)


def inspect_manifest(manifest: MasterManifest) -> ManifestInspection:
    """Produce a descriptive inspection of the manifest design.

    Parameters
    ----------
    manifest : MasterManifest
        The loaded manifest.

    Returns
    -------
    ManifestInspection
        Descriptive summary.
    """
    insp = ManifestInspection()
    insp.n_conditions = len(manifest.conditions)

    # Collect factors
    factor_names = manifest.get_attribute_names()
    insp.factor_names = factor_names
    insp.n_factors = len(factor_names)

    for name in factor_names:
        levels = manifest.get_attribute_values(name)
        insp.factor_levels[name] = levels
        # All factors from the manifest are treated as categorical
        # (binary factors have 2 levels, multi-level have more)
        insp.factor_types[name] = "binary" if len(levels) == 2 else "categorical"

    # Collect modification types
    mod_classes: Set[str] = set()
    for mod in manifest.modifications.values():
        # We don't have the modification registry here, but we can count unique IDs
        mod_classes.add(mod.modification_id)
    insp.n_modification_types = len(mod_classes)

    # Collect entity types
    entity_types: Set[str] = set()
    for ent in manifest.entities.values():
        entity_types.add(ent.entity_type)
    insp.n_entity_types = entity_types

    # Observed combinations
    observed: List[Dict[str, str]] = []
    for cid in sorted(manifest.conditions.keys()):
        factors = manifest.get_factors_for_condition(cid)
        combo = {f.factor_name: f.factor_level for f in factors}
        combo["_condition_id"] = cid
        combo["_condition_name"] = manifest.conditions[cid].condition_name
        observed.append(combo)
    insp.observed_combinations = observed

    # Expected combinations (full factorial)
    if factor_names:
        level_lists = []
        for name in factor_names:
            levels = sorted(insp.factor_levels[name])
            level_lists.append([(name, v) for v in levels])

        all_combos = list(itertools.product(*level_lists))
        insp.expected_n_conditions = len(all_combos)

        # Missing combinations
        observed_keys = set()
        for combo in observed:
            key = tuple(
                (k, combo[k]) for k in factor_names if k in combo
            )
            observed_keys.add(key)

        missing = []
        for combo_tuple in all_combos:
            combo_dict = dict(combo_tuple)
            key = tuple((k, combo_dict[k]) for k in factor_names)
            if key not in observed_keys:
                missing.append(combo_dict)
        insp.missing_combinations = missing
        insp.is_complete_factorial = len(missing) == 0

    # Condition groups
    groups: Dict[str, int] = {}
    for cond in manifest.conditions.values():
        grp = cond.condition_group or "(ungrouped)"
        groups[grp] = groups.get(grp, 0) + 1
    insp.condition_groups = groups

    return insp
