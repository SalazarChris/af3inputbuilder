#!/usr/bin/env python3
"""
af3_condition_centric_extraction.py
===================================
Simplified condition-centric AF3 extraction.

Extracts metrics from *_confidences.json files, groups by condition,
and exports four CSV tables. Condition names are read from *_data.json
files inside each condition folder.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict


def parse_ranking_scores(ranking_file: Path) -> Dict[tuple, float]:
    """Parse ranking scores CSV and return dict mapping (seed, sample) -> score."""
    df = pd.read_csv(ranking_file)
    scores = {}
    for _, row in df.iterrows():
        scores[(int(row['seed']), int(row['sample']))] = float(row['ranking_score'])
    return scores


def extract_metrics(input_dir: str, output_dir: str = None, verbose: bool = True) -> Dict[str, Any]:
    """Run the complete extraction pipeline."""
    input_dir = Path(input_dir)
    
    if not output_dir:
        output_dir = input_dir.parent / "outputs" / f"{input_dir.name}_condition_centric"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    errors: List[str] = []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  AF3 Condition-Centric Extraction")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")
    
    # ----------------------------------------------------------------
    # Step 1: Discover conditions
    # ----------------------------------------------------------------
    conditions: Dict[str, Dict[str, Any]] = {}      # cond_id -> metadata
    condition_by_folder: Dict[str, str] = {}        # folder_name -> cond_id
    
    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.') or folder.name == 'outputs':
            continue
        
        # Find any *_data.json and read the 'name' field
        data_json_files = list(folder.glob("*_data.json"))
        condition_name = folder.name  # fallback
        
        if data_json_files:
            try:
                with open(data_json_files[0], 'r') as f:
                    data = json.load(f)
                condition_name = data.get('name', folder.name)
            except Exception as e:
                errors.append(f"Failed to read {data_json_files[0]}: {e}")
                if verbose:
                    print(f"  Warning: could not read {data_json_files[0]}: {e}")
        
        cond_id = f"cond_{len(conditions) + 1:03d}"
        conditions[cond_id] = {
            'condition_id': cond_id,
            'condition_name': condition_name,
            'folder': folder.name,
            'replicates': []
        }
        condition_by_folder[folder.name] = cond_id
        
        if verbose:
            print(f"  Discovered: {folder.name} -> {condition_name} ({cond_id})")
    
    if not conditions:
        print("  No condition folders found.")
        return {'conditions': 0, 'replicates': 0, 'errors': errors, 'output_dir': str(output_dir)}
    
    # ----------------------------------------------------------------
    # Step 2: Extract metrics from confidence files
    # ----------------------------------------------------------------
    confidence_files = list(input_dir.rglob("*_confidences.json"))
    
    if verbose:
        print(f"\n  Found {len(confidence_files)} confidence file(s)")
    
    # Load ranking scores from all *_ranking_scores.csv files
    ranking_scores: Dict[tuple, float] = {}  # (seed, sample) -> score
    ranking_files = list(input_dir.rglob("*_ranking_scores.csv"))
    for rank_file in ranking_files:
        if verbose:
            print(f"  Loading ranking scores from: {rank_file.name}")
        try:
            scores = parse_ranking_scores(rank_file)
            ranking_scores.update(scores)
        except Exception as e:
            errors.append(f"Failed to read {rank_file}: {e}")
            if verbose:
                print(f"  Warning: could not read {rank_file}: {e}")
    
    if verbose:
        print(f"  Loaded {len(ranking_scores)} ranking score(s) from {len(ranking_files)} file(s)")
    
    for i, conf_file in enumerate(confidence_files, 1):
        if verbose and (i == 1 or i % 100 == 0):
            print(f"  [{i}/{len(confidence_files)}] {conf_file.name}")
        
        # Resolve condition from top-level folder name
        try:
            rel = conf_file.relative_to(input_dir)
            folder_name = rel.parts[0]
        except (ValueError, IndexError):
            folder_name = conf_file.parent.name
        
        cond_id = condition_by_folder.get(folder_name)
        if not cond_id:
            errors.append(f"No condition for folder: {folder_name}")
            continue
        
        # Parse confidence JSON
        try:
            with open(conf_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            errors.append(f"Failed to read {conf_file}: {e}")
            continue
        
        replicate_id = conf_file.stem.replace('_confidences', '')
        
        # Extract seed and sample from replicate_id for ranking score matching
        seed, sample = None, None
        for part in replicate_id.split('_'):
            if part.startswith('seed-'):
                try:
                    seed = int(part.split('-')[1])
                except (ValueError, IndexError):
                    pass
            elif part.startswith('sample-'):
                try:
                    sample = int(part.split('-')[1])
                except (ValueError, IndexError):
                    pass
        
        # Get ranking score if available
        ranking_score = None
        if seed is not None and sample is not None:
            ranking_score = ranking_scores.get((seed, sample))
        # For aggregate files (no seed/sample), use average of all ranking scores
        elif not seed and not sample:
            if ranking_scores:
                ranking_score = float(np.mean(list(ranking_scores.values())))
        
        # --- Global metrics ---
        global_metrics = {}
        
        atom_plddts = data.get('atom_plddts', [])
        if atom_plddts:
            arr = np.array(atom_plddts)
            global_metrics['pLDDT_mean'] = float(np.mean(arr))
            global_metrics['pLDDT_max'] = float(np.max(arr))
            global_metrics['pLDDT_min'] = float(np.min(arr))
            global_metrics['pLDDT_median'] = float(np.median(arr))
        
        contact_probs = data.get('contact_probs', [])
        if contact_probs:
            arr = np.array(contact_probs)
            global_metrics['contact_prob_mean'] = float(np.mean(arr))
            global_metrics['contact_prob_max'] = float(np.max(arr))
            global_metrics['contact_prob_min'] = float(np.min(arr))
            global_metrics['contact_prob_median'] = float(np.median(arr))
        
        pae = data.get('pae', [])
        if pae:
            arr = np.array(pae)
            global_metrics['pae_mean'] = float(np.mean(arr))
            global_metrics['pae_max'] = float(np.max(arr))
            global_metrics['pae_min'] = float(np.min(arr))
            global_metrics['pae_median'] = float(np.median(arr))
        
        if ranking_score is not None:
            global_metrics['ranking_score'] = ranking_score
        
        # --- Chain metrics ---
        chains = []
        token_chain_ids = data.get('token_chain_ids', [])
        atom_chain_ids = data.get('atom_chain_ids', [])
        
        if token_chain_ids and atom_chain_ids and atom_plddts:
            token_counts = defaultdict(int)
            for cid in token_chain_ids:
                token_counts[cid] += 1
            
            for chain_id in sorted(token_counts.keys()):
                indices = [i for i, cid in enumerate(atom_chain_ids) if cid == chain_id]
                chain_plddts = [atom_plddts[i] for i in indices]
                avg_plddt = sum(chain_plddts) / len(chain_plddts) if chain_plddts else 0.0
                
                chains.append({
                    'chain_id': chain_id,
                    'residue_count': token_counts[chain_id],
                    'avg_plddt': round(avg_plddt, 2)
                })
        
        conditions[cond_id]['replicates'].append({
            'replicate_id': replicate_id,
            'global_metrics': global_metrics,
            'chains': chains
        })
    
    total_replicates = sum(len(c['replicates']) for c in conditions.values())
    if verbose:
        print(f"\n  Extracted {total_replicates} replicate(s) across {len(conditions)} condition(s)")
    
    # ----------------------------------------------------------------
    # Step 3: Export CSVs
    # ----------------------------------------------------------------
    if verbose:
        print("\n  Exporting CSV tables...")
    
    # 1. condition_registry.csv
    registry_rows = []
    for c in conditions.values():
        registry_rows.append({
            'condition_id': c['condition_id'],
            'condition_name': c['condition_name'],
            'n_replicates': len(c['replicates']),
            'replicate_ids': json.dumps([r['replicate_id'] for r in c['replicates']])
        })
    pd.DataFrame(registry_rows).to_csv(output_dir / "condition_registry.csv", index=False)
    if verbose:
        print(f"    condition_registry.csv  ({len(registry_rows)} rows)")
    
    # 2. metrics_replicates.csv
    rep_rows = []
    for c in conditions.values():
        for rep in c['replicates']:
            row = {
                'condition_id': c['condition_id'],
                'condition_name': c['condition_name'],
                'replicate_id': rep['replicate_id']
            }
            row.update(rep['global_metrics'])
            for chain in rep['chains']:
                row[f"chain_{chain['chain_id']}_residues"] = chain['residue_count']
                row[f"chain_{chain['chain_id']}_plddt"] = chain['avg_plddt']
            rep_rows.append(row)
    if rep_rows:
        pd.DataFrame(rep_rows).to_csv(output_dir / "metrics_replicates.csv", index=False)
        if verbose:
            print(f"    metrics_replicates.csv  ({len(rep_rows)} rows)")
    
    # 3. metrics_conditions.csv (aggregated)
    all_metric_names = sorted({
        k for c in conditions.values()
        for r in c['replicates']
        for k in r['global_metrics']
    })
    
    cond_rows = []
    for c in conditions.values():
        row = {
            'condition_id': c['condition_id'],
            'condition_name': c['condition_name'],
            'n_replicates': len(c['replicates'])
        }
        
        # Seed count
        seeds = set()
        for rep in c['replicates']:
            for part in rep['replicate_id'].split('_'):
                if part.startswith('seed-'):
                    try:
                        seeds.add(int(part.split('-')[1]))
                    except (ValueError, IndexError):
                        pass
        row['n_seeds'] = len(seeds)
        
        # Aggregate global metrics
        for mname in all_metric_names:
            values = [r['global_metrics'].get(mname) for r in c['replicates']]
            values = [v for v in values if v is not None]
            if values:
                arr = np.array(values)
                row[f'{mname}_mean'] = float(np.mean(arr))
                row[f'{mname}_sd'] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                row[f'{mname}_median'] = float(np.median(arr))
                row[f'{mname}_min'] = float(np.min(arr))
                row[f'{mname}_max'] = float(np.max(arr))
                row[f'{mname}_cv'] = float(np.std(arr, ddof=1) / np.mean(arr)) if np.mean(arr) != 0 else 0.0
            else:
                for suffix in ['_mean', '_sd', '_median', '_min', '_max', '_cv']:
                    row[f'{mname}{suffix}'] = np.nan
        
        # Aggregate chain metrics
        chain_data = defaultdict(lambda: defaultdict(list))
        for rep in c['replicates']:
            for chain in rep['chains']:
                chain_data[chain['chain_id']]['residue_count'].append(chain['residue_count'])
                chain_data[chain['chain_id']]['avg_plddt'].append(chain['avg_plddt'])
        
        for chain_id, metrics in chain_data.items():
            for mname, vals in metrics.items():
                if vals:
                    arr = np.array(vals)
                    row[f'chain_{chain_id}_{mname}_mean'] = float(np.mean(arr))
                    row[f'chain_{chain_id}_{mname}_sd'] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        
        cond_rows.append(row)
    
    if cond_rows:
        pd.DataFrame(cond_rows).to_csv(output_dir / "metrics_conditions.csv", index=False)
        if verbose:
            print(f"    metrics_conditions.csv  ({len(cond_rows)} rows)")
    
    # 4. condition_manifest.csv
    manifest_rows = []
    for c in conditions.values():
        seeds = set()
        for rep in c['replicates']:
            for part in rep['replicate_id'].split('_'):
                if part.startswith('seed-'):
                    try:
                        seeds.add(int(part.split('-')[1]))
                    except (ValueError, IndexError):
                        pass
        manifest_rows.append({
            'condition_id': c['condition_id'],
            'condition_name': c['condition_name'],
            'replicates': len(c['replicates']),
            'seeds': len(seeds),
            'status': 'Complete' if c['replicates'] else 'Empty'
        })
    pd.DataFrame(manifest_rows).to_csv(output_dir / "condition_manifest.csv", index=False)
    if verbose:
        print(f"    condition_manifest.csv  ({len(manifest_rows)} rows)")
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Done")
        print(f"  Conditions:  {len(conditions)}")
        print(f"  Replicates:  {total_replicates}")
        if errors:
            print(f"  Errors:      {len(errors)}")
            for e in errors[:5]:
                print(f"    - {e}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        print(f"{'='*60}\n")
    
    return {
        'conditions': len(conditions),
        'replicates': total_replicates,
        'errors': errors,
        'output_dir': str(output_dir)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simplified AF3 condition-centric extraction")
    parser.add_argument("input_dir", help="Input directory containing AF3 prediction outputs")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    result = extract_metrics(args.input_dir, args.output_dir, args.verbose)
    sys.exit(0 if not result['errors'] else 1)


if __name__ == "__main__":
    main()